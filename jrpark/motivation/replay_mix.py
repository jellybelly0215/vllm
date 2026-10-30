#!/usr/bin/env python3
"""motivation 1부: 긴 문서 QA(LEval) + 채팅(ShareGPT) 혼합 워크로드 재생기.

native vLLM의 OpenAI 호환 서버에 요청을 보내고 (기본 /v1/completions, 본 실험과 같은 문자열 프롬프트),
요청마다 보낸 시각, 첫 토큰 시각, 이후 토큰(스트림 조각) 도착 시각을 jsonl로 남긴다.
첫 줄은 {"meta": {"t0": 시작 절대 시각}}이고, 이후 시각은 모두 t0 기준 상대 시각(초).
서버 로그와 맞추려면 클라이언트를 서버와 같은 머신에서 실행한다(같은 시계).

워크로드 구성
- 문서 요청: 도착률 --doc-rate (요청/s, Poisson). 활성 문서 --doc-pool개 중 하나를 골라
  다음 질문을 보낸다 (풀 운영은 본 실험 build_longctx_workload.py와 같음). 프롬프트 = 문서 + 질문 → 같은 문서의 이전 질문과 문서 부분이 겹쳐 hit이 생기고,
  시간이 지나 GPU·CPU에서 밀려나면 SSD hit이 된다. 질문을 다 쓴 문서는 새 문서로 바뀐다.
- 대화 세션: 시작률 --chat-rate (세션/s, Poisson). 세션 안의 턴은 앞 턴이 끝난 뒤
  생각 시간(평균 --think-mean 초 지수분포, 하한 --think-min 초)이 지나면 보낸다. 이전 턴은 데이터셋의 응답으로 채워
  프롬프트가 실행마다 같다.
- 출력 길이: 데이터셋 응답 길이(out_len)를 max_tokens로, ignore_eos로 고정.
  도착 시각·문서 선택·질문 순서·생각 시간은 --seed로 미리 정해 실행마다 같다.

사용 예:
  python3 replay_mix.py --url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
      --sessions sessions_mix.jsonl --doc-rate 0.8 --chat-rate 1.0 --duration 1800 --out mix_r1.jsonl
"""
import argparse, asyncio, json, random, sys, time

import aiohttp


def build_schedule(sessions, a):
    rng = random.Random(a.seed)
    docs = [s for s in sessions if s["kind"] == "doc"]
    chats = [s for s in sessions if s["kind"] == "chat"]   # 대화 세션은 prep_mix.py가 만든 순서 그대로 씀
    if not docs or not chats:
        sys.exit("문서 세션과 대화 세션이 모두 있어야 함")

    # 문서 요청: (도착 시각, 문서, 질문 번호, 출력 길이, 머리말)
    # 본 실험 build_longctx_workload.py와 같은 풀 운영: 활성 문서 --doc-pool개, 질문을 다 쓴 문서는 빠지고
    # backlog에서 채움. --recycle이면 backlog와 풀이 모두 비었을 때 전체 문서를 다시 섞어 backlog로.
    backlog = list(docs); rng.shuffle(backlog)
    pool = []                                   # [문서, 다음 질문 번호]
    doc_reqs, t = [], 0.0
    while a.doc_rate > 0:
        t += rng.expovariate(a.doc_rate)
        if t >= a.duration:
            break
        if a.recycle and not backlog and not pool:
            backlog = list(docs); rng.shuffle(backlog)
        while backlog and len(pool) < a.doc_pool:
            pool.append([backlog.pop(), 0])
        if not pool:
            break
        k = rng.randrange(len(pool))
        d, qi = pool[k]
        # 확률 new_frac로 새 세션 머리말(→ miss), 확률 long_frac로 long_output
        header = f"[session {rng.getrandbits(64):016x}]\n" if (a.doc_new_frac > 0 and rng.random() < a.doc_new_frac) else ""
        if a.doc_out > 0:
            out = a.doc_long_out if (a.doc_long_frac > 0 and rng.random() < a.doc_long_frac) else a.doc_out
        else:
            out = d["turns"][qi]["out_len"]
        doc_reqs.append((t, d, qi, out, header))
        pool[k][1] += 1
        if pool[k][1] >= len(d["turns"]):      # 질문을 다 쓴 문서는 풀에서 빠짐
            pool.pop(k)

    # 대화 세션: (시작 시각, 세션, 턴별 생각 시간)
    chat_starts, t, ci = [], 0.0, 0
    while a.chat_rate > 0:
        t += rng.expovariate(a.chat_rate)
        if t >= a.duration:
            break
        s = chats[ci % len(chats)]; ci += 1
        thinks = [max(a.think_min, rng.expovariate(1.0 / a.think_mean)) if a.think_mean > 0 else a.think_min
                  for _ in s["turns"]]
        chat_starts.append((t, s, thinks, ci))
    if ci > len(chats):
        print(f"[경고] 대화 세션 {ci}개 필요, 데이터 {len(chats)}개 → 재사용됨(prefix가 겹칠 수 있음)", file=sys.stderr)
    return doc_reqs, chat_starts


class Runner:
    def __init__(self, a):
        self.a = a
        self.fh = open(a.out, "w", encoding="utf-8")
        self.inflight = {"doc": 0, "doc_new": 0, "chat": 0}
        self.done = {"doc": 0, "doc_new": 0, "chat": 0}
        self.fail = 0
        self.fail_run = 0
        self.t0 = None
        self.abort = asyncio.Event()

    def now(self):
        return time.time() - self.t0

    async def send(self, http, kind, sid, turn, prompt, out_len, t_plan):
        """prompt: completions API면 문자열, chat API면 메시지 목록."""
        a = self.a
        body = {"model": a.model, "max_tokens": out_len, "temperature": 0.0,
                "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
        if a.api == "chat":
            body["messages"] = prompt; path = "/v1/chat/completions"
        else:
            body["prompt"] = prompt; path = "/v1/completions"
        rec = {"kind": kind, "sid": sid, "turn": turn, "t_plan": round(t_plan, 4), "max_tokens": out_len}
        self.inflight[kind] += 1
        rec["t_send"] = self.now()
        chunks = []
        try:
            async with http.post(a.url + path, json=body,
                                 timeout=aiohttp.ClientTimeout(total=a.timeout)) as r:
                if r.status != 200:
                    rec["status"] = f"http {r.status}: {(await r.text())[:200]}"
                else:
                    async for raw in r.content:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        ev = json.loads(data)
                        rec.setdefault("rid", ev.get("id"))
                        if ev.get("usage"):
                            rec["prompt_tokens"] = ev["usage"].get("prompt_tokens")
                            rec["completion_tokens"] = ev["usage"].get("completion_tokens")
                            det = ev["usage"].get("prompt_tokens_details") or {}
                            if det.get("cached_tokens") is not None:
                                rec["cached_tokens"] = det["cached_tokens"]
                        for ch in ev.get("choices") or []:
                            if ch.get("text") or (ch.get("delta") or {}).get("content"):
                                chunks.append(round(self.now(), 4))
                    rec["status"] = "ok"
        except Exception as e:  # noqa: BLE001 - 실패도 기록으로 남김
            rec["status"] = f"error {type(e).__name__}: {e}"[:300]
        rec["t_end"] = self.now()
        rec["t_chunks"] = chunks
        rec["t_first"] = chunks[0] if chunks else None
        self.inflight[kind] -= 1
        self.done[kind] += 1
        if rec["status"] != "ok":
            self.fail += 1
            self.fail_run += 1
            if self.fail <= 3:
                print(f"[실패 예] {rec['status']}", file=sys.stderr, flush=True)
        else:
            self.fail_run = 0
        if sum(self.done.values()) == 20 and self.fail == 20:   # 처음 20개가 전부 실패 → 설정 문제
            print(f"[중단] 처음 20개 요청이 모두 실패: {rec['status']}\n"
                  f"       서버 주소(--url {self.a.url})와 모델 이름(--model)을 확인", file=sys.stderr, flush=True)
            self.abort.set()
        elif self.fail_run == self.a.max_fail_run:            # 도중에 연속 실패 → 서버가 죽었을 가능성
            print(f"[중단] 연속 {self.fail_run}건 실패 (마지막: {rec['status']}) → 서버 로그 확인",
                  file=sys.stderr, flush=True)
            self.abort.set()
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fh.flush()
        return rec

    async def doc_task(self, http, t, d, qi, out, header):
        await asyncio.sleep(max(0.0, t - self.now()))
        turn = d["turns"][qi]
        text = header + d["context"] + "\n\nQuestion: " + turn["q"] + "\nAnswer:"   # 본 실험 build_longctx_workload.py와 같은 형식
        prompt = text if self.a.api == "completions" else [{"role": "user", "content": text}]
        await self.send(http, "doc_new" if header else "doc", d["sid"], qi, prompt, out, t)

    async def chat_task(self, http, t, s, thinks, idx):
        await asyncio.sleep(max(0.0, t - self.now()))
        a = self.a
        sid = f'{s["sid"]}#{idx}'
        hist = [] if a.api == "chat" else ""
        for k, turn in enumerate(s["turns"]):
            if k > 0:
                if self.now() + thinks[k] >= a.duration:   # 다음 턴이 재생 시간을 넘기면 기다리지 않고 끝냄
                    return
                await asyncio.sleep(thinks[k])
            if a.api == "chat":
                prompt = hist + [{"role": "user", "content": turn["user"]}]
                await self.send(http, "chat", sid, k, prompt, turn["out_len"], self.now())
                hist = prompt + [{"role": "assistant", "content": turn["assistant"]}]
            else:                                      # 이전 턴을 이어 붙인 문자열 (앞부분이 그대로 겹쳐 hit이 생김)
                prompt = hist + a.user_fmt.format(user=turn["user"])
                await self.send(http, "chat", sid, k, prompt, turn["out_len"], self.now())
                hist = prompt + a.asst_fmt.format(assistant=turn["assistant"])

    async def progress(self):
        while True:
            await asyncio.sleep(self.a.report)
            print(f"[{self.now():7.0f}s] 진행 중 문서 {self.inflight['doc'] + self.inflight['doc_new']} 대화 {self.inflight['chat']} | "
                  f"완료 문서 {self.done['doc']}+새 {self.done['doc_new']} 대화 {self.done['chat']} | 실패 {self.fail}", flush=True)

    async def run(self, doc_reqs, chat_starts):
        # 요청마다 새 연결: 서버(uvicorn)가 쉬는 연결을 닫는 순간 재사용하면 Connection reset이 남
        conn = aiohttp.TCPConnector(limit=0, force_close=True)
        async with aiohttp.ClientSession(connector=conn) as http:
            self.t0 = time.time()
            # 첫 줄: 시작 시각(절대) — 서버 로그(time.time())와 맞출 때 t_abs = t0 + t_rel
            self.fh.write(json.dumps({"meta": {"t0": self.t0, "args": vars(self.a)}}, ensure_ascii=False) + "\n")
            self.fh.flush()
            prog = asyncio.create_task(self.progress())
            tasks = [asyncio.create_task(self.doc_task(http, *x)) for x in doc_reqs]
            tasks += [asyncio.create_task(self.chat_task(http, *x)) for x in chat_starts]
            allt = asyncio.gather(*tasks)
            stop = asyncio.create_task(self.abort.wait())
            done, _ = await asyncio.wait({allt, stop}, timeout=self.a.duration + self.a.drain,
                                         return_when=asyncio.FIRST_COMPLETED)
            if allt not in done:
                if stop in done:
                    print("[중단] 남은 요청 취소", file=sys.stderr)
                else:
                    print(f"[경고] drain {self.a.drain}s 초과 → 남은 요청 취소", file=sys.stderr)
                for x in tasks:
                    x.cancel()
                try:
                    await allt                         # 취소 결과를 회수 (경고 방지)
                except BaseException:
                    pass
            stop.cancel()
            prog.cancel()
        self.fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api", choices=["completions", "chat"], default="completions",
                    help="본 실험과 같은 /v1/completions(문자열 프롬프트)가 기본")
    # 본 실험 build_longctx_workload.py의 --output-tokens / --long-output / --long-frac
    # 기본값 = 본 실험 run_lio.sh: --output-tokens 32 --long-frac 0.12 --long-output 1024 --recycle
    ap.add_argument("--doc-out", type=int, default=32, help="문서 질문의 출력 길이 (0이면 데이터셋 정답 길이)")
    ap.add_argument("--doc-long-out", type=int, default=1024, help="긴 출력 길이")
    ap.add_argument("--doc-long-frac", type=float, default=0.12, help="긴 출력을 받는 문서 질문 비율")
    ap.add_argument("--no-recycle", dest="recycle", action="store_false",
                    help="문서가 모두 떨어지면 재생을 멈춤 (기본은 전체 문서를 다시 섞어 재사용, 본 실험 --recycle)")
    ap.add_argument("--doc-new-frac", type=float, default=0.0,
                    help="새 세션 머리말을 붙여 miss로 만드는 문서 질문 비율 (본 실험 --new-frac)")
    # 본 실험 build_sharegpt_workload.py와 같은 형식: prompt += f"User: {q}\nAssistant:" → 응답 뒤 f" {ans}\n\n"
    ap.add_argument("--user-fmt", default="User: {user}\nAssistant:", help="completions API에서 대화 턴 형식")
    ap.add_argument("--asst-fmt", default=" {assistant}\n\n", help="completions API에서 이전 턴 응답 형식")
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--doc-rate", type=float, required=True, help="문서 요청/s")
    ap.add_argument("--chat-rate", type=float, required=True, help="대화 세션/s")
    ap.add_argument("--doc-pool", type=int, default=64, help="동시에 질문을 받는 활성 문서 수 (본 실험 POOL=64)")
    ap.add_argument("--think-mean", type=float, default=60.0, help="대화 턴 사이 평균 생각 시간(초, 본 실험 60)")
    ap.add_argument("--think-min", type=float, default=5.0, help="생각 시간 하한(초, 본 실험 5)")
    ap.add_argument("--duration", type=float, default=1800)
    ap.add_argument("--drain", type=float, default=300, help="재생 끝난 뒤 진행 중 요청을 기다리는 시간")
    ap.add_argument("--timeout", type=float, default=600, help="요청 하나의 최대 시간")
    ap.add_argument("--report", type=float, default=30)
    ap.add_argument("--max-fail-run", type=int, default=30, help="연속 실패가 이만큼이면 중단")
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sessions = [json.loads(l) for l in open(a.sessions, encoding="utf-8") if l.strip()]
    doc_reqs, chat_starts = build_schedule(sessions, a)
    n_chat_turns = sum(len(s["turns"]) for _, s, _, _ in chat_starts)
    print(f"예정: 문서 요청 {len(doc_reqs)}개, 대화 세션 {len(chat_starts)}개(최대 {n_chat_turns}턴), "
          f"재생 {a.duration:.0f}s", flush=True)
    asyncio.run(Runner(a).run(doc_reqs, chat_starts))
    print("끝:", a.out)


if __name__ == "__main__":
    main()