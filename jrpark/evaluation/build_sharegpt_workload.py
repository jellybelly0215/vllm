#!/usr/bin/env python3
"""ShareGPT 멀티턴 대화 워크로드 (CachedAttention/AttentionStore와 같은 데이터셋).
턴 k 프롬프트 = 이전 대화 전체 (User/Assistant) + 새 User 메시지. Assistant 부분은 데이터셋의 실제 답변을 넣는다
(arm 간 프롬프트 동일). 턴 k+1 프롬프트는 턴 k 프롬프트 + 답변 + 새 질문 → 이전 턴 전체가 prefix hit.
세션은 Poisson 도착, 턴 사이 생각 시간 ~ 지수분포. replay의 세션 순서 대기(prev_id)로 closed-loop.
출력 형식은 build_multiturn_workload.py와 같음.
사용:
  python3 build_sharegpt_workload.py --src ~/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
     --model ~/huggingface/Llama-3.1-8B-Instruct --session-rps 0.3 --duration 1200 --out sg_s0.3.jsonl
"""
import argparse, json, random, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--min-turns", type=int, default=3); ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--min-ctx", type=int, default=1024, help="마지막 턴 프롬프트가 이보다 짧은 대화는 제외")
    ap.add_argument("--max-input", type=int, default=30000)
    ap.add_argument("--session-rps", type=float, required=True)
    ap.add_argument("--think-mean", type=float, default=60.0); ap.add_argument("--think-min", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=1200)
    ap.add_argument("--out-min", type=int, default=16); ap.add_argument("--out-max", type=int, default=512)
    ap.add_argument("--max-convs", type=int, default=20000, help="토큰화할 대화 수 상한 (속도)")
    ap.add_argument("--kv-gib", type=float, default=32); ap.add_argument("--cpu-gib", type=float, default=32)
    ap.add_argument("--seed", type=int, default=51); ap.add_argument("--out", required=True)
    ap.add_argument("--fake-tokenizer", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.fake_tokenizer:
        ntok = lambda s: max(1, len(s) // 4)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    data = json.load(open(a.src))
    rng = random.Random(a.seed); rng.shuffle(data)
    sessions = []
    for c in data[: a.max_convs]:
        msgs = c.get("conversations") or []
        pairs, i = [], 0
        while i + 1 < len(msgs):                                   # human → gpt 쌍만 순서대로
            h, g = msgs[i], msgs[i + 1]
            if h.get("from") == "human" and g.get("from") == "gpt" and h.get("value", "").strip() and g.get("value", "").strip():
                pairs.append((h["value"].strip(), g["value"].strip())); i += 2
            else:
                break
        pairs = pairs[: a.max_turns]
        if len(pairs) < a.min_turns: continue
        prompt, turns = "", []
        for q, ans in pairs:
            prompt = prompt + f"User: {q}\nAssistant:"
            n = ntok(prompt)
            if n > a.max_input: break
            turns.append((prompt, n, max(a.out_min, min(a.out_max, ntok(ans)))))
            prompt = prompt + f" {ans}\n\n"
        if len(turns) < a.min_turns or turns[-1][1] < a.min_ctx: continue
        sessions.append((c.get("id", f"sg{len(sessions)}"), turns))
    if not sessions: sys.exit("조건에 맞는 대화 없음")
    print(f"대화 {len(sessions)}개 사용 가능, 평균 턴 {sum(len(t) for _, t in sessions)/len(sessions):.1f}, "
          f"마지막 턴 길이 중앙값 {sorted(t[-1][1] for _, t in sessions)[len(sessions)//2]} tok")

    reqs, t, used = [], 0.0, 0
    while True:
        t += rng.expovariate(a.session_rps)
        if t >= a.duration: break
        if used >= len(sessions): print("[경고] 대화 소진"); break
        sid, turns = sessions[used]; used += 1
        tt = t
        for k, (prompt, n, out) in enumerate(turns):
            think = 0.0 if k == 0 else max(a.think_min, rng.expovariate(1.0 / a.think_mean))
            tt += think
            reqs.append(dict(timestamp=int(tt * 1000), doc_id=f"sharegpt#{sid}", task="sharegpt", q_idx=k, session=used - 1,
                             turn=k, think_s=round(think, 2), prompt=prompt, input_length=n, output_length=out,
                             phase_rps=a.session_rps))
    reqs.sort(key=lambda r: r["timestamp"])
    key = {(r["session"], r["turn"]): i for i, r in enumerate(reqs)}
    for i, r in enumerate(reqs):
        r["id"] = i; r["prev_id"] = key.get((r["session"], r["turn"] - 1)) if r["turn"] > 0 else None
    with open(a.out, "w") as f:
        f.write(json.dumps({"_meta": vars(a) | {"n_sessions": used, "n_requests": len(reqs)}}) + "\n")
        for r in reqs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    T = a.duration; n_req = len(reqs)
    inp = sorted(r["input_length"] for r in reqs); q = lambda p: inp[min(len(inp) - 1, int(p * len(inp)))]
    new = [r["input_length"] - (reqs[r["prev_id"]]["input_length"] if r["prev_id"] is not None else 0) for r in reqs]
    reuse = 1 - sum(new) / sum(inp)
    sess_len = sum(r["think_s"] for r in reqs) / max(used, 1)                     # 세션당 평균 생각 시간 합
    active = a.session_rps * sess_len                                             # 동시에 살아 있는 세션 수 (Little)
    ctx = sum(r["input_length"] for r in reqs if r["turn"] > 0) / max(1, sum(r["turn"] > 0 for r in reqs))
    ws_tok = active * ctx; mem_tok = (a.kv_gib + a.cpu_gib) * 2**30 / 131072
    print(f"세션 {used}개, 요청 {n_req}개 ({n_req/T:.2f} req/s), 턴1 비율 {sum(r['turn']==0 for r in reqs)/n_req:.0%}")
    print(f"입력 p10/p50/p90: {q(.1)}/{q(.5)}/{q(.9)} tok, 출력 평균 {sum(r['output_length'] for r in reqs)/n_req:.0f}")
    print(f"재사용 가능 토큰 {reuse:.1%}, 요청당 새 토큰(쓰기) 평균 {sum(new)/n_req:,.0f}")
    print(f"작업 집합 추정: 동시 세션 ≈ {active:.0f}개 × 평균 문맥 {ctx:,.0f} tok ≈ {ws_tok/1e3:,.0f}K tok "
          f"vs GPU+CPU {mem_tok/1e3:,.0f}K tok → {'SSD hit 많음' if ws_tok > 2 * mem_tok else ('일부 SSD hit' if ws_tok > mem_tok else 'SSD hit 거의 없음 (메모리에 다 들어감)')}")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()