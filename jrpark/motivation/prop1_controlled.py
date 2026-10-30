#!/usr/bin/env python3
"""명제 1: 같은 요청을 같은 상태에서 load / recompute로 처리해 비교하는 통제 실험.

사이클 (서버 재시작, 빈 캐시에서 시작)
  1. 준비   : 이 사이클의 측정 요청 P와 I/O 큐용 문서를 한 번씩 보냄 (출력 1토큰) → CPU tier + SSD에 저장
              SSD 쓰기가 끝날 때까지 fs tier 디렉터리 크기가 멈출 때까지 기다림
  2. 쓰기 차단: NO_SSD_STORE 플래그를 켬 → 이후 생기는 KV는 SSD 쓰기 큐를 만들지 않음 (쓰기 큐 = 0 고정)
  3. 채움   : 머리말 붙인 LEval 문서 FILL 토큰으로 GPU·CPU를 채워 P와 I/O 문서를 SSD에만 남김
  4. 그룹마다 세 변형을 무작위 순서로
       base      : 측정 요청 없음
       load      : P 그대로        → SSD hit (vLLM 기본 동작대로 load)
       recompute : 머리말 + P      → miss (같은 내용을 처음부터 계산; vLLM 0.23에는 요청별 load 끄기가 없음)
     시도 하나: decode 요청 N개 시작 → 모두 첫 토큰을 받은 뒤 1초 → t0에 I/O 문서(합 Q토큰, 각각 SSD hit) 전송
               → t0+0.05초에 측정 요청 → 측정·I/O 요청이 끝난 뒤 2초, 최소 15초 → decode 요청 취소 → 서버가 빌 때까지 대기

조건 (--grids)
  ssd : hit M, decode N ∈ {10, 30} × 읽기 큐 Q ∈ {0, 15K, 30K, 60K, 100K}                (decode = ShareGPT 첫 턴)
  gpu : hit M, Q ∈ {60K, 100K} × N ∈ {10, 20, 30, 60, 100}                                 (decode = ShareGPT 첫 턴)
  ctx : hit M, Q = 100K, N = 10, decode 문맥 C ∈ {512, 2048, 4096, 8192} 토큰               (decode = LEval 문서 앞부분)
        (N=10·Q=100K는 짧은 decode에서 recompute가 이기던 조건 → 문맥이 길어지면 load로 바뀌는지 봄.
         N=30 × 4K 이상은 GPU KV 32 GiB를 넘을 위험이 있어 피함)
  aux : hit S (N=10, Q ∈ {0,30K,60K,100K}), hit L (N=30, Q ∈ {0,30K})

출력 (--out 디렉터리)
  attempts.jsonl  : 시도마다 원시 시각 (측정 요청, I/O 요청, decode 요청별 청크 시각·누적 토큰 수)
  groups.csv      : 그룹마다 내 지연(TTFT), decode 요청들의 지연, 검증 결과  (--analyze-only 로 다시 만들 수 있음)
  plan.json       : 조건·그룹 순서

예)
  python3 prop1_controlled.py --dry-run --tokenizer $HOME/huggingface/Llama-3.1-8B-Instruct ...
  python3 prop1_controlled.py --grids ssd,gpu,ctx --reps 6 --out motiv3/prop1 ...
  python3 prop1_controlled.py --analyze-only --out motiv3/prop1
"""
import argparse, asyncio, json, os, random, statistics as st, sys, time

import aiohttp

from motiv_common import (Tok, Server, chat_prompt, clear_dir, doc_prompt, load_leval,
                          load_sharegpt_first_turns, load_sharegpt_pairs, run_batch, set_flag, stream_completion)

SIZES = {"S": 1024, "M": 6144, "L": 14336}
DEFAULT_STOP = ('pkill -f "launch.sh l" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; '
                'for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 1; done; '
                'pkill -9 -f "vllm serve" 2>/dev/null; rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null; true')


def build_conditions(grids):
    conds = []
    for g in grids:
        if g == "ssd":
            conds += [dict(size="M", N=n, Q=q, C=0) for n in (10, 30) for q in (0, 15000, 30000, 60000, 100000)]
        elif g == "gpu":
            conds += [dict(size="M", N=n, Q=q, C=0) for q in (60000, 100000) for n in (10, 20, 30, 60, 100)]
        elif g == "ctx":
            conds += [dict(size="M", N=10, Q=100000, C=c) for c in (512, 2048, 4096, 8192)]
        elif g == "pair":   # motivation 두 그림
            # A. GPU가 붐비면 load: hit M, 읽기 큐 60K 고정, decode 수 5→60 (decode 5에서는 recompute가 유리할 것으로 예상)
            conds += [dict(size="M", N=n, Q=60000, C=0) for n in (5, 10, 20, 30, 60)]
            # B. SSD가 붐비면 recompute: hit S(실제 SSD hit 중앙값 ~1K), decode 10 고정, 읽기 큐 0→100K
            conds += [dict(size="S", N=10, Q=q, C=0) for q in (0, 15000, 30000, 60000, 100000)]
        elif g == "aux":
            conds += [dict(size="S", N=10, Q=q, C=0) for q in (0, 30000, 60000, 100000)]
            conds += [dict(size="L", N=30, Q=q, C=0) for q in (0, 30000)]
        else:
            raise SystemExit(f"알 수 없는 grid: {g}")
    seen, out = set(), []
    for c in conds:  # 단면이 겹치는 조건은 한 번만
        k = (c["size"], c["N"], c["Q"], c["C"])
        if k not in seen:
            seen.add(k); out.append(c)
    return out


def plan_groups(conds, reps, gpc, seed):
    rng = random.Random(seed)
    groups = [dict(cond=c, rep=r) for c in conds for r in range(1, reps + 1)]
    rng.shuffle(groups)
    for gid, g in enumerate(groups):
        g["gid"] = gid
        g["cycle"] = gid // gpc
        order = ["base", "load", "recompute"]; rng.shuffle(order)
        g["order"] = order
    return groups


class Prompts:
    """사이클·그룹마다 겹치지 않는 프롬프트를 만든다. 앞머리 태그로 블록 hash를 서로 다르게 한다."""

    def __init__(self, tok, docs, chats, seed):
        self.tok, self.docs, self.chats = tok, docs, chats
        self.rng = random.Random(seed)

    def _doc(self):
        return self.rng.choice(self.docs)

    def measured(self, tag, size):
        d = self._doc()
        q = self.rng.choice(d["questions"])
        qn = self.tok.count(f"\n\nQuestion: {q}\nAnswer:")
        body = self.tok.truncate(d["doc"], max(64, SIZES[size] - qn - 8))
        return doc_prompt(f"[{tag}] {body}", q)

    def io_docs(self, tag, Q, chunk=10000):
        out, left, k = [], Q, 0
        while left > 0:
            d = self._doc()
            n = min(chunk, left, d["n_tok"])
            if n < 512:
                n = min(512, d["n_tok"])
            body = self.tok.truncate(d["doc"], n)
            out.append(doc_prompt(f"[{tag}-{k}] {body}", self.rng.choice(d["questions"])))
            left -= n; k += 1
        return out

    def decode(self, tag, N, C):
        """[(프롬프트, 출력 길이 또는 None)]. chats 항목이 (질문, 답 토큰 수)면 그 길이를 출력 길이로 쓴다."""
        out = []
        for i in range(N):
            if C:
                d = self._doc()
                body = self.tok.truncate(d["doc"], C)
                out.append((doc_prompt(f"[{tag}-{i}] {body}", "Summarize the document in detail."), None))
            else:
                c = self.rng.choice(self.chats)
                msg, n = c if isinstance(c, tuple) else (c, None)
                out.append((chat_prompt(f"[{tag}-{i}] " + msg), n))
        return out

    def fill(self, tag, total):
        out, acc, k = [], 0, 0
        while acc < total:
            d = self._doc()
            out.append(doc_prompt(f"[{tag}-{k}] {d['doc']}", d["questions"][0]))
            acc += d["n_tok"]; k += 1
        return out


def dir_bytes(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


async def wait_store_drain(path, stable_s, timeout, expect_bytes):
    """fs tier 디렉터리 크기가 stable_s초 동안 변하지 않을 때까지 기다린다 (SSD 쓰기 완료 판정)."""
    t_end, last, t_last = time.time() + timeout, -1, time.time()
    while time.time() < t_end:
        b = dir_bytes(path)
        if b != last:
            last, t_last = b, time.time()
        elif time.time() - t_last >= stable_s:
            break
        await asyncio.sleep(2)
    frac = last / expect_bytes if expect_bytes else float("nan")
    print(f"    SSD 저장: {last / 2**30:.1f} GiB (예상의 {frac:.0%})", flush=True)
    return last


async def attempt(sess, a, srv, g, variant, P, io, dec_prompts):
    """시도 하나. 측정 요청은 variant가 load면 P, recompute면 머리말+P, base면 없음."""
    url, model = srv.base + "/v1/completions", a.model
    m0 = await srv.metrics(sess)
    dec = [dict() for _ in dec_prompts]
    dtasks = [asyncio.create_task(stream_completion(sess, url, model, p, n or a.decode_max_tokens, r, ignore_eos=True,
                                                    timeout=a.complete_timeout + 600))
              for (p, n), r in zip(dec_prompts, dec)]
    t_wait = time.time() + a.decode_start_timeout
    while time.time() < t_wait and not all(r.get("t_first") or r.get("error") for r in dec):
        await asyncio.sleep(0.05)
    started = all(r.get("t_first") for r in dec)
    await asyncio.sleep(1.0)
    t0 = time.time()
    ios = [dict() for _ in io]
    itasks = [asyncio.create_task(stream_completion(sess, url, model, p, 1, r)) for p, r in zip(io, ios)]
    meas, mtask = dict(), None
    if variant != "base":
        await asyncio.sleep(0.05)
        prompt = P if variant == "load" else f"[rc-{g['gid']}] " + P
        mtask = asyncio.create_task(stream_completion(sess, url, model, prompt, a.measured_out if a.complete else 1,
                                                      meas, ignore_eos=a.complete))
    await asyncio.gather(*itasks, *( [mtask] if mtask else []))
    t_done = time.time()
    if a.complete:   # decode 요청이 모두 끝날 때까지 (E2E를 그대로 잰다)
        done, pend = await asyncio.wait(dtasks, timeout=a.complete_timeout)
        for t in pend:
            t.cancel()
        await asyncio.gather(*dtasks, return_exceptions=True)
        t_cut = time.time()
    else:
        await asyncio.sleep(max(2.0, a.min_window - (time.time() - t0)))
        t_cut = time.time()
        for t in dtasks:
            t.cancel()
        await asyncio.gather(*dtasks, return_exceptions=True)
    idle = await srv.wait_idle(sess, timeout=a.idle_timeout)
    m1 = await srv.metrics(sess)
    pre = (m1["preemptions"] - m0["preemptions"]) if (m0 and m1) else None
    slim = lambda r: dict({k: r.get(k) for k in ("t_send", "t_first", "prompt_tokens", "cached_tokens", "error")},
                          t_last=(r.get("tok_t") or [None])[-1], n_out=(r.get("tok_n") or [0])[-1])
    return dict(cycle=g["cycle"], gid=g["gid"], rep=g["rep"], variant=variant, **g["cond"],
                complete=bool(a.complete), dec_out=[n for _, n in dec_prompts], t0=t0, t_done=t_done, t_cut=t_cut, decode_started=started, idle=idle, preempt=pre,
                measured=slim(meas) if meas else None, io=[slim(r) for r in ios],
                decode=[dict(t_send=r.get("t_send"), t_first=r.get("t_first"), prompt_tokens=r.get("prompt_tokens"),
                             tok_t=r.get("tok_t", []), tok_n=r.get("tok_n", []), error=r.get("error")) for r in dec])


async def run_cycle(sess, a, srv, cyc, groups, pr, fout):
    print(f"== cycle {cyc}: 그룹 {[g['gid'] for g in groups]}", flush=True)
    srv.stop(); clear_dir(a.fs_dir); set_flag(a.no_store_flag, False)
    srv.start(f"c{cyc}"); await srv.wait_healthy(sess)
    if a.model == "auto":
        a.model = await srv.model_id(sess); print(f"  모델: {a.model}", flush=True)
    url = srv.base + "/v1/completions"
    # 1. 준비
    P, IO = {}, {}
    for g in groups:
        c = g["cond"]
        P[g["gid"]] = pr.measured(f"m-{cyc}-{g['gid']}", c["size"])
        for v in ("base", "load", "recompute"):
            IO[(g["gid"], v)] = pr.io_docs(f"io-{cyc}-{g['gid']}-{v}", c["Q"]) if c["Q"] else []
    prep = list(P.values()) + [p for v in IO.values() for p in v]
    print(f"  준비: {len(prep)}개 요청", flush=True)
    recs = await run_batch(sess, url, a.model, prep, 1, a.prep_concurrency)
    tok_prep = sum((r.get("prompt_tokens") or 0) for r in recs)
    bad = sum(1 for r in recs if r.get("error"))
    if bad:
        print(f"  [경고] 준비 요청 오류 {bad}개", flush=True)
    await srv.wait_idle(sess, timeout=a.idle_timeout)
    await wait_store_drain(a.fs_dir, a.store_stable, a.store_timeout, tok_prep * a.kv_bytes_per_token)
    # 2. 쓰기 차단, 3. 채움
    set_flag(a.no_store_flag, True)
    fill = pr.fill(f"fill-{cyc}", a.fill_tokens)
    print(f"  채움: {len(fill)}개 문서", flush=True)
    await run_batch(sess, url, a.model, fill, 1, a.prep_concurrency)
    await srv.wait_idle(sess, timeout=a.idle_timeout)
    await asyncio.sleep(a.settle)
    # 4. 시도
    for g in groups:
        c = g["cond"]
        dec_prompts = pr.decode(f"d-{cyc}-{g['gid']}", c["N"], c["C"])
        for v in g["order"]:
            r = await attempt(sess, a, srv, g, v, P[g["gid"]], IO[(g["gid"], v)], dec_prompts)
            fout.write(json.dumps(r) + "\n"); fout.flush()
            m = r["measured"] or {}
            ttft = (m.get("t_first") - m.get("t_send")) * 1000 if m.get("t_first") else float("nan")
            print(f"    g{g['gid']} {c} {v:9s} TTFT {ttft:8.0f} ms  cached {m.get('cached_tokens')}  preempt {r['preempt']}", flush=True)
    set_flag(a.no_store_flag, False)


# ---------------------------------------------------------------- 분석


def _after_t0(d, t0, t_cut):
    """decode 요청 하나: t0 이후 (경과 초, 누적 토큰 증가) 목록과 t0 시점 누적 토큰."""
    n0 = 0
    pts = []
    for t, n in zip(d["tok_t"], d["tok_n"]):
        if t <= t0:
            n0 = n
        elif t <= t_cut:
            pts.append((t - t0, n))
    return n0, [(dt, n - n0) for dt, n in pts]


def _time_to(pts, k):
    for dt, m in pts:
        if m >= k:
            return dt
    return None


def analyze(out_dir, ok_load=0.8, ok_rc=0.2, dec_max=2048):
    import csv
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "attempts.jsonl"))]
    by = {}
    for r in rows:
        by.setdefault(r["gid"], {})[r["variant"]] = r
    res = []
    for gid, vs in sorted(by.items()):
        if set(vs) != {"base", "load", "recompute"}:
            continue
        b = vs["base"]
        o = dict(cycle=b["cycle"], gid=gid, rep=b["rep"], size=b["size"], N=b["N"], Q=b["Q"], C=b["C"])
        reasons = []
        # decode 요청끼리 짝짓기: 같은 인덱스 = 같은 프롬프트
        per = {v: [_after_t0(d, vs[v]["t0"], vs[v]["t_cut"]) for d in vs[v]["decode"]] for v in vs}
        tot = {v: 0.0 for v in vs}
        n_pair = 0
        for i in range(b["N"]):
            ks = [per[v][i][1][-1][1] if per[v][i][1] else 0 for v in vs]
            k = min(ks)
            if k <= 0:
                continue
            ts = {v: _time_to(per[v][i][1], k) for v in vs}
            if any(t is None for t in ts.values()):
                continue
            n_pair += 1
            for v in vs:
                tot[v] += ts[v]
        o["n_dec_paired"] = n_pair
        o["decode_kv_tokens"] = sum((d.get("prompt_tokens") or 0) + per["base"][i][0] for i, d in enumerate(b["decode"]))
        o["q_actual"] = sum((x.get("prompt_tokens") or 0) for x in vs["load"]["io"])
        for v, key in (("load", "load"), ("recompute", "rc")):
            m = vs[v]["measured"] or {}
            o[f"{key}_ttft_ms"] = (m["t_first"] - m["t_send"]) * 1000 if m.get("t_first") else None
            o[f"{key}_dec_ms"] = (tot[v] - tot["base"]) * 1000
            o[f"{key}_cached"] = m.get("cached_tokens")
            o[f"{key}_prompt_tokens"] = m.get("prompt_tokens")
            o[f"{key}_io_ttft_sum_ms"] = sum(((x["t_first"] - x["t_send"]) * 1000) for x in vs[v]["io"] if x.get("t_first"))
        o["base_io_ttft_sum_ms"] = sum(((x["t_first"] - x["t_send"]) * 1000) for x in b["io"] if x.get("t_first"))
        complete = all(vs[v].get("complete") for v in vs)
        o["complete"] = complete
        if complete:
            # 요청별 E2E = 마지막 토큰 시각 − 보낸 시각. 평균은 측정 요청 1개 + decode 요청 N개 (base는 decode N개)
            def e2e(r):
                return (r["tok_t"][-1] - r["t_send"]) if r.get("tok_t") else None
            exp = vs["base"].get("dec_out") or []
            for v, key in (("load", "load"), ("recompute", "rc"), ("base", "base")):
                de = [e2e(d) for d in vs[v]["decode"]]
                m = vs[v]["measured"] or {}
                me = (m["t_last"] - m["t_send"]) if m.get("t_last") else None
                vals = [x for x in de if x is not None] + ([me] if me is not None else [])
                o[f"{key}_e2el_mean_s"] = sum(vals) / len(vals) if vals else None
                o[f"{key}_dec_e2el_mean_s"] = (sum(x for x in de if x is not None) / max(1, sum(x is not None for x in de)))
                if me is not None:
                    o[f"{key}_meas_e2e_s"] = me
            # 끝까지 못 간 decode, 측정 요청 전에 이미 끝난 decode가 있으면 무효
            n_short = sum(1 for v in vs for d, n in zip(vs[v]["decode"], exp)
                          if n and (not d["tok_n"] or d["tok_n"][-1] < n))
            n_early = sum(1 for v in ("load", "recompute") for d in vs[v]["decode"]
                          if d["tok_t"] and vs[v]["measured"] and d["tok_t"][-1] < vs[v]["measured"]["t_send"])
            o["n_dec_incomplete"], o["n_dec_done_before_meas"] = n_short, n_early
            if n_short:
                reasons.append(f"decode {n_short}개가 끝까지 못 감")
            if n_early:
                reasons.append(f"decode {n_early}개가 측정 요청 전에 끝남")
        # 검증
        if any(not vs[v]["decode_started"] for v in vs):
            reasons.append("decode 시작 실패")
        if any(vs[v]["preempt"] for v in vs):
            reasons.append("preemption")
        if o["load_ttft_ms"] is None or o["rc_ttft_ms"] is None:
            reasons.append("측정 요청 오류")
        exp = (o["load_prompt_tokens"] or SIZES[o["size"]])
        if o["load_cached"] is not None and o["load_cached"] < ok_load * exp:
            reasons.append(f"load가 hit이 아님(cached {o['load_cached']})")
        if o["rc_cached"] is not None and o["rc_cached"] > ok_rc * exp:
            reasons.append(f"recompute가 miss가 아님(cached {o['rc_cached']})")
        # decode 요청이 측정 창 안에서 max_tokens에 닿으면 KV가 가득 차 측정 요청이 그 완료를 기다린 것 (N이 클 때)
        n_fin = 0 if complete else sum(1 for v in vs for d in vs[v]["decode"] if d["tok_n"] and d["tok_n"][-1] >= dec_max)
        o["n_dec_finished"] = n_fin
        if n_fin:
            reasons.append(f"decode {n_fin}개가 측정 중 끝남(KV 포화 의심)")
        if n_pair < 0.8 * b["N"]:
            reasons.append(f"짝지은 decode {n_pair}/{b['N']}")
        o["valid"] = not reasons
        o["reason"] = "; ".join(reasons)
        if o["valid"]:
            o["win_mine"] = "load" if o["load_ttft_ms"] <= o["rc_ttft_ms"] else "rc"
            lt, rt = o["load_ttft_ms"] + o["load_dec_ms"], o["rc_ttft_ms"] + o["rc_dec_ms"]
            o["win_total"] = "load" if lt <= rt else "rc"
        res.append(o)
    path = os.path.join(out_dir, "groups.csv")
    keys = list(res[0].keys()) if res else []
    for r in res:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res)
    nv = sum(r["valid"] for r in res)
    print(f"→ {path}: 그룹 {len(res)}개 중 유효 {nv}개")
    if res and nv < len(res):
        from collections import Counter
        print("  제외 사유:", Counter(r["reason"] for r in res if not r["valid"]).most_common(6))
    return res


async def check_server(a):
    """서버를 한 번 띄워 health → 모델 이름 → 같은 요청 두 번(두 번째의 cached_tokens) 확인 후 내린다."""
    srv = Server(a.host, a.port, a.start_cmd, a.stop_cmd, os.path.join(a.out, "logs"))
    srv.stop()
    srv.start("check")
    try:
        async with aiohttp.ClientSession() as sess:
            t = time.time(); await srv.wait_healthy(sess)
            print(f"서버 기동 {time.time() - t:.0f}s")
            model = a.model if a.model != "auto" else await srv.model_id(sess)
            print(f"모델: {model}")
            p = "[check] " + "hello world " * 600
            r1, r2 = dict(), dict()
            await stream_completion(sess, srv.base + "/v1/completions", model, p, 4, r1)
            await stream_completion(sess, srv.base + "/v1/completions", model, p, 4, r2)
            if r1.get("error") or r2.get("error"):
                raise RuntimeError(f"요청 실패: {r1.get('error') or r2.get('error')}")
            print(f"요청 OK: prompt {r2.get('prompt_tokens')} 토큰, 두 번째 cached_tokens = {r2.get('cached_tokens')}"
                  + ("  (None이면 --enable-prompt-tokens-details 꺼짐: hit/miss 자동 검증 생략)" if r2.get("cached_tokens") is None else ""))
            m = await srv.metrics(sess)
            print(f"/metrics: {'OK' if m else '읽지 못함 (서버 유휴·preemption 판정 불가)'}")
    finally:
        srv.stop()


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="motiv3/prop1")
    ap.add_argument("--grids", default="ssd,gpu,ctx")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--groups-per-cycle", type=int, default=4)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--max-cycles", type=int, default=0, help="0이면 전부")
    ap.add_argument("--start-cycle", type=int, default=0, help="중단 후 이어서 돌릴 때")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=os.environ.get("MODEL", "auto"),
                    help="서버의 served model name. auto면 /v1/models의 첫 모델")
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/huggingface/Llama-3.1-8B-Instruct"))
    ap.add_argument("--leval", default=os.path.expanduser("~/huggingface/trace/LEval/LEval/*/*.jsonl"))
    ap.add_argument("--sharegpt", default=os.path.expanduser("~/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json"))
    ap.add_argument("--start-cmd", default=os.environ.get(
        "START_CMD", "CONNECTOR=policy POLICY=always_load MOTIV_LOG=1 ONLINE_HW=0 KV_GIB=32 CPU_GIB=32 MODEL=$HOME/huggingface/Llama-3.1-8B-Instruct DISK_ROOT=/tmp/kvcache/fs HF_HUB_OFFLINE=1 ./launch.sh l3"))
    ap.add_argument("--stop-cmd", default=os.environ.get("STOP_CMD", DEFAULT_STOP),
                    help="서버 종료 명령 (pkill -f 패턴이 이 드라이버의 명령줄과 겹치지 않게: START_CMD는 환경변수로 넘길 것)")
    ap.add_argument("--check-server", action="store_true", help="서버를 한 번 띄워 health·모델·요청 1개만 확인하고 끝")
    ap.add_argument("--fs-dir", default="/tmp/kvcache/fs")
    ap.add_argument("--no-store-flag", default="/tmp/NO_SSD_STORE")
    ap.add_argument("--fill-tokens", type=int, default=600000)
    ap.add_argument("--decode-max-tokens", type=int, default=2048)
    ap.add_argument("--decode-start-timeout", type=float, default=120)
    ap.add_argument("--min-window", type=float, default=15.0)
    ap.add_argument("--idle-timeout", type=float, default=300)
    ap.add_argument("--settle", type=float, default=5.0)
    ap.add_argument("--store-stable", type=float, default=15.0)
    ap.add_argument("--store-timeout", type=float, default=1200)
    ap.add_argument("--prep-concurrency", type=int, default=16)
    ap.add_argument("--kv-bytes-per-token", type=int, default=131072)
    ap.add_argument("--gpu-kv-tokens", type=int, default=262144, help="GPU KV 32 GiB / 128 KiB")
    ap.add_argument("--complete", action="store_true",
                    help="decode 요청을 끝까지 실행해 요청별 E2E를 그대로 잰다 (출력 길이 = ShareGPT 첫 답의 토큰 수)")
    ap.add_argument("--out-min", type=int, default=256, help="--complete: ShareGPT 답 길이 하한 (측정 요청이 올 때 decode 중이어야 함)")
    ap.add_argument("--out-max", type=int, default=1024, help="--complete: ShareGPT 답 길이 상한")
    ap.add_argument("--measured-out", type=int, default=32, help="--complete: 측정 요청의 출력 토큰 수 (LEval 기본 32)")
    ap.add_argument("--complete-timeout", type=float, default=600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    # 어디서 실행하든 motiv3/의 상위(~/jrpark/jr)를 기준으로: ./launch.sh, --out 상대 경로가 여기서 풀린다
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if a.analyze_only:
        analyze(a.out); return
    if a.check_server:
        asyncio.run(check_server(a)); return
    os.makedirs(a.out, exist_ok=True)
    conds = build_conditions(a.grids.split(","))
    groups = plan_groups(conds, a.reps, a.groups_per_cycle, a.seed)
    cycles = sorted({g["cycle"] for g in groups})
    json.dump(dict(args=vars(a), conds=conds, groups=groups), open(os.path.join(a.out, "plan.json"), "w"), indent=1)

    # 예산 점검: GPU KV (decode는 측정 창 동안 약 500토큰 더 생성한다고 봄), 사이클당 SSD 사용량
    print(f"조건 {len(conds)}개 × 반복 {a.reps} = 그룹 {len(groups)}개, 사이클 {len(cycles)}개")
    for c in conds:
        ctx = c["C"] or 300
        need = c["N"] * (ctx + 500) + c["Q"] + SIZES[c["size"]]
        flag = "  ← GPU KV 부족 위험" if need > 0.9 * a.gpu_kv_tokens else ""
        print(f"  {c}: 시도 중 KV 약 {need / 1000:.0f}K 토큰 / {a.gpu_kv_tokens / 1000:.0f}K{flag}")
    worst = max(sum(SIZES[g["cond"]["size"]] + 3 * g["cond"]["Q"] for g in groups if g["cycle"] == cy) for cy in cycles)
    print(f"  사이클당 SSD 최대 약 {worst * a.kv_bytes_per_token / 2**30:.0f} GiB 쓰기")
    if a.dry_run:
        return

    tok = Tok(a.tokenizer)
    docs = load_leval(a.leval, tok)
    if a.complete:
        pairs = load_sharegpt_pairs(a.sharegpt)
        chats = []
        for q, r in pairs:
            n = tok.count(r)
            if a.out_min <= n <= a.out_max:
                chats.append((q, n))
            if len(chats) >= 5000:
                break
        ns = sorted(n for _, n in chats)
        print(f"ShareGPT 출력 길이 ({a.out_min}~{a.out_max}토큰, {len(ns)}개): 중앙값 {ns[len(ns) // 2]}, "
              f"p90 {ns[int(len(ns) * 0.9)]}", flush=True)
    else:
        chats = load_sharegpt_first_turns(a.sharegpt)
    pr = Prompts(tok, docs, chats, a.seed)
    srv = Server(a.host, a.port, a.start_cmd, a.stop_cmd, os.path.join(a.out, "logs"))

    async def go():
        async with aiohttp.ClientSession() as sess:
            with open(os.path.join(a.out, "attempts.jsonl"), "a") as fout:
                todo = [c for c in cycles if c >= a.start_cycle]
                if a.max_cycles:
                    todo = todo[:a.max_cycles]
                for cy in todo:
                    await run_cycle(sess, a, srv, cy, [g for g in groups if g["cycle"] == cy], pr, fout)
            srv.stop()

    try:
        asyncio.run(go())
    finally:
        set_flag(a.no_store_flag, False)
    analyze(a.out)


if __name__ == "__main__":
    main()