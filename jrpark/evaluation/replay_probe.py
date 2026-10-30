#!/usr/bin/env python3
"""Step 2: recompute-vs-load probes under real background load (Mooncake trace replay).

Phases
  1. engine warm-up (a few requests)
  2. probe pre-warm: every probe prompt is sent once so its KV reaches the SSD tier
  3. replay: background requests from the trace at scaled arrival times, streaming,
     recording per-token arrival timestamps; probes injected every --probe-interval s,
     alternating policy load / recompute over the h grid.

Outputs (prefix = --tag)
  <tag>_bg.jsonl     one line per background request:
                     {id, resp_id, t_send, t_first, tok_dt_ms:[...], input_len, output_len, n_hash}
  <tag>_probes.csv   h, policy, t_send, ttft_ms, total_ms, resp_id, probe_idx
Join with the server-side probe_log.jsonl on resp_id (server req_id starts with it)
to get (tier, num_computed, occ_reqs, occ_tokens, q_wait, q_io_*) at decision time.

Usage:
  python3 replay_probe.py --model $HOME/huggingface/Llama-3.1-8B-Instruct \
      --trace ~/huggingface/trace/mooncake_trace.jsonl --rate-scale 1.0 --duration 600 \
      --h 2048 4096 8192 16384 24576 32768 --probe-interval 4 --tag mc_x1
"""
import argparse, csv, itertools, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests

BLOCK_TRACE = 512      # tokens per Mooncake hash_id block
VOCAB_LO, VOCAB_HI = 1000, 30000

def rand_ids(n, seed):
    rng = random.Random(seed)
    return [rng.randint(VOCAB_LO, VOCAB_HI) for _ in range(n)]

_block_cache: dict[int, list[int]] = {}
def block_ids(hash_id):
    b = _block_cache.get(hash_id)
    if b is None:
        b = _block_cache[hash_id] = rand_ids(BLOCK_TRACE, 7_000_000 + hash_id)
    return b

def trace_prompt(rec, uniq_seed):
    """Token ids reproducing the trace's prefix structure: shared blocks by hash_id,
    remainder unique to this request."""
    ids = []
    for hid in rec.get("hash_ids", []):
        ids.extend(block_ids(hid))
    n = rec["input_length"]
    if len(ids) > n:
        ids = ids[:n]
    elif len(ids) < n:
        ids.extend(rand_ids(n - len(ids), uniq_seed))
    return ids

class Client:
    def __init__(self, url, model):
        self.url, self.model = url.rstrip("/"), model
        self.sess = requests.Session()
        a = requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=256)
        self.sess.mount("http://", a)

    def stream(self, ids, max_tokens, policy=None, timeout=1800, h=None):
        body = {"model": self.model, "prompt": ids, "max_tokens": max_tokens,
                "temperature": 0, "ignore_eos": True, "stream": True}
        if policy:
            body["kv_transfer_params"] = {"policy": policy, "h": int(h or 0)}   # h lets the online learner attribute a forced-recompute probe
        t_send = time.time(); times = []; rid = None
        with self.sess.post(f"{self.url}/v1/completions", json=body, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                times.append(time.time())
                if rid is None:
                    try: rid = json.loads(line[5:]).get("id")
                    except Exception: pass
        return t_send, times, rid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", default=None, help="Mooncake-format trace (hash_ids)")
    ap.add_argument("--workload", default=None,
                    help="long-context workload from build_longctx_workload.py (text prompts). "
                         "Timestamps are already Poisson-spaced; --rate-scale still applies.")
    ap.add_argument("--rate-scale", type=float, default=1.0, help="1.0 = trace timing; 2.0 = twice the request rate")
    ap.add_argument("--start-offset", type=float, default=0.0, help="skip first N seconds of (scaled) replay time")
    ap.add_argument("--trace-offset", type=float, default=0.0, help="skip first N seconds of TRACE time (e.g. 600 = start at trace minute 10)")
    ap.add_argument("--no-spread", dest="spread", action="store_false",
                    help="keep quantized trace timestamps as-is (bursty at small rate scales)")
    ap.add_argument("--duration", type=float, default=600.0, help="seconds of (scaled) replay")
    ap.add_argument("--max-output", type=int, default=256, help="cap decode length (TBT needs a stream, not 2000 tokens)")
    ap.add_argument("--max-input", type=int, default=32768, help="drop trace requests longer than this (tail requests thrash the tiers)")
    ap.add_argument("--h", type=int, nargs="+", default=[4096, 8192, 16384, 24576],
                    help="probe hit lengths. Probes ARE load: mean(h)/interval tokens/s of prefill; keep it <~15%% of GPU capacity")
    ap.add_argument("--probe-interval", type=float, default=12.0)
    ap.add_argument("--prewarm-settle", type=float, default=120.0,
                    help="seconds after pre-warm for the SSD write cascade to drain before replay starts")
    ap.add_argument("--probe-warmup-wait", type=float, default=60.0,
                    help="seconds of background before first probe (lets GPU/CPU tiers churn)")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=256)
    ap.add_argument("--session-grace", type=float, default=600.0,
                    help="multi-turn: skip turns that could only start later than duration + this (s)")
    ap.add_argument("--closed", type=int, default=int(os.environ.get("CLOSED_N", "0")),
                    help="폐루프 처리량 측정: 동시 세션 수 N (0 = 기존 개방형). 도착 시각은 무시하고, 세션 N개가 "
                         "각자 턴을 순서대로 보내며 한 세션이 끝나면 다음 세션을 시작. 환경변수 CLOSED_N으로도 줄 수 있음")
    ap.add_argument("--think-scale", type=float, default=float(os.environ.get("THINK_SCALE", "0")),
                    help="폐루프에서 턴 사이 think time = 워크로드 think_s × 이 값 (기본 0 = 바로 다음 턴)")
    a = ap.parse_args()

    S = a.seed if a.seed is not None else int(time.time()) % 10_000_000
    print(f"seed={S}", flush=True)
    c = Client(a.url, a.model)

    if not (a.trace or a.workload):
        sys.exit("give --trace (Mooncake) or --workload (long-context)")
    text_mode = a.workload is not None
    # ---- load trace window
    recs = [json.loads(l) for l in open(a.workload if text_mode else a.trace) if l.strip()]
    recs = [r for r in recs if "_meta" not in r]          # workload header line (build_trace_workload.py)
    t0 = recs[0]["timestamp"] + a.trace_offset * 1000.0
    recs = [r for r in recs if r["timestamp"] >= t0]
    # Trace timestamps are quantized (many requests share one value). Stretching
    # by 1/rate_scale would turn each bucket into a burst followed by a gap, so
    # spread requests sharing a timestamp uniformly up to the next distinct one.
    if a.spread:
        i = 0
        while i < len(recs):
            j = i
            while j < len(recs) and recs[j]["timestamp"] == recs[i]["timestamp"]:
                j += 1
            nxt = recs[j]["timestamp"] if j < len(recs) else recs[i]["timestamp"] + 1000
            for k in range(i, j):
                recs[k] = dict(recs[k]); recs[k]["timestamp"] += (nxt - recs[i]["timestamp"]) * (k - i) / (j - i)
            i = j
    win, dropped = [], 0
    for r in recs:
        t = (r["timestamp"] - t0) / 1000.0 / a.rate_scale
        if t < a.start_offset: continue
        if t - a.start_offset > a.duration: break
        if r.get("input_length", 0) > a.max_input:
            dropped += 1; continue
        r = dict(r); r["_t"] = t - a.start_offset
        win.append(r)
    tok = sum(r.get("input_length", 0) for r in win)
    print(f"background: {len(win)} requests over {a.duration:.0f}s "
          f"(~{len(win)/a.duration:.2f} rps, ~{tok/a.duration/1000:.1f}K prompt tok/s, trace x{a.rate_scale}); "
          f"dropped {dropped} with input > {a.max_input}", flush=True)

    # ---- 폐루프: 도착 시각과 무관하게 세션 단위로 묶음 (워크로드 순서대로, 세션 안은 턴 순서)
    sessions = []
    if a.closed > 0:
        groups, order = {}, []
        for idx, r in enumerate(recs):
            if r.get("input_length", 0) > a.max_input: continue
            key = r.get("session") if r.get("session") is not None else ("_single", idx)
            if key not in groups: groups[key] = []; order.append(key)
            groups[key].append(r)
        for key in order:
            g = groups[key]
            g.sort(key=lambda r: (r.get("turn") if r.get("turn") is not None else 0, r["timestamp"]))
            sessions.append(g)
        n_req = sum(len(g) for g in sessions)
        print(f"closed loop: N={a.closed} concurrent sessions, {len(sessions)} sessions / {n_req} requests available, "
              f"think x{a.think_scale}, duration {a.duration:.0f}s (probes off)", flush=True)

    # ---- probe plan: enough probes for the run, alternate policy, cycle h
    n_probes = 0 if a.closed > 0 else int((a.duration - a.probe_warmup_wait) / a.probe_interval)
    plan = []
    for i in range(n_probes):
        h = a.h[i // 2 % len(a.h)]          # each h gets a load and a recompute back-to-back
        pol = "load" if i % 2 == 0 else "recompute"
        plan.append((i, h, pol))
    mean_h = sum(a.h) / len(a.h)
    print(f"probes: {n_probes} ({n_probes//len(a.h)//2} per (h, policy)); "
          f"probe prefill load ~{mean_h/a.probe_interval/1000:.1f}K tok/s", flush=True)

    # ---- phase 1: engine warm-up
    for i in range(3):
        c.stream(rand_ids(2048, S - 10 - i), 4)

    if text_mode:
        print(f"text-prompt workload: {len({r.get('doc_id') for r in win})} distinct documents", flush=True)

    # ---- phase 2: pre-warm probe prompts into SSD tier
    print("pre-warming probe prompts ...", flush=True)
    probe_prompts = {}
    for i, h, pol in plan:
        ps = S + 900_000_000 + i
        probe_prompts[i] = rand_ids(h, ps)
        c.stream(probe_prompts[i] + rand_ids(a.tail, ps * 10), 1)
    tok = sum(len(v) for v in probe_prompts.values())
    print(f"pre-warm done: {len(plan)} prompts, {tok/1000:.0f}K tokens (~{tok*128/1024/1024:.0f} GiB KV) -> "
          f"waiting {a.prewarm_settle:.0f}s for SSD cascade to drain", flush=True)
    time.sleep(a.prewarm_settle if plan else 0)

    # ---- phase 3: replay + probes
    bg_f = open(f"{a.tag}_bg.jsonl", "w")
    pr_f = open(f"{a.tag}_probes.csv", "w", newline="")
    pr_w = csv.writer(pr_f); pr_w.writerow(["probe_idx", "h", "policy", "t_send", "ttft_ms", "total_ms", "resp_id"])
    lock = threading.Lock()
    T0 = time.time()

    sess_done, sess_ev = {}, {}          # multi-turn: workload id -> 완료 시각 / Event
    in_win = {r.get("id") for r in win if r.get("id") is not None}

    def _ev(i):
        with lock:
            return sess_ev.setdefault(i, threading.Event())

    def bg_job(r, k):
        wid, pid, waited = r.get("id"), r.get("prev_id"), 0.0
        try:
            if pid is not None and pid in in_win:
                tw = time.time()
                _ev(pid).wait()
                t_go = sess_done[pid] + float(r.get("think_s") or 0.0)
                if t_go - T0 > a.duration + a.session_grace:
                    with lock:
                        bg_f.write(json.dumps({"id": k, "skipped": "session_grace", "session": r.get("session"),
                                               "turn": r.get("turn")}) + "\n")
                    return
                time.sleep(max(0.0, t_go - time.time()))
                waited = time.time() - tw
            ids = r["prompt"] if text_mode else trace_prompt(r, S + 100_000_000 + k)
            mt = max(1, min(a.max_output, r["output_length"]))
            t_send, times, rid = c.stream(ids, mt)
            if not times: return
            dts = [round((times[i] - times[i-1]) * 1000, 2) for i in range(1, len(times))]
            line = {"id": k, "resp_id": rid, "t_send": t_send, "t_first": times[0],
                    "tok_dt_ms": dts, "input_len": r.get("input_length", len(ids)), "output_len": mt,
                    "n_hash": len(r.get("hash_ids", [])),
                    "doc_id": r.get("doc_id"), "task": r.get("task"), "q_idx": r.get("q_idx"),
                    "phase_rps": r.get("phase_rps"), "session": r.get("session"), "turn": r.get("turn"),
                    "t_sched": T0 + r["_t"], "wait_prev_s": round(waited, 3)}
            with lock:
                bg_f.write(json.dumps(line) + "\n")
        except Exception as e:
            with lock:
                bg_f.write(json.dumps({"id": k, "error": str(e)[:200]}) + "\n")
        finally:
            if wid is not None:
                with lock:
                    sess_done[wid] = time.time()
                _ev(wid).set()

    def probe_job(i, h, pol):
        try:
            ps = S + 900_000_000 + i
            ids = probe_prompts[i] + rand_ids(a.tail, ps * 10 + 1 + (pol == "recompute"))
            t_send, times, rid = c.stream(ids, 1, policy=pol, h=len(probe_prompts[i]))
            ttft = (times[0] - t_send) * 1000 if times else -1
            total = (times[-1] - t_send) * 1000 if times else -1
            with lock:
                pr_w.writerow([i, h, pol, t_send, round(ttft, 1), round(total, 1), rid]); pr_f.flush()
            print(f"  probe {i:>3} h={h:>6} {pol:<9} TTFT={ttft:8.1f} ms  (t={time.time()-T0:6.1f}s)", flush=True)
        except Exception as e:
            print(f"  probe {i} failed: {e}", flush=True)

    if a.closed > 0:
        # ---- 폐루프 처리량 측정: 워커 N개가 세션을 하나씩 가져가 턴을 순서대로 보냄
        s_it = iter(sessions); s_lock = threading.Lock(); kc = itertools.count(); used = [0]

        def next_session():
            with s_lock:
                s = next(s_it, None)
                if s is not None: used[0] += 1
                return s

        def closed_worker(w):
            while time.time() - T0 < a.duration:
                sess = next_session()
                if sess is None:
                    print(f"  [worker {w}] 세션이 바닥남 — 워크로드 세션 수를 늘려야 함 (t={time.time()-T0:.0f}s)", flush=True)
                    return
                for j, r in enumerate(sess):
                    if time.time() - T0 >= a.duration: return
                    if j > 0 and a.think_scale > 0:
                        time.sleep(min(float(r.get("think_s") or 0.0) * a.think_scale, max(0.0, a.duration - (time.time() - T0))))
                    k = next(kc)
                    try:
                        ids = r["prompt"] if text_mode else trace_prompt(r, S + 100_000_000 + k)
                        mt = max(1, min(a.max_output, r["output_length"]))
                        t_send, times, rid = c.stream(ids, mt)
                        if not times: continue
                        dts = [round((times[i] - times[i-1]) * 1000, 2) for i in range(1, len(times))]
                        line = {"id": k, "resp_id": rid, "t_send": t_send, "t_first": times[0],
                                "tok_dt_ms": dts, "input_len": r.get("input_length", len(ids)), "output_len": mt,
                                "n_hash": len(r.get("hash_ids", [])), "doc_id": r.get("doc_id"), "task": r.get("task"),
                                "q_idx": r.get("q_idx"), "session": r.get("session"), "turn": r.get("turn"),
                                "t_sched": t_send, "wait_prev_s": 0.0, "closed_n": a.closed, "worker": w}
                        with lock:
                            bg_f.write(json.dumps(line) + "\n")
                    except Exception as e:
                        with lock:
                            bg_f.write(json.dumps({"id": k, "error": str(e)[:200], "closed_n": a.closed}) + "\n")

        print("closed-loop start", flush=True)
        ths = [threading.Thread(target=closed_worker, args=(w,), daemon=True) for w in range(a.closed)]
        for t in ths: t.start()
        nrep = 15.0
        while any(t.is_alive() for t in ths) and time.time() - T0 < a.duration:
            time.sleep(1.0)
            if time.time() - T0 >= nrep:
                print(f"  t={time.time()-T0:6.1f}s  sessions started={used[0]}/{len(sessions)}", flush=True); nrep += 15.0
        print("draining ...", flush=True)
        for t in ths: t.join()
        bg_f.close(); pr_f.close()
        print(f"done -> {a.tag}_bg.jsonl  (closed N={a.closed}, sessions used {used[0]})")
        return

    ex = ThreadPoolExecutor(max_workers=a.workers)
    bi, pi = 0, 0
    next_probe = a.probe_warmup_wait
    next_report = 15.0
    print("replay start", flush=True)
    while True:
        now = time.time() - T0
        if now > a.duration and bi >= len(win): break
        # background arrivals
        while bi < len(win) and win[bi]["_t"] <= now:
            ex.submit(bg_job, win[bi], bi); bi += 1
        # probes
        if pi < len(plan) and now >= next_probe:
            ex.submit(probe_job, *plan[pi]); pi += 1; next_probe += a.probe_interval
        if now >= next_report:
            print(f"  t={now:6.1f}s  bg sent={bi}/{len(win)}  probes={pi}/{len(plan)}", flush=True)
            next_report += 15.0
        # sleep until next event
        nxt = min(win[bi]["_t"] if bi < len(win) else 1e9, next_probe if pi < len(plan) else 1e9)
        time.sleep(max(0.0, min(0.05, nxt - (time.time() - T0))))
    print("draining ...", flush=True)
    ex.shutdown(wait=True)
    bg_f.close(); pr_f.close()
    print(f"done -> {a.tag}_bg.jsonl, {a.tag}_probes.csv")

if __name__ == "__main__":
    main()