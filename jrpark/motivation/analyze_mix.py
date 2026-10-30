#!/usr/bin/env python3
"""motivation 1부 분석 (native vLLM + 읽기 전용 로그).

입력
  --client  replay_mix.py 출력 (첫 줄 meta.t0, 이후 요청별 기록; 시각은 t0 기준 상대)
  --server  커넥터 결정 로그 (PROBE_LOG, POLICY=always_load MOTIV_LOG=1)
출력 (화면 표 + CSV)
  A. SSD load 시간 vs SSD I/O 큐         → 3쪽 카드 1      (motiv1_loads.csv)
  B. prefill 구간 중 decode 요청의 지연   → 3쪽 카드 2·3    (motiv1_windows.csv)
     native vLLM에서 recompute는 miss prefill로만 일어나므로, miss prefill이 decode에 준 지연이
     "recompute했다면 생겼을 지연"의 실측이다. 같은 방법을 SSD load 구간에도 적용해 비교한다.
  C. SSD hit 순간의 상태 분포             → 2부 조건 근거
  D. 요약 (요청 수, 종류·tier별 TTFT; tier = ssd / dram / gpu(GPU hit만) / miss)
  E. 스텝 단위 (스텝 로그)                  → 3쪽 카드 2·3
     decode만 있는 스텝의 시간: SSD load 진행 중 vs 아님 (load는 decode를 늦추지 않음)
     prefill이 섞인 스텝의 시간: prefill 토큰 수별, 같은 요청 수의 decode만 있는 스텝 대비 증가분
     (B는 혼합 부하에서 구간끼리 겹쳐 분리가 어려움 → 카드 2·3의 근거는 E로)

decode 지연 측정 (13쪽 정확도와 같은 방법)
  요청마다 토큰 간격 중 어떤 prefill 구간과도 겹치지 않은 간격의 중앙값을 그 요청의 기준으로 두고,
  구간과 겹친 간격이 기준을 넘는 만큼을 지연으로 본다. 여러 구간과 겹치면 겹친 구간 수로 나눠 배분.
"""
import argparse, bisect, csv, json, statistics as st
from collections import Counter, defaultdict

BS = 16  # 블록 크기 (토큰)


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v); i = min(len(v) - 1, max(0, int(round(q / 100 * (len(v) - 1)))))
    return v[i]


def load_client(path):
    t0, recs = None, []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "meta" in r:
            t0 = r["meta"]["t0"]; continue
        recs.append(r)
    if t0 is None:
        raise SystemExit("클라이언트 기록에 meta.t0이 없음 (replay_mix.py 최신 버전으로 다시 실행)")
    for r in recs:                                    # 절대 시각으로 변환
        for k in ("t_send", "t_first", "t_end", "t_plan"):
            if r.get(k) is not None:
                r[k] = t0 + r[k]
        r["t_chunks"] = [t0 + x for x in r.get("t_chunks", [])]
    return t0, recs


def load_server(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def matcher(client_ids):
    ids = set(client_ids)
    def m(sid):
        x = sid
        for _ in range(3):                            # vLLM 내부 ID에 붙는 접미사 제거
            if x in ids:
                return x
            if "-" not in x:
                break
            x = x.rsplit("-", 1)[0]
        return None
    return m


class Windows:
    """구간 [s, e) 목록에서 주어진 간격과 겹치는 구간 찾기."""
    def __init__(self, wins):
        self.w = sorted(wins, key=lambda x: x["s"])
        self.starts = [x["s"] for x in self.w]
        self.maxlen = max((x["e"] - x["s"] for x in self.w), default=0.0)

    def overlap(self, a, b):
        lo = bisect.bisect_left(self.starts, a - self.maxlen)
        hi = bisect.bisect_left(self.starts, b)
        return [w for w in self.w[lo:hi] if w["s"] < b and w["e"] > a]


def table(rows, key_r, key_c, val, r_edges, c_edges, r_name, c_name, fmt="{:.0f}"):
    def lab(edges, i):
        lo = edges[i]; hi = edges[i + 1] if i + 1 < len(edges) else None
        return f"{lo}-{hi}" if hi is not None else f"{lo}+"
    cells = defaultdict(list)
    for r in rows:
        i = bisect.bisect_right(r_edges, r[key_r]) - 1; j = bisect.bisect_right(c_edges, r[key_c]) - 1
        if i >= 0 and j >= 0:
            cells[(i, j)].append(r[val])
    head = f"{r_name:>14} | " + " | ".join(f"{c_name} {lab(c_edges, j):>10}" for j in range(len(c_edges)))
    print(head); print("-" * len(head))
    for i in range(len(r_edges)):
        out = []
        for j in range(len(c_edges)):
            v = cells.get((i, j), [])
            out.append(f"{(fmt.format(st.median(v)) if v else '-'):>8} (n={len(v):<3})"
                       .rjust(len(c_name) + 11))
        print(f"{lab(r_edges, i):>14} | " + " | ".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--warmup", type=float, default=300, help="앞부분 제외(초)")
    ap.add_argument("--min-pf", type=int, default=256, help="decode 지연 계산에 쓸 prefill 구간의 최소 토큰")
    ap.add_argument("--prefix", default="motiv1")
    ap.add_argument("--steps", default=None, help="스텝 로그 (기본: --server 이름 + _steps.jsonl)")
    ap.add_argument("--step-lag", type=int, default=2,
                    help="스텝 시간과 구성의 정렬 (vLLM 비동기 스케줄링: 호출 j-1~j 간격 = 호출 j-2의 구성)")
    ap.add_argument("--idle-ms", type=float, default=500, help="이보다 긴 스텝 간격은 유휴로 보고 제외")
    a = ap.parse_args()

    t0, cli = load_client(a.client)
    srv = load_server(a.server)
    tw = t0 + a.warmup
    ok = [r for r in cli if r.get("status") == "ok" and r.get("rid")]
    by_rid = {r["rid"]: r for r in ok}
    m = matcher(by_rid)

    # 서버 행을 요청별로 모음
    rows = defaultdict(list)
    unmatched = 0
    for x in srv:
        rid = m(x.get("req_id", ""))
        if rid is None:
            unmatched += 1; continue
        rows[rid].append(x)

    # 요청별 tier (첫 결정 행 기준)
    tier, first_dec = {}, {}
    for rid, xs in rows.items():
        for x in sorted(xs, key=lambda y: y["ts"]):
            d = x.get("decision")
            if d in ("load_pending", "load", "miss", "recompute"):
                tier[rid] = {"load_pending": "ssd", "load": "dram", "recompute": "rc"}.get(d) or \
                    ("gpu" if (x.get("num_computed") or 0) > 0 else "miss")      # miss 행: GPU hit 있으면 gpu
                first_dec[rid] = x
                break

    # ---------------- A. SSD load 시간 vs I/O 큐
    loads = []
    for rid, xs in rows.items():
        d = next((x for x in xs if x.get("decision") == "load_pending" and x.get("tier") == "ssd"), None)
        l = next((x for x in xs if x.get("decision") == "landed"), None)
        if not d or not l or d["ts"] < tw:
            continue
        h = d.get("h_ssd") or d.get("h") or 0
        rq = d.get("io_promo_blocks", 0) * BS; wq = d.get("io_store_blocks", 0) * BS
        r = by_rid[rid]
        loads.append({"rid": rid, "kind": r["kind"], "t": d["ts"] - t0, "h_ssd": h, "h_dram": d.get("h_dram", 0),
                      "read_q_tok": rq, "write_q_tok": wq, "q_tok": rq + wq,
                      "promo_jobs": d.get("io_promo_jobs", 0), "n_dec": d.get("m_n_dec", -1),
                      "load_ms": (l["ts"] - d["ts"]) * 1000,
                      "ttft_ms": (r["t_first"] - r["t_send"]) * 1000 if r.get("t_first") else None})

    # ---------------- B. prefill / load 구간과 decode 지연
    wins = []
    for rid, xs in rows.items():
        r = by_rid[rid]
        x = first_dec.get(rid)
        if x is None or r.get("t_first") is None:
            continue
        pf = (x.get("prompt_len", 0) or 0) - (x.get("num_computed", 0) or 0)
        if x["decision"] == "miss" and pf >= a.min_pf:
            wins.append({"type": "prefill", "rid": rid, "kind": r["kind"], "s": x["ts"], "e": r["t_first"],
                         "pf_tok": pf, "n_dec": x.get("m_n_dec", -1)})
        if x["decision"] == "load_pending":
            l = next((y for y in xs if y.get("decision") == "landed"), None)
            if l:
                wins.append({"type": "ssd_load", "rid": rid, "kind": r["kind"], "s": x["ts"], "e": l["ts"],
                             "pf_tok": 0, "h_ssd": x.get("h_ssd", 0), "n_dec": x.get("m_n_dec", -1)})
    pfw = Windows([w for w in wins if w["type"] == "prefill"])
    allw = Windows(wins)
    for w in wins:
        w["delay_ms"] = 0.0; w["n_hit_req"] = 0; w["clean"] = True
    for w in wins:                                     # load 구간이 prefill 구간과 겹치면 '깨끗하지 않음'
        if w["type"] == "ssd_load" and pfw.overlap(w["s"], w["e"]):
            w["clean"] = False

    n_int = n_ovl = 0; clean_iv, ovl_iv = [], []
    for r in ok:
        c = r["t_chunks"]
        if len(c) < 3:
            continue
        ivs = [(c[i], c[i + 1]) for i in range(len(c) - 1)]
        base = [b - a_ for a_, b in ivs if not pfw.overlap(a_, b)]
        if len(base) < 3:
            continue
        med = st.median(base)
        touched = set()
        for a_, b in ivs:
            if a_ < tw:
                continue
            n_int += 1
            ws = [w for w in allw.overlap(a_, b) if w["rid"] != r["rid"]]
            if not ws:
                clean_iv.append(b - a_); continue
            if any(w["type"] == "prefill" for w in ws):
                ovl_iv.append(b - a_)
            n_ovl += 1
            ex = max(0.0, (b - a_) - med) * 1000 / len(ws)
            for w in ws:
                w["delay_ms"] += ex
                if (id(w), r["rid"]) not in touched:
                    touched.add((id(w), r["rid"])); w["n_hit_req"] += 1

    # ---------------- 출력
    kinds = Counter(r["kind"] for r in ok)
    print(f"요청: 성공 {len(ok)} (문서 {kinds['doc']}, 새 세션 문서 {kinds['doc_new']}, 대화 {kinds['chat']}), 실패 {len(cli) - len(ok)}, "
          f"서버 행 {len(srv)} (클라이언트와 못 맞춘 행 {unmatched})")
    tc = Counter(tier.values())
    print("요청별 tier:", dict(tc))
    print("\n[D] 종류·tier별 TTFT (ms, 워밍업 이후)")
    grp = defaultdict(list)
    for rid, t in tier.items():
        r = by_rid[rid]
        if r.get("t_first") and r["t_send"] >= tw:
            grp[(r["kind"], t)].append((r["t_first"] - r["t_send"]) * 1000)
    for k in sorted(grp):
        v = grp[k]; print(f"  {k[0]:>4} {k[1]:>9}: n={len(v):5}  p50 {pct(v, 50):8.0f}  p95 {pct(v, 95):8.0f}")

    print(f"\n[A] SSD load 시간 (ms, 중앙값) — 행: SSD I/O 큐(읽기+쓰기, 토큰), 열: 내 요청의 SSD hit 토큰. load {len(loads)}건")
    qe = [0, 4096, 16384, 65536, 262144]; he = [0, 4096, 8192, 16384]
    table(loads, "q_tok", "h_ssd", "load_ms", qe, he, "I/O 큐", "hit")
    print("\n    읽기 큐만 / 쓰기 큐만 (열: hit)")
    table(loads, "read_q_tok", "h_ssd", "load_ms", qe, he, "읽기 큐", "hit")
    table(loads, "write_q_tok", "h_ssd", "load_ms", qe, he, "쓰기 큐", "hit")

    pw = [w for w in wins if w["type"] == "prefill" and w["s"] >= tw]
    lw = [w for w in wins if w["type"] == "ssd_load" and w["s"] >= tw and w["clean"]]
    print(f"\n[B] 구간 하나가 decode 요청들에 준 지연의 합 (ms, 중앙값) — 행: 구간 시작 시 decode 요청 수")
    print(f"    토큰 간격 {n_int}개 중 다른 요청의 구간과 겹친 간격 {n_ovl}개. "
          f"간격 중앙값: 겹치지 않음 {pct(clean_iv, 50) * 1000:.1f} ms / prefill과 겹침 {pct(ovl_iv, 50) * 1000:.1f} ms")
    de = [0, 8, 32, 128]
    print(f"  prefill 구간 {len(pw)}개 (열: prefill 토큰)")
    table(pw, "n_dec", "pf_tok", "delay_ms", de, [256, 1024, 4096, 16384], "decode 수", "prefill")
    print(f"  SSD load 구간 {len(lw)}개 (miss prefill 구간과 안 겹친 것만, 열: SSD hit 토큰)")
    print("  ※ 부하가 높으면 구간 안에 다른 prefill(load 후 나머지 프롬프트, GPU hit의 새 토큰 등)이 섞여 분리되지 않음 → E 참고")
    table(lw, "n_dec", "h_ssd", "delay_ms", de, [0, 4096, 8192, 16384], "decode 수", "hit")

    ssd_rows = [first_dec[rid] for rid, t in tier.items() if t == "ssd" and first_dec[rid]["ts"] >= tw]
    print(f"\n[C] SSD hit 순간의 상태 분포 (n={len(ssd_rows)}) — 2부 조건 근거")
    for name, f in [("decode 요청 수", lambda x: x.get("m_n_dec", -1)),
                    ("prefill 남은 요청 수", lambda x: x.get("m_n_pf", -1)),
                    ("읽기 큐 (토큰)", lambda x: x.get("io_promo_blocks", 0) * BS),
                    ("쓰기 큐 (토큰)", lambda x: x.get("io_store_blocks", 0) * BS),
                    ("진행 중 load 수", lambda x: x.get("io_promo_jobs", 0)),
                    ("SSD hit 토큰", lambda x: x.get("h_ssd", 0))]:
        v = [f(x) for x in ssd_rows]
        print(f"  {name:>16}: p10 {pct(v, 10):>8} p50 {pct(v, 50):>8} p90 {pct(v, 90):>8} max {max(v) if v else '-':>8}")

    # ---------------- E. 스텝 단위
    steps_path = a.steps or a.server.replace(".jsonl", "_steps.jsonl")
    try:
        stl = [json.loads(l) for l in open(steps_path, encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        stl = []
        print(f"\n[E] 스텝 로그 없음 ({steps_path})")
    stl.sort(key=lambda x: x["ts"])
    ldw = Windows([w for w in wins if w["type"] == "ssd_load"])
    st_rows = []
    L = a.step_lag
    for j in range(L, len(stl)):
        seg = stl[j - L + 1: j + 1]
        if any(x.get("step_ms", 0) > a.idle_ms for x in seg):
            continue                                   # 사이에 유휴 간격이 있으면 정렬이 깨짐
        comp, dur = stl[j - L], stl[j]["step_ms"]
        n_req, n_tok = comp.get("n_req", 0), comp.get("n_tok", 0)
        if n_req <= 0 or comp["ts"] < tw:
            continue
        e_ = stl[j]["ts"]; s_ = e_ - dur / 1000
        st_rows.append({"t": s_ - t0, "step_ms": dur, "n_req": n_req, "pf_tok": max(0, n_tok - n_req),
                        "ssd_loads": len(ldw.overlap(s_, e_))})
    if st_rows:
        dec = [r for r in st_rows if r["pf_tok"] == 0]
        mix = [r for r in st_rows if r["pf_tok"] > 0]
        print(f"\n[E] 스텝 시간 (ms, 중앙값) — 스텝 {len(st_rows)}개 (decode만 {len(dec)}, prefill 포함 {len(mix)}), "
              f"정렬 lag {L}")
        print("    prefill 토큰 = 스텝 토큰 수 − 요청 수 (decode 요청은 1토큰). 요청 수 ≈ decode 요청 수")
        re_ = [1, 16, 32, 64, 96, 128]
        for r in dec:
            r["load_on"] = 1 if r["ssd_loads"] > 0 else 0
        print("  decode만 있는 스텝: SSD load 진행 중(열 1) vs 아님(열 0)")
        table(dec, "n_req", "load_on", "step_ms", re_, [0, 1], "요청 수", "load", fmt="{:.1f}")
        print("  prefill이 섞인 스텝 (열: 그 스텝의 prefill 토큰)")
        table(mix, "n_req", "pf_tok", "step_ms", re_, [1, 256, 1024, 2048], "요청 수", "prefill", fmt="{:.1f}")
        base = defaultdict(list)
        for r in dec:
            base[bisect.bisect_right(re_, r["n_req"]) - 1].append(r["step_ms"])
        for r in mix:
            b = base.get(bisect.bisect_right(re_, r["n_req"]) - 1)
            r["extra_ms"] = r["step_ms"] - st.median(b) if b else float("nan")
        mix_ok = [r for r in mix if r["extra_ms"] == r["extra_ms"]]
        for r in mix_ok:
            r["extra_x_req"] = r["extra_ms"] * r["n_req"]
        print("  prefill로 늘어난 스텝 시간 (같은 요청 수의 decode만 있는 스텝 대비)")
        table(mix_ok, "n_req", "pf_tok", "extra_ms", re_, [1, 256, 1024, 2048], "요청 수", "prefill", fmt="{:.1f}")
        print("  → 그 스텝의 decode 요청 전원이 받는 지연의 합 = 늘어난 시간 × 요청 수 (ms)")
        table(mix_ok, "n_req", "pf_tok", "extra_x_req", re_, [1, 256, 1024, 2048], "요청 수", "prefill")
        fn = f"{a.prefix}_steps.csv"
        with open(fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for d in st_rows for k in d})); w.writeheader(); w.writerows(st_rows)
        print("저장:", fn)

    for name, data in (("loads", loads), ("windows", wins)):
        if not data:
            continue
        fn = f"{a.prefix}_{name}.csv"
        keys = sorted({k for d in data for k in d})
        with open(fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(data)
        print("저장:", fn)


if __name__ == "__main__":
    main()