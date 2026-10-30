#!/usr/bin/env python3
"""LEval·LooGLE·NQ 결과를 always_load vs ours로 정리 + raw data CSV 생성 (bg.jsonl만 읽음).
사용: python3 summary_raw.py TAG [TAG ...]        (각 {tag}_bg.jsonl)
출력: raw_summary.csv (태그별 지표), raw_requests.csv (요청별), 화면에 AL vs ours 표
지표:
  TTFT       = t_first - t_send
  E2E        = TTFT + sum(tok_dt_ms)                        (마지막 토큰까지)
  TPOT       = sum(tok_dt_ms) / (output_len - 1)            (요청별, 평균은 요청 평균)
  ITL p99    = 요청별 토큰 간격 p99의 중앙값 / p95          (eval_policy의 'TBT p99 per req'와 같은 개념)
  fail@T     = error 또는 첫 토큰 없음 또는 TTFT > T초 인 요청 비율
  collapse   = fail@30s >= 10%  (TTFT p50 > 30 s 기준은 LooGLE 1.3/s AL처럼 절반이 실패해도 놓침)
"""
import json, re, sys
import numpy as np
import pandas as pd

THR = [10, 30, 60]

def parse(tag):
    m = re.match(r"(.*)_(always_load|oursv1|ours_b\d+|ours|tokenflow|static_5120)$", tag)
    grp, arm = (m.group(1), m.group(2)) if m else (tag, "?")
    wl, rate, seed = "?", np.nan, ""
    if re.search(r"LIO2_scb", grp):
        wl = "SCBench-MT"; s = re.search(r"scb[a-z]*s(\d+)_r", grp); seed = s.group(1) if s else "51"
    elif re.search(r"LIO2_sg[a-z]*(s\d+)?_r", grp):
        wl = "ShareGPT-MT"; s = re.search(r"sg[a-z]*s(\d+)_r", grp); seed = s.group(1) if s else "51"
    elif "cold" in grp:
        wl = {"k32": "LEval-cold", "lg": "LooGLE-cold", "nq": "NarrativeQA-cold"}.get(re.search(r"LIO2_(k32|lg|nq)cold", grp).group(1), "cold")
        s = re.search(r"cold[a-z]*s(\d+)_r", grp); seed = s.group(1) if s else "51"
    elif grp.startswith("LIO2_k32"):
        wl = "LEval"; s = re.search(r"k32s(\d+)", grp); seed = s.group(1) if s else "51"
    elif "_lg_" in grp: wl = "LooGLE"
    elif "_nq_" in grp: wl = "NarrativeQA"
    elif grp.startswith("LBF"): wl = "LEval-LBF"
    r = re.search(r"_r(\d+)p(\d+)", grp)
    if r: rate = float(f"{r.group(1)}.{r.group(2)}")
    if ("cold" in tag or wl in ("SCBench-MT", "ShareGPT-MT")) and arm.startswith("ours") and arm != "oursv1":
        return wl, rate, seed, arm, "online"
    v1_runs = wl in ("LEval", "LEval-LBF") and "cold" not in tag or re.search(r"_nq_r0p(48|55)_", tag)
    ver = "v1" if arm == "oursv1" or (arm == "ours" and v1_runs) else ("v2" if arm == "ours" else "")
    return wl, rate, seed, ("ours" if arm.startswith("ours") else arm), ver

args = sys.argv[1:]
WARM = 0.0
if args and args[0] == "--warmup": WARM = float(args[1]); args = args[2:]
reqs, rows = [], []
for tag in args:
    wl, rate, seed, arm, ver = parse(tag)
    try:
        lines = open(f"{tag}_bg.jsonl").read().splitlines()
    except FileNotFoundError:
        print(f"[skip] {tag}_bg.jsonl 없음"); continue
    R = []
    for ln in lines:
        try: r = json.loads(ln)
        except json.JSONDecodeError: continue
        if "t_send" not in r: continue
        dt = r.get("tok_dt_ms") or []
        ok = r.get("t_first") is not None and not r.get("error")
        ttft = (r["t_first"] - r["t_send"]) if ok else np.nan
        dec = sum(dt) / 1000 if ok else np.nan
        out = r.get("output_len", len(dt) + 1)
        R.append(dict(workload=wl, rate=rate, seed=seed, arm=arm, version=ver, tag=tag, id=r.get("id"), t_send=r["t_send"],
                      doc_id=r.get("doc_id"), input_len=r.get("input_len"), output_len=out, ok=ok,
                      ttft_s=ttft, e2e_s=ttft + dec if ok else np.nan,
                      tpot_ms=1000 * dec / max(out - 1, 1) if ok and out > 1 else np.nan,
                      itl_p99_ms=np.percentile(dt, 99) if ok and len(dt) >= 2 else np.nan))
    if not R: print(f"[skip] {tag}: 요청 없음"); continue
    g = pd.DataFrame(R).sort_values("t_send")
    g["first_seen"] = ~g["doc_id"].duplicated()          # 빈 캐시 시작이면 첫 등장 = miss
    g["t_rel"] = g["t_send"] - g["t_send"].min()
    if WARM > 0: g = g[g.t_rel >= WARM]
    reqs.append(g)
    q = lambda c, p: np.nanpercentile(g[c], p) if g[c].notna().any() else np.nan
    row = dict(workload=wl, rate=rate, seed=seed, arm=arm, version=ver, tag=tag, n=len(g),
               err=int((~g.ok).sum()), collapse=bool(((~g.ok) | (g.ttft_s > 30)).mean() >= 0.10),
               ttft_p50=q("ttft_s", 50), ttft_p95=q("ttft_s", 95), ttft_p99=q("ttft_s", 99),
               e2e_p50=q("e2e_s", 50), e2e_p95=q("e2e_s", 95), e2e_p99=q("e2e_s", 99), e2e_mean=g.e2e_s.mean(),
               tpot_mean_ms=g.tpot_ms.mean(), itl_p99_p50_ms=q("itl_p99_ms", 50), itl_p99_p95_ms=q("itl_p99_ms", 95))
    for t in THR:
        row[f"fail@{t}s_%"] = 100 * ((~g.ok) | (g.ttft_s > t)).mean()
    row["first_seen_%"] = 100 * g.first_seen.mean()
    for nm, sub in (("hit", g[~g.first_seen]), ("miss", g[g.first_seen])):
        row[f"{nm}_ttft_p50"] = sub.ttft_s.median() if len(sub) else np.nan
        row[f"{nm}_ttft_p99"] = np.nanpercentile(sub.ttft_s, 99) if sub.ttft_s.notna().any() else np.nan
        row[f"{nm}_e2e_mean"] = sub.e2e_s.mean() if len(sub) else np.nan
    okT = g[g.ok & (g.ttft_s <= 30)]
    row["ok30_ttft_p50"] = okT.ttft_s.median(); row["ok30_ttft_p95"] = np.nanpercentile(okT.ttft_s, 95) if len(okT) else np.nan
    rows.append(row)

S = pd.DataFrame(rows).sort_values(["workload", "rate", "seed", "arm"])
pd.concat(reqs).to_csv("raw_requests.csv", index=False)
S.to_csv("raw_summary.csv", index=False)
print(f"→ raw_summary.csv ({len(S)} 태그), raw_requests.csv ({sum(len(r) for r in reqs)} 요청)" + (f", 처음 {WARM:g}s 제외" if WARM else "") + "\n")

pd.set_option("display.width", 250)
cols = ["n", "first_seen_%", "collapse", "fail@30s_%", "ttft_p50", "ttft_p95", "ttft_p99", "e2e_p50", "e2e_p95", "e2e_mean",
        "tpot_mean_ms", "itl_p99_p50_ms"]
for wl, w in S.groupby("workload", sort=False):
    print(f"===== {wl}")
    w = w[w.arm.str.match(r"^(always_load|ours(_b\d+)?)$")]
    print(w.set_index(["rate", "seed", "arm", "version"])[cols].round(2).to_string()); print()

print("===== ours 계열 vs always_load (둘 다 collapse 아닌 짝만, 상대 변화)")
out = []
S2 = S[S.arm.str.match(r"^(always_load|ours(_b\d+)?)$")]
for (wl, rate, seed), g in S2.groupby(["workload", "rate", "seed"]):
    if "always_load" not in set(g.arm): continue
    al = g[g.arm == "always_load"].iloc[0]
    for _, ou in g[g.arm != "always_load"].iterrows():
        rec = dict(workload=wl, rate=rate, seed=seed, arm=ou.arm, AL_collapse=al.collapse, ours_collapse=ou.collapse,
                   AL_fail30=round(al["fail@30s_%"], 1), ours_fail30=round(ou["fail@30s_%"], 1))
        if not (al.collapse or ou.collapse):
            for c in ["ttft_p50", "ttft_p99", "e2e_p50", "e2e_p95", "e2e_mean", "tpot_mean_ms", "itl_p99_p50_ms",
                      "hit_ttft_p50", "hit_ttft_p99", "miss_ttft_p50", "miss_ttft_p99"]:
                rec[c] = f"{ou[c] / al[c] - 1:+.0%}" if al[c] else "nan"
        out.append(rec)
print(pd.DataFrame(out).to_string(index=False))