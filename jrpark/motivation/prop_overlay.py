#!/usr/bin/env python3
"""명제 2 그림: 실제 혼합 부하의 SSD hit 순간 상태를 명제 1의 승자 영역 위에 겹친다.

  python3 prop_overlay.py --prop1 motiv3/prop1/groups.csv --hits motiv3/prop2/main_s51/hits_with_kv.csv \
      motiv3/prop2/main_s52/hits_with_kv.csv --out motiv3/prop2/figs

- 명제 1 승자: 조건(크기·N·Q, decode 문맥 = ShareGPT)마다 반복 중앙값의 (내 지연 + decode 지연)이 작은 쪽
- SSD hit 하나 → hit 크기가 가장 가까운 크기 등급(S≈1K, M≈6K, L≈14K, 로그 거리)
                → 그 등급의 측정 격자에서 (읽기 큐, decode 수)가 가장 가까운 조건의 승자
  격자 사이를 잇는 근사이므로 비율은 "양쪽 모두 상당하다" 수준으로만 해석한다.
- 출력: 크기 등급별 산점도 (점 색 = 배정된 승자, 네모 = 명제 1 측정 조건), assigned.csv, 비율 표
"""
import argparse, os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prop1_plot import BLUE, RED, set_font

SIZE_TOK = {"S": 1024, "M": 6144, "L": 14336}
Q_COLS = ["read_q_tok", "ioq_read", "read_q", "rq", "q_read", "ioq"]
N_COLS = ["n_decode", "n_dec", "decode", "num_decode", "decode_n_client"]
H_COLS = ["h_ssd", "hit_ssd", "ssd_hit_tokens", "hit_tokens", "hit", "h"]


def pick(cols, cands, name, given):
    if given:
        return given
    for c in cands:
        if c in cols:
            return c
    raise SystemExit(f"{name} 열을 찾지 못함 (후보 {cands}). --{name} 로 지정. 열: {list(cols)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop1", required=True)
    ap.add_argument("--hits", nargs="+", required=True)
    ap.add_argument("--out", default="motiv3/prop2/figs")
    ap.add_argument("--q-col"); ap.add_argument("--n-col"); ap.add_argument("--hit-col")
    a = ap.parse_args()
    set_font(); os.makedirs(a.out, exist_ok=True)

    g = pd.read_csv(a.prop1)
    g = g[(g.valid == True) & (g.C == 0)].copy()  # noqa: E712
    g["L"] = g.load_ttft_ms + g.load_dec_ms; g["R"] = g.rc_ttft_ms + g.rc_dec_ms
    grid = g.groupby(["size", "N", "Q"])[["L", "R"]].median().reset_index()
    grid["win"] = np.where(grid.L <= grid.R, "load", "recompute")

    H = pd.concat([pd.read_csv(f).assign(src=os.path.basename(os.path.dirname(f)) or f) for f in a.hits], ignore_index=True)
    qc = pick(H.columns, Q_COLS, "q-col", a.q_col)
    nc = pick(H.columns, N_COLS, "n-col", a.n_col)
    hc = pick(H.columns, H_COLS, "hit-col", a.hit_col)
    sizes = [s for s in SIZE_TOK if s in set(grid["size"])]
    st = np.array([SIZE_TOK[s] for s in sizes])
    H["size"] = [sizes[int(np.argmin(np.abs(np.log(max(h, 1)) - np.log(st))))] for h in H[hc]]
    qmax, nmax = max(grid.Q.max(), 1), max(grid.N.max(), 1)

    def assign(row):
        cand = grid[grid["size"] == row["size"]]
        # 측정 격자의 decode 수 범위를 크게 벗어난 hit은 외삽하지 않는다
        if row[nc] > 1.5 * cand.N.max() or row[nc] < cand.N.min() / 1.5:
            return "범위 밖"
        d = ((np.log1p(cand.Q) - np.log1p(row[qc])) / np.log1p(qmax)) ** 2 + ((cand.N - row[nc]) / nmax) ** 2
        return cand.iloc[int(np.argmin(d.to_numpy()))]["win"]

    H["win"] = H.apply(assign, axis=1)
    H.to_csv(os.path.join(a.out, "assigned.csv"), index=False)
    tab = H.groupby(["src", "size"]).win.value_counts(normalize=True).unstack(fill_value=0).round(3)
    tab["n"] = H.groupby(["src", "size"]).size()
    print(tab.to_string())
    print("\n전체:", H.win.value_counts(normalize=True).round(3).to_dict())
    tab.to_csv(os.path.join(a.out, "fractions.csv"))

    for s in sizes:
        hs = H[H["size"] == s]; gs = grid[grid["size"] == s]
        if hs.empty:
            continue
        f, ax = plt.subplots(figsize=(5.8, 4.2))
        for w, col in (("load", BLUE), ("recompute", RED), ("범위 밖", "#AAAAAA")):
            p = hs[hs.win == w]
            ax.scatter(p[qc], p[nc], s=6, color=col, alpha=0.25, linewidths=0, label=w)
            q = gs[gs.win == w]
            if len(q):
                ax.scatter(q.Q, q.N, s=110, marker="s", color=col, edgecolors="white", linewidths=1.5)
        ax.set_xscale("symlog", linthresh=1000, linscale=0.5)
        ticks = [0, 1000, 10000, 100000, 1000000]
        ax.set_xticks(ticks); ax.set_xticklabels(["0", "1K", "10K", "100K", "1M"]); ax.minorticks_off()
        ax.set_xlabel("SSD 읽기 큐 (토큰)"); ax.set_ylabel("decode 요청 수")
        lg = ax.legend(frameon=False, loc="upper left", markerscale=3)
        for h in lg.legend_handles:
            h.set_alpha(1)
        f.tight_layout(); f.savefig(os.path.join(a.out, f"overlay_{s}.png"), dpi=220, bbox_inches="tight"); plt.close(f)
    print(f"→ {a.out}: overlay_*.png, assigned.csv, fractions.csv")


if __name__ == "__main__":
    main()