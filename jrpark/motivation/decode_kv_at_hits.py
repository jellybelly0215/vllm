#!/usr/bin/env python3
"""SSD hit 순간마다 decode 중인 요청 수와 그 요청들의 KV 양(문맥 토큰 합)을 클라이언트 로그로 재구성한다.

서버 로그 패치 없이 계산한다. 클라이언트와 서버가 같은 호스트라 time.time()이 같은 시계라고 본다.

  python3 decode_kv_at_hits.py --hits motiv1/xxx_loads.csv --client "motiv3/prop2/main_s51/**/*_bg.jsonl" \
      --out motiv3/prop2/main_s51/hits_with_kv.csv

decode 중 = 첫 토큰을 받았고 마지막 토큰은 아직 안 받은 요청.
문맥 토큰 = 입력 길이 + 그 시각까지 생성한 토큰 수. 스트리밍 청크 하나에 토큰이 여러 개 묶이는 경우가 있어
            청크 수를 (출력 길이 − 1) / 청크 수 비율로 보정한다.
서버 로그의 decode 요청 수 열이 있으면 클라이언트 재구성 값과 상관을 출력해 시계 정렬을 확인한다.
"""
import argparse, glob, json

import numpy as np
import pandas as pd

TIME_COLS = ["t_decide", "t_hit", "t_lookup", "t", "ts", "time", "t_start"]
NDEC_COLS = ["n_decode", "n_dec", "decode", "num_decode", "n_running_decode"]


def pick(cols, cands, name, given):
    if given:
        return given
    for c in cands:
        if c in cols:
            return c
    raise SystemExit(f"{name} 열을 찾지 못함 (후보 {cands}). --{name} 로 지정하세요. 열: {list(cols)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True)
    ap.add_argument("--client", required=True, help="bg.jsonl glob (재귀 ** 가능)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-col"); ap.add_argument("--ndec-col")
    a = ap.parse_args()

    H = pd.read_csv(a.hits)
    tc = pick(H.columns, TIME_COLS, "time-col", a.time_col)
    R, metas = [], []
    for f in glob.glob(a.client, recursive=True):
        for l in open(f):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "meta" in r or (isinstance(r.get("type"), str) and r["type"] == "meta"):
                metas.append(r.get("meta", r)); continue
            if not r.get("t_first") or r.get("error") or r.get("status", "ok") != "ok":
                continue
            inp = next((r[k] for k in ("input_len", "prompt_tokens", "n_prompt", "in_tok", "prompt_len", "n_in") if r.get(k)), 0)
            if "tok_dt_ms" in r:                       # run_lio 계열 bg.jsonl
                dt = r.get("tok_dt_ms") or []
                out = r.get("output_len") or (len(dt) + 1)
                cum = np.concatenate([[0.0], np.cumsum(dt) / 1000.0])
            else:                                      # replay_mix 계열 client.jsonl: 청크 도착 시각 목록
                ch = r.get("t_chunks") or [r["t_first"]]
                cum = np.array(ch, dtype=float) - r["t_first"]
                out = r.get("output_len") or r.get("completion_tokens") or len(ch)
            scale = (out - 1) / max(len(cum) - 1, 1)
            R.append([r["t_first"], r["t_first"] + cum[-1], inp, cum, scale])
    if R and not any(x[2] for x in R):
        print("  [참고] 클라이언트 기록에 입력 길이 필드가 없음 → KV 양은 생성 토큰만 반영 (decode 수는 정확)")
    # 시계 맞추기: 한쪽만 상대 시각(작은 값)이면 meta의 epoch 값으로 옮긴다
    ht = H[tc].to_numpy(dtype=float)
    c_rel, h_rel = (max(x[0] for x in R) < 1e8) if R else False, np.nanmax(ht) < 1e8
    if R and c_rel != h_rel:
        ep = [v for m in metas if isinstance(m, dict) for v in m.values() if isinstance(v, (int, float)) and v > 1e9]
        if not ep:
            raise SystemExit("클라이언트와 서버 로그의 시각 기준이 다르고 meta에 기준 시각이 없음 → --time-col 확인")
        off = ep[0] if c_rel else -ep[0]
        for x in R:
            x[0] += off; x[1] += off
        print(f"  시계 보정: 클라이언트 시각에 {off:+.0f}s")
    if not R:
        raise SystemExit("클라이언트 로그가 없음")
    R.sort(key=lambda x: x[0])
    t_first = np.array([x[0] for x in R]); t_end = np.array([x[1] for x in R])
    n_dec, kv = [], []
    for t in ht:
        idx = np.nonzero((t_first <= t) & (t < t_end))[0]
        n_dec.append(len(idx))
        s = 0.0
        for i in idx:
            _, _, inp, cum, scale = R[i]
            gen = 1 + scale * (np.searchsorted(cum, t - R[i][0], side="right") - 1)
            s += inp + gen
        kv.append(s)
    H["decode_n_client"] = n_dec
    H["decode_kv_tokens"] = np.round(kv).astype(int)
    H.to_csv(a.out, index=False)
    print(f"→ {a.out}: SSD hit {len(H)}개, 클라이언트 요청 {len(R)}개")
    try:
        nc = pick(H.columns, NDEC_COLS, "ndec-col", a.ndec_col)
        c = np.corrcoef(H[nc], H.decode_n_client)[0, 1]
        print(f"  서버 decode 수({nc}) vs 클라이언트 재구성: 상관 {c:.3f}, 중앙값 {H[nc].median():.0f} vs {H.decode_n_client.median():.0f}")
        if c < 0.8:
            print("  [경고] 상관이 낮음 → 시계 기준(서버 로그의 시간 열)이 클라이언트와 같은지 확인")
    except SystemExit:
        print("  (서버 decode 수 열이 없어 교차 확인 생략)")
    print(H[["decode_n_client", "decode_kv_tokens"]].describe(percentiles=[.1, .5, .9]).round(0).to_string())


if __name__ == "__main__":
    main()