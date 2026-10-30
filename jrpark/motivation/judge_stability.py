#!/usr/bin/env python3
"""재생 클라이언트 기록으로 과부하 여부 판정.
  python3 motiv3/judge_stability.py <*_client.jsonl>
안정 = 실패 0, TTFT 15 s 초과 < 2%, 뒤쪽 3분 TTFT 중앙값 < 앞쪽 3분 × 1.8 (처음 1분 제외)."""
import json, sys, statistics as st
rows = []
for l in open(sys.argv[1]):
    if not l.strip() or "meta" in l:
        continue
    try: rows.append(json.loads(l))
    except Exception: pass
rows = [r for r in rows if r.get("t_send", 0) > 60]          # 처음 1분 워밍업 제외
if not rows: print("unstable no-data"); sys.exit()
fail = sum(1 for r in rows if r.get("status") != "ok" or r.get("t_first") is None)
ok = [r for r in rows if r.get("status") == "ok" and r.get("t_first") is not None]
tt = [(r["t_first"] - r["t_send"], r["t_send"]) for r in ok]
over = sum(1 for t, _ in tt if t > 15) / max(1, len(tt))
t_end = max(s for _, s in tt)
first = [t for t, s in tt if s < 60 + 180]; last = [t for t, s in tt if s > t_end - 180]
ratio = (st.median(last) / st.median(first)) if first and last and st.median(first) > 0 else 99
p50 = st.median([t for t, _ in tt]); p95 = sorted(t for t, _ in tt)[int(0.95 * (len(tt) - 1))]
good = fail == 0 and over < 0.02 and ratio < 1.8
print(f"{'stable' if good else 'unstable'} n={len(rows)} fail={fail} over15s={over:.1%} "
      f"TTFT p50={p50:.2f}s p95={p95:.2f}s 뒤/앞 중앙값={ratio:.2f}")