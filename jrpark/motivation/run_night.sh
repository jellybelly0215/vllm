#!/usr/bin/env bash
# 밤새 실행: (1) SSD를 끈 vLLM이 버티는 LEval 부하를 찾고 (2) 그 부하로 실제 부하 관찰을 한꺼번에 돌린다.
#
#   cd ~/jrpark/jr   (tmux 안에서)
#   ./motiv3/run_night.sh 2>&1 | tee motiv3/run_night.log
#   R=0.5 ./motiv3/run_night.sh 2>&1 | tee motiv3/run_night.log      # 부하를 이미 정했으면 시험 생략
#
# 1. LEval 부하 정하기: 후보 RATES(기본 "0.5 0.35 0.25")를 차례로 SSD 끔(l2)으로 10분씩 재생.
#    안정 판정 = 실패 0, TTFT 15 s 초과 비율 < 2%, 뒤쪽 3분 TTFT 중앙값이 앞쪽 3분의 1.8배 미만(대기열이 계속 쌓이지 않음).
#    처음으로 안정인 rate를 R로 쓴다. 모두 불안정이면 LEval은 건너뛴다.
# 2. 본 측정 (각 30분 × seed 51·52):
#    a. ShareGPT 1.0 새 대화/s, SSD 끔           → motiv3/prop2_sharegpt_nossd   (SSD 켬은 이미 motiv3/prop2_sharegpt)
#    b. LEval R req/s, SSD 켬 (vLLM 기본)        → motiv3/prop2_leval_d<R>
#    c. LEval R req/s, SSD 끔                    → motiv3/prop2_leval_d<R>_nossd
# 3. 끝: 정리 후 training_validation.py 한 번.
# 예상 시간: 부하 정하기 15~45분 + 본 측정 약 4시간.
set -uo pipefail
cd "$(dirname "$0")/.."
RATES=${RATES:-"0.5 0.35 0.25"}
TEST_SEC=${TEST_SEC:-600}
log() { echo "[$(date '+%F %T')] $*"; }
SUMMARY=()

cleanup() {
  pkill -f "[r]eplay_mix.py" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 1; done
  pkill -9 -f "vllm serve" 2>/dev/null || true
  rm -f /dev/shm/vllm_offload_*.mmap /tmp/NO_SSD_STORE 2>/dev/null || true
  find /tmp/kvcache/fs -mindepth 1 -delete 2>/dev/null || true
}

# ---- 0. 사전 점검
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  log "[중단] GPU를 쓰는 프로세스가 있음 → pkill -f \"[t]raining_validation.py\" 등으로 정리 후 다시"; exit 1; fi
for f in run_motiv1.sh motiv3/run_prop2_single.sh; do [[ -x $f ]] || { log "[중단] $f 없음"; exit 1; }; done
grep -q 'L2_CPU_GIB' run_motiv1.sh || { log "[중단] run_motiv1.sh가 L2_CPU_GIB를 넘기지 않음 → 수정본으로"; exit 1; }
grep -q '_nossd' motiv3/run_prop2_single.sh || { log "[중단] run_prop2_single.sh가 TIER를 받지 않음 → 수정본으로"; exit 1; }

judge() {   # $1 = client.jsonl → "stable ..." 또는 "unstable ..."
  python3 - "$1" <<'EOF'
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
EOF
}

# ---- 1. LEval 부하 정하기 (SSD 끔). R=<rate>를 주면 시험 없이 그 rate로 본 측정
R=${R:-}
[[ -n $R ]] && { log "== LEval 부하 R=$R (지정됨, 시험 생략)"; RATES=""; }
for r in $RATES; do
  log "== 부하 시험: LEval $r req/s, SSD 끔(l2), ${TEST_SEC}s"
  cleanup
  POST= TIER=l2 OUTDIR="motiv3/night_test/r$r" SEED=51 ./run_motiv1.sh one "$r" 0 "$TEST_SEC" > "motiv3/night_test_r$r.log" 2>&1
  f=$(ls motiv3/night_test/r$r/*_client.jsonl 2>/dev/null | head -1)
  res=$([[ -n $f ]] && judge "$f" || echo "unstable no-client-log")
  log "   결과: $res"; SUMMARY+=("부하 시험 LEval $r: $res")
  if [[ $res == stable* ]]; then R=$r; break; fi
done
cleanup

# ---- 2. 본 측정
run() {   # $1 = 설명, 나머지 = 환경변수와 명령
  local name=$1; shift
  log "########## 시작: $name"
  if env "$@" 2>&1 | tee -a "motiv3/run_night_detail.log"; then SUMMARY+=("완료  $name"); else SUMMARY+=("실패  $name"); fi
  cleanup
}
rm -rf motiv3/prop2_sharegpt_nossd
run "ShareGPT 1.0, SSD 끔" POST= TIER=l2 WL=sharegpt OUT=motiv3/prop2_sharegpt_nossd ./motiv3/run_prop2_single.sh
if [[ -n $R ]]; then
  run "LEval $R, SSD 켬" POST= WL=leval DOCS="$R" OUT="motiv3/prop2_leval_d$R" ./motiv3/run_prop2_single.sh
  run "LEval $R, SSD 끔" POST= TIER=l2 WL=leval DOCS="$R" OUT="motiv3/prop2_leval_d${R}_nossd" ./motiv3/run_prop2_single.sh
else
  SUMMARY+=("건너뜀 LEval 본 측정 (후보 $RATES 모두 불안정)")
fi

# ---- 3. 요약, 상태 줄 모으기, 후속 작업
log "########## 밤 실행 완료 (LEval 부하 R=${R:-없음})"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
grep -h "상태" motiv3/run_night_detail.log 2>/dev/null | tail -12
cleanup
POST=${POST-/home/work/jrpark/training_validation.py}
if [ -n "$POST" ] && [ -s "$POST" ]; then
  log "########## 후속 작업: $POST"
  ( cd "$(dirname "$POST")" && python3 "$POST" ) 2>&1 | tee -a motiv3/run_night_post.log
fi