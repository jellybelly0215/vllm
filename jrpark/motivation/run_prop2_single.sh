#!/usr/bin/env bash
# 명제 2: 단일 워크로드에서 SSD hit 시점의 GPU·SSD 상태 분포 (vLLM 기본 동작 그대로, 읽기 전용 로그)
#
#   WL=leval    : load-heavy 워크로드   — LEval 1.0 req/s          (evaluation 표의 지점)
#   WL=sharegpt : decode-heavy 워크로드 — ShareGPT 1.0 새 대화/s   (evaluation 표의 지점)
# 서버 설정은 evaluation과 같음 (GPU KV 32 GiB, CPU tier 32 GiB, 로컬 SSD). CPU_GIB=8 처럼 주면 민감도 실험.
# 순서: 3분 동작 점검(한쪽 rate 0으로 재생되는지; 안 되면 0.001로) → 30분 본 측정 × seed 51, 52 → 후처리
#
# 기존 1부 파이프라인 run_motiv1.sh (서버 기동 = launch.sh l3, DISK_ROOT=/tmp/kvcache/fs, KV·CPU 32 GiB,
# replay_mix.py 재생, analyze_mix.py 분석)를 그대로 쓴다. run_motiv1.sh는 SEED를 받도록 한 줄 수정본이 필요.
#
#   WL=leval    ./motiv3/run_prop2_single.sh 2>&1 | tee motiv3/prop2_leval_run.log
#   WL=sharegpt ./motiv3/run_prop2_single.sh 2>&1 | tee motiv3/prop2_sharegpt_run.log
#   POLICY=static:100000000 WL=leval ./motiv3/run_prop2_single.sh   # SSD hit만 항상 recompute → motiv3/prop2_leval_rc
#   TIER=l2 WL=leval ./motiv3/run_prop2_single.sh                   # SSD offloading 끔 (CPU tier 32 GiB만) → motiv3/prop2_leval_nossd
#   (끝나면 training_validation.py 실행. 두 개를 이어 돌릴 때는 앞쪽에 POST= 를 붙임)
set -uo pipefail
cd "$(dirname "$0")/.."            # ~/jrpark/jr

WL=${WL:-leval}
case $WL in
  leval)    DOCS=${DOCS:-1.0}; CHATS=${CHATS:-0} ;;
  sharegpt) DOCS=${DOCS:-0};   CHATS=${CHATS:-1.0} ;;
  *) echo "WL=leval|sharegpt"; exit 1 ;;
esac
SEEDS=${SEEDS:-"51 52"}
POLICY=${POLICY:-always_load}; export POLICY           # always_load | static:100000000 (SSD hit만 항상 recompute)
TIER=${TIER:-l3}; export TIER                           # l3 = CPU + SSD tier, l2 = CPU tier만 (SSD offloading 끔)
PTAG=$([ "$POLICY" = always_load ] && echo "" || echo "_rc")$([ "$TIER" = l2 ] && echo "_nossd" || echo "")
OUT=${OUT:-motiv3/prop2_$WL$PTAG}
STAGE=${STAGE:-all}
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

cleanup() {
  pkill -f "launch.sh l" 2>/dev/null || true; pkill -f "vllm serve" 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 1; done
  pkill -9 -f "vllm serve" 2>/dev/null || true
  rm -f /dev/shm/vllm_offload_*.mmap "${NO_SSD_STORE_FLAG:-/tmp/NO_SSD_STORE}" 2>/dev/null || true
  rm -rf /tmp/kvcache/fs.trash.* 2>/dev/null || true
  find /tmp/kvcache/fs -mindepth 1 -delete 2>/dev/null || true
}
trap cleanup EXIT

preflight() {
  [[ -x ./run_motiv1.sh ]] || { log "[중단] run_motiv1.sh 없음"; exit 1; }
  grep -q 'seed "${SEED:-51}"' run_motiv1.sh || { log "[중단] run_motiv1.sh가 SEED를 받지 않음 → 수정본으로 교체"; exit 1; }
  grep -q 'POLICY="${POLICY:-always_load}"' run_motiv1.sh || { log "[중단] run_motiv1.sh가 POLICY를 받지 않음 → 수정본으로 교체"; exit 1; }
  [[ -s replay_mix.py && -s analyze_mix.py ]] || { log "[중단] replay_mix.py / analyze_mix.py 없음"; exit 1; }
}

collapsed() {  # 기존 1부와 같은 기준: 실패 또는 TTFT > 30s 비율 >= 10%
  python3 - "$1" <<'EOF'
import glob, json, sys
rows = []
for f in glob.glob(sys.argv[1] + "/*_client.jsonl"):
    for l in open(f):
        if not l.strip() or "meta" in l:
            continue
        try: rows.append(json.loads(l))
        except Exception: pass
if not rows: print("no-data"); sys.exit(0)
bad = sum(1 for r in rows if r.get("status") != "ok" or r.get("t_first") is None or (r["t_first"] - r["t_send"]) > 30)
print("collapsed" if bad / len(rows) >= 0.10 else f"ok fail30={bad / len(rows):.1%} n={len(rows)}")
EOF
}

if [[ $STAGE == all ]]; then
  preflight
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    log "[중단] GPU를 쓰는 프로세스가 있음 — 정리 후 다시 실행"; exit 1; fi
  # 1. 3분 동작 점검: 대화 0으로 재생이 되는지. 안 되면 대화를 거의 0(0.001 세션/s, 30분에 2개 안팎)으로
  log "== [$WL, POLICY=$POLICY, TIER=$TIER] 동작 점검: 문서 $DOCS + 대화 $CHATS, 3분 (CPU tier ${CPU_GIB:-32} GiB)"
  POST= OUTDIR="$OUT/check" SEED=51 ./run_motiv1.sh one "$DOCS" "$CHATS" 180
  st=$(collapsed "$OUT/check"); log "   점검 상태: $st"
  if [[ $st == no-data ]] || grep -q "Traceback" "$OUT"/check/*_replay.log 2>/dev/null; then
    [[ $CHATS == 0 ]] && CHATS=0.001; [[ $DOCS == 0 ]] && DOCS=0.001
    log "   rate 0으로는 재생이 안 됨 → DOCS=$DOCS CHATS=$CHATS 로 진행"
    rm -rf "$OUT/check"; POST= OUTDIR="$OUT/check" SEED=51 ./run_motiv1.sh one "$DOCS" "$CHATS" 180
    st=$(collapsed "$OUT/check"); log "   점검 상태: $st"
  fi
  if [[ $st == collapsed* || $st == no-data ]]; then log "[중단] 점검 실패 → $OUT/check 확인"; exit 1; fi
  # 2. 30분 본 측정 × seed
  for s in $SEEDS; do
    log "== 본 측정 seed $s: 30분"
    POST= OUTDIR="$OUT/main_s$s" SEED="$s" DOCS="$DOCS" CHATS="$CHATS" ./run_motiv1.sh main
    log "   상태: $(collapsed "$OUT/main_s$s")"
  done
fi

# 3. 후처리 (실패해도 실험 결과에는 영향 없음): SSD hit마다 decode 요청들의 KV 양
for s in $SEEDS; do
  od="$OUT/main_s$s"; tag="d${DOCS}_c${CHATS}"
  hits=$(ls "$od"/${tag}*loads*.csv 2>/dev/null | head -1 || true)
  if [[ -z "$hits" ]]; then log "[참고] $od 에 SSD hit CSV(*loads*.csv)가 없음: $(ls "$od" 2>/dev/null | tr '\n' ' ')"; continue; fi
  python3 motiv3/decode_kv_at_hits.py --hits "$hits" --client "$od/${tag}_client.jsonl" --out "$od/hits_with_kv.csv" \
    || { log "[참고] decode KV 재구성 실패 → SSD hit CSV를 그대로 사용"; cp "$hits" "$od/hits_with_kv.csv"; }
done
if [[ -s motiv3/prop1/groups.csv ]] && ls "$OUT"/main_s*/hits_with_kv.csv >/dev/null 2>&1; then
  python3 motiv3/prop_overlay.py --prop1 motiv3/prop1/groups.csv --hits "$OUT"/main_s*/hits_with_kv.csv --out "$OUT/figs" \
    || log "[참고] 겹치기 그림 실패 (열 이름 확인: --q-col --n-col --hit-col)"
fi

# 4. 끝: 정리 후 후속 작업 (run_all.sh에서 부를 때는 POST="" 로 꺼짐)
cleanup
log "########## motivation 명제 2 단일 워크로드 $WL (DOCS=$DOCS CHATS=$CHATS, seed $SEEDS) 완료"
POST=${POST-/home/work/jrpark/training_validation.py}
if [ -n "$POST" ] && [ -s "$POST" ]; then
  log "########## 후속 작업: $POST"
  ( cd "$(dirname "$POST")" && python3 "$POST" ) 2>&1 | tee -a "$PWD/$OUT/run_prop2_${WL}_post.log" || log "[경고] 후속 작업이 오류로 끝남"
fi