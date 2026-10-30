#!/usr/bin/env bash
# A2 — I/O-busy regime (LIO-S): hit-heavy, decode-light workload at rising arrival rates, local SSD tier.
# tmux 안에서:
#   PILOT=1 ./run_lio.sh        # 파일럿: rate 2.0에서 always_load + ours만 (~35분). 먼저 이것부터.
#   ./run_lio.sh                # 본 실험: rate 스윕 × 전체 arm (밤새)
#   RATES="1.0 2.0" ./run_lio.sh
#
# 워크로드: LEval 4–30K 문서, 응답 32 토큰 + 12%는 1024 토큰(decode 모집단 → recompute의 Opp가 0이 아님),
#   풀 64 문서(재방문 사이 ~500K 토큰의 다른 KV → 대부분 SSD hit), --recycle(LEval 소진 시 재순환).
# 비용 모델: LD_INDEX=promo (A0 적합: BW1 1.52 / BWsat 2.23 GB/s, γ 1.15) — ours와 tokenflow가 같은 T_load를 쓴다.
set -uo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-$HOME/huggingface/Llama-3.1-8B-Instruct}
LEVAL=${LEVAL:-$HOME/huggingface/trace/LEval}
# 다른 데이터셋: SRC=<변환된 jsonl 디렉터리> DSET=loogle (워크로드/태그 이름 구분), 긴 문서는 MAX_LEN=65536 MAX_INPUT=62000
SRC=${SRC:-$LEVAL}; DSET=${DSET:-}
MIN_INPUT=${MIN_INPUT:-4000}; MAX_INPUT=${MAX_INPUT:-30000}; MAX_LEN=${MAX_LEN:-32768}
PROFILE_LOCAL=${PROFILE_LOCAL:?"로컬 tier LB 실험의 프로파일 경로 (예: PROFILE_LOCAL=LB_profile.json)"}
DURATION=${DURATION:-600}
POOL=${POOL:-64}
MAX_DOCS=${MAX_DOCS:-96}      # 고정 코퍼스 → arm마다 SSD에 prewarm (정상 상태 측정)
log() { echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a run_lio.log; }

# ---- 공통 서버 환경 (run_regime.sh -> launch.sh -> vllm 로 상속)
export DISK_ROOT=/tmp/kvcache/fs
export LD_INDEX=promo IO_SIGNAL=promo ONLINE_IO_EDGES="1,257,769,1537"
export LDQ_BW1_GBPS=${LDQ_BW1_GBPS:-1.52} LDQ_BWSAT_GBPS=${LDQ_BWSAT_GBPS:-2.23} LDQ_GAMMA=${LDQ_GAMMA:-1.15}
export LDQ_STORE_W=${LDQ_STORE_W:-0} RCQ_THR_TOKS=${RCQ_THR_TOKS:-11500} BETA_IO=${BETA_IO:-1.0}
export OURS_PROFILE=$(readlink -f "$PROFILE_LOCAL")
export HEALTH_TIMEOUT_S=1800 STOP_ON_FAIL=0
[ -s "$OURS_PROFILE" ] || { echo "profile not found: $PROFILE_LOCAL"; exit 1; }
rm -f "${NO_SSD_STORE_FLAG:-/tmp/NO_SSD_STORE}"        # A0가 남긴 store 차단 플래그가 있으면 제거

if [ "${PILOT:-0}" = 1 ]; then
  RATES=${RATES:-"1.5"}; ARMS_HI="always_load ours"; ARMS_LO="always_load ours"
else
  # 이 코퍼스(평균 문서 11.9K)에서 always_load 포화 ≈ 1.9 rps, 혼합 상한 ≈ 3 rps
  RATES=${RATES:-"1.0 1.5 1.75 2.0 2.5"}
  # always_recompute는 GPU 용량(~1.4 req/s) 위에서는 큐가 무한히 쌓여 drain만 수십 분 → 저율에서만
  ARMS_LO="always_load always_recompute static:5120 tokenflow ours ours@betaio=0"
  ARMS_HI="always_load static:5120 tokenflow ours"
fi

for R in $RATES; do
  WSUF=""; [ "$MAX_DOCS" != 96 -o "$POOL" != 64 ] && WSUF="_d${MAX_DOCS}p${POOL}"   # 코퍼스가 바뀌면 워크로드 파일도 새로
  [ "${SEED:-51}" != 51 ] && WSUF="${WSUF}_s${SEED}"                                   # 반복 실험: SEED=52 SUF=_k32s52
  [ "${NEW_FRAC:-0}" != 0 ] && WSUF="${WSUF}_n${NEW_FRAC}"                              # 새 문서 혼합: NEW_FRAC=0.1
  [ -n "${WARMUP_S:-}" ] && WSUF="${WSUF}_w${WARMUP_S}"
  [ "${MAXQ:-0}" != 0 ] && WSUF="${WSUF}_q${MAXQ}"                              # 워밍업 구간 포함 워크로드
  [ -n "$DSET" ] && WSUF="_${DSET}${WSUF}"
  [ "$MAX_INPUT" != 30000 ] && WSUF="${WSUF}_L${MAX_INPUT}"
  W="lio2${WSUF}_r${R}.jsonl"; P="LIO2${SUF:-}_r${R/./p}"      # SUF=_c32 등으로 설정별 결과 분리
  [ -s "$W" ] || python3 build_longctx_workload.py --src "$SRC" --model "$MODEL" \
      --min-input "$MIN_INPUT" --max-input "$MAX_INPUT" --pool "$POOL" --rps "$R" ${WARMUP_S:+--rps-schedule "${WARMUP_RPS:-0.45}@${WARMUP_S},${R}@$((DURATION-WARMUP_S))"} --output-tokens 32 \
      --long-frac 0.12 --long-output 1024 --recycle --max-docs "$MAX_DOCS" --new-frac "${NEW_FRAC:-0}" ${MAXQ:+$([ "$MAXQ" != 0 ] && echo --max-q-per-doc $MAXQ)} --duration "$DURATION" --seed "${SEED:-51}" \
      --out "$W" | tail -n 7 | tee -a run_lio.log
  ARMS=$ARMS_HI; awk "BEGIN{exit !($R <= 1.5)}" && ARMS=$ARMS_LO
  [ -n "${ARMS_OVERRIDE:-}" ] && ARMS=$ARMS_OVERRIDE      # 예: ARMS_OVERRIDE="ours" (기존 always_load 재사용)
  log "########## $P: rate $R rps, arms [$ARMS]"
  PREWARM_CMD="$([ "${PREWARM:-1}" = 1 ] && echo "python3 lio_prewarm.py --workload $W --model $MODEL --dev sda")" \
  MAX_INPUT=$MAX_INPUT KV_GIB=${KV_GIB:-16} CPU_GIB=${CPU_GIB:-12} MAX_LEN=$MAX_LEN MAX_OUTPUT=1024 DURATION=$DURATION ARMS="$ARMS" \
    ./run_regime.sh "$P" "$W" || log "$P 중단"
  python3 a2_check.py --tags $(for a in $ARMS; do a=${a//:/_}; a=${a//@betaio=/_bio}; echo "${P}_$a"; done) \
    2>&1 | tee "${P}_a2check.txt" | tee -a run_lio.log
done
log "########## 완료"
pkill -f "vllm serve" 2>/dev/null || true