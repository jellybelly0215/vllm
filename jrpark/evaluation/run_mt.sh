#!/usr/bin/env bash
# 미리 만든 멀티턴 워크로드(.jsonl)를 run_regime.sh로 돌린다 (run_lio.sh와 같은 서버 환경, 사전 저장 없음).
#   W=/tmp/scb_0.075.jsonl P=LIO2_scb_r0p075 ARMS="always_load ours" KV_GIB=32 CPU_GIB=32 MAX_LEN=32768 MAX_INPUT=30000 DURATION=1500 ./run_mt.sh
# DURATION은 워크로드의 마지막 '예정' 턴까지 덮어야 함 (세션이 끝까지 돌도록). replay의 --session-grace(600 s)가 지연을 흡수.
set -uo pipefail
cd "$(dirname "$0")"
log() { echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a run_mt.log; }
: "${W:?워크로드 파일 W}"; : "${P:?태그 접두 P}"
[ -s "$W" ] || { log "워크로드 없음: $W"; exit 1; }
grep -q "sess_done" replay_probe.py || { log "replay_probe.py에 세션 순서 패치 없음 (patch_replay_session.py)"; exit 1; }
MODEL=${MODEL:-$HOME/huggingface/Llama-3.1-8B-Instruct}
PROFILE_LOCAL=${PROFILE_LOCAL:-LMIX_profile.json}

export DISK_ROOT=/tmp/kvcache/fs
export LD_INDEX=promo IO_SIGNAL=promo ONLINE_IO_EDGES="1,257,769,1537"
export LDQ_BW1_GBPS=${LDQ_BW1_GBPS:-1.52} LDQ_BWSAT_GBPS=${LDQ_BWSAT_GBPS:-2.23} LDQ_GAMMA=${LDQ_GAMMA:-1.15}
export LDQ_STORE_W=${LDQ_STORE_W:-2.4} RCQ_THR_TOKS=${RCQ_THR_TOKS:-11500} BETA_IO=${BETA_IO:-1.0}
export OURS_PROFILE=$(readlink -f "$PROFILE_LOCAL")
export HEALTH_TIMEOUT_S=1800 STOP_ON_FAIL=0
export ONLINE_HW=${ONLINE_HW:-1} OHW_LOG=${OHW_LOG:-$PWD/ohw_compute.jsonl}
rm -f "${NO_SSD_STORE_FLAG:-/tmp/NO_SSD_STORE}"

log "########## $P: W=$W arms [${ARMS:-always_load ours}] KV=${KV_GIB:-32}G CPU=${CPU_GIB:-32}G maxlen=${MAX_LEN:-32768} dur=${DURATION:-1500}s"
PREWARM_CMD="" MODEL=$MODEL MAX_INPUT=${MAX_INPUT:-30000} KV_GIB=${KV_GIB:-32} CPU_GIB=${CPU_GIB:-32} MAX_LEN=${MAX_LEN:-32768} \
  MAX_OUTPUT=${MAX_OUTPUT:-1024} DURATION=${DURATION:-1500} ARMS="${ARMS:-always_load ours}" \
  ./run_regime.sh "$P" "$W" || log "$P 중단"
log "########## $P 완료"