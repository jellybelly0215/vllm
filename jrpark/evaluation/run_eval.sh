#!/usr/bin/env bash
# 평가용 부하별 재측정 (모델 공통, seed 51, always_load + ours = 이전 최종 정책)
#   LEval    0.25 0.5 0.75 1.0 1.25 req/s                (Tutti와 같은 축, 이전에 1.28까지 무너지지 않음)
#   LooGLE   0.2 0.3 0.4 0.5 0.6 0.7 req/s               (Tutti 축 + 한 칸, 세 모델 공통 상한)
#   SCBench  0.04 0.055 0.075 0.09 새 대화/s             (이전에 0.09까지 무너지지 않음)
#   ShareGPT 1.0 1.5 2.0 2.5 3.0 새 대화/s               (이전에 3.0까지 무너지지 않음)
# 결과 태그는 기존과 섞이지 않게 'ev'를 붙임: LIO2_k32coldev_r…, LIO2_lgcoldev_r…, LIO2_scbev_r…, LIO2_sgev_r…
# 어느 모델 디렉터리에서나: jr(Llama-3.1-8B)이면 기본값, 다른 모델은 prep_model.sh가 남긴 .model·.bkv를 읽음.
# 워크로드는 Llama 토크나이저로 생성 (모델 간 같은 문서·도착 시각). 이미 있는 파일은 그대로 씀.
# 워밍업 rate: LEval min(0.45, 0.6×rate) 180 s, LooGLE min(0.30, 0.7×rate) 300 s (낮은 rate에서 워밍업이 더 무겁지 않게)
#   DRY=1 ./run_eval.sh  /  ./run_eval.sh  /  WLS="leval lg" ./run_eval.sh  /  LG_RATES="… 0.8" ./run_eval.sh
#   다시 실행하면 끝난 arm은 건너뜀. 끝나면 eval_report.txt (vs always_load 표)
set -uo pipefail
cd "$(dirname "$0")"
log() { echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a run_eval.log; }
trap 'log "interrupted"; pkill -f "launch.sh l" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; exit 130' INT TERM
HF=$HOME/huggingface; TR=$HF/trace; LLAMA=$HF/Llama-3.1-8B-Instruct; SG=$HF/ShareGPT_V3_unfiltered_cleaned_split.json
SCB=$HF/SCBench/data
SCB_SRC="$SCB/multi_turn_qa_eng.jsonl $SCB/multi_turn_summary.jsonl $SCB/multi_turn_repoqa.jsonl $SCB/multi_turn_many_shot.jsonl $SCB/multi_turn_choice_eng.jsonl"
WLS=${WLS:-"leval lg scb sg"}
LEVAL_RATES=${LEVAL_RATES-"0.25 0.5 0.75 1.0 1.25"}
LG_RATES=${LG_RATES-"0.2 0.3 0.4 0.5 0.6 0.7"}
SCB_RATES=${SCB_RATES-"0.04 0.055 0.075 0.09"}
SG_RATES=${SG_RATES-"1.0 1.5 2.0 2.5 3.0"}

# ---- 모델·KV·메모리·프로파일
export MODEL=${MODEL:-$(cat .model 2>/dev/null || echo "$LLAMA")}
export B_KV=${B_KV:-$(cat .bkv 2>/dev/null || echo 131072)}
GiB() { python3 -c "print(int($1 * $B_KV / 131072 + 0.5))"; }
KV_L=$(GiB 32); CPU_L=$(GiB 32); KV_SG=$(GiB 16); CPU_SG=$(GiB 4)
PROF=MODEL_profile.json; [ -s $PROF ] || PROF=LMIX_profile.json
MN=$(basename "$MODEL")
[ -s "$MODEL/config.json" ] && [ -s "$PROF" ] || { log "모델($MODEL) 또는 프로파일($PROF) 없음"; exit 1; }
[ "$B_KV" = 524288 ] && { log "MHA 모델은 부하를 따로 맞춘 실험이라 이 범위에 넣지 않음"; exit 1; }
[ "$CPU_L" -le 36 ] || { log "CPU tier ${CPU_L} GiB가 /dev/shm(40G)에 비해 큼"; exit 1; }

# ---- 점검 (이전 최종 정책)
grep -q 'B_KV = int(os.environ.get("B_KV"' online_hw.py || { log "online_hw.py의 B_KV가 환경변수를 안 읽음"; exit 1; }
grep -q "OHW_MIX" online_hw.py && grep -q 'OHW_FIFO", "1"' online_hw.py || { log "online_hw.py가 b_mix 버전이 아님"; exit 1; }
grep -q 'OHW_LDEXCL", "1"' online_hw.py && grep -q 'OHW_LAG", "1"' online_hw.py || { log "LDEXCL·LAG 패치 누락"; exit 1; }
grep -q 'OHW_IDLE", "1"' online_hw.py && grep -q 'OHW_GUARD", "1"' online_hw.py || { log "빈 스텝·거부 패치 누락"; exit 1; }
grep -q 'OHW_OFF", "1"' online_hw.py && grep -q '_cur_a = num_computed_tokens' policy_connector.py || { log "위치 오프셋 패치 누락"; exit 1; }
grep -q "sess_done" replay_probe.py && grep -q "dram_only" policy_connector.py || { log "replay/connector 패치 확인"; exit 1; }
[ "${OHW_TOK:-0}" = 0 ] && [ "${OHW_LAG:-1}" = 1 ] && [ "${OHW_MIX:-1}" = 1 ] && [ "${OHW_FIFO:-1}" = 1 ] && [ "${OHW_CALIB:-0}" = 0 ] \
  || { log "OHW_TOK=0, OHW_LAG=1, OHW_MIX=1, OHW_FIFO=1, OHW_CALIB=0 이어야 함"; exit 1; }
for v in OHW_V2 OHW_FIFO_SSD OHW_IO_DONE OHW_OPP_IO_FUT OHW_PF_QUEUE OHW_MIX_DIFF OHW_C_CFG OHW_F_AVG OHW_PF_FUT CLOSED_N; do
  [ "${!v:-0}" = 0 ] || { log "$v=${!v} — 이전 최종 정책으로 돌리려면 꺼야 함"; exit 1; }; done
[ -s profile.json ] || cp "$PROF" profile.json
for f in build_longctx_workload.py build_multiturn_workload.py build_sharegpt_workload.py run_lio.sh run_mt.sh summary_raw.py; do
  [ -s $f ] || { log "$f 없음"; exit 1; }; done
if [ "${DRY:-0}" != 1 ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  log "GPU를 쓰는 프로세스가 있음 — 정리 후 실행"; exit 1; fi
export ONLINE_HW=1 OHW_LOG=$PWD/ohw_compute.jsonl
COMMON="PILOT=1 PROFILE_LOCAL=$PROF LDQ_STORE_W=2.4 PREWARM=0 MAX_DOCS=0"

# ---- 워크로드 (없으면 Llama 토크나이저로 생성; 기존 실험과 같은 인자)
wr() { python3 -c "print(min($2, round($3*$1, 3)))"; }
mkw() { local k=$1 R=$2 W
  case $k in
    leval) W="lio2_d0p64_w180_r$R.jsonl"
      [ -s "$W" ] || python3 build_longctx_workload.py --src $TR/LEval --model $LLAMA --min-input 4000 --max-input 30000 --pool 64 \
          --rps $R --rps-schedule "$(wr $R 0.45 0.6)@180,${R}@600" --output-tokens 32 --long-frac 0.12 --long-output 1024 \
          --recycle --max-docs 0 --new-frac 0 --duration 780 --seed 51 --out $W > /dev/null 2>&1 ;;
    lg) W="lio2_loogle_d0p40_w300_q6_r$R.jsonl"
      [ -s "$W" ] || python3 build_longctx_workload.py --src $TR/LooGLE_conv --model $LLAMA --min-input 4000 --max-input 30000 --pool 40 \
          --rps $R --rps-schedule "$(wr $R 0.30 0.7)@300,${R}@600" --output-tokens 32 --long-frac 0.12 --long-output 1024 \
          --recycle --max-docs 0 --new-frac 0 --max-q-per-doc 6 --duration 900 --seed 51 --out $W > /dev/null 2>&1 ;;
    scb) W="mt_scb_r${R}_s51.jsonl"
      [ -s "$W" ] || python3 build_multiturn_workload.py --src $SCB_SRC --model $LLAMA --truncate-doc 28000 --max-input 30000 \
          --min-input 4000 --min-turns 3 --max-turns 6 --session-rps $R --think-mean 60 --duration 1200 --long-frac 0.12 \
          --seed 51 --out $W > /dev/null 2>&1 ;;
    sg) W="mt_sg_r${R}_s51.jsonl"
      [ -s "$W" ] || python3 build_sharegpt_workload.py --src $SG --model $LLAMA --session-rps $R --duration 1200 \
          --kv-gib 16 --cpu-gib 4 --seed 51 --out $W > /dev/null 2>&1 ;;
  esac
  [ -s "$W" ] || { log "  워크로드 생성 실패: $W"; return 1; }
}

plan=()
for w in $WLS; do case $w in
  leval) for r in $LEVAL_RATES; do plan+=("leval $r LIO2_k32coldev_r${r/./p}"); done ;;
  lg)    for r in $LG_RATES;    do plan+=("lg $r LIO2_lgcoldev_r${r/./p}"); done ;;
  scb)   for r in $SCB_RATES;   do plan+=("scb $r LIO2_scbev_r${r/./p}"); done ;;
  sg)    for r in $SG_RATES;    do plan+=("sg $r LIO2_sgev_r${r/./p}"); done ;;
esac; done
todo=(); for p in "${plan[@]}"; do read k r T <<<"$p"
  arms=""; for a in always_load ours; do [ -s "${T}_${a}_bg.jsonl" ] || arms="$arms $a"; done
  [ -n "$arms" ] && todo+=("$p ${arms# }")
done
log "$MN (B_KV=$B_KV, GPU/CPU ${KV_L}/${CPU_L} GiB, ShareGPT ${KV_SG}/${CPU_SG} GiB, 프로파일 $PROF): 실행 예정 ${#todo[@]}지점"
if [ "${DRY:-0}" = 1 ]; then printf '  %s\n' "${todo[@]}"; exit 0; fi

i=0
for p in "${todo[@]}"; do i=$((i+1)); read k r T arms <<<"$p"
  mkw $k $r || continue
  log "########## [$i/${#todo[@]}] ${T}  arms [$arms]"
  case $k in
    leval)
      env $COMMON SRC=$TR/LEval DSET="" MAX_LEN=32768 MAX_INPUT=30000 KV_GIB=$KV_L CPU_GIB=$CPU_L POOL=64 MAXQ=0 \
          WARMUP_S=180 WARMUP_RPS=$(wr $r 0.45 0.6) DURATION=780 SEED=51 SUF="_k32coldev" RATES="$r" ARMS_OVERRIDE="$arms" \
          ./run_lio.sh || log "${T}: run_lio.sh 비정상 종료" ;;
    lg)
      env $COMMON SRC=$TR/LooGLE_conv DSET=loogle MAX_LEN=32768 MAX_INPUT=30000 KV_GIB=$KV_L CPU_GIB=$CPU_L POOL=40 MAXQ=6 \
          WARMUP_S=300 WARMUP_RPS=$(wr $r 0.30 0.7) DURATION=900 SEED=51 SUF="_lgcoldev" RATES="$r" ARMS_OVERRIDE="$arms" \
          ./run_lio.sh || log "${T}: run_lio.sh 비정상 종료" ;;
    scb)
      W=mt_scb_r${r}_s51.jsonl P=$T ARMS="$arms" PROFILE_LOCAL=$PROF KV_GIB=$KV_L CPU_GIB=$CPU_L \
          MAX_LEN=32768 MAX_INPUT=30000 DURATION=1500 ./run_mt.sh || log "${T}: run_mt.sh 비정상 종료" ;;
    sg)
      W=mt_sg_r${r}_s51.jsonl P=$T ARMS="$arms" PROFILE_LOCAL=$PROF KV_GIB=$KV_SG CPU_GIB=$CPU_SG \
          MAX_LEN=32768 MAX_INPUT=30000 DURATION=1500 ./run_mt.sh || log "${T}: run_mt.sh 비정상 종료" ;;
  esac
  for a in $arms; do [ -s "${T}_${a}_bg.jsonl" ] || log "  [경고] ${T}_${a}_bg.jsonl 없음"; done
done

# ---- 요약: vs always_load (summary_raw.py 정의) → eval_report.txt
{
  echo "===== $MN  평가 재측정 (seed 51)"
  for spec in "k32coldev 180 $LEVAL_RATES" "lgcoldev 300 $LG_RATES" "scbev 300 $SCB_RATES" "sgev 300 $SG_RATES"; do
    read pre warm rs <<<"$spec"; tags=()
    for r in $rs; do for a in always_load ours; do [ -s LIO2_${pre}_r${r/./p}_${a}_bg.jsonl ] && tags+=(LIO2_${pre}_r${r/./p}_$a); done; done
    [ ${#tags[@]} -gt 0 ] && python3 summary_raw.py --warmup $warm "${tags[@]}" 2>&1 | sed -n '/vs always_load/,$p' | tail -n +2
  done
} | tee eval_report.txt

pkill -f "launch.sh l" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null
for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 1; done
pkill -9 -f "vllm serve" 2>/dev/null; rm -f /dev/shm/vllm_offload_*.mmap "${NO_SSD_STORE_FLAG:-/tmp/NO_SSD_STORE}" 2>/dev/null
rm -rf /tmp/kvcache/fs.trash.* 2>/dev/null
find /tmp/kvcache/fs -mindepth 1 -delete 2>/dev/null
log "########## 평가 재측정 $MN 완료"
POST=${POST-/home/work/jrpark/training_validation.py}
if [ -n "$POST" ] && [ -s "$POST" ]; then
  log "########## 후속 작업: $POST"; cd "$(dirname "$POST")" && python3 "$POST" 2>&1 | tee -a "$OLDPWD/run_eval_post.log"
fi
