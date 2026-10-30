#!/usr/bin/env bash
# One-shot policy comparison for a fixed regime (server config + workload).
#
#   ./run_regime.sh R2 longctx_v2.jsonl
#
# Environment (all optional, defaults = regime 2):
#   KV_GIB=16 CPU_GIB=12 MAX_LEN=32768 DURATION=1800 MAX_OUTPUT=1024
#   ARMS="always_load static:24576 tokenflow ours"   # ours is always run last, with the
#                                                     # profile built from the arms before it
#   LD_INDEX=occ  BETA=1.0  SWEEP=sweep_zero_load.csv
#   OURS_PROFILE=path.json  # use this profile for ours instead of building one from the other arms
#   "ours@beta=0.5" in ARMS runs ours with that Opp weight (tag ours_b0.5); several allowed
#   "ours@kv=0.9"   runs ours with the KV-pressure guard (no recompute when KV usage >= 90%)
#   "online"        self-learning selector, cold start (fresh tables + startup calibration via sweep.py)
#   "online@warm=FILE"  same, but starting from a saved online_state.json
#   CALIBRATE=1 (default) runs sweep.py right after the server is healthy for online arms
#   "table" also runs after the profile is built and reads winner_a4 from it (in-deployment table)
#   SKIP_DONE=1   # skip an arm whose <PREFIX>_<arm>_bg.jsonl already exists (resume)
#
# Per arm: kill any vllm, clear /dev/shm leftovers, launch (launch.sh l3), wait for /health,
# replay, kill, wait for the port to free. Server log -> server_<tag>.log, client -> client_<tag>.log.
# After the non-"ours" arms: build_profile_v2.py over their logs -> <PREFIX>_profile.json.
# Finally: eval_policy.py + tier_mix.py for every tag -> <PREFIX>_results.txt.
set -uo pipefail
cd "$(dirname "$0")"

PREFIX=${1:?usage: $0 PREFIX WORKLOAD.jsonl}
WORKLOAD=${2:?usage: $0 PREFIX WORKLOAD.jsonl}
MODEL=${MODEL:-$HOME/huggingface/Llama-3.1-8B-Instruct}
KV_GIB=${KV_GIB:-16}; CPU_GIB=${CPU_GIB:-12}; MAX_LEN=${MAX_LEN:-32768}
DURATION=${DURATION:-1800}; MAX_OUTPUT=${MAX_OUTPUT:-1024}
ARMS=${ARMS:-"always_load static:24576 tokenflow ours"}
LD_INDEX=${LD_INDEX:-occ}; BETA=${BETA:-1.0}
SWEEP=${SWEEP:-sweep_zero_load.csv}
SKIP_DONE=${SKIP_DONE:-1}
PORT=${PORT:-8000}
PROFILE_OUT="${PREFIX}_profile.json"
RESULTS="${PREFIX}_results.txt"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${PREFIX}_run.log"; }

kill_server() {
  pkill -f "launch.sh l" 2>/dev/null || true       # the launcher too, or it starts a server later
  pkill -f "vllm serve" 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 1; done
  pkill -9 -f "vllm serve" 2>/dev/null || true
  rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
  # wait for the port to be released
  for _ in $(seq 1 30); do (echo >/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
trap 'log "interrupted — killing server"; kill_server; exit 130' INT TERM

wait_health() {   # $1 = server log, $2 = launcher pid
  local t=0
  until curl -sf -m 2 "http://localhost:$PORT/health" >/dev/null; do
    sleep 5; t=$((t+5))
    # "dead" only if BOTH the launcher (may still be clearing the tier / importing) and vllm are gone
    if ! kill -0 "$2" 2>/dev/null && ! pgrep -f "vllm serve" >/dev/null; then
      log "server died during startup (see $1)"; return 1
    fi
    if [ $t -ge "${HEALTH_TIMEOUT_S:-1800}" ]; then log "server not healthy after ${HEALTH_TIMEOUT_S:-1800}s (see $1)"; return 1; fi
  done
  sleep 15   # let cuda graphs / warmup settle
}

tag_of() {                                 # always_load -> P_always_load ; static:24576 -> P_static_24576 ; ours@beta=0.5 -> P_ours_b0.5 ; ours@kv=0.9 -> P_ours_kv0.9
  local a=${1//:/_}; a=${a//@betaio=/_bio}; a=${a//@beta=/_b}; a=${a//@kv=/_kv}; a=${a//@warm=*/_warm}; echo "${PREFIX}_${a}"
}
kv_of()   { if [[ "$1" == *@kv=* ]]; then echo "${1##*@kv=}"; else echo ""; fi; }
warm_of() { if [[ "$1" == *@warm=* ]]; then echo "${1##*@warm=}"; else echo ""; fi; }
is_online(){ [[ "$1" == online* ]]; }
# arms that need the in-deployment profile: run after the profiling arms, with that profile
is_ours() { [[ "$1" == ours* || "$1" == table ]]; }
beta_of() { if [[ "$1" == *@beta=* ]]; then echo "${1##*@beta=}"; else echo "$BETA"; fi; }
betaio_of() { if [[ "$1" == *@betaio=* ]]; then echo "${1##*@betaio=}"; else echo "${BETA_IO:-1.0}"; fi; }

run_arm() {
  local arm=$1 profile=${2:-}
  local tag; tag=$(tag_of "$arm")
  local policy=${arm%%@*}                    # strip @beta=... / @kv=... for the POLICY env
  local beta; beta=$(beta_of "$arm")
  local betaio; betaio=$(betaio_of "$arm")
  local kvg; kvg=$(kv_of "$arm")
  local conn=policy online_cold=1 warm; warm=$(warm_of "$arm")
  if is_online "$arm"; then conn=online; [ -n "$warm" ] && online_cold=0; fi
  if [ "$SKIP_DONE" = 1 ] && [ -s "${tag}_bg.jsonl" ]; then log "skip $arm (found ${tag}_bg.jsonl)"; return 0; fi
  log "=== arm=$arm  tag=$tag  KV=${KV_GIB}G CPU=${CPU_GIB}G  tier=${DISK_ROOT:-nfs-default}  profile=${profile:-none} beta=$beta kv_guard=${kvg:-off} ==="
  kill_server
  if [ "$conn" = online ]; then
    # fresh state file per tag unless warm-starting from a given file
    if [ -n "$warm" ]; then cp -f "$warm" "${tag}_online_state.json"; else rm -f "${tag}_online_state.json"; fi
  fi
  MAX_LEN=$MAX_LEN CONNECTOR=$conn POLICY=$policy KV_GIB=$KV_GIB CPU_GIB=$CPU_GIB MODEL=$MODEL \
    PROFILE=${profile:-$PWD/profile.json} LD_INDEX=$LD_INDEX BETA=$beta BETA_IO=$betaio KV_GUARD=$kvg \
    ONLINE_STATE="$PWD/${tag}_online_state.json" ONLINE_COLD=$online_cold \
    ./launch.sh l3 > "server_${tag}.log" 2>&1 &
  local launch_pid=$!
  wait_health "server_${tag}.log" "$launch_pid" || { kill_server; return 1; }
  grep -E "PolicyOffloadingConnector arm=|online selector|GPU KV budget|cpu_bytes_to_use|Created secondary tier" "server_${tag}.log" | head -5 | sed 's/^/    /' | tee -a "${PREFIX}_run.log"
  if [ -n "${PREWARM_CMD:-}" ]; then
    # A2: put the workload's documents into the SSD tier before replay (steady state, not cold start)
    local tp=$(date +%s)
    eval "$PREWARM_CMD" > "${tag}_prewarm.log" 2>&1 || log "    prewarm FAILED (see ${tag}_prewarm.log)"
    log "    prewarm done in $(( $(date +%s) - tp ))s: $(tail -n 1 "${tag}_prewarm.log")"
  fi
  if [ "$conn" = online ] && [ "${CALIBRATE:-1}" = 1 ]; then
    # startup calibration: zero-load probes through the forced-policy path fill the idle columns
    local t0=$(date +%s)
    # grid stops at 24576: h + tail must stay under max-model-len 32768. reps = N_MIN + 2 for margin.
    python3 sweep.py --model "$MODEL" --h 512 2048 8192 16384 24576 --reps "${CALIB_REPS:-$(( ${ONLINE_NMIN:-3} + 2 ))}" \
      --flush-gib $((CPU_GIB + 2)) --out "${tag}_calib.csv" > "${tag}_calib.log" 2>&1
    log "    calibration done in $(( $(date +%s) - t0 ))s ($(grep -c . "${tag}_calib.csv" 2>/dev/null || echo 0) rows)"
  fi
  # --max-input: replay_probe는 기본으로 32768 토큰 초과 요청을 버린다 → 긴 컨텍스트(NarrativeQA 64K)에서는 MAX_INPUT 전달 필수
  python3 replay_probe.py --model "$MODEL" --workload "$WORKLOAD" --duration "$DURATION" --max-input "${MAX_INPUT:-32768}" \
    --probe-warmup-wait 9999 --max-output "$MAX_OUTPUT" --tag "$tag" > "client_${tag}.log" 2>&1
  local rc=$?
  tail -n 2 "client_${tag}.log" | sed 's/^/    /' | tee -a "${PREFIX}_run.log"
  # persist end-of-run server metrics next to the client log so the evaluator can read them
  curl -s "http://localhost:$PORT/metrics" > "${tag}_metrics.txt" 2>/dev/null
  grep -E "^vllm:num_preemptions_total" "${tag}_metrics.txt" | sed 's/^/    /' | tee -a "${PREFIX}_run.log"
  # sanity: a run where most requests errored (engine crash, 5xx) is a failure, not a result
  local n_all n_err
  # grep -c exits 1 on zero matches, so `|| echo 0` would append a second line;
  # use `|| true` and default empties to 0 instead.
  n_all=$( { wc -l < "${tag}_bg.jsonl"; } 2>/dev/null | tr -d ' ' )
  n_err=$( { grep -c '"error"' "${tag}_bg.jsonl" || true; } 2>/dev/null | head -1 )
  n_all=${n_all:-0}; n_err=${n_err:-0}
  log "    requests: $n_all total, $n_err errors"
  if [ "$n_all" -gt 0 ] && [ $((n_err * 5)) -gt "$n_all" ]; then      # > 20% errors
    log "    !!! $tag FAILED: $n_err/$n_all requests errored — see server_${tag}.log"
    grep -nE "Traceback|Error" "server_${tag}.log" | head -3 | sed 's/^/    /' | tee -a "${PREFIX}_run.log"
    mv "${tag}_bg.jsonl" "${tag}_bg.FAILED.jsonl"; mv "${tag}_probes.csv" "${tag}_probes.FAILED.csv" 2>/dev/null
    rc=1
  fi
  kill_server
  # Start deleting this arm's tier files NOW in the background, so the next launch (which clears
  # synchronously) finds little or nothing left. On NFS this hides 5-15 min per arm.
  local root=${DISK_ROOT:-$HOME/jrpark/jr/kvcache/fs}
  if [ -n "$(ls -A "$root" 2>/dev/null | head -1)" ]; then
    local trash="${root%/}.trash.early.$$.$RANDOM"
    mv "$root" "$trash" 2>/dev/null && mkdir -p "$root" && nohup sh -c "find '$trash' -delete" >/dev/null 2>&1 &
  fi
  return $rc
}

log "regime $PREFIX: workload=$WORKLOAD arms=[$ARMS] duration=${DURATION}s out=${MAX_OUTPUT}"

STOP_ON_FAIL=${STOP_ON_FAIL:-1}
PROFILING_TAGS=(); OURS_ARMS=()
for arm in $ARMS; do
  if is_ours "$arm"; then OURS_ARMS+=("$arm"); continue; fi
  if run_arm "$arm"; then
    PROFILING_TAGS+=("$(tag_of "$arm")")
  else
    log "arm $arm FAILED"
    if [ "$STOP_ON_FAIL" = 1 ]; then log "STOP_ON_FAIL=1 — aborting regime $PREFIX"; kill_server; exit 1; fi
  fi
done

if [ ${#OURS_ARMS[@]} -gt 0 ]; then
  if [ -n "${OURS_PROFILE:-}" ]; then
    PROFILE_USE=$(readlink -f "$OURS_PROFILE"); log "using external profile $PROFILE_USE"
  else
    log "=== building in-deployment profile from: ${PROFILING_TAGS[*]} ==="
    python3 build_profile_v2.py --tags "${PROFILING_TAGS[@]}" --sweep "$SWEEP" --out "$PROFILE_OUT" \
      > "${PREFIX}_profile_build.log" 2>&1 && tail -n 20 "${PREFIX}_profile_build.log" | tee -a "${PREFIX}_run.log" \
      || log "profile build FAILED (see ${PREFIX}_profile_build.log)"
    PROFILE_USE="$PWD/$PROFILE_OUT"
  fi
  if [ -s "$PROFILE_USE" ]; then
    for arm in "${OURS_ARMS[@]}"; do run_arm "$arm" "$PROFILE_USE" || log "arm $arm FAILED"; done
  else
    log "no profile — skipping ours"
  fi
fi

log "=== evaluation ==="
TAGS=(); for arm in $ARMS; do TAGS+=("$(tag_of "$arm")"); done
{
  echo "regime $PREFIX  KV=${KV_GIB}G CPU=${CPU_GIB}G maxlen=$MAX_LEN out=$MAX_OUTPUT tier=${DISK_ROOT:-nfs-default} workload=$WORKLOAD"
  echo; python3 eval_policy.py --tag "${TAGS[@]}"
  echo; echo "# looser TBT SLO"; python3 eval_policy.py --tag "${TAGS[@]}" --slo-tbt-ms 100 --slo-ttft-ms 4000
  for t in "${TAGS[@]}"; do echo; echo "--- $t"; python3 tier_mix.py --tag "$t" 2>/dev/null; done
} | tee "$RESULTS"
log "done -> $RESULTS"