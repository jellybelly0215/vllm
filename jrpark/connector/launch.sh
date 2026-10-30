#!/usr/bin/env bash
# vLLM 0.23.0 native KV offloading launch configs for recompute-vs-load motivation experiment.
# Usage: ./launch.sh l2 | l3        (l2 = DRAM tier only, l3 = DRAM + SSD fs tier)
#
# Llama-3.1-8B: 128 KB/token (FP16)  ->  16-token block = 2 MiB.
# cpu_bytes_to_use controls which tier a hit lands in:
#   L2 config: large CPU tier  -> probes hit in DRAM
#   L3 config: CPU tier ~2x max probe KV -> background/flush traffic evicts probe blocks
#              from DRAM; they remain on SSD -> probes hit in SSD (promoted via CPU).
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
PORT=${PORT:-8000}
DISK_ROOT=${DISK_ROOT:-$HOME/jrpark/jr/kvcache/fs}   # default: NFSv3 (Isilon NAS) - measured 1.6 GB/s rand read @QD16, 80 us/token. Local RAID via overlay: DISK_ROOT=/tmp/kvcache/fs - 3.1 GB/s, 42 us/token
MAX_LEN=${MAX_LEN:-131072}   # Mooncake trace inputs reach 126K; 8B KV budget on 80GB ~470K tokens
CONNECTOR_DIR=${CONNECTOR_DIR:-$PWD}           # dir containing probe_connector.py
CPU_GIB=${CPU_GIB:-10}                          # l3 CPU staging tier; sweep.py uses 5 so a 4x16K flush evicts probes fast
# GPU KV budget. Unset -> vLLM infers from gpu-memory-utilization (~60 GiB = 470K tokens
# for Llama-3.1-8B), which is large enough that an hour of Mooncake replay almost never
# reaches the offload tiers. Set e.g. KV_GIB=8 to put the L2/L3 decision on the critical path.
KV_GIB=${KV_GIB:-}
FS_READ_THREADS=${FS_READ_THREADS:-16}          # vLLM default. 32 measured 26% more SSD bandwidth (2025 vs 1610 MB/s)
FS_WRITE_THREADS=${FS_WRITE_THREADS:-16}

export PYTHONHASHSEED=0        # stable block hashes -> fs tier stays warm across restarts
export VLLM_SERVER_DEV_MODE=1  # enables POST /reset_prefix_cache (used by smoke.py / sweeps)
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export PROBE_LOG=${PROBE_LOG:-./probe_log.jsonl}

GiB=$((1<<30))

# vLLM's CPU offload tier lives in /dev/shm/vllm_offload_*.mmap and is not removed on
# Ctrl-C. Four leftovers (10 GiB each) fill a 40 GiB /dev/shm -> madvise EFAULT at start.
if ! pgrep -f "vllm serve" >/dev/null; then
  rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
fi
trap 'rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null' EXIT

case "${1:-}" in
  l2)
    # 40 GiB DRAM tier (L2_CPU_GIB로 변경 가능) = ~320K tokens of Llama-3.1-8B KV. Everything hits in DRAM.
    EXTRA=$(cat <<EOF
{"spec_name":"CPUOffloadingSpec",
 "cpu_bytes_to_use":$((${L2_CPU_GIB:-40}*GiB)),
 "block_size":16,
 "eviction_policy":"lru"}
EOF
);;
  l3)
    # 10 GiB DRAM tier = ~80K tokens: must fit a 32K probe (4 GiB) as promotion staging,
    # yet small enough that a few background requests evict probe blocks from DRAM.
    # FS tier on SSD holds everything (cascade).
    mkdir -p "$DISK_ROOT"
    # The fs tier never evicts: every offloaded block stays as one file. A few 25-min
    # replays left millions of files, which made lookup/store stall the scheduler.
    # Probes are pre-warmed per run, so clearing between launches is safe.
    # Clearing is SYNCHRONOUS by default: a background unlink of 30K-100K files on NFS
    # takes minutes and overlaps the next run's first phase with metadata traffic (an
    # uncontrolled, run-dependent perturbation). Costs a few minutes of startup on NFS,
    # seconds on the local tier. ASYNC_TRASH=1 restores the old behaviour.
    if [ "${KEEP_FS:-0}" != "1" ]; then
      PARENT=$(dirname "${DISK_ROOT%/}"); BASE=$(basename "${DISK_ROOT%/}")
      LEFTOVER=$(ls -d "$PARENT/$BASE".trash.* 2>/dev/null || true)   # no match -> ls exits 2; do not trip set -e
      if [ -n "$(ls -A "$DISK_ROOT" 2>/dev/null | head -1)" ]; then
        TRASH="${DISK_ROOT%/}.trash.$$"
        mv "$DISK_ROOT" "$TRASH" && mkdir -p "$DISK_ROOT"
        LEFTOVER="$LEFTOVER $TRASH"
      fi
      if [ -n "$(echo $LEFTOVER)" ]; then
        if [ "${ASYNC_TRASH:-0}" = "1" ]; then
          echo "clearing fs tier in background -> $LEFTOVER"
          nohup sh -c "find $LEFTOVER -delete; true" >/dev/null 2>&1 &
        else
          echo "clearing fs tier (synchronous) -> $LEFTOVER"
          # a background deletion of the same trash (started when the previous arm ended) may race
          # with this one: entries vanish underneath, find exits non-zero -> must not trip set -e
          t0=$(date +%s); find $LEFTOVER -delete 2>/dev/null || true; echo "  cleared in $(( $(date +%s) - t0 ))s"
        fi
      fi
    fi
    EXTRA=$(cat <<EOF
{"spec_name":"TieringOffloadingSpec",
 "cpu_bytes_to_use":$((CPU_GIB*GiB)),
 "block_size":16,
 "eviction_policy":"lru",
 "secondary_tiers":[{"type":"fs","root_dir":"$DISK_ROOT","n_read_threads":$FS_READ_THREADS,"n_write_threads":$FS_WRITE_THREADS}]}
EOF
);;
  *) echo "usage: $0 l2|l3"; exit 1;;
esac

# Custom connector. Default: ProbeOffloadingConnector (probe force switch + logging).
# CONNECTOR=policy -> PolicyOffloadingConnector, applies $POLICY to every SSD hit
# (always_load | static:<tokens> | table | tokenflow | ours), reads $PROFILE (profile.json).
CONNECTOR=${CONNECTOR:-probe}
if [ "$CONNECTOR" = "online" ]; then
  CONN_CLASS=OnlineOffloadingConnector; CONN_MODULE=online_connector
  export POLICY=online PROFILE=${PROFILE:-$CONNECTOR_DIR/profile.json} BETA=${BETA:-1.0}
  export ONLINE_STATE=${ONLINE_STATE:-$CONNECTOR_DIR/online_state.json} ONLINE_COLD=${ONLINE_COLD:-0}
  export ONLINE_EPS=${ONLINE_EPS:-0.05} ONLINE_EPS_IDLE=${ONLINE_EPS_IDLE:-0.25} ONLINE_NMIN=${ONLINE_NMIN:-3} ONLINE_EPS_LOW=${ONLINE_EPS_LOW:-0.10} ONLINE_ALPHA=${ONLINE_ALPHA:-0.1}
  echo "online selector: state=$ONLINE_STATE cold=$ONLINE_COLD eps=$ONLINE_EPS nmin=$ONLINE_NMIN beta=$BETA"
elif [ "$CONNECTOR" = "native" ]; then
  CONN_CLASS=OffloadingConnector; CONN_MODULE=vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector
elif [ "$CONNECTOR" = "policy" ]; then
  CONN_CLASS=PolicyOffloadingConnector; CONN_MODULE=policy_connector
  export POLICY=${POLICY:-always_load} PROFILE=${PROFILE:-$CONNECTOR_DIR/profile.json} BETA=${BETA:-1.0}
  echo "policy arm: $POLICY  (profile $PROFILE, beta $BETA)"
else
  CONN_CLASS=ProbeOffloadingConnector; CONN_MODULE=probe_connector
fi
KV_CFG=$(python3 - "$EXTRA" "$CONN_CLASS" "$CONN_MODULE" <<'EOF'
import json, sys
extra = json.loads(sys.argv[1])
print(json.dumps({
  "kv_connector": sys.argv[2],
  "kv_connector_module_path": sys.argv[3],
  "kv_role": "kv_both",
  "kv_connector_extra_config": extra,
}))
EOF
)

export PYTHONPATH="$CONNECTOR_DIR:${PYTHONPATH:-}"

# (not exec: keep this shell alive so the EXIT trap cleans /dev/shm afterwards)
KV_ARG=()
[ -n "$KV_GIB" ] && KV_ARG=(--kv-cache-memory-bytes $((KV_GIB*GiB))) && echo "GPU KV budget: ${KV_GIB} GiB (~$((KV_GIB*1024/128*1000/1000))K tokens @128KB/token)"

vllm serve "$MODEL" \
  "${KV_ARG[@]}" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --block-size 16 \
  --enable-prefix-caching \
  --kv-transfer-config "$KV_CFG"
# NOTE: keep chunked prefill ON (default). It is the mechanism by which recompute
# leaks into other requests' TBT (Sarathi-Serve) — i.e. the Opp term we want to measure.