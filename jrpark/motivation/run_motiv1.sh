#!/usr/bin/env bash
# motivation 1부: LEval + ShareGPT 혼합 워크로드를 native vLLM(always_load + 읽기 전용 로그)으로 실행.
#
# 사용:
#   ./run_motiv1.sh smoke     10분 점검 1회 (문서 0.73 요청/s, 대화 1.0 세션/s)
#   ./run_motiv1.sh main      30분 x 4조건 (문서 DOCS="0.73 1.03" x 대화 CHATS="0.5 1.0")
#   ./run_motiv1.sh native    점검과 같은 조건을 NATIVE_CONN 커넥터로 1회 → 점검 결과와 비교
#   ./run_motiv1.sh one <문서 rps> <대화 rps> <초>   임의 조건 1회
#   끝나면 training_validation.py 실행 (run_batch3.sh와 같음). 끄려면 POST= ./run_motiv1.sh ...
#
# 실행마다 서버를 새로 띄우며, launch.sh가 DISK_ROOT(fs tier)를 비우므로 빈 캐시에서 시작한다.
# 결과: $OUTDIR/<tag>_client.jsonl, <tag>_server.log, <tag>_analysis.txt, CSV
#       서버 결정 로그는 $LOGDIR/<tag>.jsonl (+ _steps.jsonl), 끝나면 $OUTDIR로 복사.
set -u
cd "$(dirname "$0")"

MODEL=${MODEL:-$HOME/huggingface/Llama-3.1-8B-Instruct}
PORT=${PORT:-8000}
SESS=${SESS:-sessions_mix.jsonl}
LOGDIR=${LOGDIR:-/dev/shm/motiv1}         # 서버 로그는 SSD tier와 다른 곳에
OUTDIR=${OUTDIR:-motiv1}
TIER=${TIER:-l3}
# 서버 설정: 본 실험 LEval과 같음
SRV_ENV=(DISK_ROOT=/tmp/kvcache/fs KV_GIB=${KV_GIB:-32} CPU_GIB=${CPU_GIB:-32} L2_CPU_GIB=${CPU_GIB:-32} MAX_LEN=${MAX_LEN:-32768}
         PYTHONHASHSEED=0)
NATIVE_CONN=${NATIVE_CONN:-native}         # native 비교: launch.sh에 추가한 CONNECTOR=native (vLLM 기본 OffloadingConnector)

mkdir -p "$LOGDIR" "$OUTDIR"
log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$OUTDIR/run_motiv1.log"; }

grep -q MOTIV_LOG policy_connector.py || { log "policy_connector.py 미패치 → python3 patch_motiv_log.py policy_connector.py"; exit 1; }
[ -s profile.json ] || { log "profile.json 없음 (PolicyOffloadingConnector가 기동 시 읽음)"; exit 1; }
[ -s "$SESS" ] || { log "$SESS 없음 → prep_mix.py 먼저"; exit 1; }

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  for _ in $(seq 1 60); do pgrep -f "vllm serve" >/dev/null || return 0; sleep 2; done
  pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 5
}

start_server() {   # $1 = tag, $2 = connector (policy | NATIVE_CONN 값)
  local tag=$1 conn=$2
  stop_server
  rm -f /tmp/NO_SSD_STORE                  # SSD 쓰기 차단 플래그가 남아 있으면 쓰기 큐가 안 생김
  rm -rf __pycache__; touch policy_connector.py probe_connector.py   # .pyc 함정 방지
  rm -f "$LOGDIR/${tag}_rand.jsonl"       # 이전 실행의 rand 배정 기록이 섞이지 않도록
  log "서버 기동: $tag (connector=$conn)"
  env -u NO_SSD_STORE "${SRV_ENV[@]}" MODEL="$MODEL" PORT="$PORT" CONNECTOR="$conn" \
      POLICY="${POLICY:-always_load}" MOTIV_LOG=1 ONLINE_HW=0 PROBE_LOG="$LOGDIR/$tag.jsonl" \
      nohup ./launch.sh "$TIER" > "$OUTDIR/${tag}_server.log" 2>&1 &
  for i in $(seq 1 360); do                # 최대 30분
    if curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1; then log "  준비됨 ($((i * 5))s)"; return 0; fi
    if [ "$i" -gt 12 ] && ! pgrep -f "vllm serve" >/dev/null; then
      log "  서버가 죽음 → $OUTDIR/${tag}_server.log"; tail -20 "$OUTDIR/${tag}_server.log"; return 1
    fi
    sleep 5
  done
  log "  30분 안에 준비 안 됨"; return 1
}

run_one() {   # $1 tag, $2 문서 rps, $3 대화 rps, $4 초, $5 connector
  local tag=$1 dr=$2 cr=$3 dur=$4 conn=${5:-policy}
  start_server "$tag" "$conn" || return 1
  log "재생: 문서 $dr 요청/s, 대화 $cr 세션/s, ${dur}s"
  python3 replay_mix.py --url "http://localhost:$PORT" --model "$MODEL" --sessions "$SESS" \
      --doc-rate "$dr" --chat-rate "$cr" --duration "$dur" --drain 300 --seed "${SEED:-51}" \
      --out "$OUTDIR/${tag}_client.jsonl" 2>&1 | tee "$OUTDIR/${tag}_replay.log" | grep -E "예정|중단|실패 예|경고|끝" || true
  stop_server
  if [ -s "$LOGDIR/$tag.jsonl" ]; then
    cp "$LOGDIR/$tag.jsonl" "$LOGDIR/${tag}_steps.jsonl" "$OUTDIR/" 2>/dev/null || true
    cp "$LOGDIR/${tag}_rand.jsonl" "$OUTDIR/" 2>/dev/null || true   # POLICY=rand:... 배정 기록
    python3 analyze_mix.py --client "$OUTDIR/${tag}_client.jsonl" --server "$OUTDIR/$tag.jsonl" \
        --steps "$OUTDIR/${tag}_steps.jsonl" \
        --warmup "${WARMUP:-$((dur / 6))}" --prefix "$OUTDIR/$tag" > "$OUTDIR/${tag}_analysis.txt" 2>&1
    log "분석: $OUTDIR/${tag}_analysis.txt"; head -12 "$OUTDIR/${tag}_analysis.txt"
  else
    log "서버 결정 로그 없음 ($LOGDIR/$tag.jsonl) — native 커넥터이면 정상"
  fi
}

client_summary() {   # 클라이언트 기록만으로 TTFT·토큰 간격 요약 (native 비교용)
  python3 - "$@" <<'PY'
import json, sys
def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] if v else float("nan")
for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f) if l.strip() and "meta" not in l]
    ok = [r for r in rs if r.get("status") == "ok" and r.get("t_first") is not None]
    out = []
    for kind in ("doc", "chat"):
        k = [r for r in ok if r["kind"] == kind and r["t_send"] > 100]
        tt = [(r["t_first"] - r["t_send"]) * 1000 for r in k]
        iv = [(b - a) * 1000 for r in k for a, b in zip(r["t_chunks"], r["t_chunks"][1:])]
        out.append(f"{kind}: n={len(k)} TTFT p50 {pct(tt, 50):.0f} p95 {pct(tt, 95):.0f} ms | 간격 p50 {pct(iv, 50):.1f} p95 {pct(iv, 95):.1f} ms")
    print(f"{f}: 실패 {len(rs) - len(ok)} | " + " | ".join(out))
PY
}

case "${1:-}" in
  smoke)  run_one smoke_d0.73_c1.0 0.73 1.0 600 ;;
  main)
    for dr in ${DOCS:-0.73 1.03}; do for cr in ${CHATS:-0.5 1.0}; do
      run_one "d${dr}_c${cr}" "$dr" "$cr" 1800 || log "실패: d${dr}_c${cr} (다음 조건으로 진행)"
    done; done
    log "끝. 요약:"; for f in "$OUTDIR"/d*_analysis.txt; do echo "== $f"; head -12 "$f"; done ;;
  native)
    grep -q "\"\$CONNECTOR\" = \"$NATIVE_CONN\"" launch.sh || { log "launch.sh에 CONNECTOR=$NATIVE_CONN 분기 없음"; exit 1; }
    [ -s "$OUTDIR/smoke_d0.73_c1.0_client.jsonl" ] || { log "smoke 결과가 먼저 있어야 함"; exit 1; }
    run_one native_d0.73_c1.0 0.73 1.0 600 "$NATIVE_CONN"
    client_summary "$OUTDIR/smoke_d0.73_c1.0_client.jsonl" "$OUTDIR/native_d0.73_c1.0_client.jsonl" ;;
  one)    run_one "d${2}_c${3}_t${4}" "$2" "$3" "$4" ;;
  *)      sed -n 2,13p "$0"; exit 1 ;;
esac
stop_server
log "########## 완료"
POST=${POST-/home/work/jrpark/training_validation.py}
if [ -n "$POST" ] && [ -s "$POST" ]; then
  log "########## 후속 작업: $POST"; cd "$(dirname "$POST")" && python3 "$POST" 2>&1 | tee -a "$OLDPWD/$OUTDIR/run_motiv1_post.log"
fi
