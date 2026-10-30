#!/usr/bin/env bash
# 새 모델 실험 준비 (기존 Llama 디렉터리 jr에서 모델마다 한 번):
#   MODEL=$HOME/huggingface/Llama-3.2-3B-Instruct   QDIR=../jrl3  ./prep_model.sh
#   MODEL=$HOME/huggingface/Phi-4-mini-instruct     QDIR=../jrphi ./prep_model.sh
#   MODEL=$HOME/huggingface/Qwen3-4B-Instruct-2507  QDIR=../jrq3  ./prep_model.sh
# 1) config로 토큰당 KV(B_KV) 계산 → QDIR/.bkv, 모델 경로 → QDIR/.model
# 2) 코드·Llama seed 51 워크로드 15개 복사 (같은 텍스트·도착 시각)
# 3) 복사 시간 표를 KV 비율로 조정한 MODEL_profile.json과 profile.json
# 4) 이 모델 토크나이저로 워크로드 길이 확인 (32768 초과 요청 수)
set -uo pipefail
cd "$(dirname "$0")"
: "${MODEL:?MODEL=<모델 경로>}"; : "${QDIR:?QDIR=<새 디렉터리>}"
[ -s "$MODEL/config.json" ] || { echo "모델 없음: $MODEL"; exit 1; }
grep -q 'B_KV = int(os.environ.get("B_KV"' online_hw.py || { echo "online_hw.py의 B_KV가 환경변수를 안 읽음"; exit 1; }
[ -s run_model.sh ] || { echo "run_model.sh를 이 디렉터리에 먼저 두기"; exit 1; }
BKV=$(python3 - "$MODEL" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json")); c = c.get("text_config", c); A = c["num_attention_heads"]
print(2 * c["num_hidden_layers"] * c.get("num_key_value_heads", A) * (c.get("head_dim") or c["hidden_size"] // A) * 2)
EOF
)
echo "$(basename "$MODEL"): 토큰당 KV $BKV 바이트 (Llama-3.1-8B 대비 $(python3 -c "print(round($BKV/131072, 3))")배)"

mkdir -p "$QDIR"
cp -n *.py *.sh LMIX_profile.json "$QDIR"/ 2>/dev/null
cp -f run_model.sh "$QDIR"/; [ -s model_report.sh ] && cp -f model_report.sh "$QDIR"/
for f in lio2_d0p64_w180_r0.73.jsonl lio2_d0p64_w180_r1.03.jsonl lio2_d0p64_w180_r1.25.jsonl \
         mt_scb_r0.055_s51.jsonl mt_scb_r0.075_s51.jsonl mt_scb_r0.09_s51.jsonl \
         mt_sg_r1.5_s51.jsonl mt_sg_r2.5_s51.jsonl mt_sg_r3.0_s51.jsonl \
         $(ls lio2_loogle_d0p40_w300_q6_r*.jsonl 2>/dev/null); do
  [ -s "$f" ] && cp -n "$f" "$QDIR"/ || echo "[경고] 워크로드 없음: $f"
done
echo "$MODEL" > "$QDIR/.model"; echo "$BKV" > "$QDIR/.bkv"

python3 - "$QDIR" "$BKV" <<'PY'
import json, sys
q, bkv = sys.argv[1], int(sys.argv[2]); r = bkv / 131072
p = json.load(open(f"{q}/LMIX_profile.json"))
sc = lambda v: [x * r for x in v] if isinstance(v, list) else (v * r if isinstance(v, (int, float)) else v)
p["ld_dram_ms"] = {k: sc(v) for k, v in p["ld_dram_ms"].items()}
p["source"] = f"{p.get('source', '')} | ld_dram_ms scaled x{r:.4f} ({bkv} B/token)"
for n in ("MODEL_profile.json", "profile.json"): json.dump(p, open(f"{q}/{n}", "w"), indent=1)
print(f"MODEL_profile.json, profile.json: ld_dram_ms ×{r:.4f}")
PY

python3 - "$QDIR" "$MODEL" <<'PY'
import glob, json, sys
from transformers import AutoTokenizer
q, m = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(m)
print("%-40s %6s %8s %8s %8s %6s" % ("워크로드", "요청", "Llama중앙", "이모델중앙", "최대", "초과"))
for f in sorted(glob.glob(f"{q}/lio2_*.jsonl") + glob.glob(f"{q}/mt_*.jsonl")):
    R = [json.loads(l) for l in open(f) if l.strip()]
    R = [r for r in R if "prompt" in r and r.get("input_length", 0) <= 30000]
    seen, ql, ll, over = {}, [], [], 0
    for r in R:
        p = r["prompt"]
        n = seen.get(p) or seen.setdefault(p, len(tok(p).input_ids))
        ql.append(n); ll.append(r.get("input_length", 0)); over += n + min(1024, r.get("output_length", 32)) > 32768
    if not ql: continue
    ql.sort(); ll.sort()
    print("%-40s %6d %8d %8d %8d %6d" % (f.split("/")[-1][-40:], len(ql), ll[len(ll) // 2], ql[len(ql) // 2], ql[-1], over))
PY
echo "준비 완료 → $QDIR   다음: cd $QDIR && DRY=1 ./run_model.sh"