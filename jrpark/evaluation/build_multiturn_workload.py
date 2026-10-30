#!/usr/bin/env python3
"""문서 기반 멀티턴 대화 워크로드 ("문서를 올려두고 질문을 이어서 하는" 세션).
턴 k 프롬프트 = 문서 + Q1 A1 … Q(k−1) A(k−1) + Qk   (A는 데이터셋 정답 텍스트 → arm 간 프롬프트 동일, 공정 비교)
턴 k+1 프롬프트는 턴 k 프롬프트 + " A_k" + 다음 질문 → 이전 턴 전체가 prefix hit, 새 부분(질문·답)은 매 턴 저장(쓰기)
세션은 Poisson 도착, 턴 사이 생각 시간 ~ 지수분포. timestamp는 '예정' 시각이며, replay의 세션 순서 대기(--session-order)가
이전 턴 완료 전에는 다음 턴을 보내지 않게 막는다.
출력 형식은 build_longctx_workload.py와 같음 (+ session, turn, prev_id, think_s).
사용 예:
  python3 build_multiturn_workload.py --src ~/huggingface/trace/LEval --model ~/huggingface/Llama-3.1-8B-Instruct \
     --session-rps 0.08 --max-turns 5 --think-mean 60 --duration 1800 --seed 51 --out mt_leval_s0.08.jsonl
"""
import argparse, json, math, os, random, sys
from collections import defaultdict

DOC_KEYS = ["input", "context", "document", "text", "story", "passage", "article", "content"]
Q_KEYS = ["instructions", "questions", "instruction", "question", "query", "queries"]
A_KEYS = ["outputs", "answers", "output", "answer", "response"]


def iter_records(src):
    paths = []
    for root, _, fs in os.walk(src) if os.path.isdir(src) else [(os.path.dirname(src), None, [os.path.basename(src)])]:
        paths += [os.path.join(root, f) for f in fs if f.endswith((".jsonl", ".json"))]
    for p in sorted(paths):
        with open(p) as fh:
            txt = fh.read().strip()
        recs = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
        for r in recs:
            yield p, r


def pick(rec, keys, exclude=()):
    for k in keys:
        if k in rec and k not in exclude and rec[k]:
            return k, rec[k]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--min-input", type=int, default=4000); ap.add_argument("--max-input", type=int, default=30000)
    ap.add_argument("--max-turns", type=int, default=5, help="세션당 최대 턴 수 (질문 수가 적으면 그만큼)")
    ap.add_argument("--min-turns", type=int, default=3)
    ap.add_argument("--truncate-doc", type=int, default=0,
                    help="문서를 앞에서부터 이 토큰 수로 자름 (0=안 자름). 시스템 성능 실험에서는 답의 정답 여부가 아니라 길이·턴 구조가 중요")
    ap.add_argument("--session-rps", type=float, required=True, help="세션 도착률 (요청률 ≈ session-rps × 평균 턴 수)")
    ap.add_argument("--think-mean", type=float, default=60.0, help="턴 사이 생각 시간 평균 (s, 지수분포)")
    ap.add_argument("--think-min", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=1800, help="이 시간 안에 시작하는 세션만 (턴은 넘어갈 수 있음)")
    ap.add_argument("--out-min", type=int, default=32); ap.add_argument("--out-max", type=int, default=256)
    ap.add_argument("--long-frac", type=float, default=0.0); ap.add_argument("--long-output", type=int, default=1024)
    ap.add_argument("--doc-reuse", action="store_true", help="문서 소진 시 재사용 (다른 세션이 같은 문서로 시작 → 첫 턴도 hit 가능)")
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake-tokenizer", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.fake_tokenizer:
        ntok = lambda s: max(1, len(s) // 4)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    docs = []
    for src in a.src:
        for p, rec in iter_records(src):
            if isinstance(rec.get("multi_turns"), list):                    # SCBench 멀티턴 형식
                doc = rec.get("context")
                qs, ans = [], []
                for tt in rec["multi_turns"]:
                    q = tt.get("input") or ""
                    if tt.get("options"): q += "\nOptions: " + " / ".join(map(str, tt["options"]))
                    x = tt.get("answer")
                    x = ", ".join(map(str, x)) if isinstance(x, list) else ("" if x is None else str(x))
                    qs.append(q); ans.append(x)
                if not isinstance(doc, str) or not qs: continue
            else:
                dk, doc = pick(rec, DOC_KEYS)
                qk, qs = pick(rec, Q_KEYS, exclude=(dk,))
                ak, ans = pick(rec, A_KEYS, exclude=(dk, qk))
                if not isinstance(doc, str) or not qs or not ans: continue
                qs = [qs] if isinstance(qs, str) else qs; ans = [ans] if isinstance(ans, str) else ans
            pairs = [(q.strip(), (x if isinstance(x, str) else str(x)).strip()) for q, x in zip(qs, ans)
                     if isinstance(q, str) and q.strip() and x]
            if len(pairs) < a.min_turns: continue
            if a.truncate_doc and not a.fake_tokenizer:
                ids = tok(doc, add_special_tokens=False)["input_ids"]
                if len(ids) > a.truncate_doc: doc = tok.decode(ids[: a.truncate_doc])
            elif a.truncate_doc and len(doc) > 4 * a.truncate_doc:
                doc = doc[: 4 * a.truncate_doc]
            n = ntok(doc)
            if n < a.min_input: continue
            pairs = pairs[:a.max_turns]
            # 마지막 턴 프롬프트 길이가 max-input을 넘지 않도록 턴 수를 줄인다
            qa_tok = [ntok(f"\n\nQ: {q}\nA: {x}") for q, x in pairs]
            while pairs and n + sum(qa_tok[:len(pairs)]) > a.max_input: pairs = pairs[:-1]
            if len(pairs) < a.min_turns: continue
            task = os.path.splitext(os.path.basename(p))[0]
            docs.append((f"{task}#{len(docs)}", task, doc, pairs, n))
    if not docs: sys.exit("조건에 맞는 문서 없음")
    print(f"문서 {len(docs)}개, 평균 문서 {sum(d[4] for d in docs)/len(docs):,.0f} tok, "
          f"평균 턴 {sum(len(d[3]) for d in docs)/len(docs):.1f}")

    rng = random.Random(a.seed)
    order = list(range(len(docs))); rng.shuffle(order)
    reqs, t, s_id, used = [], 0.0, 0, 0
    while True:
        t += rng.expovariate(a.session_rps)
        if t >= a.duration: break
        if used >= len(order):
            if not a.doc_reuse: print("[경고] 문서 소진 → 세션 생성 중단 (--doc-reuse 또는 도착률 낮추기)"); break
            rng.shuffle(order); used = 0
        did, task, doc, pairs, n = docs[order[used]]; used += 1
        prompt, tt, prev = doc, t, None
        for k, (q, ans) in enumerate(pairs):
            prompt = prompt + f"\n\nQ: {q}\nA:"
            out = max(a.out_min, min(a.out_max, ntok(ans)))
            if a.long_frac and rng.random() < a.long_frac: out = a.long_output
            rid = len(reqs)
            think = 0.0 if k == 0 else max(a.think_min, rng.expovariate(1.0 / a.think_mean))
            tt = tt + think
            reqs.append(dict(timestamp=int(tt * 1000), doc_id=did, task=task, q_idx=k, session=s_id, turn=k,
                             prev_id=prev, think_s=round(think, 2), prompt=prompt,
                             input_length=ntok(prompt), output_length=out, phase_rps=a.session_rps))
            prev = rid
            prompt = prompt + f" {ans}"
        s_id += 1
    reqs.sort(key=lambda r: r["timestamp"])
    # 정렬 후 인덱스로 id·prev_id 다시 매핑
    by_key = {(r["session"], r["turn"]): i for i, r in enumerate(reqs)}
    for r in reqs:
        r["prev_id"] = by_key.get((r["session"], r["turn"] - 1)) if r["turn"] > 0 else None
        r["id"] = by_key[(r["session"], r["turn"])]
    with open(a.out, "w") as f:
        f.write(json.dumps({"_meta": vars(a) | {"n_sessions": s_id, "n_requests": len(reqs)}}) + "\n")
        for r in reqs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    T = a.duration
    in_tok = [r["input_length"] for r in reqs]; new_tok = []
    for r in reqs:
        new_tok.append(r["input_length"] if r["turn"] == 0 else r["input_length"] - reqs[r["prev_id"]]["input_length"])
    hit_tok = sum(in_tok) - sum(new_tok)
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    print(f"세션 {s_id}개, 요청 {len(reqs)}개 ({len(reqs)/T:.3f} req/s), 턴1 비율 {sum(r['turn']==0 for r in reqs)/len(reqs):.0%}")
    print(f"입력 p10/p50/p90: {q(in_tok,.1)}/{q(in_tok,.5)}/{q(in_tok,.9)} tok, 출력 평균 {sum(r['output_length'] for r in reqs)/len(reqs):.0f}")
    print(f"재사용 가능 토큰 비율(이전 턴 prefix): {hit_tok/sum(in_tok):.1%}, 요청당 새 토큰(쓰기) 평균 {sum(new_tok)/len(reqs):,.0f}")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()