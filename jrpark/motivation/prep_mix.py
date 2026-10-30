#!/usr/bin/env python3
"""motivation 1부용 세션 파일 만들기 (LEval 문서 세션 + ShareGPT 대화 세션).

출력: 한 줄에 세션 하나인 jsonl
  문서 세션: {"kind":"doc",  "sid":..., "context":문서, "turns":[{"q":질문, "out_len":N}, ...], "ctx_tokens":N}
  대화 세션: {"kind":"chat", "sid":..., "turns":[{"user":..., "assistant":..., "out_len":N}, ...]}

out_len은 데이터셋 정답/응답의 토큰 수(상한 --max-out). 재생 때 ignore_eos와 함께 쓰여
실행마다 출력 길이가 같아진다(2부에서 같은 상태를 다시 만들 때 필요).

사용 예:
  python3 prep_mix.py --leval "$HOME/huggingface/trace/LEval/LEval/*/*.jsonl" \
      --sharegpt $HOME/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
      --tokenizer $HOME/huggingface/Llama-3.1-8B-Instruct --out sessions_mix.jsonl
LEval 파일 키 이름이 다르면 --leval-keys input,instructions,outputs 로 바꾼다.
"""
import argparse, glob, json, random, sys


def make_counter(path):
    if not path:
        print("[경고] --tokenizer 없음 → 글자 수/4로 토큰 수 근사", file=sys.stderr)
        return lambda s: max(1, len(s) // 4)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def load_leval(pattern, keys, count, max_ctx, max_out, min_turns, out_min, max_q, min_ctx):
    k_in, k_q, k_out = keys.split(",")
    out = []
    for f in sorted(glob.glob(pattern)):
        for i, line in enumerate(open(f, encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ctx, qs = r.get(k_in), r.get(k_q) or []
            outs = r.get(k_out) or [""] * len(qs)
            if not ctx or len(qs) < min_turns:
                continue
            n_ctx = count(ctx)
            if n_ctx > max_ctx or n_ctx < min_ctx:
                continue
            turns = [{"q": q, "out_len": min(max_out, max(out_min, count(a or "")))} for q, a in zip(qs, outs)]
            if max_q > 0:
                turns = turns[:max_q]                     # 본 실험 --max-q-per-doc (0이면 제한 없음)
            out.append({"kind": "doc", "sid": f"doc:{f.split('/')[-1]}:{i}",
                        "context": ctx, "ctx_tokens": n_ctx, "turns": turns})
    return out


def load_sharegpt(path, count, a):
    """본 실험 build_sharegpt_workload.py와 같은 선택: 원본을 --seed로 섞고 앞 --max-convs개만 봄 →
    human→gpt 쌍만 순서대로(빈 발화 제외), 최대 --chat-max-turns턴, 최소 --chat-min-turns턴,
    마지막 턴 문맥 --chat-min-ctx 이상, 입력 --chat-max-input 이하. 통과한 순서를 그대로 유지."""
    out = []
    data = json.load(open(path, encoding="utf-8"))
    random.Random(a.seed).shuffle(data)
    for c in data[: a.max_convs]:
        conv = c.get("conversations") or []
        pairs, i = [], 0
        while i + 1 < len(conv):
            u, g = conv[i], conv[i + 1]
            if u.get("from") == "human" and g.get("from") == "gpt" and u.get("value", "").strip() and g.get("value", "").strip():
                pairs.append((u["value"], g["value"]))
            i += 2
        pairs = pairs[: a.chat_max_turns]
        if len(pairs) < a.chat_min_turns:
            continue
        turns, prompt = [], ""
        for u, g in pairs:                              # 재생기와 같은 문자열을 만들어 길이를 셈 (본 실험과 같은 방식)
            prompt = prompt + f"User: {u}\nAssistant:"
            ctx = count(prompt)
            if ctx > a.chat_max_input:
                break
            turns.append({"user": u, "assistant": g, "ctx_tokens": ctx,
                          "out_len": max(a.out_min, min(a.chat_out_max, count(g)))})
            prompt = prompt + f" {g}\n\n"
        if len(turns) < a.chat_min_turns or turns[-1]["ctx_tokens"] < a.chat_min_ctx:
            continue
        out.append({"kind": "chat", "sid": f"chat:{c.get('id', len(out))}", "turns": turns})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leval", required=True, help="LEval jsonl glob")
    ap.add_argument("--leval-keys", default="input,instructions,outputs")
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-doc-tokens", type=int, default=30000, help="본 실험 LEval과 같은 상한 (MAX_INPUT)")
    ap.add_argument("--min-doc-tokens", type=int, default=0, help="문서 토큰 하한 (본 실험 MIN_INPUT)")
    ap.add_argument("--min-doc-turns", type=int, default=2, help="질문이 2개 이상이어야 문서 재사용(hit)이 생김")
    ap.add_argument("--max-q-per-doc", type=int, default=0, help="문서당 질문 수 상한 (본 실험 MAXQ, 0이면 제한 없음)")
    ap.add_argument("--out-min", type=int, default=16, help="출력 길이 하한 (본 실험 out_min)")
    ap.add_argument("--doc-out-max", type=int, default=1024, help="문서 질문의 출력 길이 상한")
    # 아래 기본값은 본 실험 mt_sg_*.jsonl의 _meta와 같음
    ap.add_argument("--chat-min-turns", type=int, default=3)
    ap.add_argument("--chat-max-turns", type=int, default=8)
    ap.add_argument("--chat-min-ctx", type=int, default=1024, help="마지막 턴 문맥 최소 토큰")
    ap.add_argument("--chat-max-input", type=int, default=30000)
    ap.add_argument("--chat-out-max", type=int, default=512)
    ap.add_argument("--max-convs", type=int, default=20000, help="섞은 원본에서 앞 이만큼만 봄 (본 실험 max_convs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=51, help="ShareGPT 원본을 섞는 seed (본 실험 seed와 같게)")
    a = ap.parse_args()

    count = make_counter(a.tokenizer)
    docs = load_leval(a.leval, a.leval_keys, count, a.max_doc_tokens, a.doc_out_max, a.min_doc_turns, a.out_min, a.max_q_per_doc, a.min_doc_tokens)
    chats = load_sharegpt(a.sharegpt, count, a)
    with open(a.out, "w", encoding="utf-8") as fh:
        for s in docs + chats:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    def med(v):
        v = sorted(v); return v[len(v) // 2] if v else 0
    print(f"문서 세션 {len(docs)}개 (문서 토큰 중앙값 {med([d['ctx_tokens'] for d in docs])}, "
          f"질문 수 중앙값 {med([len(d['turns']) for d in docs])})")
    print(f"대화 세션 {len(chats)}개 (턴 수 중앙값 {med([len(c['turns']) for c in chats])}, "
          f"응답 토큰 중앙값 {med([t['out_len'] for c in chats for t in c['turns']])})")


if __name__ == "__main__":
    main()