#!/usr/bin/env python3
"""Build a replay workload from long-context QA datasets (LEval / LooGLE / NarrativeQA).

Protocol follows Tutti (SSD-backed KV cache, arXiv:2605.03375): requests are drawn
round-robin from the sub-datasets to emulate a multi-session environment, and arrivals
are Poisson because these datasets carry no timestamps.

Why this produces SSD-tier hits: one document (3K-200K tokens) is shared by several
questions, and round-robin interleaving places many other documents' requests between
two questions of the same document. By the time question k+1 of document d arrives,
d's KV has been evicted from HBM and the CPU tier down to SSD.

Input: directories or files of json/jsonl. Field names are auto-detected; the script
prints what it found so you can correct with --doc-field / --q-field.

Output jsonl, one request per line:
  {"timestamp": ms, "doc_id": ..., "task": ..., "q_idx": k,
   "prompt": "<document>\n\n<question>", "input_length": tok, "output_length": tok}

Usage:
  python3 build_longctx_workload.py --src ~/huggingface/LEval ~/huggingface/LooGLE \
      --model $HOME/huggingface/Llama-3.1-8B-Instruct \
      --max-input 30000 --min-input 4000 --rps 0.5 --duration 1800 \
      --out longctx_workload.jsonl
"""
import argparse, glob, json, os, random, sys
from collections import defaultdict

DOC_KEYS = ["input", "context", "document", "text", "story", "passage", "article", "content"]
Q_KEYS = ["instructions", "questions", "instruction", "question", "query", "queries", "input"]
A_KEYS = ["outputs", "answers", "output", "answer", "response"]

def iter_records(path):
    files = []
    if os.path.isdir(path):
        for ext in ("*.json", "*.jsonl", "**/*.json", "**/*.jsonl"):
            files += glob.glob(os.path.join(path, ext), recursive=True)
    else:
        files = [path]
    for f in sorted(set(files)):
        try:
            with open(f) as fh:
                head = fh.read(2)
                fh.seek(0)
                if head.startswith("["):                 # json array
                    for r in json.load(fh):
                        if isinstance(r, dict): yield f, r
                else:                                    # jsonl
                    for line in fh:
                        line = line.strip()
                        if not line: continue
                        try:
                            r = json.loads(line)
                            if isinstance(r, dict): yield f, r
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            print(f"  [skip] {f}: {e}", file=sys.stderr)

def pick(rec, keys, exclude=()):
    for k in keys:
        if k in exclude: continue
        v = rec.get(k)
        if isinstance(v, str) and v.strip(): return k, v
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v): return k, v
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="dataset dirs or files")
    ap.add_argument("--model", required=True, help="tokenizer path (local model dir)")
    ap.add_argument("--doc-field", default=None)
    ap.add_argument("--q-field", default=None)
    ap.add_argument("--min-input", type=int, default=4000, help="drop docs shorter than this (below break-even, decision is trivial)")
    ap.add_argument("--max-input", type=int, default=30000, help="drop docs longer than this (engine max-model-len)")
    ap.add_argument("--min-questions", type=int, default=2, help="a doc needs >=2 questions to create reuse")
    ap.add_argument("--output-tokens", type=int, default=64)
    ap.add_argument("--rps", type=float, default=0.5)
    ap.add_argument("--rps-schedule", default=None,
                    help="alternate phases instead of a constant rate, e.g. '0.12@300,0.40@300' "
                         "= 5 min at 0.12 rps then 5 min at 0.40 rps, repeating. Lets one run contain "
                         "both an idle regime (recompute wins) and a busy one (load wins), which no "
                         "fixed policy can serve well. A phase with rps 0 (e.g. '0@60') is a gap: "
                         "no arrivals, so the queue from the previous phase drains before the next.")
    ap.add_argument("--output-schedule", default=None,
                    help="per-phase decode length, e.g. '64,1024' paired with --rps-schedule phases. "
                         "Decode length is what actually sets concurrency, so vary it with the rate.")
    ap.add_argument("--duration", type=float, default=1800, help="seconds of workload to emit")
    ap.add_argument("--long-frac", type=float, default=0.0,
                    help="A2: fraction of requests that decode --long-output tokens instead of the phase "
                         "length. Keeps a decode population alive so recompute has a real Opp cost.")
    ap.add_argument("--long-output", type=int, default=1024)
    ap.add_argument("--max-docs", type=int, default=0,
                    help="A2: use only this many documents (after shuffling). A fixed corpus can be pre-warmed "
                         "into the SSD tier (lio_prewarm.py) so the run measures steady-state serving, not cold start.")
    ap.add_argument("--new-frac", type=float, default=0.0,
                    help="robustness: this fraction of requests get a unique session header in front of the "
                         "document, so every block hash changes -> a never-seen document (full miss, prefill + "
                         "SSD store), not pre-warmed and never revisited. Chosen with a separate RNG so arrival "
                         "times, documents and questions stay identical to the --new-frac 0 workload.")
    ap.add_argument("--recycle", action="store_true",
                    help="A2: when every document is used up, reshuffle and start over (high-rate runs "
                         "exhaust LEval). Revisits across cycles hit the SSD tier (fs never evicts).")
    ap.add_argument("--pool", type=int, default=16,
                    help="number of documents active at once. Revisit interval ~= pool/rps requests, "
                         "so this directly sets how much other KV passes between two questions of one "
                         "document. Round-robin over ALL documents (pool=0) makes the interval "
                         "n_docs/rps, which is usually far longer than the run.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="longctx_workload.jsonl")
    ap.add_argument("--max-q-per-doc", type=int, default=0,
                    help="keep at most this many questions per document (0 = all). Bounds reuse per document so "
                         "documents retire and new ones enter during the run (steady miss ~= 1/K from a cold cache).")
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    # ---- collect (document, [questions]) grouped by task file
    docs = []            # (task, doc_text, [questions])
    field_report = defaultdict(int)
    for src in a.src:
        print(f"scanning {src} ...")
        for f, rec in iter_records(src):
            task = os.path.splitext(os.path.basename(f))[0]
            dk, doc = (a.doc_field, rec.get(a.doc_field)) if a.doc_field else pick(rec, DOC_KEYS)
            if not isinstance(doc, str) or not doc.strip(): continue
            qk, qs = (a.q_field, rec.get(a.q_field)) if a.q_field else pick(rec, Q_KEYS, exclude=(dk,))
            if isinstance(qs, str): qs = [qs]
            if not qs: continue
            field_report[f"{task}: doc={dk} q={qk}"] += 1
            docs.append((task, doc, [q for q in qs if isinstance(q, str) and q.strip()]))

    print("\nfields detected (records per file):")
    for k, v in sorted(field_report.items(), key=lambda x: -x[1])[:25]:
        print(f"  {v:>6}  {k}")
    if not docs:
        sys.exit("no (document, questions) pairs found — inspect the files and pass --doc-field/--q-field")

    # ---- tokenize and filter
    kept = defaultdict(list)     # task -> [(doc_id, doc_text, [q], doc_tokens)]
    n_short = n_long = n_fewq = 0
    for i, (task, doc, qs) in enumerate(docs):
        if len(qs) < a.min_questions: n_fewq += 1; continue
        n = len(tok(doc, add_special_tokens=False)["input_ids"])
        if n < a.min_input: n_short += 1; continue
        if n > a.max_input: n_long += 1; continue
        if a.max_q_per_doc > 0: qs = qs[:a.max_q_per_doc]
        kept[task].append((f"{task}#{i}", doc, qs, n))
    total_docs = sum(len(v) for v in kept.values())
    total_reqs = sum(len(q) for v in kept.values() for _, _, q, _ in v)
    print(f"\nkept {total_docs} documents / {total_reqs} questions across {len(kept)} tasks"
          f"   (dropped: {n_short} < {a.min_input} tok, {n_long} > {a.max_input} tok, {n_fewq} with < {a.min_questions} questions)")
    if not total_docs: sys.exit("nothing kept — widen --min-input/--max-input")
    for t in sorted(kept, key=lambda t: -len(kept[t]))[:15]:
        lens = sorted(n for _, _, _, n in kept[t])
        print(f"  {t:<28} docs={len(kept[t]):>4}  q/doc avg {sum(len(q) for _,_,q,_ in kept[t])/len(kept[t]):4.1f}"
              f"  doc tokens p50={lens[len(lens)//2]:>6} p90={lens[int(.9*len(lens))]:>6}")
    if a.stats_only: return

    # ---- sliding pool of active documents, interleaved across tasks (Tutti's multi-session
    # round-robin, but with a bounded working set so revisits happen within the run)
    rng = random.Random(a.seed)
    backlog = [(t, d) for t, v in kept.items() for d in v]
    rng.shuffle(backlog)
    if a.max_docs > 0:
        backlog = backlog[:a.max_docs]
        print(f"corpus limited to {len(backlog)} documents ({sum(d[3] for _, d in backlog)/1000:.0f}K doc tokens, "
              f"~{sum(d[3] for _, d in backlog)*128/1024/1024:.0f} GiB KV to pre-warm)")
    pool_size = a.pool if a.pool > 0 else len(backlog)
    pool = []                                   # [(task, doc, next_question_index)]
    stream = []
    all_docs = list(backlog)
    max_rps = max([float(p.partition("@")[0]) for p in a.rps_schedule.split(",")] if a.rps_schedule else [a.rps])
    target = int(max_rps * a.duration * 1.5) + 100
    while backlog or pool or (a.recycle and len(stream) < target):
        if a.recycle and not backlog and not pool:
            backlog = list(all_docs); rng.shuffle(backlog)
        if a.recycle and len(stream) >= target: break
        while backlog and len(pool) < pool_size:
            t, d = backlog.pop()
            pool.append([t, d, 0])
        if not pool: break
        i = rng.randrange(len(pool))             # next request: a random active document
        t, d, qi = pool[i]
        stream.append((t, d, qi))
        pool[i][2] += 1
        if pool[i][2] >= len(d[2]):              # questions exhausted -> retire, admit a new doc
            pool.pop(i)

    # phase schedule: [(rps, seconds, output_tokens), ...] cycled
    phases = []
    if a.rps_schedule:
        outs = [int(x) for x in a.output_schedule.split(",")] if a.output_schedule else None
        for i, part in enumerate(a.rps_schedule.split(",")):
            r, _, d = part.partition("@")
            o = outs[i % len(outs)] if outs else a.output_tokens
            phases.append((float(r), float(d or 300), o))
        print("phases: " + " | ".join(f"{r} rps for {d:.0f}s, out {o}" for r, d, o in phases))
    else:
        phases = [(a.rps, float("inf"), a.output_tokens)]

    out = open(a.out, "w")
    cold_rng = random.Random(a.seed + 7777); n_cold = 0
    t_ms, n, last_seen = 0.0, 0, {}
    gaps, in_lens = [], []
    ph_i, ph_end = 0, phases[0][1] * 1000.0
    phase_counts = defaultdict(int)
    for task, d, r in stream:
        doc_id, doc, qs, doc_tok = d
        q = qs[r]
        prompt = doc + "\n\nQuestion: " + q + "\nAnswer:"
        n_in = doc_tok + len(tok("\n\nQuestion: " + q + "\nAnswer:", add_special_tokens=False)["input_ids"])
        cold = a.new_frac > 0 and cold_rng.random() < a.new_frac
        if cold:
            header = f"[session {cold_rng.getrandbits(64):016x}]\n"
            prompt = header + prompt; n_in += len(tok(header, add_special_tokens=False)["input_ids"])
        if n_in > a.max_input: continue
        if t_ms >= ph_end and len(phases) > 1:          # advance to the next phase
            ph_i = (ph_i + 1) % len(phases); ph_end = t_ms + phases[ph_i][1] * 1000.0
        cur_rps, _, cur_out = phases[ph_i]
        if a.long_frac > 0 and rng.random() < a.long_frac:
            cur_out = a.long_output
        if cur_rps <= 0:                                 # gap phase: no arrivals, let the queue drain
            t_ms = ph_end; continue
        t_ms += rng.expovariate(cur_rps) * 1000.0      # Poisson arrivals at the phase rate
        if t_ms / 1000.0 > a.duration: break
        phase_counts[(cur_rps, cur_out)] += 1
        if doc_id in last_seen: gaps.append((t_ms - last_seen[doc_id]) / 1000.0)
        last_seen[doc_id] = t_ms
        in_lens.append(n_in)
        if cold: n_cold += 1
        out.write(json.dumps({"timestamp": round(t_ms, 1), "doc_id": doc_id + (f"#new{n_cold}" if cold else ""),
                              "cold": cold, "task": task,
                              "q_idx": r, "prompt": prompt, "input_length": n_in,
                              "output_length": cur_out, "phase_rps": cur_rps}) + "\n")
        n += 1
    out.close()

    def qv(xs, p):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    print(f"\nwrote {n} requests over {t_ms/1000:.0f}s ({n/(t_ms/1000 or 1):.2f} rps) -> {a.out}")
    if a.new_frac > 0: print(f"  new (never-seen) documents: {n_cold} requests ({100*n_cold/max(1,n):.1f}%)")
    print(f"  input tokens p10/p50/p90/max: {qv(in_lens,.1)}/{qv(in_lens,.5)}/{qv(in_lens,.9)}/{max(in_lens or [0])}")
    print(f"  prompt token throughput: {sum(in_lens)/(t_ms/1000 or 1)/1000:.1f}K tok/s"
          f"   (A100/8B measured ceiling ~8.2K tok/s)")
    if gaps:
        print(f"  revisit interval for the same document p10/p50/p90: "
              f"{qv(gaps,.1):.0f}/{qv(gaps,.5):.0f}/{qv(gaps,.9):.0f} s")
        kv_between = qv(gaps, .5) * (sum(in_lens) / (t_ms / 1000 or 1))
        print(f"  -> ~{kv_between/1000:.0f}K tokens of other requests' KV pass through between two "
              f"questions of one document; GPU KV budget is ~{6*1024//128}K tokens at KV_GIB=6")
    else:
        print("  [warn] no document was revisited — increase --duration or lower --rps")
    if len(phases) > 1:
        print("  requests per phase:", {f"{r} rps/out {o}": c for (r, o), c in sorted(phase_counts.items())})

if __name__ == "__main__":
    main()