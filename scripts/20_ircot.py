#!/usr/bin/env python
"""
IRCoT reference point (pre-registration run R4; hypothesis H3). Not a plan: its cost depends on the data.

IRCoT (Trivedi et al., ACL 2023) as published, with the choices our setting forces stated in docs/PREREGISTRATION.md:
  retrieve  BM25 (our bm25s index, the same as plan bm25@k) for the question: top K paragraphs.
  reason    the reader continues a chain of thought, in the authors' few shot format ("Wikipedia Title: ..." blocks,
            "Q: Answer the following question by reasoning step-by-step.", "A: ..."), with the authors' own
            demonstrations (data/ircot_prompts, from github.com/StonyBrookNLP/ircot, Apache 2.0; gold paragraphs
            plus M = 1 distractor, the Flan-T5 files) packed in file order up to DEMO_CHARS characters.
            Raw text completion (no chat template), greedy, at most 64 new tokens; only the FIRST generated
            sentence is kept, as in the paper.
  retrieve  BM25 with that sentence as the query: top K new paragraphs, added until MAX_PARAGRAPHS are collected.
  stop      when the chain of thought contains "answer is" or after MAX_STEPS reason steps.
  answer    a separate reader call (the paper's "IRCoT QA" with a direct reader, as for Flan-T5) on the collected
            paragraphs, with the unchanged answer prompt of tsr/agent_runner.py, so that only retrieval differs
            from the single step plans.
  cost      agent_time_s = all reason calls + the answer call (generate() time, as in 04);
            retrieval_time_s = first BM25 (cache latency, as plan bm25@k) + every later BM25 call (timed here).

Readers: qwen2.5-7b, llama3.1-8b, qwen3-14b (transformers, H3); qwen3-32b, gemma3-27b (llama.cpp, optional).
Datasets: hotpotqa, 2wikimultihopqa, musique (the same 500 queries).

Usage (resumable; one line per question in data/processed_full/r4_ircot/<reader>/<dataset>.jsonl):
  python scripts/20_ircot.py run llama3.1-8b --datasets hotpotqa          (gate G3 first)
  python scripts/20_ircot.py run qwen2.5-7b
  python scripts/20_ircot.py merge llama3.1-8b                            -> r4_ircot/labels_<reader>.csv
  python scripts/20_ircot.py status
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings

_s19 = importlib.util.spec_from_file_location("twohop19", Path(__file__).parent / "19_two_hop_plans.py")
T = importlib.util.module_from_spec(_s19)
_s19.loader.exec_module(T)               # Reader (transformers or llama.cpp), setup(), G (= script 17)

K = 4                     # paragraphs per retrieval step (paper grid {2, 4, 6, 8}; fixed a priori, see amendment)
MAX_STEPS = 8             # paper
MAX_PARAGRAPHS = 15       # paper
REASON_MAX_TOKENS = 64
ANSWER_MAX_TOKENS = 100
DEMO_CHARS = 24000        # about 6,000 tokens of demonstrations (the paper's Flan-T5 budget)
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
PROMPTS = Path(__file__).parent.parent / "data" / "ircot_prompts"
OUT = settings.PROCESSED_DATA_ROOT / "r4_ircot"
INSTR = "Q: Answer the following question by reasoning step-by-step.\n"
SENT_END = re.compile(r"[.!?]\s+")
ENUM = re.compile(r"^\s*(?:\(?\d{1,2}[.)]|\(?[a-h]\)|[Ss]tep\s*\d{1,2}\s*[.:)])\s+")   # not "A." (an initial)
# A period after an initial ("W. H. Auden"), a title or a common abbreviation does not end a sentence.
# (Bug fix 2026-10-10: the first version split there, so "The W. H. Auden ..." became the query "The W.")
ABBREV = re.compile(r"(?:\b[A-Za-z]|\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|No|Nos|vs|Inc|Ltd|Co|Corp|Mt|Ft|Gen|Col|Lt|Sgt|Capt|"
                    r"Prof|Rev|Gov|Sen|Rep|Pres|Ave|Rd|Univ|Dept|est|approx|ca|Jan|Feb|Mar|Apr|Aug|Sept|Sep|Oct|Nov|"
                    r"Dec|Vol|pp|ed|al|etc|U\.S|U\.K|e\.g|i\.e|D\.C|a\.m|p\.m))\.$")


# ====================================================================== prompt pieces
def demos(ds):
    f = PROMPTS / f"{ds}_gold_with_1_distractors_cot_flan_t5.txt"
    text = f.read_text(encoding="utf-8")
    blocks = []
    for b in text.split("# METADATA:"):
        b = b.strip()
        if not b:
            continue
        b = b.split("\n", 1)[1].strip() if b.startswith("{") else b      # drop the metadata line
        blocks.append(b)
    out, n = [], 0
    for b in blocks:
        if n + len(b) > DEMO_CHARS:
            break
        out.append(b)
        n += len(b)
    return "\n\n\n".join(out)


def para_block(title, text):
    return f"Wikipedia Title: {title}\n{text}"


def reason_prompt(demo_text, paras, question, cot):
    ctx = "\n\n".join(para_block(t, x) for t, x in paras)
    a = " ".join(cot)
    return f"{demo_text}\n\n\n{ctx}\n\n{INSTR}{question}\nA:" + (f" {a}" if a else "")


def first_sentence(gen):
    g = (gen or "").strip().split("\n")[0].strip()
    if not g:
        return ""
    g = ENUM.sub("", g)                               # a leading list marker ("1.", "2)", "Step 3:") is not a sentence
    for m in SENT_END.finditer(g):
        head, rest = g[: m.start() + 1], g[m.end():]
        if ABBREV.search(head) or (rest[:1].islower()):
            continue                                  # initial, abbreviation, or the sentence continues
        return head.strip()
    return g.strip()


def has_terms(q):
    """BM25 (bm25s) keeps only tokens of 2+ word characters; a query without any matches nothing."""
    return bool(re.search(r"\w\w", q or ""))


# ====================================================================== corpus access (title, text) by id
class Corpus:
    def __init__(self, ds):
        from data.corpus_utils import build_corpus_list
        from data.data_loader import get_processed
        from retrievers.sparse.bm25 import BM25Retriever
        print(f"  loading {ds} corpus and BM25 index ...", flush=True)
        t0 = time.time()
        corpus = get_processed(ds)[0]
        lst = build_corpus_list(corpus)
        del corpus
        self.title = {d["id"]: d["title"] for d in lst}
        self.text = {d["id"]: d["text"] for d in lst}
        self.ret = BM25Retriever()
        self.ret.build_index(lst)                  # loads the cached index of 04
        del lst
        self.ret.search("warm up", 5)
        print(f"  ready in {time.time() - t0:.0f}s", flush=True)

    def search(self, query, k, exclude):
        t0 = time.perf_counter()
        res = self.ret.search(query, k + len(exclude))
        lat = time.perf_counter() - t0
        return [r["id"] for r in res if r["id"] not in exclude][:k], lat

    def passage(self, i):
        t, x = (self.title.get(i) or "").strip(), (self.text.get(i) or "").strip()
        return t, x


# ====================================================================== raw completion for the reason step
def raw_generate(R, prompt, max_new):
    if R.agent is None:                                         # llama.cpp: /completion with the raw prompt
        body = {"prompt": prompt, "n_predict": max_new, "temperature": 0.0, "top_p": 1.0, "seed": 0,
                "cache_prompt": False, "stream": False, "stop": ["\n\n", "\nQ:"]}
        t0 = time.perf_counter()
        r = T.G.session().post(f"{R.url}/completion", json=body, timeout=600)
        wall = time.perf_counter() - t0
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        tm = j.get("timings", {})
        pm, gm = tm.get("prompt_ms"), tm.get("predicted_ms")
        return {"text": j.get("content", ""), "prompt_tokens": j.get("tokens_evaluated"),
                "gen_tokens": j.get("tokens_predicted"),
                "time_s": (pm + gm) / 1000 if pm is not None and gm is not None else wall}
    import torch
    a = R.agent
    inputs = a.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(a.model.device)
    n = int(inputs.input_ids.shape[1])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    with torch.no_grad():
        out = a.model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                               top_k=None, pad_token_id=a.tokenizer.pad_token_id)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.perf_counter() - t0
    gen = out[0][n:]
    return {"text": a.tokenizer.decode(gen, skip_special_tokens=True), "prompt_tokens": n,
            "gen_tokens": int(gen.shape[0]), "time_s": dt}


# ====================================================================== one question
def ircot_one(R, C, demo_text, question, first_ids, first_lat):
    ids = list(first_ids[:K])
    ret_time, ret_calls = first_lat, 1
    cot, steps, q_time, p_tok, g_tok = [], [], 0.0, 0, 0
    for step in range(MAX_STEPS):
        paras = [C.passage(i) for i in ids]
        o = raw_generate(R, reason_prompt(demo_text, paras, question, cot), REASON_MAX_TOKENS)
        q_time += o["time_s"]
        p_tok += o["prompt_tokens"] or 0
        g_tok += o["gen_tokens"] or 0
        s = first_sentence(o["text"])
        steps.append(s)
        if not s:
            break
        cot.append(s)
        if "answer is" in s.lower():
            break
        if len(ids) >= MAX_PARAGRAPHS or not has_terms(s):
            continue                                   # keep reasoning on the collected paragraphs (no retrieval)
        new, lat = C.search(s, K, set(ids))
        ret_time += lat
        ret_calls += 1
        ids += new[: MAX_PARAGRAPHS - len(ids)]
    texts = {i: (f"{t}. {x}" if t and x else (t or x)) for i, (t, x) in ((i, C.passage(i)) for i in ids)}
    a = R.generate(T.answer_messages(question, ids, texts), ANSWER_MAX_TOKENS)
    return {"answer": a["text"], "a_time_s": a["time_s"], "a_prompt_tokens": a["prompt_tokens"],
            "a_gen_tokens": a["gen_tokens"], "mean_logprob": a["mean_logprob"], "min_logprob": a["min_logprob"],
            "first_token_prob": a["first_token_prob"], "finish": a["finish"],
            "cot": cot, "n_reason_calls": len(steps), "reason_time_s": q_time, "reason_prompt_tokens": p_tok,
            "reason_gen_tokens": g_tok, "retrieval_time_s": ret_time, "retrieval_calls": ret_calls,
            "n_paragraphs": len(ids), "doc_ids": ids, "stopped_on_answer": any("answer is" in c.lower() for c in cot)}


# ====================================================================== stages
def stage_run(reader, url, datasets):
    from tsr.grader import apply_rule, grade_detail
    order, data = T.setup([d for d in DATASETS if d in (datasets or DATASETS)])
    d_out = OUT / reader
    d_out.mkdir(parents=True, exist_ok=True)
    R = None
    for ds in order:
        p = d_out / f"{ds}.jsonl"
        done = {x["query_id"] for x in T.read_jsonl(p)}
        todo = [q for q in data[ds]["qids"] if q not in done]
        print(f"== {reader} / {ds}: {len(done)} done, {len(todo)} to do", flush=True)
        if not todo:
            continue
        if R is None:
            R = T.Reader(reader, url)
        C = Corpus(ds)
        demo_text = demos(ds)
        gold = json.load(open(settings.PROCESSED_DATA_ROOT / ds / "answers.json", encoding="utf-8"))
        t_start, n, n_oom_retry, n_failed = time.time(), 0, 0, 0
        with open(p, "a", encoding="utf-8") as f:
            for q in todo:
                try:
                    o = ircot_one(R, C, demo_text, data[ds]["qtext"][q], data[ds]["hop1"][q], data[ds]["lat1"][q])
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    o = None
                    if "out of memory" in str(e).lower() and R.agent is not None:
                        # free cached blocks and retry once; greedy decoding makes the retry give the same answer
                        import gc
                        import torch
                        gc.collect()
                        torch.cuda.empty_cache()
                        try:
                            o = ircot_one(R, C, demo_text, data[ds]["qtext"][q], data[ds]["hop1"][q], data[ds]["lat1"][q])
                            n_oom_retry += 1
                        except KeyboardInterrupt:
                            raise
                        except Exception as e2:
                            e = e2
                            gc.collect()
                            torch.cuda.empty_cache()
                    if o is None:
                        n_failed += 1
                        print(f"  failed {q}: {str(e)[:150]}", flush=True)
                        continue
                g = grade_detail(o["answer"], gold.get(q, []))
                f.write(json.dumps({"reader": reader, "dataset": ds, "query_id": q,
                                    "label": apply_rule(g, "lenient"), "strict": apply_rule(g, "strict")} | o) + "\n")
                f.flush()
                n += 1
                if n % 25 == 0:
                    el = time.time() - t_start
                    D = pd.DataFrame(T.read_jsonl(p))
                    print(f"  {len(done) + n}/{len(data[ds]['qids'])}  {el / n:.1f} s/question  "
                          f"ETA {(len(todo) - n) * el / n / 60:.0f} min  strict {D.strict.mean():.3f}  "
                          f"reason calls {D.n_reason_calls.mean():.2f}  retrieval calls {D.retrieval_calls.mean():.2f}",
                          flush=True)
        if n_oom_retry or n_failed:
            print(f"  {ds}: {n_oom_retry} questions needed an out of memory retry; {n_failed} failed and will be "
                  f"retried on the next run", flush=True)
        del C


def stage_merge(reader, datasets):
    order, data = T.setup([d for d in DATASETS if d in (datasets or DATASETS)])
    rows = []
    for ds in order:
        got = {x["query_id"]: x for x in T.read_jsonl(OUT / reader / f"{ds}.jsonl")}
        miss = [q for q in data[ds]["qids"] if q not in got]
        if miss:
            print(f"Not merged: {ds} misses {len(miss)} questions.")
            return
        for q in data[ds]["qids"]:
            x = got[q]
            rows.append({"dataset": ds, "query_id": q, "retriever": "ircot", "depth": K, "repeat": 0,
                         "label": x["label"], "answer": x["answer"], "prompt_tokens": x["a_prompt_tokens"],
                         "gen_tokens": x["reason_gen_tokens"] + (x["a_gen_tokens"] or 0),
                         "agent_time_s": x["reason_time_s"] + x["a_time_s"],
                         "retrieval_time_s": x["retrieval_time_s"], "mean_logprob": x["mean_logprob"],
                         "min_logprob": x["min_logprob"], "first_token_prob": x["first_token_prob"],
                         "finish": x["finish"], "lm_calls": x["n_reason_calls"] + 1,
                         "retrieval_calls": x["retrieval_calls"], "n_paragraphs": x["n_paragraphs"],
                         "reason_prompt_tokens": x["reason_prompt_tokens"], "stopped_on_answer": x["stopped_on_answer"],
                         "cot": " ".join(x["cot"]), "reader": reader})
    df = pd.DataFrame(rows)
    p = OUT / f"labels_{reader}.csv"
    df.to_csv(p, index=False)
    print(f"Wrote {p.name}: {len(df)} rows")
    print(df.groupby("dataset").agg(strict_or_contains=("label", "mean"), lm_calls=("lm_calls", "mean"),
                                    retrieval_calls=("retrieval_calls", "mean"), paragraphs=("n_paragraphs", "mean"),
                                    agent_s=("agent_time_s", "mean"), stop_on_answer=("stopped_on_answer", "mean"))
          .round(3).to_string())


def stage_status():
    for r in T.ALL_READERS:
        cells = [f"{ds}:{len(T.read_jsonl(OUT / r / f'{ds}.jsonl'))}/500" for ds in DATASETS]
        print(f"{r:12s} " + "  ".join(cells) + f"   merged {(OUT / f'labels_{r}.csv').exists()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "merge", "status"])
    ap.add_argument("reader", nargs="?", choices=T.ALL_READERS)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args()
    if a.stage in ("run", "merge") and not a.reader:
        sys.exit(f"Stage {a.stage} needs a reader: {T.ALL_READERS}")
    {"run": lambda: stage_run(a.reader, a.url, a.datasets), "merge": lambda: stage_merge(a.reader, a.datasets),
     "status": stage_status}[a.stage]()
