"""
Fixed step chain of thought controllers (F_T) against IRCoT. GPU (answer calls and continuation steps).
Plan: docs/ANALYSIS_PLAN_2026-11-05.md, E2.

F_T runs IRCoT's reasoning chain (06_ircot.py: same demonstrations, prompt, K = 4 new paragraphs per retrieval,
15 paragraph cap, at most 8 reasoning steps) for exactly T retrievals with the stopping rule switched off, then
answers with the plan answer prompt. The number of calls is fixed in advance, so F_T is a plan.

Construction from the stored IRCoT chains (r4_ircot/<reader>/<dataset>.jsonl). Decoding is greedy, so a stored
chain is the path every F_T follows as long as it lasts: its sentences and its retrieved paragraphs are reused.
Where the stored chain ends (it stopped on "answer is", or ran out of steps), the walk continues with stopping off:
the last sentence is still used as a query, and new reasoning calls are made and timed live.
Costs per question and T:
  reasoning   reused calls: the question's own measured IRCoT time per reasoning call; new calls: timed live
  retrieval   first retrieval: the cached BM25 latency (as plan bm25@k and IRCoT); reused retrievals: the question's
              measured IRCoT time per later retrieval; new retrievals: timed live
  answer      a new answer call for every T, timed live (for T equal to IRCoT's own retrieval count the paragraphs
              are IRCoT's, so the answer must equal IRCoT's stored answer: a determinism check)
Stage check: for 30 questions per reader and dataset (seed 5), the stored chain's reasoning calls are rerun with
per call timing, to measure the error of the reused cost and to confirm that the rerun gives the stored sentences.

Usage (resumable):
  python scripts/34_fixed_step_controllers.py run qwen2.5-7b [--datasets 2wikimultihopqa musique ...]
  python scripts/34_fixed_step_controllers.py check qwen2.5-7b
  python scripts/34_fixed_step_controllers.py merge qwen2.5-7b      -> r5_cotfixed/labels_<reader>.csv (graded by 07)
  python scripts/34_fixed_step_controllers.py status
Readers: qwen2.5-7b, llama3.1-8b, qwen3-14b. Datasets: those with a stored IRCoT file (nq after 06 --datasets nq).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings

_s = importlib.util.spec_from_file_location("ircot06", Path(__file__).parent / "06_ircot.py")
M = importlib.util.module_from_spec(_s)
_argv, sys.argv = sys.argv, [sys.argv[0]]
_s.loader.exec_module(M)
sys.argv = _argv
T = M.T                                   # 05_two_hop_plans: Reader, setup, answer_messages, read_jsonl

READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "nq"]
TS = [1, 2, 3, 4]
IN = settings.PROCESSED_DATA_ROOT / "r4_ircot"
OUT = settings.PROCESSED_DATA_ROOT / "r5_cotfixed"
CHECK_N, CHECK_SEED = 30, 5


def stored(reader, ds):
    return {x["query_id"]: x for x in T.read_jsonl(IN / reader / f"{ds}.jsonl")}


def segments(x):
    """Replay the stored chain's control flow: which sentence triggered which retrieval.
    Returns a list of steps: (sentence, retrieved_ids or None) in order, as IRCoT executed them."""
    ids = list(x["doc_ids"][:M.K])
    pos, steps = M.K, []
    for s in x["cot"]:
        if "answer is" in s.lower():
            steps.append((s, None, "answer"))
            break
        if len(ids) >= M.MAX_PARAGRAPHS or not M.has_terms(s):
            steps.append((s, None, "noretrieve"))
            continue
        n = min(M.K, M.MAX_PARAGRAPHS - len(ids))
        new = x["doc_ids"][pos:pos + n]
        pos += n
        ids += new
        steps.append((s, new, "retrieve"))
    return steps


def walk(R, C, demo_text, question, first_ids, first_lat, x):
    """Walk the chain with stopping off up to max(TS) retrievals. Returns the paragraph ids and cost parts at each T."""
    per_call = x["reason_time_s"] / max(x["n_reason_calls"], 1)
    later = max(x["retrieval_calls"] - 1, 1)
    per_ret = max(x["retrieval_time_s"] - first_lat, 0.0) / later if x["retrieval_calls"] > 1 else 0.0
    ids = list(first_ids[:M.K])
    if ids != list(x["doc_ids"][:M.K]):
        raise RuntimeError("first retrieval differs from the stored chain")
    stored_steps = segments(x)
    cot = []
    cost = {"reason_reused": 0.0, "reason_new": 0.0, "ret": first_lat, "n_reason_reused": 0, "n_reason_new": 0,
            "n_ret_new": 0}
    snaps = {1: (list(ids), dict(cost), 0)}
    n_ret, n_steps, i = 1, 0, 0
    while n_ret < max(TS) and len(ids) < M.MAX_PARAGRAPHS and n_steps < M.MAX_STEPS:
        on_path = i < len(stored_steps)
        if on_path:
            s, new_ids, kind = stored_steps[i]
            cost["reason_reused"] += per_call
            cost["n_reason_reused"] += 1
        else:
            paras = [C.passage(j) for j in ids]
            o, _ = M.reason_call(R, demo_text, paras, question, cot)
            s, new_ids, kind = M.first_sentence(o["text"]), None, "new"
            cost["reason_new"] += o["time_s"]
            cost["n_reason_new"] += 1
            if not s:
                break                                   # an empty sentence would repeat (greedy): the chain ends
        i += 1
        n_steps += 1
        cot.append(s)
        if not M.has_terms(s):
            continue
        if kind == "retrieve":                           # the stored chain retrieved with this sentence: reuse
            ids += new_ids
            cost["ret"] += per_ret
        else:                                            # stopping is off: retrieve with the sentence live
            new, lat = C.search(s, min(M.K, M.MAX_PARAGRAPHS - len(ids)), set(ids))
            ids += new
            cost["ret"] += lat
            cost["n_ret_new"] += 1
        n_ret += 1
        snaps[n_ret] = (list(ids), dict(cost), n_steps)
    return snaps, cot


def run_one(R, C, demo_text, question, first_ids, first_lat, x):
    snaps, cot = walk(R, C, demo_text, question, first_ids, first_lat, x)
    rows = {}
    for t in TS:
        reach = max(k for k in snaps if k <= t)          # fewer retrievals possible only if the chain ended early
        ids, cost, n_steps = snaps[reach]
        texts = {i: (f"{a}. {b}" if a and b else (a or b)) for i, (a, b) in ((i, C.passage(i)) for i in ids)}
        a = R.generate(T.answer_messages(question, ids, texts), M.ANSWER_MAX_TOKENS)
        rows[str(t)] = {"reached": reach, "n_paragraphs": len(ids), "doc_ids": ids, "n_reason_steps": n_steps,
                        "answer": a["text"], "a_time_s": a["time_s"], "a_prompt_tokens": a["prompt_tokens"],
                        "a_gen_tokens": a["gen_tokens"], "mean_logprob": a["mean_logprob"],
                        "min_logprob": a["min_logprob"], "first_token_prob": a["first_token_prob"], **cost,
                        "same_ids_as_ircot": ids == list(x["doc_ids"]),
                        "same_answer_as_ircot": (a["text"].strip() == (x["answer"] or "").strip()) if ids == list(x["doc_ids"]) else None}
    return rows


def stage_run(reader, url, datasets):
    from tsr.grader import apply_rule, grade_detail
    dss = [d for d in DATASETS if d in (datasets or DATASETS) and (IN / reader / f"{d}.jsonl").exists()]
    order, data = T.setup(dss)
    (OUT / reader).mkdir(parents=True, exist_ok=True)
    R = None
    for ds in order:
        X = stored(reader, ds)
        p = OUT / reader / f"{ds}.jsonl"
        done = {x["query_id"] for x in T.read_jsonl(p)}
        todo = [q for q in data[ds]["qids"] if q in X and q not in done]
        print(f"== {reader} / {ds}: {len(done)} done, {len(todo)} to do ({len(X)} stored IRCoT chains)", flush=True)
        if not todo:
            continue
        if R is None:
            R = T.Reader(reader, url)
        C = M.Corpus(ds)
        demo_text = M.demos(ds)
        gold = json.load(open(settings.PROCESSED_DATA_ROOT / ds / "answers.json", encoding="utf-8"))
        t0, n = time.time(), 0
        with open(p, "a", encoding="utf-8") as f:
            for q in todo:
                try:
                    rows = run_one(R, C, demo_text, data[ds]["qtext"][q], data[ds]["hop1"][q], data[ds]["lat1"][q], X[q])
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"  failed {q}: {str(e)[:200]}", flush=True)
                    import gc
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                for t, r in rows.items():
                    g = grade_detail(r["answer"], gold.get(q, []))
                    r["label"], r["strict"] = apply_rule(g, "lenient"), apply_rule(g, "strict")
                f.write(json.dumps({"reader": reader, "dataset": ds, "query_id": q, "steps": rows}) + "\n")
                f.flush()
                n += 1
                if n % 25 == 0:
                    el = time.time() - t0
                    D = T.read_jsonl(p)
                    acc = {t: np.mean([d["steps"][t]["strict"] for d in D]) for t in map(str, TS)}
                    same = [d["steps"][t]["same_answer_as_ircot"] for d in D for t in map(str, TS)
                            if d["steps"][t]["same_answer_as_ircot"] is not None]
                    print(f"  {len(done) + n}/{len(todo) + len(done)}  {el / n:.1f} s/question  ETA "
                          f"{(len(todo) - n) * el / n / 60:.0f} min  strict by T {acc}  answer = IRCoT's where the "
                          f"paragraphs are the same: {np.mean(same) if same else float('nan'):.3f}", flush=True)
        del C


def stage_check(reader, url, datasets):
    """Rerun the stored chains' reasoning calls (up to 4 retrievals) for 30 questions per dataset, timing each call."""
    dss = [d for d in DATASETS if d in (datasets or DATASETS) and (IN / reader / f"{d}.jsonl").exists()]
    order, data = T.setup(dss)
    p = OUT / reader / "check.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    done = {(x["dataset"], x["query_id"]) for x in T.read_jsonl(p)}
    R = None
    for ds in order:
        X = stored(reader, ds)
        qs = sorted(X)
        rng = np.random.default_rng(CHECK_SEED)
        pick = [q for q in rng.choice(qs, min(CHECK_N, len(qs)), replace=False) if (ds, q) not in done]
        if not pick:
            continue
        if R is None:
            R = T.Reader(reader, url)
        C = M.Corpus(ds)
        demo_text = M.demos(ds)
        with open(p, "a", encoding="utf-8") as f:
            for q in pick:
                x = X[q]
                ids, cot, calls = list(x["doc_ids"][:M.K]), [], []
                for s_stored, new_ids, kind in segments(x):
                    o, _ = M.reason_call(R, demo_text, [C.passage(j) for j in ids], data[ds]["qtext"][q], cot)
                    s = M.first_sentence(o["text"])
                    calls.append({"time_s": o["time_s"], "prompt_tokens": o["prompt_tokens"], "same": s == s_stored})
                    cot.append(s_stored)                 # follow the stored path
                    if kind == "retrieve":
                        ids += new_ids
                    if len(ids) >= M.MAX_PARAGRAPHS:
                        break
                f.write(json.dumps({"reader": reader, "dataset": ds, "query_id": q, "calls": calls,
                                    "stored_per_call": x["reason_time_s"] / max(x["n_reason_calls"], 1)}) + "\n")
                f.flush()
        del C
    D = T.read_jsonl(p)
    err = [abs(sum(c["time_s"] for c in d["calls"]) - len(d["calls"]) * d["stored_per_call"]) /
           max(sum(c["time_s"] for c in d["calls"]), 1e-9) for d in D if d["calls"]]
    same = [c["same"] for d in D for c in d["calls"]]
    print(f"{reader}: {len(D)} chains checked; median relative error of the reused reasoning cost "
          f"{np.median(err):.3f}; rerun sentence equals the stored one in {np.mean(same):.3f} of {len(same)} calls")


def stage_merge(reader):
    rows = []
    for ds in DATASETS:
        for x in T.read_jsonl(OUT / reader / f"{ds}.jsonl"):
            for t, r in x["steps"].items():
                rows.append({"dataset": ds, "query_id": x["query_id"], "retriever": f"cotfixed{t}", "depth": M.K,
                             "repeat": 0, "label": r["label"], "answer": r["answer"], "prompt_tokens": r["a_prompt_tokens"],
                             "gen_tokens": r["a_gen_tokens"],
                             "agent_time_s": r["reason_reused"] + r["reason_new"] + r["a_time_s"],
                             "retrieval_time_s": r["ret"], "mean_logprob": r["mean_logprob"],
                             "min_logprob": r["min_logprob"], "first_token_prob": r["first_token_prob"],
                             "lm_calls": r["n_reason_reused"] + r["n_reason_new"] + 1, "retrieval_calls": r["reached"],
                             "n_paragraphs": r["n_paragraphs"], "reason_new": r["n_reason_new"],
                             "same_answer_as_ircot": r["same_answer_as_ircot"], "reader": reader})
    df = pd.DataFrame(rows)
    p = OUT / f"labels_{reader}.csv"
    df.to_csv(p, index=False)
    print(f"Wrote {p.name}: {len(df)} rows")
    print(df.groupby(["dataset", "retriever"]).agg(strict_or_contains=("label", "mean"), agent_s=("agent_time_s", "mean"),
                                                   retrieval_calls=("retrieval_calls", "mean"),
                                                   new_reason_calls=("reason_new", "mean")).round(3).to_string())


def stage_status():
    for r in READERS:
        cells = [f"{ds}:{len(T.read_jsonl(OUT / r / f'{ds}.jsonl'))}/{len(T.read_jsonl(IN / r / f'{ds}.jsonl'))}"
                 for ds in DATASETS]
        print(f"{r:12s} " + "  ".join(cells) + f"   check {len(T.read_jsonl(OUT / r / 'check.jsonl'))}"
              f"   merged {(OUT / f'labels_{r}.csv').exists()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "check", "merge", "status"])
    ap.add_argument("reader", nargs="?", choices=READERS)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args()
    if a.stage != "status" and not a.reader:
        sys.exit(f"Stage {a.stage} needs a reader: {READERS}")
    {"run": lambda: stage_run(a.reader, a.url, a.datasets), "check": lambda: stage_check(a.reader, a.url, a.datasets),
     "merge": lambda: stage_merge(a.reader), "status": stage_status}[a.stage]()
