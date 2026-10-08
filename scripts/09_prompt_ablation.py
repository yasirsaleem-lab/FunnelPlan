#!/usr/bin/env python
"""
Prompt ablation: does the "use only the passages" instruction cause the
"retrieval hurts a reader that already knows the answer" effect?

Main labels use SYSTEM_RAG ("You answer questions using only the given
passages ... If the passages do not contain the answer, give your best
guess."). This script reruns a small grid with a SOFT prompt that lets the
reader fall back on its own knowledge, on the SAME 500 queries and the SAME
cached retrieval lists, so the prompt is the only change.

Grid (default): NQ; BM25 (weak) and BGE M3 (strong) at depths 1 and 3;
readers Qwen2.5 7B and Llama 3.1 8B. 2 x 500 x 4 = 4,000 calls, about 30 to
40 minutes on the RTX 3080 (estimate). Closed book is unaffected by the change
and is taken from the main labels.

Stages: "run" (GPU; one fresh process per reader; resumable) then "report" (CPU).
Never run during labelling. Does not modify tsr/agent_runner.py: the soft
prompt lives in a subclass here.

Output: data/processed_full/ablation_prompt/<reader>__<dataset>__<retriever>.csv
        data/processed_full/ablation_prompt/report.txt
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings

CONFIG = {
    "stage": "report",                       # "run" | "report"
    "dataset": "2wikimultihopqa",
    "retrievers": ["bm25", "bge_m3"],
    "depths": [1, 3],
    "readers": ["qwen2.5-7b", "llama3.1-8b"],
}

SYSTEM_SOFT = ("Answer the question. Use the passages if they help. If they do not contain the "
               "answer, answer from your own knowledge. Reply with the short answer only (a few words).")

OUT = settings.PROCESSED_DATA_ROOT / "ablation_prompt"
CACHE_DIR = settings.CACHE_ROOT / "retrievals_v3"


def out_path(reader, ds, ret):
    return OUT / f"{reader}__{ds}__{ret}.csv"


def run_reader(reader):
    import torch
    from tqdm import tqdm
    from data.data_loader import get_processed
    from tsr.agent_runner import AgentRunner, SYSTEM_RAG
    import tsr.agent_runner as ar

    class SoftRunner(AgentRunner):
        def build_prompt(self, query, passages):
            # identical to the parent except the system prompt with passages
            saved = ar.SYSTEM_RAG
            ar.SYSTEM_RAG = SYSTEM_SOFT
            try:
                return super().build_prompt(query, passages)
            finally:
                ar.SYSTEM_RAG = saved

    ds = CONFIG["dataset"]
    todo = [r for r in CONFIG["retrievers"] if not out_path(reader, ds, r).exists()]
    if not todo:
        print(f"{reader}: all done")
        return
    corpus, _, _, _ = get_processed(ds)
    agent = SoftRunner(reader=reader)
    agent.set_corpus(corpus)
    agent.load()
    p = agent.build_prompt("test question", [{"text": "test passage", "title": ""}])
    assert SYSTEM_SOFT in p and SYSTEM_RAG not in p, "soft prompt not applied"
    OUT.mkdir(parents=True, exist_ok=True)
    for ret in todo:
        cache = pickle.load(open(CACHE_DIR / f"{ds}__{ret}.pkl", "rb"))
        labels = pd.read_csv(settings.PROCESSED_DATA_ROOT / f"labels_{reader}_parts" / f"{ds}__{ret}.csv",
                             dtype={"query_id": str})
        qids = list(dict.fromkeys(labels["query_id"]))          # same 500 queries, same order
        rows = []
        for qid in tqdm(qids, desc=f"{reader} {ds}/{ret} soft"):
            full = cache["results"][qid]
            for d in CONFIG["depths"]:
                o = agent.answer_with_meta(cache["queries"][qid], full[:d])
                rows.append({"dataset": ds, "query_id": qid, "retriever": ret, "depth": d,
                             "prompt": "soft", "answer": o["answer"], "prompt_tokens": o["prompt_tokens"],
                             "gen_tokens": o["gen_tokens"], "agent_time_s": o["agent_time_s"],
                             "mean_logprob": o["mean_logprob"]})
        tmp = out_path(reader, ds, ret).with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(out_path(reader, ds, ret))
    agent.unload()
    torch.cuda.empty_cache()


def stage_run():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        return run_reader(sys.argv[2])
    for reader in CONFIG["readers"]:
        print(f"=== {reader} (fresh process) ===", flush=True)
        r = subprocess.run([sys.executable, __file__, "--child", reader])
        if r.returncode:
            sys.exit(f"{reader} failed ({r.returncode})")
    print("Done. Set stage to 'report'.")


def stage_report():
    from tsr.grader import apply_rule, grade_detail
    ds = CONFIG["dataset"]
    gold = json.load(open(settings.PROCESSED_DATA_ROOT / ds / "answers.json", encoding="utf-8"))
    g = lambda q, a, rule: apply_rule(grade_detail(a if isinstance(a, str) else "", gold.get(q, [])), rule)
    L = [f"Prompt ablation on {ds}: strict prompt (main labels) vs soft prompt, same queries and passages", ""]
    rng = np.random.default_rng(0)
    for reader in CONFIG["readers"]:
        parts = settings.PROCESSED_DATA_ROOT / f"labels_{reader}_parts"
        cb = pd.read_csv(parts / f"{ds}__closed_book.csv", dtype={"query_id": str})
        cb_ok = {q: g(q, a, "strict") for q, a in zip(cb["query_id"], cb["answer"])}
        L.append(f"== {reader}: closed book (no passages, prompt unchanged) strict = {np.mean(list(cb_ok.values())):.3f}")
        for ret in CONFIG["retrievers"]:
            f = out_path(reader, ds, ret)
            if not f.exists():
                L.append(f"  {ret}: not run")
                continue
            soft = pd.read_csv(f, dtype={"query_id": str})
            main = pd.read_csv(parts / f"{ds}__{ret}.csv", dtype={"query_id": str})
            for d in CONFIG["depths"]:
                s = soft[soft.depth == d].set_index("query_id")
                m = main[main.depth == d].set_index("query_id").loc[s.index]
                ys = np.array([g(q, a, "strict") for q, a in zip(s.index, s["answer"])])
                ym = np.array([g(q, a, "strict") for q, a in zip(m.index, m["answer"])])
                yc = np.array([cb_ok[q] for q in s.index])
                diff = ys - ym
                boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                lost_m, lost_s = int(((yc == 1) & (ym == 0)).sum()), int(((yc == 1) & (ys == 0)).sum())
                L.append(f"  {ret}@{d}: strict prompt {ym.mean():.3f} | soft prompt {ys.mean():.3f} | "
                         f"soft minus strict {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                         f"known answers lost (closed book right, RAG wrong): strict {lost_m}, soft {lost_s} | "
                         f"vs closed book: strict {ym.mean() - yc.mean():+.3f}, soft {ys.mean() - yc.mean():+.3f}")
        L.append("")
    L.append("Reading: if 'vs closed book' turns from negative to positive (or the known answers lost "
             "drop sharply) under the soft prompt, the dip is a prompt effect. If it persists, it is a "
             "reader effect (context overrides stored knowledge).")
    (OUT / f"report_{ds}.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--child" in sys.argv:
        stage_run()
    else:
        {"run": stage_run, "report": stage_report}[CONFIG["stage"]]()
