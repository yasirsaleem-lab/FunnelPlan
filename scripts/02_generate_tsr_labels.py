"""
v3 TSR label generation.

What changed against the old 04 + regenerate_features_v2 pair:
  * Retrieval runs ONCE per (dataset, retriever) at the max depth and is
    cached with the query text and per query latency. Depth 10 and 50 are
    prefixes of the depth 100 list, so the three depths are truly nested.
  * Labels and features are both built from that same cache. Before, labels
    came from one retrieval run (04) and features from another
    (regenerate_features_v2), so they could describe different lists.
  * Agent output is stored with prompt length and wall time for the cost model.
  * N_REPEATS > 1 reruns the agent to measure label noise.
  * Every (dataset, retriever) block is written to its own part file, so a
    crash loses one block, not the run.

Run order (paper runs, FUNNELPLAN_CORPUS_MODE=full):
  02_generate_tsr_labels.py   -> data/cache_full/retrievals_v3/*.pkl, data/processed_full/tsr_labels_<reader>.csv
  07_grading_full.py          -> grades the answers (tsr_labels_<reader>_graded.csv)
"""

import gc
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Corpus mode must be set BEFORE config is imported. Child processes inherit it.
#   "sub200k": 200k subsample, 2 datasets, 200 queries (pilot and gate runs)
#   "full":    full corpora, 4 datasets, 500 queries (the paper runs)
CORPUS_MODE = "full"
import os
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE

from config import settings
from data.data_loader import get_processed
from data.corpus_utils import build_corpus_list, resolve_gold
from retrievers.registry import build_retriever
from tsr.agent_runner import AgentRunner
from tsr.grader import Grader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
#   One dataset + one method: runs in this process.
#   "all" for either: every (dataset, method) block not yet done runs in its
#   own fresh Python process (clean GPU each time). Finished blocks are skipped.
#   After every block, all part files are merged into tsr_labels_v3.csv.
# ============================================================================
CONFIG = {
    "dataset": "all",               # "nq", "hotpotqa", "musique", "2wikimultihopqa" or "all"
    "method": "all",                    # one name, a list, or "all". Full mode methods:
                                        # "closed_book" (depth 0), "bm25", "splade_v3", "bge_m3",
                                        # "stella_1_5b", "bge_bm25_rrf", "qwen3_reranker_0_6b"

    # Reader: "qwen2.5-7b" (bf16), "llama3.1-8b" (bf16), "qwen3-14b" (int8),
    # or "qwen2.5-7b-4bit" (the original v3 labels). See tsr/agent_runner.py.
    "reader": "qwen3-14b",
    # Depths to label. 0 = closed book. All depths must be <= 100 (cached depth).
    "depths": [0, 1, 3, 5, 10],
    "num_queries": settings.NUM_QUERIES,   # 200 in sub200k mode, 500 in full mode
    "seed": 42,
    "n_repeats": 1,          # agent runs per (query, plan). Use 3 on a subset for noise.
    "skip_done": True,       # skip blocks whose part file already exists
}
DATASETS = list(settings.DATASETS)       # full mode adds musique and 2wikimultihopqa

CACHE_DIR = settings.CACHE_ROOT / "retrievals_v3"
LEGACY_READER = "qwen2.5-7b-4bit"


def reader_paths(reader):
    """The original 4 bit labels keep their old paths; every other reader gets
    its own part folder and merged file, so readers never mix."""
    if reader == LEGACY_READER:
        return (settings.PROCESSED_DATA_ROOT / "labels_v3_parts",
                settings.PROCESSED_DATA_ROOT / settings.LABELS_FILE)
    return (settings.PROCESSED_DATA_ROOT / f"labels_{reader}_parts",
            settings.PROCESSED_DATA_ROOT / f"tsr_labels_{reader}.csv")


PARTS_DIR, OUT_CSV = reader_paths(CONFIG["reader"])


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_queries(queries, answers, n, seed):
    """Same sampling rule as the original 04, so the query set stays comparable."""
    with_gold = [qid for qid in queries.keys() if answers.get(str(qid), [])]
    if len(with_gold) > n:
        np.random.seed(seed)
        return [str(q) for q in np.random.choice(with_gold, n, replace=False).tolist()]
    return [str(q) for q in with_gold]


def cache_path(ds, ret):
    return CACHE_DIR / f"{ds}__{ret}.pkl"


def run_retrieval(ds, ret_name, corpus_list, qtexts, max_depth):
    path = cache_path(ds, ret_name)
    if path.exists():
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if set(cache["results"]) >= set(qtexts) and cache["max_depth"] >= max_depth:
            logger.info(f"  cache hit {path.name}")
            return cache
        logger.info(f"  cache {path.name} incomplete, rebuilding")

    retriever = build_retriever(ret_name)
    results, latency = {}, {}
    try:
        t0 = time.perf_counter()
        retriever.build_index(corpus_list)
        logger.info(f"  index built in {time.perf_counter() - t0:.0f}s")
        for qid, q in tqdm(list(qtexts.items()), desc=f"retrieve {ret_name}", leave=False):
            # One warm call is excluded from timing for the first query only.
            if not latency:
                retriever.search(q, max_depth)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            res = retriever.search(q, max_depth)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency[qid] = time.perf_counter() - t
            results[qid] = [{"id": str(r["id"]), "score": float(r["score"])} for r in res]
    finally:
        if hasattr(retriever, "cleanup"):
            try:
                retriever.cleanup()
            except Exception as e:
                logger.warning(f"  cleanup failed: {e}")
        del retriever
        free_gpu()

    cache = {
        "dataset": ds, "retriever": ret_name, "max_depth": max_depth,
        "queries": dict(qtexts), "results": results, "latency_s": latency,
        "meta": {
            "stella_model": settings.STELLA_MODEL, "gte_model": settings.GTE_QWEN_MODEL,
            "rerank_base": settings.RERANK_BASE, "rerank_candidates": settings.RERANK_CANDIDATES,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "pipeline": "v3",
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    return cache


CLOSED_BOOK = "closed_book"   # pseudo method: depth 0, no retrieval, run once per dataset
RETRIEVE_DEPTH = 100          # always cache 100 results so any depth <= 100 can be labelled later


def all_methods():
    return ([CLOSED_BOOK] if 0 in CONFIG["depths"] else []) + list(settings.PLAN_RETRIEVERS)


def merge_parts():
    """Rebuild this reader's merged label file from all finished part files."""
    rank = {f"{d}__{m}": i for i, (d, m) in enumerate(
        (d, m) for d in DATASETS for m in [CLOSED_BOOK] + list(settings.PLAN_RETRIEVERS))}
    parts = sorted(PARTS_DIR.glob("*.csv"), key=lambda p: (rank.get(p.stem, 10**6), p.stem))
    if not parts:
        logger.warning("No part files to merge.")
        return
    df = pd.concat([pd.read_csv(p, dtype={"query_id": str}) for p in parts], ignore_index=True)
    df["reader"] = CONFIG["reader"]
    tmp = OUT_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(OUT_CSV)
    logger.info(f"Merged {len(parts)} part files -> {OUT_CSV.name} ({len(df)} rows)\n"
                + df.groupby(["dataset", "retriever"], sort=False).size().to_string())


def part_path(ds, ret):
    return PARTS_DIR / f"{ds}__{ret}.csv"


def status():
    print(f"\nTSR label status for reader {CONFIG['reader']} ({PARTS_DIR.name}):")
    for ds in DATASETS:
        done = [m for m in all_methods() if part_path(ds, m).exists()]
        left = [m for m in all_methods() if not part_path(ds, m).exists()]
        print(f"  {ds:9s} done: {done}")
        print(f"  {'':9s} left: {left}")


def run_block(ds, ret_name):
    """Retrieval (cached) + reader + grading for ONE (dataset, method) block."""
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    closed = ret_name == CLOSED_BOOK
    depths = [0] if closed else sorted(d for d in CONFIG["depths"] if d > 0)
    if not closed and depths and depths[-1] > RETRIEVE_DEPTH:
        sys.exit(f"Depth {depths[-1]} exceeds the cached retrieval depth {RETRIEVE_DEPTH}.")

    logger.info(f"=== {ds} / {ret_name} / reader {CONFIG['reader']} / depths {depths} ===")
    corpus, queries, _, answers = get_processed(ds)
    qids = sample_queries(queries, answers, CONFIG["num_queries"], CONFIG["seed"])
    qtexts = {q: queries[q]["text"] for q in qids}
    gold = {q: resolve_gold(queries, answers, q) for q in qids}
    logger.info(f"  {len(qids)} queries; example: {qtexts[qids[0]]!r}")
    grader = Grader(ds)

    cache = None
    if not closed:
        corpus_list = build_corpus_list(corpus)
        cache = run_retrieval(ds, ret_name, corpus_list, qtexts, RETRIEVE_DEPTH)
        del corpus_list
        free_gpu()

    agent = AgentRunner(reader=CONFIG["reader"])
    agent.set_corpus(corpus)
    agent.load()
    rows = []
    try:
        for qid in tqdm(qids, desc=f"{CONFIG['reader']} {ds}/{ret_name}"):
            full = [] if closed else cache["results"].get(qid, [])
            for depth in depths:
                for rep in range(CONFIG["n_repeats"]):
                    out = agent.answer_with_meta(qtexts[qid], full[:depth])
                    ok = grader.is_correct(qtexts[qid], out["answer"], gold[qid]) if gold[qid] else False
                    rows.append({
                        "dataset": ds, "query_id": qid, "retriever": ret_name,
                        "depth": depth, "repeat": rep, "label": int(ok),
                        "answer": out["answer"], "prompt_tokens": out["prompt_tokens"],
                        "gen_tokens": out["gen_tokens"], "agent_time_s": out["agent_time_s"],
                        "retrieval_time_s": 0.0 if closed else cache["latency_s"].get(qid, np.nan),
                        "mean_logprob": out["mean_logprob"], "min_logprob": out["min_logprob"],
                        "first_token_prob": out["first_token_prob"],
                    })
    finally:
        agent.unload()
        free_gpu()

    # Temp file first: a crash mid write must not leave a part that looks finished.
    part = part_path(ds, ret_name)
    tmp = part.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(part)
    df = pd.DataFrame(rows)
    logger.info(f"  wrote {part.name}: {len(rows)} rows; TSR (lenient) by depth: "
                f"{df.groupby('depth')['label'].mean().round(3).to_dict()}")
    merge_parts()


def main():
    from logutil import setup_logging
    setup_logging("02_generate_tsr_labels", CONFIG)
    import subprocess
    # Child mode: this script calling itself for one block.
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        run_block(sys.argv[2], sys.argv[3])
        return

    ds_cfg, m_cfg = CONFIG["dataset"], CONFIG["method"]
    datasets = DATASETS if ds_cfg == "all" else [ds_cfg]
    methods = all_methods() if m_cfg == "all" else (list(m_cfg) if isinstance(m_cfg, (list, tuple)) else [m_cfg])
    bad = [d for d in datasets if d not in DATASETS] + \
          [m for m in methods if m not in [CLOSED_BOOK] + list(settings.PLAN_RETRIEVERS)]
    if bad:
        sys.exit(f"Unknown dataset/method: {bad}")

    blocks = [(d, m) for d in datasets for m in methods]
    todo = []
    for d, m in blocks:
        if CONFIG["skip_done"] and part_path(d, m).exists():
            logger.info(f"[skip] {d} / {m} already labelled ({part_path(d, m).name})")
        else:
            todo.append((d, m))

    if len(todo) == 1:
        run_block(*todo[0])
    else:
        for d, m in todo:
            logger.info(f"=== {d} / {m} (fresh process) ===")
            r = subprocess.run([sys.executable, __file__, "--child", d, m])
            if r.returncode != 0:
                sys.exit(f"Block {d}/{m} failed with exit code {r.returncode}. "
                         "Fix it and rerun; finished blocks are skipped.")
    status()


if __name__ == "__main__":
    main()
