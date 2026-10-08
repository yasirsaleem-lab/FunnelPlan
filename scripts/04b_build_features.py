#!/usr/bin/env python
"""
Build the v3 feature table from the v3 retrieval caches.

Output: data/processed/features_v3.csv, one row per (dataset, query, retriever,
depth). Column prefixes are defined in tsr/features.py (q_, plan_, p_, r_, x_,
diag_). Labels are joined in 05, not here.

Runs in minutes without a GPU when USE_RERANKER is False. With the reranker
on, it scores the union of each query's top RERANK_TOP passages across all
plans once, and caches the scores, so reruns are cheap.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from tsr.features import (
    probe_features, plan_identity, own_plan_features, reranker_features,
    standardize_by_retriever, retriever_stats, gold_in_passages,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {
    "datasets": ["nq", "hotpotqa"],
    "retrievers": settings.PLAN_RETRIEVERS,
    "depths": settings.PLAN_DEPTHS,
    "use_reranker": True,        # r_ features (GPU). False -> r_ columns omitted.
    "rerank_top": 20,
    "diagnostics": True,         # diag_ columns (needs corpus text + answers + qrels)
    "reranker_device": "cuda",
}

CACHE_DIR = settings.CACHE_ROOT / "retrievals_v3"
RERANK_CACHE_DIR = settings.CACHE_ROOT / "rerank_scores_v3"
OUT_CSV = settings.PROCESSED_DATA_ROOT / settings.FEATURES_FILE
PSTATS_JSON = settings.PROCESSED_DATA_ROOT / "features_v3_pstats.json"


def load_caches(ds):
    caches = {}
    for r in CONFIG["retrievers"]:
        p = CACHE_DIR / f"{ds}__{r}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run 04_generate_tsr_labels.py first.")
        with open(p, "rb") as f:
            caches[r] = pickle.load(f)
    qids = sorted(set.intersection(*[set(c["results"]) for c in caches.values()]))
    return caches, qids


def rerank_scores(ds, caches, qids, queries, text_by_id):
    """{(qid, doc_id): P(yes)} for the union of top passages across plans."""
    RERANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = RERANK_CACHE_DIR / f"{ds}.pkl"
    scores = pickle.load(open(path, "rb")) if path.exists() else {}
    need = {}
    for q in qids:
        docs = set()
        for c in caches.values():
            docs.update(r["id"] for r in c["results"][q][:CONFIG["rerank_top"]])
        todo = [d for d in docs if (q, d) not in scores]
        if todo:
            need[q] = todo
    if need:
        from retrievers.rerankers.qwen3_reranker_0_6b import Qwen3Reranker06B
        rr = Qwen3Reranker06B(device=CONFIG["reranker_device"])
        rr.load()
        try:
            for q, docs in tqdm(need.items(), desc=f"rerank {ds}"):
                s = rr.score_pairs(queries[q], [text_by_id.get(d, "") for d in docs])
                scores.update({(q, d): float(v) for d, v in zip(docs, s)})
        finally:
            rr.cleanup()
        pickle.dump(scores, open(path, "wb"))
    return scores


def main():
    rows = []
    for ds in CONFIG["datasets"]:
        caches, qids = load_caches(ds)
        queries = caches["bm25"]["queries"]
        logger.info(f"{ds}: {len(qids)} queries with all {len(caches)} retrievers cached")

        text_by_id, answers, qrels = None, None, None
        if CONFIG["use_reranker"] or CONFIG["diagnostics"]:
            from data.data_loader import get_processed
            from data.corpus_utils import build_text_by_id, resolve_gold
            corpus, qmap, qrels, answers_raw = get_processed(ds)
            text_by_id = build_text_by_id(corpus)
            answers = {q: resolve_gold(qmap, answers_raw, q) for q in qids}
            line_to_real = {str(k): str(v.get("_id") or v.get("id") or k) for k, v in qmap.items()}
            del corpus

        rr = rerank_scores(ds, caches, qids, queries, text_by_id) if CONFIG["use_reranker"] else None

        for q in tqdm(qids, desc=f"features {ds}"):
            qf = probe_features(queries[q], caches["bm25"]["results"][q])
            top10 = {r: [x["id"] for x in caches[r]["results"][q][:10]] for r in caches}
            top100 = {r: set(x["id"] for x in caches[r]["results"][q][:100]) for r in caches}
            for r, c in caches.items():
                full = c["results"][q]
                others = [o for o in caches if o != r]
                jacc = np.mean([len(set(top10[r]) & set(top10[o])) / max(len(set(top10[r]) | set(top10[o])), 1)
                                for o in others])
                for depth in CONFIG["depths"]:
                    res = full[:depth]
                    row = {"dataset": ds, "query_id": str(q), "retriever": r, "depth": depth}
                    row.update(qf)
                    row.update(plan_identity(r, depth, CONFIG["retrievers"]))
                    row.update(own_plan_features(res, depth))
                    ids = [x["id"] for x in res]
                    votes = [sum(d in top100[o] for o in others) / len(others) for d in ids[:10]]
                    row["x_jaccard10_mean"] = float(jacc)
                    row["x_vote10_mean"] = float(np.mean(votes)) if votes else 0.0
                    if rr is not None:
                        row.update(reranker_features([rr.get((q, d), 0.0) for d in ids[:CONFIG["rerank_top"]]]))
                    if CONFIG["diagnostics"]:
                        texts = [text_by_id.get(d, "") for d in ids]
                        row["diag_gold_pos"] = gold_in_passages(answers[q], texts)
                        row["diag_gold_in_ctx"] = int(row["diag_gold_pos"] > 0)
                        rel = set(qrels.get(line_to_real.get(str(q), str(q)), {}).keys())
                        row["diag_qrel_recall"] = len(rel & set(ids)) / len(rel) if rel else np.nan
                    rows.append(row)

    df = pd.DataFrame(rows)
    p_cols = [c for c in df.columns if c.startswith("p_")]
    stats = retriever_stats(df, p_cols)
    PSTATS_JSON.write_text(json.dumps(stats, indent=2))
    df = standardize_by_retriever(df, p_cols)
    df.to_csv(OUT_CSV, index=False)
    logger.info(f"Wrote {OUT_CSV} with {len(df)} rows and {df.shape[1]} columns")


if __name__ == "__main__":
    main()
