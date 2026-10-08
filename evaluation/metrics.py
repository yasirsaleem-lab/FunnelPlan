"""
Evaluation metrics for retrieval.
Works with a single query's results: a list of dicts with 'id' and 'score'.
"""

import numpy as np
from typing import List, Set, Dict, Any


def _dcg(scores: List[float]) -> float:
    """Compute DCG given a list of relevance scores (binary or graded)."""
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(scores))


def _idcg(relevant_ids: Set[str], total_docs: int) -> float:
    """Compute IDCG for a query: ideal DCG if all relevant docs are at the top."""
    scores = [1.0] * len(relevant_ids) + [0.0] * (total_docs - len(relevant_ids))
    return _dcg(scores[:total_docs])


def ndcg_at_k(results: List[Dict[str, Any]], relevant_ids: Set[str], k: int = 10) -> float:
    """
    Compute nDCG@k for a single query.

    Args:
        results: List of retrieved documents, each with an 'id' field (and optionally 'score').
        relevant_ids: Set of relevant document IDs.
        k: Depth cutoff.

    Returns:
        nDCG@k value (0.0 to 1.0).
    """
    if k <= 0:
        return 0.0
    # Build relevance list for top-k results
    relevance = []
    for res in results[:k]:
        doc_id = res.get("id")
        rel = 1.0 if doc_id in relevant_ids else 0.0
        relevance.append(rel)
    dcg = _dcg(relevance)
    # Ideal DCG: all relevant docs at the top (bounded by k)
    ideal_scores = [1.0] * min(k, len(relevant_ids)) + [0.0] * (k - min(k, len(relevant_ids)))
    idcg = _dcg(ideal_scores)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(results: List[Dict[str, Any]], relevant_ids: Set[str], k: int = 100) -> float:
    """
    Compute Recall@k for a single query.
    """
    retrieved_ids = {res.get("id") for res in results[:k]}
    if len(relevant_ids) == 0:
        return 0.0
    return len(retrieved_ids & relevant_ids) / len(relevant_ids)


def mrr_at_k(results: List[Dict[str, Any]], relevant_ids: Set[str], k: int = 10) -> float:
    """
    Compute MRR@k (Mean Reciprocal Rank) for a single query.
    """
    for i, res in enumerate(results[:k], start=1):
        if res.get("id") in relevant_ids:
            return 1.0 / i
    return 0.0


def map_at_k(results: List[Dict[str, Any]], relevant_ids: Set[str], k: int = 100) -> float:
    """
    Compute MAP@k (Mean Average Precision) for a single query.
    """
    if len(relevant_ids) == 0:
        return 0.0
    ap = 0.0
    num_hit = 0
    for i, res in enumerate(results[:k], start=1):
        if res.get("id") in relevant_ids:
            num_hit += 1
            ap += num_hit / i
    return ap / len(relevant_ids)