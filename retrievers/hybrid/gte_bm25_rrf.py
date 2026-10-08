import logging
from typing import List, Dict, Any

from retrievers.base import BaseRetriever
from retrievers.sparse.bm25 import BM25Retriever
from retrievers.dense.gte_qwen2_7b import GTEQwenRetriever

logger = logging.getLogger(__name__)


class HybridRRFRetriever(BaseRetriever):
    """GTE-Qwen2-7B (dense) + BM25 (sparse) fused with Reciprocal Rank Fusion.

    Change (2026-10-02): added cleanup() so the 7B model is released before the
    agent loads. Fusion pool is max(k, 100) from each side.
    """

    def __init__(self, dense_retriever=None, sparse_retriever=None, rrf_k: int = 60, pool: int = 100,
                 name: str = "gte_bm25_rrf"):
        self._name = name
        self.dense = dense_retriever or GTEQwenRetriever()
        self.sparse = sparse_retriever or BM25Retriever()
        self.rrf_k = rrf_k
        self.pool = pool
        self.corpus_ids = None

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        self.dense.build_index(corpus)
        self.sparse.build_index(corpus)
        self.corpus_ids = [d["id"] for d in corpus]

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        k = min(k, len(self.corpus_ids))
        n = min(len(self.corpus_ids), max(k, self.pool))
        dense = {r["id"]: i + 1 for i, r in enumerate(self.dense.search(query, n))}
        sparse = {r["id"]: i + 1 for i, r in enumerate(self.sparse.search(query, n))}
        fused = {}
        for d in set(dense) | set(sparse):
            fused[d] = sum(1.0 / (self.rrf_k + r[d]) for r in (dense, sparse) if d in r)
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{"id": d, "score": s} for d, s in top]

    def cleanup(self):
        for r in (self.dense, self.sparse):
            if hasattr(r, "cleanup"):
                r.cleanup()

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_gpu(self) -> bool:
        return self.dense.requires_gpu
