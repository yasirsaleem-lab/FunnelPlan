"""
One place that maps plan retriever names to constructors.

Includes "bge_bm25_rrf" (BGE-M3 + BM25 fused with RRF). The reranker base can
be any non reranker retriever (full mode reranks the hybrid).
"""

from config import settings

RERANKERS = ("qwen3_reranker_0_6b", "qwen3_reranker_4b")


def build_retriever(name: str):
    from retrievers.sparse.bm25 import BM25Retriever
    from retrievers.sparse.splade_v3 import SPLADEV3Retriever
    from retrievers.dense.stella_1_5b import StellaRetriever
    from retrievers.dense.gte_qwen2_7b import GTEQwenRetriever
    from retrievers.dense.bge_m3 import BGERetriever
    from retrievers.hybrid.gte_bm25_rrf import HybridRRFRetriever
    from retrievers.rerankers.qwen3_reranker_0_6b import Qwen3Reranker06B
    from retrievers.rerankers.qwen3_reranker_4b import Qwen3Reranker4B

    simple = {
        "bm25": BM25Retriever,
        "splade_v3": SPLADEV3Retriever,
        "stella_1_5b": StellaRetriever,
        "gte_qwen2_7b": GTEQwenRetriever,
        "bge_m3": BGERetriever,
        "gte_bm25_rrf": lambda: HybridRRFRetriever(GTEQwenRetriever(), BM25Retriever(), name="gte_bm25_rrf"),
        "bge_bm25_rrf": lambda: HybridRRFRetriever(BGERetriever(), BM25Retriever(), name="bge_bm25_rrf"),
    }
    if name in simple:
        return simple[name]()
    if name in RERANKERS:
        if settings.RERANK_BASE in RERANKERS:
            raise ValueError("RERANK_BASE cannot be a reranker")
        base = build_retriever(settings.RERANK_BASE)
        return Qwen3Reranker06B(base_retriever=base) if name == "qwen3_reranker_0_6b" else Qwen3Reranker4B(base_retriever=base)
    raise KeyError(f"Unknown retriever: {name}")


def plan_name(retriever: str, depth: int) -> str:
    return f"{retriever}@{int(depth)}"
