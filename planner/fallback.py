"""
Fallback retriever: uses BM25 (or a minimal, fast retriever) when the planner
cannot find a viable retriever within the budget, or when the TSR predictor fails.
"""

import logging
from typing import List, Dict, Any, Optional

from retrievers.sparse.bm25 import BM25Retriever
from retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


class Fallback:
    """
    Wraps a BM25 retriever as a fallback. Index must be built separately.
    """

    def __init__(self, retriever: Optional[BaseRetriever] = None):
        self.retriever = retriever or BM25Retriever()
        self._indexed = False

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        """Build BM25 index over the given corpus."""
        logger.info("Building fallback BM25 index...")
        self.retriever.build_index(corpus)
        self._indexed = True
        logger.info("Fallback index ready.")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k using BM25."""
        if not self._indexed:
            raise RuntimeError("Fallback index not built. Call build_index() first.")
        return self.retriever.search(query, k)

    @property
    def name(self) -> str:
        return "fallback_bm25"