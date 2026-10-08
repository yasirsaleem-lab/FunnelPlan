import logging
from typing import List, Dict, Any

import bm25s
from tqdm import tqdm

from config import settings
from data.corpus_utils import passage_text
from retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 (bm25s). Change (2026-10-04): the index is saved to the cache and
    reloaded, so the 5.2M passage HotpotQA index is built once, not in every
    block that uses BM25 (BM25, the hybrid and the reranker base)."""

    def __init__(self, tokenizer=None):
        self.bm25 = None
        self.corpus_ids = None

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        from retrievers.dense import emb_cache
        texts = [passage_text(d.get("title", ""), d.get("text", "")) or "[EMPTY]" for d in corpus]
        self.corpus_ids = [d["id"] for d in corpus]
        key = emb_cache.corpus_key("bm25s", self.corpus_ids, texts)
        folder = settings.CACHE_ROOT / "bm25" / key
        if folder.exists() and any(folder.iterdir()):
            try:
                self.bm25 = bm25s.BM25.load(str(folder))
                logger.info(f"BM25 index loaded from cache ({len(self.corpus_ids)} documents).")
                return
            except Exception as e:
                logger.warning(f"Could not load BM25 cache ({e}); rebuilding.")
        logger.info(f"Building BM25 index for {len(texts)} documents...")
        tokens = bm25s.tokenize(texts, show_progress=True)
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens)
        del tokens, texts
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.bm25.save(str(folder))
        except Exception as e:
            logger.warning(f"Could not save BM25 cache: {e}")
        logger.info(f"BM25 index built with {len(self.corpus_ids)} documents.")

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        k = min(k, len(self.corpus_ids))
        results, scores = self.bm25.retrieve(bm25s.tokenize(query, show_progress=False), k=k,
                                             show_progress=False)
        return [{"id": self.corpus_ids[int(i)], "score": float(s)} for i, s in zip(results[0], scores[0])]

    def cleanup(self):
        self.bm25 = None

    @property
    def name(self) -> str:
        return "bm25"
