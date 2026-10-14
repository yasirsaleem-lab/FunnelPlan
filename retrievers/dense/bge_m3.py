"""
BGE-M3 dense retriever (dense head only, via SentenceTransformers).

Corpus embeddings are float32: NumPy has no BLAS path for fp16, so fp16
search would dominate the measured latency. Titles are indexed.
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from retrievers.base import BaseRetriever
from config import settings
from data.corpus_utils import passage_text

logger = logging.getLogger(__name__)


class BGERetriever(BaseRetriever):
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 batch_size: int = 32, use_fp16: bool = True):
        self.model_name = model_name or getattr(settings, "BGE_M3_MODEL", "BAAI/bge-m3")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.model = None
        self.index = None
        self.corpus_ids = None

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        from retrievers.dense import emb_cache
        logger.info(f"Loading BGE-M3 model {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if self.use_fp16 and self.device.startswith("cuda"):
            self.model.half()
        self.model.eval()
        texts = [passage_text(d.get("title", ""), d.get("text", "")) for d in corpus]
        self.corpus_ids = [d["id"] for d in corpus]
        key = emb_cache.corpus_key(self.model_name, self.corpus_ids, texts)
        enc = lambda t: self.model.encode(t, batch_size=self.batch_size, show_progress_bar=True,
                                          convert_to_numpy=True, normalize_embeddings=True)
        emb = emb_cache.get_or_encode(key, texts, enc)
        del texts
        self.index = emb_cache.DenseIndex(emb)
        logger.info(f"BGE-M3 index ready with {len(self.corpus_ids)} documents.")

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        idx, sc = self.index.search(q, k)
        return [{"id": self.corpus_ids[i], "score": float(v)} for i, v in zip(idx, sc)]

    def cleanup(self):
        if self.index is not None:
            self.index.release()
        self.index = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return "bge_m3"

    @property
    def requires_gpu(self) -> bool:
        return self.device.startswith("cuda")
