"""
Stella 1.5B dense retriever (NovaSearch/stella_en_1.5B_v5).

FIXES (2026-10-02):
  1. settings.STELLA_MODEL pointed to infgrad/stella-base-en-v2 (~110M).
     Every earlier "stella_1_5b" number came from the base model.
  2. Stella v5 needs the "s2p_query" prompt on queries (documents get none).
     Without it, query and passage embeddings are not in the trained space.
  3. Titles are indexed (title + text).
  4. Embeddings are kept in float32. NumPy has no fast fp16 matmul.
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

QUERY_PROMPT_NAME = "s2p_query"


class StellaRetriever(BaseRetriever):
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 batch_size: int = 32, use_fp16: bool = True):
        self.model_name = model_name or settings.STELLA_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.model = None
        self.index = None
        self.corpus_ids = None

    def _load(self):
        if self.model is None:
            logger.info(f"Loading Stella model {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
            if self.use_fp16 and self.device.startswith("cuda"):
                self.model.half()
            self.model.eval()

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        from retrievers.dense import emb_cache
        self._load()
        texts = [passage_text(d.get("title", ""), d.get("text", "")) for d in corpus]
        self.corpus_ids = [d["id"] for d in corpus]
        key = emb_cache.corpus_key(self.model_name, self.corpus_ids, texts)
        enc = lambda t: self.model.encode(t, batch_size=self.batch_size, show_progress_bar=True,
                                          convert_to_numpy=True, normalize_embeddings=True)
        emb = emb_cache.get_or_encode(key, texts, enc, repair_fn=self._encode_fp32)
        del texts
        self.index = emb_cache.DenseIndex(emb)
        logger.info(f"Stella index ready with {len(self.corpus_ids)} documents.")

    def _encode_fp32(self, texts, prompt_name=None):
        """fp16 overflows to NaN on a few inputs (Qwen2 backbone); redo them in fp32."""
        half = next(self.model.parameters()).dtype == torch.float16
        if half:
            self.model.float()
        try:
            return self.model.encode(texts, prompt_name=prompt_name, batch_size=8,
                                     convert_to_numpy=True, normalize_embeddings=True)
        finally:
            if half:
                self.model.half()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        q = self.model.encode([query], prompt_name=QUERY_PROMPT_NAME,
                              normalize_embeddings=True, convert_to_numpy=True)[0]
        if not np.isfinite(q).all():
            q = self._encode_fp32([query], prompt_name=QUERY_PROMPT_NAME)[0]
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
        return "stella_1_5b"

    @property
    def requires_gpu(self) -> bool:
        return self.device.startswith("cuda")
