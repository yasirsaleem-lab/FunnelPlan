"""
SPLADE-v3 sparse retriever.

Design:
  1. Term weights are max_t log(1 + ReLU(logit)) over tokens (the SPLADE
     log saturation).
  2. Documents are encoded in batches with the attention mask applied.
  3. Titles are indexed.
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix, vstack
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

from retrievers.base import BaseRetriever
from config import settings
from data.corpus_utils import passage_text

logger = logging.getLogger(__name__)


class SPLADEV3Retriever(BaseRetriever):
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 batch_size: int = 32, max_length: int = 256):
        self.model_name = model_name or settings.SPLADE_V3_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.doc_vectors = None
        self.corpus_ids = None

    def _encode(self, texts: List[str]) -> csr_matrix:
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True,
                                max_length=self.max_length).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
            weights = torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1)
            vec = torch.max(weights, dim=1).values.float().cpu().numpy()
        return csr_matrix(vec)

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        """Encode in chunks of 100k passages; each chunk is saved as a .npz part,
        so a crash resumes at the last finished chunk and later runs load the
        cached matrix instead of re-encoding (5.2M passages is about an hour)."""
        from scipy.sparse import load_npz, save_npz
        from retrievers.dense import emb_cache
        logger.info(f"Loading SPLADE model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        if self.device.startswith("cuda"):
            self.model.half()
        self.model.eval()
        texts = [passage_text(d.get("title", ""), d.get("text", "")) for d in corpus]
        self.corpus_ids = [d["id"] for d in corpus]
        key = emb_cache.corpus_key(self.model_name, self.corpus_ids, texts, extra=f"log1p|len{self.max_length}")
        folder = settings.CACHE_ROOT / "splade" / key
        folder.mkdir(parents=True, exist_ok=True)
        step = 100_000
        parts = []
        for ci, start in enumerate(range(0, len(texts), step)):
            f = folder / f"part_{ci:04d}.npz"
            if not f.exists():
                chunk = texts[start: start + step]
                blocks = [self._encode(chunk[i: i + self.batch_size])
                          for i in tqdm(range(0, len(chunk), self.batch_size),
                                        desc=f"SPLADE chunk {ci + 1}/{(len(texts) - 1) // step + 1}")]
                tmp = folder / f"part_{ci:04d}.tmp.npz"
                save_npz(tmp, vstack(blocks).tocsr())
                tmp.replace(f)
            parts.append(load_npz(f))
        self.doc_vectors = vstack(parts).tocsr()
        logger.info(f"SPLADE index ready with {len(self.corpus_ids)} documents, "
                    f"{self.doc_vectors.nnz / len(self.corpus_ids):.0f} terms per passage.")

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self.doc_vectors is None:
            raise RuntimeError("Index not built.")
        k = min(k, len(self.corpus_ids))
        q = self._encode([query])
        scores = np.asarray(self.doc_vectors.dot(q.T).todense()).ravel()
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [{"id": self.corpus_ids[i], "score": float(scores[i])} for i in top]

    def cleanup(self):
        self.model = None
        self.doc_vectors = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return "splade_v3"

    @property
    def requires_gpu(self) -> bool:
        return self.device.startswith("cuda")
