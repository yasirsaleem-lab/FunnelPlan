"""
Qwen3-Reranker-0.6B as a retriever and as a standalone pair scorer.

Two usage modes:
  1. As a retriever: give it a base_retriever, call build_index(corpus), then
     call search(query, k). It re-ranks a FIXED pool of base candidates.
  2. As a scorer (used by feature building): score_pairs(query, passages).

_score_batch reads the "yes/no" logits at position -1, so sequences are LEFT
padded with an explicit attention mask, as in the official Qwen3-Reranker
example. With right padding, position -1 would be a pad token for every
sequence shorter than the longest one in the batch.
"""

import gc
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrievers.base import BaseRetriever
from config import settings
from data.corpus_utils import passage_text

logger = logging.getLogger(__name__)

_tf_major, _tf_minor, *_ = transformers.__version__.split(".")
if (int(_tf_major), int(_tf_minor)) < (4, 51):
    logger.warning(
        f"transformers {transformers.__version__} may be too old for Qwen3 rerankers. "
        f"Consider upgrading: pip install 'transformers>=4.51.0'"
    )


def left_pad(batch_ids: List[List[int]], pad_id: int):
    """Left pad token id lists. Returns (input_ids, attention_mask) as nested lists.

    Kept as a pure function so tests can check it without loading a model.
    """
    max_len = max(len(ids) for ids in batch_ids)
    input_ids, mask = [], []
    for ids in batch_ids:
        n_pad = max_len - len(ids)
        input_ids.append([pad_id] * n_pad + list(ids))
        mask.append([0] * n_pad + [1] * len(ids))
    return input_ids, mask


class Qwen3Reranker06B(BaseRetriever):
    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(
        self,
        base_retriever: Optional[BaseRetriever] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        candidate_k: Optional[int] = None,
        max_length: int = 2048,
    ):
        self.base_retriever = base_retriever
        self.model_name = model_name or getattr(settings, "RERANKER_0_6B", "Qwen/Qwen3-Reranker-0.6B")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.candidate_k = candidate_k or settings.RERANK_CANDIDATES
        # Passages are short (NQ / HotpotQA paragraphs). 2048 avoids huge
        # padded batches; the official limit is 8192.
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.corpus = None
        self.token_true_id = None
        self.token_false_id = None
        self.prefix_ids = None
        self.suffix_ids = None
        self.pad_id = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        logger.info(f"Loading Qwen3 reranker {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id

        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        logger.info(
            f"Reranker loaded. yes_id={self.token_true_id}, no_id={self.token_false_id}, "
            f"pad_id={self.pad_id}"
        )

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        self.load()
        self.corpus = {doc["id"]: doc for doc in corpus}
        if self.base_retriever is not None:
            self.base_retriever.build_index(corpus)
        logger.info(f"Reranker ready. Corpus size: {len(self.corpus)}")

    def cleanup(self) -> None:
        for attr in ("model", "tokenizer"):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        if self.base_retriever is not None and hasattr(self.base_retriever, "cleanup"):
            try:
                self.base_retriever.cleanup()
            except Exception:
                pass
        self.corpus = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def _build_batch_ids(self, query: str, texts: List[str]) -> List[List[int]]:
        max_content_len = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        out = []
        for text in texts:
            content = f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {text}"
            ids = self.tokenizer.encode(
                content, add_special_tokens=False, truncation=True, max_length=max_content_len
            )
            out.append(self.prefix_ids + ids + self.suffix_ids)
        return out

    def _score_batch(self, batch_ids: List[List[int]]) -> List[float]:
        ids, mask = left_pad(batch_ids, self.pad_id)
        input_ids = torch.tensor(ids, device=self.device)
        attention_mask = torch.tensor(mask, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        stacked = torch.stack([logits[:, self.token_false_id], logits[:, self.token_true_id]], dim=1)
        probs = torch.nn.functional.log_softmax(stacked.float(), dim=1)[:, 1].exp()
        return probs.cpu().tolist()

    def score_pairs(self, query: str, passages: List[Any]) -> List[float]:
        if not passages:
            return []
        if self.model is None or self.tokenizer is None:
            self.load()
        texts = []
        for p in passages:
            if isinstance(p, str):
                texts.append(p)
            elif isinstance(p, dict):
                texts.append(passage_text(p.get("title", ""), p.get("text", "")))
            else:
                texts.append(str(p))
        # Sort by length so each batch pads little; restore order after.
        order = np.argsort([len(t) for t in texts])
        scores = np.zeros(len(texts), dtype=np.float64)
        for i in range(0, len(texts), self.batch_size):
            idx = order[i:i + self.batch_size]
            batch_ids = self._build_batch_ids(query, [texts[j] for j in idx])
            scores[idx] = self._score_batch(batch_ids)
        return scores.tolist()

    # ------------------------------------------------------------------
    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self.corpus is None:
            raise RuntimeError("Index not built.")
        if self.base_retriever is None:
            raise ValueError("No base retriever set for reranking.")
        pool = min(max(k, self.candidate_k), len(self.corpus))
        candidates = self.base_retriever.search(query, pool)
        docs = [self.corpus[c["id"]] for c in candidates if c["id"] in self.corpus]
        if not docs:
            return []
        scores = self.score_pairs(query, docs)
        scored = sorted(zip([d["id"] for d in docs], scores), key=lambda x: x[1], reverse=True)
        return [{"id": d, "score": float(s)} for d, s in scored[:k]]

    @property
    def name(self) -> str:
        return "qwen3_reranker_0_6b"

    @property
    def requires_gpu(self) -> bool:
        return self.device.startswith("cuda")
