"""
GTE-Qwen2-7B-instruct dense retriever.

FIXES (2026-10-02):
  1. Last token pooling (the trained pooling), not mean pooling.
  2. Query instruction prefix "Instruct: <task>\\nQuery: <query>".
  3. Pinned to one GPU; load fails if any layer lands on CPU or disk.
  4. Titles indexed; float32 embeddings for fast NumPy search.

FIXES (2026-10-02, evening):
  5. VRAM guard. On Windows the NVIDIA driver can silently spill GPU
     allocations into shared system RAM ("Sysmem Fallback") instead of
     raising out of memory. The model then runs ~20x slower and nothing
     errors. This happened when earlier retrievers in the same process left
     GPU memory allocated. The guard checks free VRAM before loading and
     times one batch after loading, and stops with a clear message.
  6. Length sorted batches: far less padding, so fewer tokens per batch.
  7. Corpus embeddings are cached on disk (retrievers/dense/emb_cache.py),
     so 03, 04 and the RRF retriever encode each corpus once, not 3 times.
"""

import gc
import logging
import os
import time
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from retrievers.base import BaseRetriever
from retrievers.dense import emb_cache
from config import settings
from data.corpus_utils import passage_text

logger = logging.getLogger(__name__)

TASK = "Given a web search query, retrieve relevant passages that answer the query"
# fp16 weights are ~15.2 GB; activations for batch 16 x 512 need ~2 GB more.
MIN_FREE_GB = float(os.environ.get("FUNNELPLAN_GTE_MIN_FREE_GB", "17.5"))
# A healthy GPU encodes one 16 x 512 batch in ~1 s. Shared memory spill is ~20 s.
SLOW_BATCH_S = float(os.environ.get("FUNNELPLAN_GTE_SLOW_BATCH_S", "6.0"))


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Official GTE-Qwen2 pooling. Works for left or right padding."""
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lengths]


def format_query(query: str) -> str:
    return f"Instruct: {TASK}\nQuery: {query}"


def _gb(x: float) -> str:
    return f"{x / 1024**3:.1f} GB"


class GTEQwenRetriever(BaseRetriever):
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 batch_size: int = 16, use_fp16: bool = True, doc_max_length: int = 512,
                 check_speed: bool = True):
        self.model_name = model_name or settings.GTE_QWEN_MODEL
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.doc_max_length = doc_max_length
        self.check_speed = check_speed
        self.tokenizer = None
        self.model = None
        self.index = None
        self.corpus_ids = None

    # ------------------------------------------------------------------
    def _vram_guard(self):
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        mine = torch.cuda.memory_allocated()
        logger.info(f"VRAM before GTE load: free {_gb(free)} / total {_gb(total)}; "
                    f"this process already holds {_gb(mine)}")
        if free / 1024**3 < MIN_FREE_GB:
            raise RuntimeError(
                f"Only {_gb(free)} VRAM free; GTE-Qwen2-7B needs about {MIN_FREE_GB:.1f} GB. "
                f"This process holds {_gb(mine)} (leftovers from earlier models?). "
                "Run this retriever in a fresh process (03/04 now do this by default) and close "
                "other GPU programs. On Windows also set NVIDIA Control Panel > Manage 3D settings > "
                "'CUDA - Sysmem Fallback Policy' = 'Prefer No Sysmem Fallback', so a shortage raises "
                "an error instead of running 20x slower. Override with FUNNELPLAN_GTE_MIN_FREE_GB."
            )

    def _load(self):
        if self.model is not None:
            return
        logger.info(f"Loading GTE-Qwen model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        dtype = torch.float16 if (self.use_fp16 and self.device.startswith("cuda")) else torch.float32
        if self.device.startswith("cuda"):
            self._vram_guard()
            gpu = int(self.device.split(":")[1]) if ":" in self.device else 0
            kwargs = dict(trust_remote_code=True, torch_dtype=dtype, device_map={"": gpu})
            try:
                self.model = AutoModel.from_pretrained(self.model_name, attn_implementation="sdpa", **kwargs)
            except (TypeError, ValueError) as e:
                logger.warning(f"sdpa attention not available ({e}); using default attention")
                self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            dev_map = getattr(self.model, "hf_device_map", {}) or {}
            bad = {k: v for k, v in dev_map.items() if str(v) in ("cpu", "disk")}
            if bad:
                raise RuntimeError(f"GTE-Qwen2-7B layers offloaded to CPU/disk: {list(bad)[:5]}")
            free, total = torch.cuda.mem_get_info()
            logger.info(f"VRAM after GTE load: free {_gb(free)} / total {_gb(total)}")
        else:
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True,
                                                   torch_dtype=dtype).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def _encode_batch(self, texts: List[str], max_length: int) -> np.ndarray:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                               return_tensors="pt").to(self.model.device)
        hidden = self.model(**batch, use_cache=False).last_hidden_state
        emb = F.normalize(last_token_pool(hidden, batch["attention_mask"]).float(), p=2, dim=1)
        return emb.cpu().numpy()

    def _encode(self, texts: List[str], max_length: int, show_progress: bool = False) -> np.ndarray:
        # Sort by length so each batch pads to similar lengths; restore order at the end.
        order = np.argsort([-len(t) for t in texts], kind="stable")
        chunks = []
        starts = range(0, len(texts), self.batch_size)
        if show_progress:
            starts = tqdm(starts, desc="GTE encode", unit="batch")
        with torch.inference_mode():
            for n, i in enumerate(starts):
                idx = order[i:i + self.batch_size]
                t0 = time.perf_counter()
                chunks.append(self._encode_batch([texts[j] for j in idx], max_length))
                # Batches are longest first, so batch 2 is a worst case timing.
                if self.check_speed and show_progress and n == 2 and self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    if dt > SLOW_BATCH_S:
                        raise RuntimeError(
                            f"GTE batch took {dt:.1f}s (healthy is about 1s). The GPU is almost "
                            "certainly spilling into shared system memory. Run this retriever in a "
                            "fresh process, close other GPU programs, and set the NVIDIA "
                            "'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem Fallback'. "
                            "Override with FUNNELPLAN_GTE_SLOW_BATCH_S if your GPU is just slow."
                        )
        emb = np.vstack(chunks).astype(np.float32)
        out = np.empty_like(emb)
        out[order] = emb
        return out

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        texts = [passage_text(d.get("title", ""), d.get("text", "")) for d in corpus]
        self.corpus_ids = [d["id"] for d in corpus]
        key = emb_cache.corpus_key(self.model_name, self.corpus_ids, texts,
                                   extra=f"lasttok|maxlen{self.doc_max_length}")
        self._load()   # needed for query encoding even on a cache hit
        enc = lambda t: self._encode(t, self.doc_max_length, show_progress=True)
        emb = emb_cache.get_or_encode(key, texts, enc)
        del texts
        self.index = emb_cache.DenseIndex(emb)
        logger.info(f"GTE-Qwen index ready with {len(self.corpus_ids)} documents.")

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        with torch.inference_mode():
            q = self._encode_batch([format_query(query)], max_length=512)[0]
        idx, sc = self.index.search(q, k)
        return [{"id": self.corpus_ids[i], "score": float(v)} for i, v in zip(idx, sc)]

    def cleanup(self):
        if self.index is not None:
            self.index.release()
        self.index = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return "gte_qwen2_7b"

    @property
    def requires_gpu(self) -> bool:
        return self.device.startswith("cuda")
