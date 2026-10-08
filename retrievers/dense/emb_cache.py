"""
Disk cache and search index for dense corpus embeddings.

Changes (2026-10-04, full corpora):
  * get_or_encode() encodes in chunks into an fp16 .npy memmap with a progress
    file. A crash in hour 9 of a 10 hour encode resumes where it stopped.
    5.2M x 1024 dims in fp16 is 10.7 GB on disk and is never held twice in RAM.
  * DenseIndex searches on the GPU when the fp16 matrix fits next to the
    encoder (retrieval runs before the reader is loaded), else in chunks on
    the CPU. Cosine scores of normalised vectors in fp16 differ from fp32 in
    the 3rd or 4th decimal; ranks are almost always unchanged.
  * Old float32 caches (<key>.npy, the 200k subsample) still load.

The cache key covers model name, passage text function version, an extra tag
(pooling, max length) and a hash of all passage ids and text prefixes, so a
changed corpus or model never reuses stale vectors.
"""

import hashlib
import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

TEXT_VERSION = "title_dot_text_v1"   # bump if data.corpus_utils.passage_text changes
CHUNK = 50_000


def cache_dir():
    return settings.CACHE_ROOT / "embeddings"


def corpus_key(model_name: str, ids: List[str], texts: List[str], extra: str = "") -> str:
    h = hashlib.sha1()
    h.update(f"{model_name}|{TEXT_VERSION}|{extra}|{len(ids)}".encode())
    for i, t in zip(ids, texts):
        h.update(i.encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update(t[:64].encode("utf-8", "ignore"))
    return f"{model_name.replace('/', '__')}__{h.hexdigest()[:16]}"


# --- legacy float32 cache (200k subsample) -----------------------------------
def load(key: str):
    p = cache_dir() / f"{key}.npy"
    if p.exists():
        logger.info(f"Embedding cache hit (legacy fp32): {p.name}")
        return np.load(p).astype(np.float32, copy=False)
    return None


def save(key: str, emb: np.ndarray) -> None:
    cache_dir().mkdir(parents=True, exist_ok=True)
    p = cache_dir() / f"{key}.npy"
    tmp = p.with_name(p.stem + ".tmp.npy")
    np.save(tmp, emb.astype(np.float32))
    tmp.replace(p)


# --- chunked, resumable fp16 cache -------------------------------------------
def get_or_encode(key: str, texts: List[str], encode_fn: Callable[[List[str]], np.ndarray],
                  chunk: int = CHUNK, repair_fn: Optional[Callable[[List[str]], np.ndarray]] = None) -> np.ndarray:
    """Return an (n, dim) array (fp16 memmap or legacy fp32) for texts.

    repair_fn: encoder used to redo rows that came out NaN or inf (fp16
    overflow in Qwen2 based encoders such as Stella v5). Usually fp32."""
    legacy = load(key)
    if legacy is not None and legacy.shape[0] == len(texts):
        return legacy
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    path, prog = d / f"{key}.f16.npy", d / f"{key}.progress"
    n = len(texts)
    if path.exists() and not prog.exists():
        mm = np.load(path, mmap_mode="r")
        if mm.shape[0] == n:
            logger.info(f"Embedding cache hit: {path.name} {mm.shape}")
            del mm
            return _repair(path, texts, repair_fn or encode_fn)
    done = int(prog.read_text()) if (prog.exists() and path.exists()) else 0
    if done == 0:
        first = np.asarray(encode_fn(texts[:chunk]), dtype=np.float32)
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, first.shape[1]))
        mm[: len(first)] = first
        done = len(first)
        mm.flush()
        prog.write_text(str(done))
    else:
        mm = np.load(path, mmap_mode="r+")
        logger.info(f"Resuming encoding of {path.name} at {done}/{n}")
    while done < n:
        e = np.asarray(encode_fn(texts[done: done + chunk]), dtype=np.float32)
        mm[done: done + len(e)] = e
        done += len(e)
        mm.flush()
        prog.write_text(str(done))
        logger.info(f"  encoded {done}/{n} ({done / n:.1%})")
    del mm
    prog.unlink()
    return _repair(path, texts, repair_fn or encode_fn)


def _repair(path, texts, repair_fn) -> np.ndarray:
    """Find rows that are not finite and re-encode them with repair_fn.

    2026-10-04: Stella v5 in fp16 produced 108 NaN rows out of 101,958 on
    MuSiQue. torch.topk ranks NaN above every real score, so every query
    returned the same 108 passages and Stella recall@10 was 0.0. A .ok marker
    records a clean scan so a 10 GB matrix is checked only once."""
    ok = path.with_name(path.name + ".ok")
    if ok.exists():
        return np.load(path, mmap_mode="r")
    mm = np.load(path, mmap_mode="r")
    bad = []
    for i in range(0, mm.shape[0], 200_000):
        blk = np.asarray(mm[i: i + 200_000], dtype=np.float32)
        bad.extend((np.where(~np.isfinite(blk).all(axis=1))[0] + i).tolist())
    del mm
    if bad:
        logger.warning(f"{path.name}: {len(bad)} non finite rows; re-encoding them")
        mm = np.load(path, mmap_mode="r+")
        e = np.asarray(repair_fn([texts[i] for i in bad]), dtype=np.float32)
        still = ~np.isfinite(e).all(axis=1)
        e[still] = 0.0          # a zero vector scores 0 and never ranks first
        mm[bad] = e
        mm.flush()
        del mm
        logger.warning(f"  repaired {len(bad) - int(still.sum())}; zeroed {int(still.sum())}")
    ok.write_text(str(len(bad)))
    return np.load(path, mmap_mode="r")


class DenseIndex:
    """Exact inner product top k over a (possibly memmapped) embedding matrix."""

    def __init__(self, emb: np.ndarray, gpu_reserve_gb: float = 1.5, cpu_chunk: int = 500_000):
        self.emb = emb
        self.n, self.dim = emb.shape
        self.gpu = None
        self.gpu_reserve = gpu_reserve_gb * 1024 ** 3
        self.cpu_chunk = cpu_chunk
        self._try_gpu()

    def _try_gpu(self):
        try:
            import torch
        except ImportError:
            return
        if not torch.cuda.is_available():
            return
        need = self.n * self.dim * 2
        # Release blocks the caching allocator still holds (e.g. fp32 weights
        # freed by model.half()). Without this, mem_get_info under reports free
        # memory: in profiling, Stella on HotpotQA saw 9.0 GB free, fell back
        # to CPU search and took 12.5 s per query instead of ~0.07 s.
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        if free - need < self.gpu_reserve:
            logger.info(f"DenseIndex: {need / 1024**3:.1f} GB does not fit on GPU "
                        f"({free / 1024**3:.1f} GB free); using chunked CPU search")
            return
        t = torch.empty((self.n, self.dim), dtype=torch.float16, device="cuda")
        step = 200_000
        for i in range(0, self.n, step):
            # np.array(copy=True): the memmap is read only; a writable copy avoids
            # PyTorch's non writable array warning. Values are identical.
            t[i: i + step] = torch.from_numpy(np.array(self.emb[i: i + step], dtype=np.float16, copy=True)).cuda()
        self.gpu = t
        logger.info(f"DenseIndex: {self.n} x {self.dim} fp16 on GPU ({need / 1024**3:.1f} GB)")

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = min(k, self.n)
        if not np.isfinite(q).all():
            raise ValueError("query embedding is not finite (fp16 overflow?)")
        if self.gpu is not None:
            import torch
            qt = torch.from_numpy(np.asarray(q, dtype=np.float16)).cuda()
            s = torch.nan_to_num((self.gpu @ qt).float(), nan=float("-inf"))
            v, i = torch.topk(s, k)
            return i.cpu().numpy(), v.cpu().numpy()
        q32 = np.asarray(q, dtype=np.float32)
        best_i, best_s = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        for start in range(0, self.n, self.cpu_chunk):
            s = np.asarray(self.emb[start: start + self.cpu_chunk], dtype=np.float32) @ q32
            s = np.where(np.isfinite(s), s, -np.inf)
            kk = min(k, len(s))
            top = np.argpartition(-s, kk - 1)[:kk]
            best_i = np.concatenate([best_i, top + start])
            best_s = np.concatenate([best_s, s[top]])
            if len(best_i) > k:
                keep = np.argpartition(-best_s, k - 1)[:k]
                best_i, best_s = best_i[keep], best_s[keep]
        order = np.argsort(-best_s)
        return best_i[order], best_s[order]

    def release(self):
        self.gpu = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
