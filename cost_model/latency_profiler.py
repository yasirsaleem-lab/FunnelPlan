"""
Profiles search latency for each retriever on a sample of queries.

Retrievers come from the shared registry (rerankers use the same base and
fixed candidate pool as the label pipeline). Titles are indexed, CUDA is
synchronised around each timed call, and each retriever is released after
profiling.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from tqdm import tqdm

from config import settings
from data.data_loader import get_processed
from data.corpus_utils import build_corpus_list
from retrievers.registry import build_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_methods(method_config: Union[str, List[str]]):
    if method_config == "all":
        return list(settings.PLAN_RETRIEVERS)
    return list(method_config) if isinstance(method_config, list) else [method_config]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_retriever(retriever, queries, depths, warmup_runs=1, timing_runs=3):
    results = []
    for q in queries[:3]:
        retriever.search(q, max(depths))
    for depth in depths:
        times = []
        for q in tqdm(queries, desc=f"depth {depth}", leave=False):
            for _ in range(warmup_runs):
                retriever.search(q, depth)
            for _ in range(timing_runs):
                _sync()
                t = time.perf_counter()
                retriever.search(q, depth)
                _sync()
                times.append(time.perf_counter() - t)
        results.append({"depth": depth, "mean_time": float(np.mean(times)),
                        "std_time": float(np.std(times)), "query_count": len(times)})
        logger.info(f"depth {depth}: {np.mean(times)*1000:.1f} ms")
    return results


def save_profile(out: Path, name: str, result) -> Dict[str, Any]:
    """Add ONE method's profile to the JSON file without touching the others.

    * Re-reads the file right before writing, so methods saved by other runs
      in the meantime are kept (the old code read once at start, wrote hours later).
    * Refuses to continue if the existing file cannot be parsed, instead of
      replacing it with a file that holds only this method.
    * Keeps a .bak copy of the previous version.
    * Writes to a temp file and renames it, so a crash mid write cannot
      leave a half written JSON.
    """
    profiles: Dict[str, Any] = {}
    if out.exists():
        try:
            profiles = json.loads(out.read_text())
        except json.JSONDecodeError as e:
            bad = out.with_suffix(".unreadable.json")
            tmp_dump = out.with_suffix(f".{name}.json")
            tmp_dump.write_text(json.dumps({name: result}, indent=2))
            raise RuntimeError(
                f"{out} is not valid JSON ({e}). Nothing was overwritten. "
                f"This run's result was saved to {tmp_dump}. Fix or restore {out} "
                f"(see {out.with_suffix('.json.bak')}) and merge it by hand."
            )
        out.with_suffix(".json.bak").write_text(json.dumps(profiles, indent=2))
    replaced = name in profiles
    profiles[name] = result
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profiles, indent=2))
    tmp.replace(out)
    logger.info(f"{'Replaced' if replaced else 'Added'} '{name}' in {out.name}. "
                f"Methods in file: {sorted(profiles)}")
    return profiles


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 3
    except ImportError:
        return float("nan")


def _dir_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    files = [path] if path.is_file() else [f for f in path.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1024 ** 3


def _sample_queries(queries, answers, n, seed=42):
    """Same seeded sample as scripts/04 (queries with gold answers), first n."""
    with_gold = [q for q in queries.keys() if answers.get(str(q), [])]
    if len(with_gold) > settings.NUM_QUERIES:
        np.random.seed(seed)
        with_gold = [str(q) for q in np.random.choice(with_gold, settings.NUM_QUERIES, replace=False).tolist()]
    return [queries[q]["text"] for q in with_gold[:n]]


def _find_dense_index(r):
    """The DenseIndex behind a retriever, also inside fusion and reranker
    wrappers; None for purely sparse retrievers (BM25, SPLADE)."""
    from retrievers.dense.emb_cache import DenseIndex
    stack = [r]
    while stack:
        x = stack.pop()
        if x is None:
            continue
        idx = getattr(x, "index", None)
        if isinstance(idx, DenseIndex):
            return idx
        stack += [getattr(x, "dense", None), getattr(x, "base_retriever", None)]
    return None


def save_meta(ds: str, name: str, meta: Dict[str, Any]) -> None:
    """System numbers an EAB paper reports: index load time, memory, disk.
    Kept in its own file (latency_meta_<ds>.json) so the cost model fit,
    which globs latency_profiles*.json, never reads it."""
    out = settings.PROCESSED_DATA_ROOT / f"latency_meta_{ds}.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    data[name] = meta
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(out)


def main(config: Dict[str, Any]):
    """Queries are the same seeded sample the labels use; depths
    include the plan grid; index load time, peak RAM, peak VRAM, dense index
    placement (GPU or CPU) and cache size on disk are saved per retriever.
    Note: build_index on a cached corpus measures LOAD time, not encode time.
    Encode time is a one off cost and is reported from the 04 logs."""
    dataset = config.get("dataset", "nq")
    out = Path(config.get("output", settings.PROCESSED_DATA_ROOT / "latency_profiles.json"))
    corpus, queries, _, answers = get_processed(dataset)
    corpus_list = build_corpus_list(corpus)
    qtexts = _sample_queries(queries, answers, config.get("num_profile_queries", 50))
    for name in get_methods(config.get("method", "all")):
        logger.info(f"=== profiling {name} on {dataset} ({len(corpus_list)} passages) ===")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        r = build_retriever(name)
        try:
            t0 = time.perf_counter()
            r.build_index(corpus_list)
            load_s = time.perf_counter() - t0
            result = profile_retriever(r, qtexts, config.get("depths", settings.PROFILE_DEPTHS))
            save_profile(out, name, result)   # saved right after each method
            idx = _find_dense_index(r)
            save_meta(dataset, name, {
                "n_passages": len(corpus_list),
                "index_load_s": round(load_s, 1),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else 0.0,
                "rss_gb_after": round(_rss_gb(), 2),
                "dense_on_gpu": (getattr(idx, "gpu", None) is not None) if idx is not None else None,
                "cache_gb": {k: round(_dir_gb(settings.CACHE_ROOT / k), 2) for k in ("embeddings", "bm25", "splade")},
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        finally:
            if hasattr(r, "cleanup"):
                r.cleanup()
            del r
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main({"dataset": "nq", "method": "all", "num_profile_queries": 50,
          "depths": settings.PROFILE_DEPTHS, "merge": True})
