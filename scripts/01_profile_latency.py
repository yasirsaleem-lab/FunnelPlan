"""
Profile retrieval latency. Configure with the CONFIG block, then run.

  * One method + one dataset: profiled in this process.
  * "all" for method or dataset: every (dataset, method) pair not yet done
    runs in its own fresh Python process, so the GPU starts clean each time.
  * Pairs already profiled are skipped, so rerunning is safe.

Output, one file per dataset (keyed by retriever):
  nq        -> data/processed/latency_profiles.json   (existing file, kept)
  hotpotqa  -> data/processed/latency_profiles_hotpotqa.json
Then run:  python -m cost_model.fit_cost_model   (averages over the dataset files)

Note: gte_qwen2_7b encodes the whole corpus of each dataset (a few hours each).
The embeddings are cached in data/cache/embeddings/, so gte_bm25_rrf and the
label script (04) reuse them instead of encoding again.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Corpus mode must be set BEFORE config is imported. Child processes inherit it.
#   "sub200k": old 200k subsample profiles (data/processed)
#   "full":    full corpora profiles (data/processed_full), the ones the paper uses
CORPUS_MODE = "full"
import os
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE

from config import settings

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "dataset": "all",                 # "nq", "hotpotqa", "musique", "2wikimultihopqa" or "all"
    "method": "all",                  # one name, a list, or "all". Full mode methods:
                                      # "closed_book" (depth 0), "bm25", "splade_v3", "bge_m3",
                                      # "stella_1_5b", "bge_bm25_rrf", "qwen3_reranker_0_6b"
}
# Do NOT run while 04 is labelling: both share the GPU and both timings break.

NUM_PROFILE_QUERIES = 50              # the first 50 of the same seeded sample 04 labels
DEPTHS = [1, 3, 5, 10, 20, 100] if CORPUS_MODE == "full" else [10, 50, 100, 200]
DATASETS = list(settings.DATASETS)


def profile_path(ds: str) -> Path:
    if CORPUS_MODE == "full":
        return settings.PROCESSED_DATA_ROOT / f"latency_profiles_{ds}.json"
    name = "latency_profiles.json" if ds == "nq" else f"latency_profiles_{ds}.json"
    return settings.PROCESSED_DATA_ROOT / name


def is_done(ds: str, method: str) -> bool:
    p = profile_path(ds)
    return p.exists() and method in json.loads(p.read_text())


def run_one(ds: str, method: str):
    from cost_model.latency_profiler import main as profile_main
    profile_main({"dataset": ds, "method": method, "merge": True,
                  "num_profile_queries": NUM_PROFILE_QUERIES, "depths": DEPTHS,
                  "output": str(profile_path(ds))})


def status():
    for ds in DATASETS:
        done = [m for m in settings.PLAN_RETRIEVERS if is_done(ds, m)]
        left = [m for m in settings.PLAN_RETRIEVERS if not is_done(ds, m)]
        print(f"  {ds:9s} done: {done}")
        print(f"  {'':9s} left: {left}")


def main():
    from logutil import setup_logging
    setup_logging("01_profile_latency", CONFIG)
    # Child mode: called by this script itself for one (dataset, method) pair.
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        run_one(sys.argv[2], sys.argv[3])
        return

    ds_cfg, m_cfg = CONFIG["dataset"], CONFIG["method"]
    datasets = DATASETS if ds_cfg == "all" else [ds_cfg]
    methods = settings.PLAN_RETRIEVERS if m_cfg == "all" else [m_cfg]
    bad = [d for d in datasets if d not in DATASETS] + [m for m in methods if m not in settings.PLAN_RETRIEVERS]
    if bad:
        sys.exit(f"Unknown dataset/method: {bad}")

    todo = [(d, m) for d in datasets for m in methods if not is_done(d, m)]
    for d, m in [(d, m) for d in datasets for m in methods if is_done(d, m)]:
        print(f"[skip] {d} / {m} already profiled")

    if len(todo) == 1:
        run_one(*todo[0])
    else:
        for d, m in todo:
            print(f"=== {d} / {m} (fresh process) ===", flush=True)
            r = subprocess.run([sys.executable, __file__, "--child", d, m])
            if r.returncode != 0:
                sys.exit(f"Profiling {d}/{m} failed with exit code {r.returncode}")

    print("\nProfiling status:")
    status()


if __name__ == "__main__":
    main()
