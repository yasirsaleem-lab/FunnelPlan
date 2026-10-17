import os

# ----------------------------------------------------------------------------
# Hugging Face offline mode.
# By default transformers / sentence-transformers send a HEAD request to
# huggingface.co to check for updates even when the model is cached. From
# mainland China that request often fails (SSL EOF) and stalls every run.
# All models are already in the local cache, so load them offline.
# To download a NEW model: set FUNNELPLAN_HF_OFFLINE=0, and optionally
# HF_ENDPOINT=https://hf-mirror.com, for that one run.
# This must run before transformers / huggingface_hub are imported, so every
# script imports config before them.
# ----------------------------------------------------------------------------
if os.environ.get("FUNNELPLAN_HF_OFFLINE", "1") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional


class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).parent.resolve()
    DATA_ROOT: Path = PROJECT_ROOT / "data"
    RAW_DATA_ROOT: Path = DATA_ROOT / "raw"
    PROCESSED_DATA_ROOT: Path = DATA_ROOT / "processed"
    CACHE_ROOT: Path = DATA_ROOT / "cache"
    RESULTS_ROOT: Path = PROJECT_ROOT / "results"

    PASSAGE_CAP: int = 200_000
    VRAM_LIMIT_BYTES: int = 20 * 1024**3
    RAM_LIMIT_BYTES: int = 64 * 1024**3

    DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
        "nq": {
            "corpus": "nq/corpus.jsonl",
            "queries": "nq/queries.jsonl",
            "qrels": "nq/qrels.tsv",
        },
        "hotpotqa": {
            "corpus": "hotpotqa/corpus.jsonl",
            "queries": "hotpotqa/queries.jsonl",
            "qrels": "hotpotqa/qrels.tsv",
        },
    }

    BM25_IMPLEMENTATION: str = "bm25"
    SPLADE_V3_MODEL: str = "naver/splade-v3"
    # Stella 1.5B v5 (not stella-base-en-v2, a ~110M model).
    STELLA_MODEL: str = "NovaSearch/stella_en_1.5B_v5"
    GTE_QWEN_MODEL: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    BGE_M3_MODEL: str = "BAAI/bge-m3"
    RERANKER_0_6B: str = "Qwen/Qwen3-Reranker-0.6B"
    RERANKER_4B: str = "Qwen/Qwen3-Reranker-4B"
    AGENT_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

    # Reranker plans re-rank the top RERANK_CANDIDATES of RERANK_BASE.
    # The candidate pool is fixed, so "depth" only controls how many reranked
    # passages reach the agent.
    RERANK_BASE: str = "bm25"
    RERANK_CANDIDATES: int = 100

    # The plan grid used for labels, training and planning.
    PLAN_RETRIEVERS: List[str] = [
        "bm25", "splade_v3", "stella_1_5b", "gte_qwen2_7b", "bge_m3",
        "gte_bm25_rrf", "qwen3_reranker_0_6b", "qwen3_reranker_4b",
    ]
    PLAN_DEPTHS: List[int] = [10, 50, 100]

    # Label file written by scripts/02_generate_tsr_labels.py (v3 pipeline).
    LABELS_FILE: str = "tsr_labels_v3.csv"
    FEATURES_FILE: str = "features_v3.csv"

    # Corpus mode fields; see the block after `settings = Settings()`.
    CORPUS_MODE: str = "sub200k"
    DATASETS: List[str] = ["nq", "hotpotqa"]
    NUM_QUERIES: int = 200

    PROFILE_DEPTHS: List[int] = [10, 50, 100, 200, 500, 1000]
    BUDGET_STEPS: List[int] = [50, 100, 200, 500, 1000]

    class Config:
        env_prefix = "FUNNELPLAN_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ----------------------------------------------------------------------------
# Corpus mode.
#   "sub200k": the original 200k gold preserving subsample (pilot, gate run).
#   "full":    full corpora, 4 datasets, 500 queries (the paper runs).
# Set with the environment variable FUNNELPLAN_CORPUS_MODE=full, or edit the
# default below. In full mode every data, cache and results path moves to its
# own *_full folder, so the two modes can never mix files.
# ----------------------------------------------------------------------------
CORPUS_MODE = os.environ.get("FUNNELPLAN_CORPUS_MODE", "sub200k")
settings.CORPUS_MODE = CORPUS_MODE
settings.DATASETS = ["nq", "hotpotqa"]
settings.NUM_QUERIES = 200
if CORPUS_MODE == "full":
    settings.PROCESSED_DATA_ROOT = settings.DATA_ROOT / "processed_full"
    settings.CACHE_ROOT = settings.DATA_ROOT / "cache_full"
    settings.RESULTS_ROOT = settings.PROJECT_ROOT / "results_full"
    settings.PASSAGE_CAP = 10**9          # never subsample, even on a fallback path
    settings.DATASETS = ["nq", "hotpotqa", "musique", "2wikimultihopqa"]
    settings.NUM_QUERIES = 500            # IRCoT (ACL 2023) uses 500 test questions per dataset
    # GTE-Qwen2-7B and the 4B reranker are dropped: ~80 GPU hours to encode the
    # full corpora, and the 4B reranker added cost without gain in the pilot.
    settings.PLAN_RETRIEVERS = ["bm25", "splade_v3", "bge_m3", "stella_1_5b",
                                "bge_bm25_rrf", "qwen3_reranker_0_6b"]
    settings.RERANK_BASE = "bge_bm25_rrf"   # rerank a strong hybrid, not BM25
    settings.PLAN_DEPTHS = [1, 3, 5, 10, 20]
elif CORPUS_MODE != "sub200k":
    raise ValueError(f"FUNNELPLAN_CORPUS_MODE must be 'sub200k' or 'full', got {CORPUS_MODE!r}")

settings.DATA_ROOT.mkdir(exist_ok=True)
settings.RAW_DATA_ROOT.mkdir(exist_ok=True)
settings.PROCESSED_DATA_ROOT.mkdir(exist_ok=True)
settings.CACHE_ROOT.mkdir(exist_ok=True)
settings.RESULTS_ROOT.mkdir(exist_ok=True)
