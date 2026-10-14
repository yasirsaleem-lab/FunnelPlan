"""
data_loader.py

Handles loading, subsampling, and caching of datasets (NQ, HotpotQA, BRIGHT).
Also loads gold answers for each query.
- NQ and HotpotQA: read JSONL files directly from data/raw/
- BRIGHT: uses Hugging Face datasets, loading both documents, query examples, and answers.
- Gold answers for NQ and HotpotQA are automatically downloaded from HuggingFace
  and matched to the BEIR queries.

Subsampling preserves gold documents: when capping the corpus, all documents
that appear as relevant in qrels are kept, and the remainder is filled by
random sampling. This ensures evaluation and TSR labelling can reach the
gold passages for every query.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from tqdm import tqdm

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seed for reproducible subsampling
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Cache for BRIGHT documents per subset
_BRIGHT_DOCS_CACHE = {}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_jsonl(data: Dict[str, Any], path: Path) -> None:
    """Save a dict of records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            if not isinstance(value, dict):
                raise TypeError(f"Value for key {key} is not a dict, got {type(value)}")
            if "id" not in value:
                value["id"] = key
            f.write(json.dumps(value, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> Dict[str, Any]:
    """Load a JSONL file into a dict keyed by the line index (as string)."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            if "id" not in item:
                item["id"] = str(idx)
            result[str(idx)] = item
    return result


def _save_qrels(qrels: Dict[str, Dict[str, int]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid, doc_scores in qrels.items():
            for doc_id, score in doc_scores.items():
                f.write(f"{qid}\t{doc_id}\t{score}\n")


def _load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, doc_id, score_str = parts[0], parts[1], parts[2]
            # Skip header lines like "query-id	corpus-id	score"
            try:
                score = int(score_str)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[doc_id] = score
    return qrels


def subsample_corpus(
    corpus: Dict[str, Dict[str, Any]],
    cap: int,
    qrels: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Subsample a corpus down to `cap` documents.

    If `qrels` is provided, ALL gold (relevant) documents are preserved.
    The remainder of the cap is filled by random sampling.

    IMPORTANT: `corpus` is keyed by line index. Each value may carry a real
    BEIR id in `_id` or `id`. Gold docs in qrels use the real BEIR ids, so we
    match on the value's `_id`/`id` field, not on the dict key.
    """
    if len(corpus) <= cap:
        return corpus

    # Build map: line_key -> real_id
    real_id_of = {
        k: str(v.get("_id") or v.get("id") or k)
        for k, v in corpus.items()
    }

    # Collect gold real ids from qrels
    gold_real_ids = set()
    if qrels is not None:
        for doc_scores in qrels.values():
            gold_real_ids.update(str(d) for d in doc_scores.keys())

    # Find corpus keys that correspond to gold docs
    gold_keys = [k for k, realid in real_id_of.items() if realid in gold_real_ids]

    if not gold_keys:
        # No gold docs to preserve -> fall back to pure random
        logger.warning(
            f"No gold documents found in corpus (qrels may use different IDs). "
            f"Falling back to random sampling for {cap} docs."
        )
        ids = list(corpus.keys())
        selected = random.sample(ids, cap)
        return {k: corpus[k] for k in selected}

    # If gold alone exceeds cap, take gold only (with a warning)
    if len(gold_keys) >= cap:
        logger.warning(
            f"Gold documents ({len(gold_keys)}) meet or exceed cap ({cap}). "
            f"Keeping only gold documents."
        )
        return {k: corpus[k] for k in gold_keys}

    # Otherwise, keep all gold + fill remainder with random
    gold_set = set(gold_keys)
    remaining = [k for k in corpus.keys() if k not in gold_set]
    extra_n = cap - len(gold_keys)
    extra = random.sample(remaining, min(extra_n, len(remaining)))

    selected = gold_set | set(extra)
    logger.info(
        f"Subsample: kept {len(gold_keys)} gold docs + {len(extra)} random = "
        f"{len(selected)} / {len(corpus)} (target cap {cap})"
    )
    return {k: corpus[k] for k in selected}


# ------------------------------------------------------------
# Gold answer helpers
# ------------------------------------------------------------

def _try_load_hf_dataset(names, config=None, split=None):
    """Try multiple HuggingFace dataset names until one works."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed; cannot download gold answers.")
        return None
    last_error = None
    for name in names:
        try:
            if config and split:
                return load_dataset(name, config, split=split, trust_remote_code=True)
            elif split:
                return load_dataset(name, split=split, trust_remote_code=True)
            elif config:
                return load_dataset(name, config, trust_remote_code=True)
            else:
                return load_dataset(name, trust_remote_code=True)
        except Exception as e:
            last_error = e
            continue
    logger.warning(f"All HF dataset names failed for split='{split}'. Last error: {last_error}")
    return None


def _prepare_nq_gold(queries: Dict) -> Dict[str, List[str]]:
    """Download NQ-open answers from HuggingFace and match to BEIR queries."""
    logger.info("Downloading NQ-open answers from HuggingFace...")
    ds = _try_load_hf_dataset(
        ["google-research-datasets/nq_open", "nq_open"],
        split="validation"
    )
    if ds is None:
        logger.warning("Could not load NQ-open dataset. Gold answers will be missing.")
        return {}

    gold_by_text = {}
    for item in tqdm(ds, desc="NQ gold", leave=False):
        q = item.get("question", "").strip().lower()
        ans = item.get("answer", [])
        if q and ans:
            gold_by_text[q] = ans if isinstance(ans, list) else [ans]

    gold_by_id = {}
    for qid, qinfo in queries.items():
        q_text = qinfo.get("text", "").strip().lower()
        if q_text in gold_by_text:
            gold_by_id[qid] = gold_by_text[q_text]

    logger.info(f"NQ: Matched {len(gold_by_id)}/{len(queries)} queries")
    return gold_by_id


def _prepare_hotpotqa_gold(queries: Dict) -> Dict[str, List[str]]:
    """Download HotpotQA answers from HuggingFace and match to BEIR queries."""
    logger.info("Downloading HotpotQA answers from HuggingFace...")
    ds = _try_load_hf_dataset(
        ["hotpotqa/hotpot_qa", "hotpotqa", "hotpot_qa", "facebook/hotpot_qa"],
        config="distractor",
        split="validation"
    )
    if ds is None:
        logger.warning("Could not load HotpotQA dataset. Gold answers will be missing.")
        return {}

    gold_by_text = {}
    for item in tqdm(ds, desc="HotpotQA gold", leave=False):
        q = item.get("question", "").strip().lower()
        ans = item.get("answer", "")
        if q and ans:
            gold_by_text[q] = [ans] if isinstance(ans, str) else ans

    gold_by_id = {}
    for qid, qinfo in queries.items():
        q_text = qinfo.get("text", "").strip().lower()
        if q_text in gold_by_text:
            gold_by_id[qid] = gold_by_text[q_text]

    logger.info(f"HotpotQA: Matched {len(gold_by_id)}/{len(queries)} queries")
    return gold_by_id


# ------------------------------------------------------------
# Dataset loaders with answers
# ------------------------------------------------------------

def load_beir_dataset(dataset_name: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load a BEIR dataset (NQ, HotpotQA) by reading JSONL files directly.
    Returns (corpus, queries, qrels, answers) where answers is a dict mapping qid -> list of strings.
    """
    data_path = settings.RAW_DATA_ROOT / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(f"BEIR data not found at {data_path}. Please download first.")

    corpus_path = data_path / "corpus.jsonl"
    queries_path = data_path / "queries.jsonl"
    if not corpus_path.exists() or not queries_path.exists():
        raise FileNotFoundError(f"Missing corpus.jsonl or queries.jsonl in {data_path}")

    corpus = _load_jsonl(corpus_path)
    queries = _load_jsonl(queries_path)

    # Load qrels - try both locations
    qrels_path = data_path / "qrels.tsv"
    if qrels_path.exists():
        qrels = _load_qrels(qrels_path)
    else:
        qrels_path = data_path / "qrels" / "test.tsv"
        if qrels_path.exists():
            qrels = _load_qrels(qrels_path)
        else:
            raise FileNotFoundError(f"No qrels file found in {data_path}.")

    # Load answers from the queries file itself (if present)
    answers = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # BEIR uses "_id" as the key
            qid = item.get("_id") or item.get("id")
            if qid is None:
                continue
            qid = str(qid)
            if "answers" in item and isinstance(item["answers"], list):
                answers[qid] = item["answers"]
            elif "answer" in item and isinstance(item["answer"], str):
                answers[qid] = [item["answer"]]
            elif "answer" in item and isinstance(item["answer"], list):
                answers[qid] = item["answer"]

    # If no answers found, try downloading gold from HF (for NQ and HotpotQA)
    if not answers:
        if dataset_name == "nq":
            answers = _prepare_nq_gold(queries)
        elif dataset_name == "hotpotqa":
            answers = _prepare_hotpotqa_gold(queries)

    logger.info(f"Loaded answers for {len(answers)} queries in {dataset_name}.")
    return corpus, queries, qrels, answers


def _load_bright_documents(subset_name: str) -> Dict[str, str]:
    """Load the BRIGHT document corpus for a specific subset."""
    global _BRIGHT_DOCS_CACHE
    if subset_name in _BRIGHT_DOCS_CACHE:
        return _BRIGHT_DOCS_CACHE[subset_name]

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Please install 'datasets' to use BRIGHT: pip install datasets")

    split_name = subset_name
    if subset_name == "theoremqa":
        split_name = "theoremqa_questions"

    try:
        docs_ds = load_dataset("xlangai/BRIGHT", "documents", split=split_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load BRIGHT document split '{split_name}'. "
            f"Error: {e}"
        )

    first = docs_ds[0]
    id_key = None
    if "id" in first:
        id_key = "id"
    elif "doc_id" in first:
        id_key = "doc_id"
    else:
        logger.warning(f"No 'id' or 'doc_id' field found; using index as doc_id.")
        id_key = None

    doc_map = {}
    for idx, doc in enumerate(docs_ds):
        if id_key is not None:
            doc_id = str(doc[id_key])
        else:
            doc_id = str(idx)
        text = doc.get("text", "")
        doc_map[doc_id] = text

    logger.info(f"Loaded {len(doc_map)} BRIGHT documents for subset '{subset_name}'. "
                f"Using key '{id_key or 'index'}'.")
    _BRIGHT_DOCS_CACHE[subset_name] = doc_map
    return doc_map


def load_bright_subset(subset_name: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load a BRIGHT subset by combining documents and examples."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Please install 'datasets' to use BRIGHT: pip install datasets")

    split_name = subset_name
    if subset_name == "theoremqa":
        split_name = "theoremqa_questions"

    try:
        examples_ds = load_dataset("xlangai/BRIGHT", "examples", split=split_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load BRIGHT subset '{subset_name}'. Error: {e}"
        )

    doc_map = _load_bright_documents(subset_name)

    corpus = {}
    queries = {}
    qrels = {}
    answers = {}

    for example in examples_ds:
        qid = str(example.get("id", len(queries)))
        query_text = example.get("query", "")
        queries[qid] = {"id": qid, "text": query_text}

        gold_answer = example.get("gold_answer", "")
        if gold_answer:
            if isinstance(gold_answer, str):
                answers[qid] = [gold_answer]
            else:
                answers[qid] = list(gold_answer) if isinstance(gold_answer, list) else [str(gold_answer)]
        else:
            answers[qid] = []

        gold_ids = example.get("gold_ids", [])
        if not gold_ids:
            gold_ids = example.get("gold_ids_long", [])
        qrels[qid] = {}
        for doc_id in gold_ids:
            doc_id_str = str(doc_id)
            if doc_id_str not in corpus:
                doc_text = doc_map.get(doc_id_str, "")
                corpus[doc_id_str] = {"id": doc_id_str, "text": doc_text}
            qrels[qid][doc_id_str] = 1

    logger.info(f"Loaded BRIGHT subset '{subset_name}': "
                f"{len(queries)} queries, {len(corpus)} documents.")

    return corpus, queries, qrels, answers


# ------------------------------------------------------------
# Save / load processed data
# ------------------------------------------------------------

def save_processed(
    dataset_name: str,
    corpus: Dict,
    queries: Dict,
    qrels: Dict,
    answers: Dict,
    subset: Optional[str] = None,
) -> None:
    base = settings.PROCESSED_DATA_ROOT / dataset_name
    if subset is not None:
        base = base / subset
    _ensure_dir(base)

    _save_jsonl(corpus, base / "corpus.jsonl")
    _save_jsonl(queries, base / "queries.jsonl")
    _save_qrels(qrels, base / "qrels.tsv")
    with open(base / "answers.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)
    logger.info(f"Saved processed {dataset_name}" + (f"/{subset}" if subset else ""))


def load_processed(
    dataset_name: str,
    subset: Optional[str] = None,
) -> Tuple[Dict, Dict, Dict, Dict]:
    base = settings.PROCESSED_DATA_ROOT / dataset_name
    if subset is not None:
        base = base / subset
    corpus_path = base / "corpus.jsonl"
    queries_path = base / "queries.jsonl"
    qrels_path = base / "qrels.tsv"
    answers_path = base / "answers.json"
    if not all(p.exists() for p in (corpus_path, queries_path, qrels_path, answers_path)):
        raise FileNotFoundError(f"Processed data missing at {base}")
    corpus = _load_jsonl(corpus_path)
    queries = _load_jsonl(queries_path)
    qrels = _load_qrels(qrels_path)
    with open(answers_path, "r", encoding="utf-8") as f:
        answers = json.load(f)
    return corpus, queries, qrels, answers


def get_processed(
    dataset_name: str,
    subset: Optional[str] = None,
    force_reload: bool = False,
) -> Tuple[Dict, Dict, Dict, Dict]:
    if not force_reload:
        try:
            return load_processed(dataset_name, subset)
        except FileNotFoundError:
            logger.info(f"Processed data not found for {dataset_name}/{subset}, loading raw...")
        except Exception as e:
            logger.warning(f"Error loading processed data: {e}, reloading raw...")

    if dataset_name.lower() in ("nq", "hotpotqa"):
        corpus, queries, qrels, answers = load_beir_dataset(dataset_name.lower())
    elif dataset_name.lower() == "bright":
        if subset is None:
            raise ValueError("For BRIGHT, you must specify a subset.")
        corpus, queries, qrels, answers = load_bright_subset(subset)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Subsample corpus, preserving gold documents
    cap = settings.PASSAGE_CAP
    if len(corpus) > cap:
        logger.info(f"Subsampling {dataset_name} corpus from {len(corpus)} to {cap} (preserving gold)")
        corpus = subsample_corpus(corpus, cap, qrels=qrels)
    else:
        logger.info(f"Corpus size {len(corpus)} <= cap, no subsampling needed.")

    save_processed(dataset_name, corpus, queries, qrels, answers, subset)
    return corpus, queries, qrels, answers


# ------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------

def get_nq(force_reload: bool = False) -> Tuple[Dict, Dict, Dict, Dict]:
    return get_processed("nq", force_reload=force_reload)


def get_hotpotqa(force_reload: bool = False) -> Tuple[Dict, Dict, Dict, Dict]:
    return get_processed("hotpotqa", force_reload=force_reload)


def get_bright_subset(subset: str, force_reload: bool = False) -> Tuple[Dict, Dict, Dict, Dict]:
    return get_processed("bright", subset=subset, force_reload=force_reload)


def get_all_bright_subsets(force_reload: bool = False) -> Dict[str, Tuple[Dict, Dict, Dict, Dict]]:
    result = {}
    for subset in settings.BRIGHT_SUBSETS:
        result[subset] = get_bright_subset(subset, force_reload=force_reload)
    return result