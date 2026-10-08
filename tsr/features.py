"""
Feature definitions for the v3 TSR predictor.

Column prefixes say WHEN a feature is available. This matters because the
planner must decide before it pays for a plan.

  q_     decision time: query text + one BM25 probe (about 2 ms). Same for
         every plan of a query. Safe for the planner.
  plan_  plan identity (retriever one hot, depth). Safe for the planner.
  p_     post retrieval, own plan only: needs that plan's retrieval to run.
         Safe for a cascade step on the plan already run; NOT safe for
         choosing among unrun plans.
  r_     own plan reranker features: also need a reranker pass (~0.5 s).
  x_     cross plan: need several retrievers to have run. Analysis only.
  diag_  uses gold answers or qrels. NEVER a feature. Diagnostics only.

The old FeatureExtractor fed p_ and r_ style features of EVERY plan to the
planner. To get them the planner ran every retriever, so it paid the cost of
all plans before choosing one. That design could not respect a budget.
"""

import re
from typing import Dict, List, Sequence

import numpy as np

SAFE_PREFIXES = ("q_", "plan_")
CASCADE_PREFIXES = ("q_", "plan_", "p_")
ALL_FEATURE_PREFIXES = ("q_", "plan_", "p_", "r_", "x_")
RELEVANCE_THRESHOLD = 0.5


def feature_columns(columns: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    return [c for c in columns if c.startswith(tuple(prefixes)) and not c.startswith("diag_")]


# ----------------------------------------------------------------------
def query_lexical(q: str) -> Dict[str, float]:
    toks = q.split()
    n = max(len(toks), 1)
    ql = q.lower().strip()
    return {
        "q_len_tokens": float(len(toks)),
        "q_len_chars": float(len(q)),
        "q_wh_who": float(ql.startswith("who")),
        "q_wh_when": float(ql.startswith(("when", "what year"))),
        "q_wh_where": float(ql.startswith("where")),
        "q_wh_what_which": float(ql.startswith(("what", "which"))),
        "q_wh_how": float(ql.startswith("how")),
        "q_yesno": float(ql.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can "))),
        "q_has_digit": float(bool(re.search(r"\d", q))),
        "q_n_capitalized": float(sum(t[:1].isupper() for t in toks[1:])),
        "q_type_token_ratio": float(len({t.lower() for t in toks}) / n),
    }


def score_stats(scores: Sequence[float], prefix: str) -> Dict[str, float]:
    """Score shape statistics used in query performance prediction.

    nqc is the standard deviation of the top k scores divided by the absolute
    mean score (a normalised form of NQC, Shtok et al., ACM TOIS 2012).
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        s = np.zeros(1)
    top10 = s[:10]
    mean_all = float(np.mean(s))
    out = {
        f"{prefix}top1": float(s[0]),
        f"{prefix}gap12": float(s[0] - s[1]) if s.size > 1 else 0.0,
        f"{prefix}gap1_10": float(s[0] - top10[-1]),
        f"{prefix}mean10": float(top10.mean()),
        f"{prefix}std10": float(top10.std()),
        f"{prefix}nqc": float(s.std() / (abs(mean_all) + 1e-9)),
        f"{prefix}top1_over_mean": float(s[0] / (abs(mean_all) + 1e-9)),
    }
    return out


def probe_features(query: str, bm25_results: List[Dict], probe_depth: int = 100) -> Dict[str, float]:
    """Decision time features: query text + BM25 probe at probe_depth."""
    feats = query_lexical(query)
    scores = [r["score"] for r in bm25_results[:probe_depth]]
    feats.update(score_stats(scores, "q_bm25_"))
    # Query term coverage is not available without the corpus text, so we
    # keep the probe to scores only. It costs one BM25 call.
    return feats


def plan_identity(retriever: str, depth: int, retrievers: Sequence[str]) -> Dict[str, float]:
    f = {f"plan_ret_{r}": float(r == retriever) for r in retrievers}
    f["plan_log_depth"] = float(np.log(depth))
    return f


def own_plan_features(results: List[Dict], depth: int) -> Dict[str, float]:
    """p_ features from the plan's own list (raw scores; z scoring per retriever
    is done later over the whole table, see standardize_by_retriever)."""
    return score_stats([r["score"] for r in results[:depth]], "p_")


def reranker_features(scores_in_order: Sequence[float]) -> Dict[str, float]:
    """r_ features from reranker P(yes) of the plan's top passages, in plan order."""
    s = np.asarray(scores_in_order, dtype=np.float64)
    if s.size == 0:
        return {k: 0.0 for k in ("r_max", "r_top1", "r_mean5", "r_n_rel", "r_first_rel_pos", "r_any_rel")}
    rel = np.where(s > RELEVANCE_THRESHOLD)[0]
    return {
        "r_max": float(s.max()),
        "r_top1": float(s[0]),
        "r_mean5": float(s[:5].mean()),
        "r_n_rel": float(rel.size),
        # Position of the first relevant passage. Large values mean the
        # evidence sits deep in a long prompt ("lost in the middle").
        "r_first_rel_pos": float(rel[0] + 1) if rel.size else float(s.size + 1),
        "r_any_rel": float(rel.size > 0),
    }


def retriever_stats(df, cols: Sequence[str], retriever_col: str = "retriever") -> Dict:
    """{retriever: {col: [mean, std]}}. Saved with the model so the planner
    standardises p_ features exactly as training did."""
    out = {}
    for r, sub in df.groupby(retriever_col):
        out[str(r)] = {c: [float(sub[c].mean()), float(sub[c].std() + 1e-9)] for c in cols}
    return out


def apply_retriever_stats(feats: Dict[str, float], retriever: str, stats: Dict) -> Dict[str, float]:
    s = stats.get(retriever, {})
    return {k: ((v - s[k][0]) / s[k][1] if k in s else v) for k, v in feats.items()}


def standardize_by_retriever(df, cols: Sequence[str], retriever_col: str = "retriever"):
    """Raw scores are not comparable across retrievers (BM25 vs cosine vs
    P(yes)). Z score each p_ column within retriever. Uses no labels, so it is
    safe to compute on the full table."""
    df = df.copy()
    for c in cols:
        g = df.groupby(retriever_col)[c]
        df[c] = (df[c] - g.transform("mean")) / (g.transform("std") + 1e-9)
    return df


# ----------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(t: str) -> str:
    t = _PUNCT.sub(" ", str(t).lower())
    return " ".join(t.split())


def gold_in_passages(gold: Sequence[str], passages: Sequence[str]) -> int:
    """diag_: 1 based position of the first passage that contains a gold string, 0 if none."""
    golds = [normalize_answer(g) for g in gold if normalize_answer(g)]
    for i, p in enumerate(passages):
        pn = f" {normalize_answer(p)} "
        if any(f" {g} " in pn for g in golds):
            return i + 1
    return 0
