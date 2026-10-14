"""
Answer grading for NQ and HotpotQA (and BRIGHT).

Notes:
  1. SQuAD normalisation: lowercase, strip punctuation AND the articles
     a / an / the (Rajpurkar et al., EMNLP 2016). Without article removal,
     "Mariinsky Ballet" did not exactly match "The Mariinsky Ballet".
  2. grade_detail() returns the evidence behind a decision (exact match,
     gold contained in answer, best token F1, answer length) instead of a
     bare yes/no. scripts/04d_regrade_labels.py stores this evidence so any
     rule can be applied later without rerunning the agent.
  3. Several named rules. Which one the paper uses is decided by the hand
     audit in 04d, not by intuition:
       strict    exact match OR gold appears as a word sequence in the answer
       f1_07     strict OR token F1 >= 0.7
       lenient   strict OR token F1 >= 0.4   (the rule used to make the v3 labels)
       proposed  alternative rule: strict, else skip F1 when a multi token
                 gold has a much shorter answer, else F1 >= 0.7
"""

import re
from collections import Counter
from typing import Dict, List, Optional

RULES = ("strict", "f1_07", "lenient", "proposed")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: Optional[str]) -> str:
    if text is None:
        return ""
    t = _PUNCT.sub(" ", str(text).lower())
    t = _ARTICLES.sub(" ", t)
    return " ".join(t.split())


def contains_words(haystack: str, needle: str) -> bool:
    """needle appears in haystack as a whole word sequence (both normalised)."""
    if not needle:
        return False
    return re.search(r"\b" + re.escape(needle) + r"\b", haystack) is not None


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def grade_detail(answer: str, golds: List[str]) -> Dict:
    """Evidence for one answer against all gold aliases."""
    a = normalize(answer)
    a_tokens = a.split()
    out = {"em": 0, "contains": 0, "f1": 0.0, "f1_gold": "", "n_answer_tokens": len(a_tokens),
           "n_gold_tokens": 0, "proposed": 0}
    for g in golds or []:
        gn = normalize(g)
        if not gn:
            continue
        g_tokens = gn.split()
        if a and a == gn:
            out["em"] = 1
        if a and contains_words(a, gn):
            out["contains"] = 1
        f = token_f1(answer, g)
        if f > out["f1"]:
            out.update(f1=f, f1_gold=g, n_gold_tokens=len(g_tokens))
        # 'proposed' rule, evaluated per gold alias
        if a and (a == gn or contains_words(a, gn)):
            out["proposed"] = 1
        elif a and not (len(g_tokens) >= 2 and len(a_tokens) < 0.75 * len(g_tokens)) and f >= 0.7:
            out["proposed"] = 1
    return out


def apply_rule(d: Dict, rule: str) -> int:
    strict = int(d["em"] or d["contains"])
    if rule == "strict":
        return strict
    if rule == "f1_07":
        return int(strict or d["f1"] >= 0.7)
    if rule == "lenient":
        return int(strict or d["f1"] >= 0.4)
    if rule == "proposed":
        return int(d["proposed"])
    raise ValueError(f"unknown rule {rule}; choose from {RULES}")


class Grader:
    """Backward compatible wrapper used by 04 and 04c."""

    def __init__(self, dataset_name: str, rule: str = "lenient"):
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in ("nq", "hotpotqa", "musique", "2wikimultihopqa", "bright"):
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule}; choose from {RULES}")
        self.rule = rule

    def detail(self, answer: str, ground_truth: Optional[List[str]]) -> Dict:
        return grade_detail(answer, ground_truth or [])

    def is_correct(self, query: str, answer: str, ground_truth: Optional[List[str]] = None) -> bool:
        if not answer or not ground_truth:
            return False
        return bool(apply_rule(grade_detail(answer, ground_truth), self.rule))
