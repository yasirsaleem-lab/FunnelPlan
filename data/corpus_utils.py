"""
Shared helpers for corpus and query handling.

Every script builds its corpus list with these helpers, so the "title" field
is kept, indexed and shown to the reader. For HotpotQA the title is often the
bridge entity.
"""

from typing import Any, Dict, List, Optional


def doc_id(v: Dict[str, Any], fallback: Any) -> str:
    return str(v.get("_id") or v.get("id") or fallback)


def build_corpus_list(corpus: Dict[Any, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Corpus as a list of {"id", "title", "text"} for retriever.build_index()."""
    return [
        {"id": doc_id(v, k), "title": v.get("title", "") or "", "text": v.get("text", "") or ""}
        for k, v in corpus.items()
    ]


def passage_text(title: str, text: str) -> str:
    """One canonical passage string. Title first, as in BEIR and most QA readers."""
    title = (title or "").strip()
    text = (text or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def build_text_by_id(corpus: Dict[Any, Dict[str, Any]]) -> Dict[str, str]:
    """doc id -> canonical passage text (title + text)."""
    return {
        doc_id(v, k): passage_text(v.get("title", ""), v.get("text", ""))
        for k, v in corpus.items()
    }


def get_query_text(queries: Dict[Any, Any], qid: Any) -> Optional[str]:
    """Resolve query text for qid stored as int or str. Never falls back to str(qid)."""
    candidates = [qid, str(qid)]
    if str(qid).isdigit():
        candidates.append(int(qid))
    for k in candidates:
        if k in queries:
            v = queries[k]
            return v.get("text") if isinstance(v, dict) else str(v)
    return None


def resolve_gold(queries: Dict[Any, Any], answers: Dict[str, List[str]], qid: Any) -> List[str]:
    gold = answers.get(str(qid), [])
    if gold:
        return gold
    q = queries.get(str(qid)) or queries.get(qid) or {}
    alt = str(q.get("_id") or q.get("id") or qid) if isinstance(q, dict) else str(qid)
    return answers.get(alt, [])
