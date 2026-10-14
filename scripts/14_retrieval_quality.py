"""
Retrieval quality per retriever and depth, and how well each reader uses a context that holds the answer.

Metrics (500 evaluation queries per dataset, the cached lists the readers saw):
  hit@k      at least one gold (qrels) passage in the top k
  recall@k   share of the gold passages in the top k (multi hop: the supporting passages)
  all@k      every gold passage in the top k (a multi hop question needs all of them)
  ndcg@10    binary relevance, for comparison with published BEIR numbers (NQ, HotpotQA)
  ans@k      a gold answer string occurs in the top k passages (title + text, normalised as in grading)

Reader utilisation (joins the final labels): for every reader and retrieval plan, success when the answer
is in the context vs when it is not. Separates "the retriever missed it" from "the reader missed it".

Stages:
  "texts"  stream each corpus once and keep only the passages that appear in some top 20 list
           (data/processed_full/retrieval_quality/ctx_<dataset>.json). CPU, about 1 to 3 minutes per dataset.
  "report" metrics and the LaTeX table: results_full/tables/tab_retrieval.tex (+ tables_retrieval.txt)

Usage: python scripts/14_retrieval_quality.py texts [dataset]   then   python scripts/14_retrieval_quality.py report
"""

import collections
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings
from tsr.grader import normalize

DATASETS = ["nq", "hotpotqa", "2wikimultihopqa", "musique"]
D_TEX = {"nq": "NQ", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki", "musique": "MuSiQue"}
RETRIEVERS = ["bm25", "splade_v3", "bge_m3", "stella_1_5b", "bge_bm25_rrf", "qwen3_reranker_0_6b"]
R_TEX = {"bm25": "BM25", "splade_v3": "SPLADE-v3", "bge_m3": "BGE-M3", "stella_1_5b": "Stella 1.5B",
         "bge_bm25_rrf": "BGE+BM25 RRF", "qwen3_reranker_0_6b": "Qwen3 reranker"}
KS = [1, 3, 5, 10, 20]
READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
PROC = settings.PROCESSED_DATA_ROOT
CACHE = settings.CACHE_ROOT / "retrievals_v3"
OUTD = PROC / "retrieval_quality"
TAB = settings.RESULTS_ROOT / "tables"
ID_RE = re.compile(r'^\{"_id":\s*"((?:[^"\\]|\\.)*)"')


def caches(ds):
    out = {}
    for r in RETRIEVERS:
        f = CACHE / f"{ds}__{r}.pkl"
        if f.exists():
            out[r] = pickle.load(open(f, "rb"))
    return out


def query_ids(ds):
    """cache query key (line index as string) -> the dataset's own _id (the key qrels use)."""
    with open(PROC / ds / "queries.jsonl", encoding="utf-8") as f:
        return {str(i): json.loads(l)["_id"] for i, l in enumerate(f)}


def qrels(ds):
    g = collections.defaultdict(set)
    with open(PROC / ds / "qrels.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                try:
                    if int(p[2]) > 0:
                        g[p[0]].add(p[1])
                except ValueError:
                    pass
    return g


def stage_texts(only=None):
    OUTD.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        if only and ds != only:
            continue
        out = OUTD / f"ctx_{ds}.json"
        if out.exists():
            print(f"{ds}: done already ({out.name})")
            continue
        need = set()
        for c in caches(ds).values():
            for res in c["results"].values():
                need.update(x["id"] for x in res[:max(KS)])
        got = {}
        with open(PROC / ds / "corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                m = ID_RE.match(line)
                if m and m.group(1) in need:
                    d = json.loads(line)
                    got[d["_id"]] = normalize((d.get("title") or "") + " " + (d.get("text") or ""))
                    if len(got) == len(need):
                        break
        tmp = out.with_suffix(".tmp")
        json.dump(got, open(tmp, "w", encoding="utf-8"))
        tmp.replace(out)
        print(f"{ds}: {len(got)} of {len(need)} passages kept", flush=True)


def ndcg10(top, gold):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(top[:10]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / idcg if idcg else 0.0


def stage_report():
    rows, util = [], []
    for ds in DATASETS:
        ctx = json.load(open(OUTD / f"ctx_{ds}.json", encoding="utf-8"))
        qid, gold = query_ids(ds), qrels(ds)
        answers = json.load(open(PROC / ds / "answers.json", encoding="utf-8"))
        ans_norm = {q: [" " + normalize(a) + " " for a in v if normalize(a)] for q, v in answers.items()}
        cs = caches(ds)
        present = {}                     # (retriever, k, query) -> answer in context
        for r, c in cs.items():
            for k in KS:
                acc = collections.defaultdict(list)
                for q, res in c["results"].items():
                    top = [x["id"] for x in res[:k]]
                    g = gold.get(qid[q], set())
                    if g:
                        inter = len(g & set(top))
                        acc["hit"].append(inter > 0)
                        acc["recall"].append(inter / len(g))
                        acc["all"].append(inter == len(g))
                        if k == 10:
                            acc["ndcg10"].append(ndcg10(top, g))
                    text = " " + " ".join(ctx.get(d, "") for d in top) + " "
                    a = any(x in text for x in ans_norm.get(q, []))
                    acc["ans"].append(a)
                    present[(r, k, q)] = a
                rows.append({"dataset": ds, "retriever": r, "k": k, "n": len(acc["ans"]),
                             **{m: float(np.mean(v)) for m, v in acc.items()}})
        for rd in READERS:
            f = PROC / f"tsr_labels_{rd}_graded.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, dtype={"query_id": str}, keep_default_na=False,
                             usecols=["dataset", "query_id", "retriever", "depth", "repeat", "label_final"])
            df = df[(df.dataset == ds) & (df["repeat"].astype(int) == 0) & (df.retriever.isin(cs))]
            df["depth"] = df["depth"].astype(int)
            df = df[df.depth.isin(KS)]
            df["ans"] = [present.get((r, k, q), np.nan) for r, k, q in zip(df.retriever, df.depth, df.query_id)]
            df = df.dropna(subset=["ans"])
            df["y"] = df["label_final"].astype(int)
            for a, g in df.groupby("ans"):
                util.append({"dataset": ds, "reader": rd, "answer_in_context": bool(a), "n": len(g),
                             "success": g.y.mean()})
    R, U = pd.DataFrame(rows), pd.DataFrame(util)
    OUTD.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUTD / "retrieval_quality.csv", index=False)
    U.to_csv(OUTD / "reader_utilisation.csv", index=False)

    # ---- LaTeX: recall@k and ans@k at k = 1, 5, 20; nDCG@10
    TAB.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Retrieval quality on the 500 evaluation queries. R@k: share of gold passages in the top $k$ "
             "(supporting passages for multi hop data). A@k: a gold answer string occurs in the top $k$. nDCG@10 with "
             "binary relevance.}", "\\label{tab:retrieval}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{l" + "rrrr" * len(DATASETS) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{D_TEX[d]}}}" for d in DATASETS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(DATASETS))),
             "Retriever & " + " & ".join(["R@5", "R@20", "A@5", "nDCG@10"] * len(DATASETS)) + " \\\\", "\\midrule"]
    txt = []
    for r in RETRIEVERS:
        cells = []
        for d in DATASETS:
            g = R[(R.dataset == d) & (R.retriever == r)].set_index("k")
            if g.empty:
                cells += ["--"] * 4
                continue
            cells += [f"{g.at[5, 'recall']:.3f}", f"{g.at[20, 'recall']:.3f}", f"{g.at[5, 'ans']:.3f}",
                      f"{g.at[10, 'ndcg10']:.3f}"]
        lines.append(f"{R_TEX[r]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (TAB / "tab_retrieval.tex").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 200)
    for d in DATASETS:
        t = R[R.dataset == d].pivot_table(index="retriever", columns="k", values=["recall", "all", "ans"]).round(3)
        nd = R[(R.dataset == d) & (R.k == 10)].set_index("retriever")["ndcg10"].round(3)
        txt.append(f"== {d}\n{t.to_string()}\nnDCG@10:\n{nd.to_string()}\n")
    ut = U.pivot_table(index=["dataset", "reader"], columns="answer_in_context", values="success").round(3)
    txt.append("== reader success given the answer IS / IS NOT in the retrieved context (retrieval plans, depth 1 to 20)\n"
               + ut.to_string())
    (TAB / "tables_retrieval.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "texts":
        stage_texts(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        stage_report()
