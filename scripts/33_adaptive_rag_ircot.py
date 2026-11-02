"""
Adaptive RAG with IRCoT as its top rung. CPU only; uses the
stored IRCoT answers (06_ircot.py, graded by 08: Qwen2.5 7B, Llama 3.1 8B, Qwen3 14B on HotpotQA, 2Wiki, MuSiQue).

Concern: in the paper's single step plan set, Adaptive RAG's top rung is a single step plan, and in the two hop set
it is the fixed two hop plan. The published Adaptive RAG (Jeong et al., NAACL 2024) routes among no retrieval, single
step retrieval and multi step retrieval, its multi step strategy being IRCoT. Here it gets that ladder:
  rung 0  closed book
  rung 1  the most cost efficient single step plan on the training folds (as in 11 and 12)
  rung 2  IRCoT (cost: its measured time per question; plan cost = mean over the block's questions, as for C(p))
Silver labels as published: the simplest rung that answers; if none answers, the strongest rung (all three datasets
are multi hop). Two classifiers on the question only: TF IDF logistic regression (ar_ircot_tfidf, the paper's primary
Adaptive RAG classifier) and logistic regression on BGE-M3 embeddings (ar_ircot_emb).

Scored with the paper's protocol (5 folds x 5 seeds; settings and hulls chosen on the training folds) against
  fixed      the hull of the fixed plans (single step plans; and with the fixed two hop plans)
  with_ircot the hull of the fixed plans plus IRCoT as one more plan (a random mix may run IRCoT too)
Tests per reader and dataset (9) and pooled per reader (3); plain and corrected (Nadeau and Bengio) tests; Holm within
each plan set, comparator and level. Also reported: the share of questions sent to IRCoT, cost and success.

Usage:  python scripts/33_adaptive_rag_ircot.py        (about 5 to 10 minutes)
Output: results_full/analysis/adaptive_rag_ircot/  tests.csv, folds.csv, summary.txt
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import settings

OUT = settings.RESULTS_ROOT / "analysis" / "adaptive_rag_ircot"
OUT.mkdir(parents=True, exist_ok=True)
READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
IR = "ircot@0"
LINES = []


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


K = _load("inference27", "27_inference_refit.py")
A, R = K.A, K.R


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def load_ircot(reader):
    f = settings.PROCESSED_DATA_ROOT / "r4_ircot" / f"labels_{reader}_graded.csv"
    d = pd.read_csv(f, dtype={"query_id": str}, keep_default_na=False)
    d["y"] = d[f"label_{A.CONFIG['label']}"].astype(int)
    d["cost"] = pd.to_numeric(d.agent_time_s, errors="coerce") + pd.to_numeric(d.retrieval_time_s, errors="coerce")
    return d


def make_policy(B, Yaug, Caug, kind, E):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    single = [p for p in B.plans if B.depth_of[p] > 0 and not A.is_multi(p)]
    cb = "closed_book@0"

    def ladder(sel):
        acc = Yaug.loc[sel].mean()
        r1 = max(single, key=lambda p: (acc[p] - acc.get(cb, 0)) / max(Caug[p], 1e-6))
        return [cb, r1, IR] if cb in Yaug.columns else [r1, IR]

    def fit(sel):
        rungs = ladder(sel)
        Ys = Yaug.loc[sel, rungs].to_numpy()
        y = np.where(Ys.max(1) == 1, Ys.argmax(1), len(rungs) - 1)       # none answers: strongest rung (multi hop)
        if len(set(y)) < 2:
            return (rungs, None, None, int(y[0]))
        if kind == "tfidf":
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            X = vec.fit_transform([B.text[q] for q in sel])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)
        else:
            vec = None
            clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(E.loc[sel].to_numpy(), y)
        return (rungs, clf, vec, None)

    def apply_(cfg, idx):
        rungs, clf, vec, const = cfg
        if clf is None:
            k = np.full(len(idx), const)
        else:
            k = clf.predict(vec.transform([B.text[q] for q in idx]) if vec is not None else E.loc[idx].to_numpy())
        chosen = [rungs[int(i)] for i in k]
        s = np.array([Yaug.at[q, p] for q, p in zip(idx, chosen)], float)
        return s, float(np.mean([Caug[p] for p in chosen])), float(np.mean([p == IR for p in chosen]))
    return fit, apply_


def run_block(B, Yaug, Caug, kind, E, tag, rd, ds):
    fit, apply_ = make_policy(B, Yaug, Caug, kind, E)
    k = A.CONFIG["k_folds"]
    rows = []
    for sd in A.CONFIG["seeds"]:
        rng = np.random.default_rng(sd)
        q = B.q.copy()
        rng.shuffle(q)
        folds = [q[i::k] for i in range(k)]
        for i in range(k):
            test = folds[i]
            sel = np.concatenate([folds[j] for j in range(k) if j != i])
            cfg = fit(sel)
            s, c, share = apply_(cfg, test)
            hf = A.hull_at(B.Y, B.cost, sel, c, eval_idx=test)
            hi = A.hull_at(Yaug, Caug, sel, c, eval_idx=test)
            rows.append({"plan_set": tag, "reader": rd, "dataset": ds, "method": f"ar_ircot_{kind}", "seed": sd, "fold": i,
                         "success": s.mean(), "cost": c, "ircot_share": share, "hull_fixed": hf, "hull_with_ircot": hi,
                         "gain_fixed": s.mean() - hf, "gain_with_ircot": s.mean() - hi,
                         "n_test": len(test), "n_train": len(sel), "rungs": "|".join(cfg[0])})
    return rows


def main():
    t0 = time.time()
    say("Adaptive RAG with IRCoT as its top rung; label " + A.CONFIG["label"])
    rows = []
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        tag = "with two hop" if r3 else "single step"
        for rd in READERS:
            df, _ = A.load_reader(rd)
            ir = load_ircot(rd)
            for ds in DATASETS:
                B = A.Block(df, ds)
                g = ir[ir.dataset == ds].set_index("query_id")
                if not set(B.q) <= set(g.index):
                    say(f"  {rd} {ds}: IRCoT answers missing for {len(set(B.q) - set(g.index))} questions; skipped")
                    continue
                Yaug = B.Y.copy()
                Yaug[IR] = g.y.reindex(B.Y.index).astype(float)
                Caug = B.cost.copy()
                Caug[IR] = float(g.cost.reindex(B.Y.index).mean())
                E = R.load_emb(ds).reindex([str(q) for q in B.q])
                E.index = B.q
                for kind in ("tfidf", "emb"):
                    rows += run_block(B, Yaug, Caug, kind, E, tag, rd, ds)
                print(f"  {tag} {rd} {ds} done ({time.time() - t0:.0f}s); IRCoT {Yaug[IR].mean():.3f} at {Caug[IR]:.1f}s",
                      flush=True)
    A.CONFIG["include_r3"] = False
    FO = pd.DataFrame(rows)
    FO.to_csv(OUT / "folds.csv", index=False)
    out = []
    for comp in ("fixed", "with_ircot"):
        for level, keys in (("dataset", ["plan_set", "method", "reader", "dataset"]), ("pooled", ["plan_set", "method", "reader"])):
            for key, g in FO.groupby(keys):
                cells = [K.nb_cell(x[f"gain_{comp}"], x.n_test.mean(), x.n_train.mean()) for _, x in g.groupby("dataset")]
                r = K.nb_row(*K.nb_pooled(cells))
                out.append({"comparator": comp, "level": level, **dict(zip(keys, key)), "dataset": key[3] if level == "dataset" else "pooled",
                            "success": g.success.mean(), "cost": g.cost.mean(), "ircot_share": g.ircot_share.mean(),
                            "hull": g[f"hull_{comp}"].mean(), **r})
    T = pd.DataFrame(out)
    T["p_nb_holm"] = np.nan
    for _, idx in T.groupby(["comparator", "level", "plan_set", "method"]).groups.items():
        T.loc[idx, "p_nb_holm"] = A.holm(T.loc[idx, "p_nb"].to_numpy())
    T["positive"] = (T.p_nb_holm < 0.05) & (T.nb_low > 0)
    T["negative"] = (T.p_nb_holm < 0.05) & (T.nb_high < 0)
    T.to_csv(OUT / "tests.csv", index=False)
    for (comp, level, tag, m), g in T.groupby(["comparator", "level", "plan_set", "method"], sort=False):
        say(f"\n== {m} | {tag} | against the {comp} hull | {level}: positive {int(g.positive.sum())}, negative "
            f"{int(g.negative.sum())} of {len(g)}")
        say(g[["reader", "dataset", "success", "cost", "ircot_share", "hull", "gain", "nb_low", "nb_high", "p_nb_holm"]]
            .round(4).to_string(index=False))
    say(f"\nCorrected test (Nadeau and Bengio); Holm within comparator, level, plan set and classifier. Done in "
        f"{time.time() - t0:.0f}s.")
    (OUT / "summary.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
