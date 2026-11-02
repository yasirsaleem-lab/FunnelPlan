"""
Routers trained on the difference between two plans. CPU only.

Concern: the planners and per plan routers predict each plan's success and then compare the predictions (S or T
learners in treatment effect terms). Their predictions across plans correlate at 0.99 and their pair AUROC is at
chance (Table X). Learners that target the difference directly can do better when the difference is simpler than
the outcomes (Kuenzel et al., PNAS 2019). The plan matrix observes every plan on every training question, so the
difference s(q, b) - s(q, a) is observed, yet no tested policy is trained on it. RouteLLM (Ong et al., ICLR 2025)
trains routers on such pairwise targets.

Two routers, same features as the per plan router (BGE-M3 question embedding, PCA 64 fitted on the training data):
  diff_clf  logistic regression on the training questions where plans a and b DISAGREE; target: b answers.
  diff_reg  ridge regression on all training questions; target: s(q, b) - s(q, a) in {-1, 0, 1}.
Policy: run b when the score is above a threshold, else a. The pair (a, b) is chosen among the vertices of the
training fold hull (a cheaper than b), and the threshold among 22 quantiles of the score; both are chosen on the
training folds with cross fitted (5 inner folds) scores to maximise the gain over the training fold hull.
Scored with the paper's protocol (5 folds x 5 seeds, held out hull at the policy's mean cost), without and with the
router's own cost (24.6 ms unless the chosen plan embeds the query with BGE-M3 anyway, as in 24 and 27).

Reported: pooled gains with the plain and the corrected (Nadeau and Bengio) test, Holm over the 10 tests of each
plan set and cost; and the pair AUROC on held out questions where the two chosen plans disagree, next to the pair
AUROC of the per plan router (plan_emb: p_b - p_a) on the SAME questions and pair.

Usage:  python scripts/28_difference_router.py            both plan sets, all readers (about 20 to 40 minutes)
        python scripts/28_difference_router.py --quick    one reader and dataset, to test the code
Output: results_full/analysis/difference_router/  tests.csv, per_query.csv, folds.csv, pair_auroc.csv,
        pair_scores.csv, choices.csv, summary.txt
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

ARGS = sys.argv[1:]


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


QUICK = "--quick" in ARGS
OWN_MS = float(arg("--overhead_ms", "24.6"))
OUT = settings.RESULTS_ROOT / "analysis" / ("difference_router_quick" if QUICK else "difference_router")
OUT.mkdir(parents=True, exist_ok=True)


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


K = _load("inference27", "27_inference_refit.py")      # protocol with fold gains, corrected test, Holm
A, R = K.A, K.R
BGE_RETRIEVERS = {"bge_m3", "bge_bm25_rrf", "qwen3_reranker_0_6b"}
LINES, PAIRS, CHOICES = [], [], []
_MEMO = {}


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def eff_cost(B, delta):
    return {p: float(B.cost[p]) + (0.0 if p.split("@")[0] in BGE_RETRIEVERS else delta) for p in B.plans}


def _pca(B, E, tr, te):
    """PCA 64 fitted on tr; returns (Z_tr, Z_te). Memoised: both costs and both routers share it."""
    from sklearn.decomposition import PCA
    key = ("pca", id(B), hash(tuple(tr)), hash(tuple(te)))
    if key not in _MEMO:
        X = E.loc[tr].to_numpy()
        p = PCA(n_components=min(R.CONFIG["pca_dim"], len(tr) - 1), random_state=0).fit(X)
        _MEMO[key] = (p.transform(X), p.transform(E.loc[te].to_numpy()))
    return _MEMO[key]


def scores(B, E, kind, tr, te, a, b):
    """Score of 'run b instead of a' for questions te, from a model trained on tr. None if it cannot be trained."""
    from sklearn.linear_model import LogisticRegression, Ridge
    key = (kind, id(B), hash(tuple(tr)), hash(tuple(te)), a, b)
    if key in _MEMO:
        return _MEMO[key]
    Ztr, Zte = _pca(B, E, tr, te)
    ya, yb = B.Y.loc[tr, a].to_numpy(float), B.Y.loc[tr, b].to_numpy(float)
    out = None
    if kind == "diff_clf":
        m = ya != yb
        if m.sum() >= 10 and 0 < yb[m].mean() < 1:
            out = LogisticRegression(C=1.0, max_iter=2000).fit(Ztr[m], yb[m]).predict_proba(Zte)[:, 1]
    else:
        d = yb - ya
        if d.min() < d.max():
            out = Ridge(alpha=1.0).fit(Ztr, d).predict(Zte)
    _MEMO[key] = out
    return out


def plan_emb_diff(B, E, tr, te, a, b):
    """The per plan router's score for the same pair: p_b - p_a (PCA 64 + one logistic regression per plan)."""
    from sklearn.linear_model import LogisticRegression
    Ztr, Zte = _pca(B, E, tr, te)
    p = []
    for x in (a, b):
        y = B.Y.loc[tr, x].to_numpy(float)
        p.append(np.full(len(te), y[0]) if y.min() == y.max() else
                 LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, y).predict_proba(Zte)[:, 1])
    return p[1] - p[0]


def m_diff(B, E, kind, delta, tag, rd):
    ce = eff_cost(B, delta)

    def run(a, b, thr, sc, idx):
        go = sc > thr
        r = B.rows(idx)
        s = np.where(go, B.Yn[r, B.pcol[b]], B.Yn[r, B.pcol[a]])
        return s, float(np.mean(np.where(go, ce[b], ce[a])))

    def fit(sel):
        sel = np.asarray(sel)
        V = [h[2] for h in A._hull_vertices(B.Y, B.cost, sel)]
        inner = R.inner_splits(len(sel), A._CUR_SEED[0])
        best = None
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                a, b = V[i], V[j]
                oof = np.empty(len(sel))
                ok = True
                for te in inner:
                    tr = np.setdiff1d(np.arange(len(sel)), te)
                    sc = scores(B, E, kind, sel[tr], sel[te], a, b)
                    if sc is None:
                        ok = False
                        break
                    oof[te] = sc
                if not ok:
                    continue
                for thr in np.concatenate([[-np.inf], np.unique(np.quantile(oof, np.linspace(0, 1, 21)))]):
                    s, c = run(a, b, thr, oof, sel)
                    g = s.mean() - A.hull_at(B.Y, B.cost, sel, c)
                    if best is None or g > best[0]:
                        best = (g, a, b, float(thr), float(np.mean(oof > thr)))
        if best is None:                                   # no pair can be trained: run the cheapest vertex
            return (sel, V[0], V[0], np.inf, 0.0)
        return (sel, best[1], best[2], best[3], best[4])

    def apply_(cfg, idx):
        sel, a, b, thr, rate = cfg
        idx = np.asarray(idx)
        sc = scores(B, E, kind, sel, idx, a, b) if a != b else None
        if sc is None:
            sc = np.full(len(idx), -np.inf)
        s, c = run(a, b, thr, sc, idx)
        if len(idx) < len(sel) and a != b:                 # held out fold: pair AUROC data and the choice made
            ya, yb = B.Y.loc[idx, a].to_numpy(float), B.Y.loc[idx, b].to_numpy(float)
            disc = ya != yb
            pe = plan_emb_diff(B, E, sel, idx, a, b)
            for q, x, y_, z in zip(idx[disc], sc[disc], yb[disc], pe[disc]):
                PAIRS.append((tag, kind, delta, rd, B.ds, A._CUR_SEED[0], str(q), a, b, float(x), float(z), int(y_)))
            CHOICES.append((tag, kind, delta, rd, B.ds, A._CUR_SEED[0], a, b, rate, float(np.mean(sc > thr))))
        return s, c
    return fit, apply_


def main():
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    say(f"Plan difference routers; own cost {OWN_MS} ms; seeds {A.CONFIG['seeds']}" + ("; QUICK TEST RUN" if QUICK else ""))
    folds, perq = [], []
    variants = [("diff_clf", 0.0), ("diff_reg", 0.0), ("diff_clf", OWN_MS / 1000), ("diff_reg", OWN_MS / 1000)]
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        tag = "with two hop" if r3 else "single step"
        for rd in (A.readers()[:1] if QUICK else A.readers()):
            df, _ = A.load_reader(rd)
            for ds in (sorted(df.dataset.unique())[:1] if QUICK else sorted(df.dataset.unique())):
                B = A.Block(df, ds)
                E = R.load_emb(ds).reindex([str(q) for q in B.q])
                E.index = B.q
                _MEMO.clear()
                for kind, dl in variants:
                    name = f"{kind}{'_own' if dl else ''}"
                    D, fr = K.repeated_folds(B, lambda: m_diff(B, E, kind, dl, tag, rd))
                    folds += [{"plan_set": tag, "reader": rd, "dataset": ds, "method": name, **f} for f in fr]
                    perq.append(pd.DataFrame({"plan_set": tag, "reader": rd, "dataset": ds, "method": name,
                                              "query_id": D.index.astype(str), "gain": D.gain.to_numpy()}))
                print(f"  {tag} {rd} {ds} done ({time.time() - t0:.0f}s)", flush=True)
    A.CONFIG["include_r3"] = False
    FO, PQ = pd.DataFrame(folds), pd.concat(perq, ignore_index=True)
    FO.to_csv(OUT / "folds.csv", index=False)
    PQ.to_csv(OUT / "per_query.csv", index=False)

    rows = []
    for (tag, rd, m), g in PQ.groupby(["plan_set", "reader", "method"]):
        f = FO[(FO.plan_set == tag) & (FO.reader == rd) & (FO.method == m)]
        cells = [K.nb_cell(x.gain, x.n_test.mean(), x.n_train.mean()) for _, x in f.groupby("dataset")]
        rows.append({"plan_set": tag, "own_cost": m.endswith("_own"), "reader": rd, "method": m, "n": len(g),
                     **K.nb_row(*K.nb_pooled(cells)), **K.plain_row(g.gain)})
    T = K.add_holm(pd.DataFrame(rows), ["plan_set", "own_cost"])
    T.to_csv(OUT / "tests.csv", index=False)

    PP = pd.DataFrame(PAIRS, columns=["plan_set", "kind", "delta", "reader", "dataset", "seed", "query_id", "a", "b",
                                      "score", "plan_emb_diff", "b_wins"])
    PP.to_csv(OUT / "pair_scores.csv", index=False)
    pd.DataFrame(CHOICES, columns=["plan_set", "kind", "delta", "reader", "dataset", "seed", "a", "b",
                                   "train_rate_b", "test_rate_b"]).to_csv(OUT / "choices.csv", index=False)
    au = []
    for (tag, kind, dl, rd), g in PP.groupby(["plan_set", "kind", "delta", "reader"]):
        per_seed = [(roc_auc_score(x.b_wins, x.score), roc_auc_score(x.b_wins, x.plan_emb_diff), len(x))
                    for _, x in g.groupby("seed") if 0 < x.b_wins.mean() < 1]
        if per_seed:
            au.append({"plan_set": tag, "method": kind + ("_own" if dl else ""), "reader": rd,
                       "pair_auroc": np.mean([x[0] for x in per_seed]), "pair_auroc_plan_emb": np.mean([x[1] for x in per_seed]),
                       "n_discordant_per_seed": np.mean([x[2] for x in per_seed])})
    AU = pd.DataFrame(au)
    AU.to_csv(OUT / "pair_auroc.csv", index=False)

    for (tag, own), g in T.groupby(["plan_set", "own_cost"], sort=False):
        say(f"\n== {tag} | {'with' if own else 'without'} own cost: {len(g)} tests")
        for lab, p, lo, hi, ub in (("plain", "p_plain_holm", "plain_low", "plain_high", "ub95_plain"),
                                   ("corrected", "p_nb_holm", "nb_low", "nb_high", "ub95_nb")):
            pos = g[(g[p] < 0.05) & (g[lo] > 0)]
            say(f"  {lab:9s} positive {len(pos)}" + (f" ({', '.join(f'{r.reader} {r.method} {r.gain:+.3f}' for r in pos.itertuples())})"
                                                     if len(pos) else "")
                + f"  negative {int(((g[p] < 0.05) & (g[hi] < 0)).sum())}  upper bound below 0.03: {int((g[ub] < 0.03).sum())}")
        say(g.sort_values("gain", ascending=False)[["reader", "method", "gain", "nb_low", "nb_high", "p_nb_holm",
                                                     "plain_low", "plain_high"]].round(4).to_string(index=False))
    say("\n== pair AUROC on held out questions where the two chosen plans disagree (mean over seeds of the AUROC pooled "
        "over datasets and folds); the per plan router's score p_b - p_a on the same questions and pair")
    say(AU.round(3).to_string(index=False))
    say(AU.groupby(["plan_set", "method"])[["pair_auroc", "pair_auroc_plan_emb"]].mean().round(3).to_string())
    say(f"\nDone in {time.time() - t0:.0f}s. Output: {OUT}")
    (OUT / "summary.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
