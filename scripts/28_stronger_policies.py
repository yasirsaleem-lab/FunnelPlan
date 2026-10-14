"""
Stronger learned policies (amendment 2026-10-13 20:45, item (c)). Concern: the routers may be weak, and the ACL 2025
uncertainty baselines were missing. Two policies, same protocol as 10 (imported from it: same queries, plans, mean plan
costs, 5 folds x 5 seeds, settings chosen on the training folds, gain against the hull fitted on the training folds):

  c1  clf_gate      gradient boosted trees on the closed book answer's mean_logprob, min_logprob, first_token_prob,
                    gen_tokens and n_answer_tokens predict whether the closed book answer is correct. Retrieve with plan
                    p when the predicted probability is below tau. p and tau are chosen on the training folds as in
                    gate_closed_book; tau is chosen on CROSS FITTED predictions (inner 5 folds inside the training
                    folds), so the threshold is not set on in sample probabilities. Cost: closed book always runs,
                    plus p when retrieving.
  c2  planner_sk    planner_tsr (10) with the same five closed book features added to its decision features. The closed
                    book call is charged to every query (unless the chosen plan is closed book itself, which then costs
                    nothing extra), plus the BM25 probe as in planner_tsr.

Family: 2 policies x 5 readers = 10 pooled tests (n = 2,000 per test), one sample t test, Holm over the 10, in each plan
set. A result counts only if it survives Holm with its CI above 0 in BOTH plan sets (rule of 12 Oct 20:30).
Learning curve for c2 (descriptive, single step plan set): training on 20, 40, 60, 80, 100% of the training folds.

Usage:  python scripts/28_stronger_policies.py              (CPU; about 1 to 2 hours, mostly c2 and its curve)
        python scripts/28_stronger_policies.py --no-curve   (skip the learning curve)
Output: results_full/analysis/stronger_policies/  family.csv, per_dataset.csv, per_query_gains[_r3].csv,
        learning_curve.csv, summary.txt
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import settings

_spec = importlib.util.spec_from_file_location("analysis10", Path(__file__).parent / "10_analysis_full.py")
A = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]          # 10 must not see our flags
_spec.loader.exec_module(A)
sys.argv = _argv

CB = "closed_book@0"
CB_FEATS = ["mean_logprob", "min_logprob", "first_token_prob", "gen_tokens", "n_answer_tokens"]
TAUS_P = list(np.round(np.linspace(0.0, 1.0, 21), 2))     # thresholds on a predicted probability
FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
OUT = settings.RESULTS_ROOT / "analysis" / "stronger_policies"
XGB = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.05, "min_child_weight": 5, "subsample": 0.8,
       "colsample_bytree": 0.8, "lambda": 5.0, "verbosity": 0}


def cb_features(df, ds, qids):
    """The five closed book features per query (NaN filled with the dataset median)."""
    d = df[(df["dataset"] == ds) & (df["retriever"] == "closed_book")].copy()
    d = d.drop_duplicates("query_id").set_index("query_id")
    X = d[CB_FEATS].apply(pd.to_numeric, errors="coerce").reindex(qids)
    return X.fillna(X.median()).fillna(0.0)


# ---------------------------------------------------------------- c1: classifier gate
def m_clf_gate(B, X):
    import xgboost as xgb
    if CB not in B.plans:
        return None
    rplans = [p for p in B.plans if B.depth_of[p] > 0]
    ycb = B.Y[CB]

    def train(idx):
        d = xgb.DMatrix(X.loc[idx].to_numpy(np.float32), label=ycb.loc[idx].to_numpy())
        return xgb.train({**XGB, "seed": A._CUR_SEED[0]}, d, 150)

    def predict(bst, idx):
        return bst.predict(xgb.DMatrix(X.loc[idx].to_numpy(np.float32)))

    def run_p(phat, tau, p, idx):
        r = B.rows(idx)
        go = phat < tau                                        # closed book likely wrong -> retrieve
        succ = np.where(go, B.Yn[r, B.pcol[p]], B.Yn[r, B.pcol[CB]])
        return succ, float(B.cost[CB] + go.mean() * B.cost[p])

    def fit(sel):
        sel = np.asarray(sel)
        rng = np.random.default_rng(A._CUR_SEED[0] + 101)
        perm = rng.permutation(len(sel))
        inner = [perm[i::5] for i in range(5)]
        oof = pd.Series(np.nan, index=sel)
        for k in range(5):                                     # cross fitted probabilities on the training folds
            te = sel[inner[k]]
            tr = sel[np.concatenate([inner[j] for j in range(5) if j != k])]
            oof.loc[te] = predict(train(tr), te)
        ph = oof.loc[sel].to_numpy()
        best = None
        for p in rplans:
            for tau in TAUS_P:
                s, c = run_p(ph, tau, p, sel)
                g = s.mean() - A.hull_at(B.Y, B.cost, sel, c)
                if best is None or g > best[0]:
                    best = (g, tau, p)
        return (train(sel), best[1], best[2])

    def apply_(cfg, idx):
        bst, tau, p = cfg
        return run_p(predict(bst, idx), tau, p, idx)
    return fit, apply_


# ---------------------------------------------------------------- c2: planner_tsr with self knowledge
def m_planner_sk(B, F, frac=1.0):
    import xgboost as xgb
    probe_cost = float(B.ret_cost.get("bm25@1", 0.0)) if "bm25" in B.cache else 0.0
    cb_cost = float(B.cost[CB]) if CB in B.plans else 0.0
    plans = B.plans
    P = np.eye(len(plans))
    cost_vec = B.cost[plans].to_numpy()

    def design(idx):
        Xq = F.loc[idx].to_numpy(np.float32)
        return np.concatenate([np.repeat(Xq, len(plans), 0), np.tile(P, (len(idx), 1))], 1)

    def apply_(cfg, idx):
        bst, prior, lam = cfg
        base = np.tile(np.log(prior / (1 - prior)).to_numpy(), len(idx))
        p = bst.predict(xgb.DMatrix(design(idx), base_margin=base)).reshape(len(idx), len(plans))
        pick = (p - lam * cost_vec[None, :]).argmax(1)
        chosen = [plans[i] for i in pick]
        succ = np.array([B.Y.at[q, pl] for q, pl in zip(idx, chosen)])
        cost = np.mean([B.cost[pl] + (0.0 if B.retr_of[pl] == "bm25" else probe_cost)
                        + (0.0 if pl == CB else cb_cost) for pl in chosen])
        return succ, cost

    def fit(sel):
        sel = np.asarray(sel)
        if frac < 1.0:                                         # learning curve: a seeded subset of the training folds
            rng = np.random.default_rng(A._CUR_SEED[0] + 7)
            sel = sel[np.sort(rng.permutation(len(sel))[: max(20, int(round(frac * len(sel))))])]
        prior = B.Y.loc[sel].mean().clip(0.02, 0.98)
        y = B.Y.loc[sel].to_numpy().ravel()
        base = np.tile(np.log(prior / (1 - prior)).to_numpy(), len(sel))
        bst = xgb.train({**XGB, "seed": A._CUR_SEED[0]}, xgb.DMatrix(design(sel), label=y, base_margin=base), 200)
        best = None
        for lam in A.CONFIG["lambdas"]:
            s, c = apply_((bst, prior, lam), sel)
            g = s.mean() - A.hull_at(B.Y, B.cost, sel, c)
            if best is None or g > best[0]:
                best = (g, lam)
        return (bst, prior, best[1])
    return fit, apply_


# ---------------------------------------------------------------- driver
def run_plan_set(r3, L, curve):
    A.CONFIG["include_r3"] = r3
    tag = "two hop" if r3 else "single step"
    rows, pq, lc = [], [], []
    t0 = time.time()
    for rd in A.readers():
        df, col = A.load_reader(rd)
        for ds in sorted(df["dataset"].unique()):
            B = A.Block(df, ds)
            F = A.decision_features(B)
            X = cb_features(df, ds, B.q)
            F2 = F.join(X.add_prefix("cb_"))
            makers = {"clf_gate": lambda: m_clf_gate(B, X), "planner_sk": lambda: m_planner_sk(B, F2)}
            for name, mk in makers.items():
                print(f"  [{tag}] {rd} {ds} {name} ({time.time() - t0:.0f}s)", flush=True)
                res = A.repeated(B, mk)
                if res is None:
                    continue
                d, seed_means, halves, cfgs, qids = res
                pq.append(pd.DataFrame({"reader": rd, "dataset": ds, "method": name, "query_id": qids.astype(str),
                                        "gain": d}))
                lo, hi = A.boot_ci(d, A.CONFIG["n_boot"], A.CONFIG["seed"])
                rows.append({"plan_set": tag, "reader": rd, "dataset": ds, "method": name, "gain": d.mean(),
                             "ci_low": lo, "ci_high": hi, "p": A.p_two_sided(d),
                             "cost": np.mean([h["cost"] for h in halves]),
                             "gain_training_folds": np.mean([h["gain_selection_half"] for h in halves])})
            if curve:
                for fr in FRACS:
                    print(f"  [curve] {rd} {ds} frac {fr} ({time.time() - t0:.0f}s)", flush=True)
                    d, *_ = A.repeated(B, lambda: m_planner_sk(B, F2, frac=fr))
                    lc.append({"reader": rd, "dataset": ds, "frac": fr, "n_train": int(round(fr * 400)),
                               "gain": d.mean(), "sum_gain": d.sum(), "n": len(d)})
    PQ = pd.concat(pq, ignore_index=True)
    PQ.to_csv(OUT / f"per_query_gains{'_r3' if r3 else ''}.csv", index=False)
    fam = []
    for (rd, m), g in PQ.groupby(["reader", "method"]):
        d = g["gain"].to_numpy(float)
        lo, hi = A.boot_ci(d, A.CONFIG["n_boot"], A.CONFIG["seed"])
        fam.append({"plan_set": tag, "reader": rd, "method": m, "gain": d.mean(), "ci_low": lo, "ci_high": hi,
                    "p": A.p_two_sided(d), "n": len(d)})
    Fm = pd.DataFrame(fam)
    Fm["p_holm"] = A.holm(Fm["p"].to_numpy())
    L.append(f"\n== {tag} plans: {len(Fm)} pooled tests, Holm over {len(Fm)}")
    for _, r in Fm.sort_values("gain", ascending=False).iterrows():
        flag = ("POSITIVE" if r.ci_low > 0 else "negative") if r.p_holm < 0.05 else ""
        L.append(f"  {r.reader:12s} {r.method:11s} {r.gain:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  "
                 f"Holm p {r.p_holm:.4f}  {flag}")
    return pd.DataFrame(rows), Fm, pd.DataFrame(lc)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curve = "--no-curve" not in sys.argv
    L = [f"Stronger policies (amendment 13 Oct 20:45 (c)). Label {A.CONFIG['label']}, seeds {A.CONFIG['seeds']}, "
         f"{A.CONFIG['k_folds']} folds, cost axis {A.CONFIG['cost_axis']}."]
    D1, F1, LC = run_plan_set(False, L, curve)
    D2, F2, _ = run_plan_set(True, L, False)
    pd.concat([D1, D2]).to_csv(OUT / "per_dataset.csv", index=False)
    Fam = pd.concat([F1, F2], ignore_index=True)
    Fam.to_csv(OUT / "family.csv", index=False)
    pos = Fam[(Fam.p_holm < 0.05) & (Fam.ci_low > 0)].groupby(["reader", "method"]).plan_set.nunique()
    both = pos[pos == 2]
    L.append("\n== Decision rule (both plan sets): " + (f"{len(both)} positive: " + ", ".join(f"{r} {m}" for r, m in both.index)
                                                      if len(both) else "0 positive"))
    if len(LC):
        LC.to_csv(OUT / "learning_curve.csv", index=False)
        L.append("\n== Learning curve, planner_sk, single step plans (pooled gain over the hull; descriptive)")
        for rd, g in LC.groupby("reader"):
            pts = g.groupby("frac").apply(lambda x: x.sum_gain.sum() / x.n.sum())
            L.append(f"  {rd:12s} " + "  ".join(f"{int(f * 100):3d}%: {v:+.3f}" for f, v in pts.items()))
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nWritten {OUT / 'summary.txt'}")


if __name__ == "__main__":
    main()
