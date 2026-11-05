"""
Refit bootstrap on randomly drawn tests. CPU only. Plan: docs/ANALYSIS_PLAN_2026-11-05.md, CPU analysis 2.

Concern: the inflation factor 1.64 is the largest ratio of the refit bootstrap spread to the corrected standard error
over four tests chosen because they were near a decision. Does it cover other tests?

Draw: 12 tests (method, reader, plan set) with numpy seed 20261105 from the primary and secondary families, both
plan sets, excluding the four tests already calibrated. Policies that cannot be refitted on resampled questions here
are outside the draw, stated before the run: the reader cascades (two reader blocks), the FLAN-T5 Adaptive RAG
classifier (a GPU fine tune per replicate) and the METIS style router (profiles stored per question).
Each test: 100 replicates; in each, the 500 questions of every dataset are resampled with replacement, copies of a
question stay in one fold, plan costs are recomputed on the resample, and policy, hull and weights are refitted
(one fold split per replicate, as 24_review_checks.py). The spread (SD over replicates of the pooled gain) is
compared with the per question and the corrected standard errors of tests_nb.csv.
If the largest ratio to the corrected error exceeds 1.64, the larger ratio replaces 1.64 (written to calibration_random.csv).

Usage:  python scripts/37_refit_random.py [--workers 4] [--reps 100]
        python scripts/37_refit_random.py --draw-only        (prints the draw and stops)
Output: results_full/analysis/inference/refit_random.csv, refit_random_reps.csv, summary_refit_random.txt
"""
import copy
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import settings

ARGS = sys.argv[1:]
arg = lambda n, d=None: ARGS[ARGS.index(n) + 1] if n in ARGS else d
REPS = int(arg("--reps", "100"))
WORKERS = int(arg("--workers", "4"))
SEED, N_DRAW = 20261105, 12
POOL = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade",
        "ak_paper", "ar_emb", "plan_emb", "plan_knn"]
DONE = {("plan_emb", "qwen3-32b", "single step"), ("plan_emb", "qwen3-32b", "with two hop"),
        ("conf_cascade", "qwen2.5-7b", "with two hop")}
I = settings.RESULTS_ROOT / "analysis" / "inference"


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


def draw():
    T = pd.read_csv(I / "tests_nb.csv")
    T = T[T.family.isin(["primary", "secondary"]) & T.method.isin(POOL)]
    T = T[[(m, r, p) not in DONE for m, r, p in zip(T.method, T.reader, T.plan_set)]].reset_index(drop=True)
    T = T.sort_values(["plan_set", "family", "reader", "method"]).reset_index(drop=True)
    pick = np.random.default_rng(SEED).choice(len(T), N_DRAW, replace=False)
    return T.loc[sorted(pick), ["plan_set", "family", "reader", "method", "gain", "se_plain", "se_nb"]].reset_index(drop=True)


def resample_full(B, d_ds, take):
    """A copy of block B on resampled questions (copy j of question q is 'q#j'), with every per question
    attribute remapped (answers, confidences, query text, retrieval caches) and plan costs recomputed."""
    nb = copy.copy(B)
    new = np.array([f"{q}#{j}" for j, q in enumerate(take)])
    nb.Y = B.Y.loc[take].set_axis(new)
    nb.conf = B.conf.loc[take].set_axis(new)
    rt = d_ds.pivot_table(index="query_id", columns="plan", values="retrieval_time_s").reindex(index=B.q, columns=B.plans)
    at = d_ds.pivot_table(index="query_id", columns="plan", values="agent_time_s").reindex(index=B.q, columns=B.plans)
    nb.ret_cost = rt.loc[take].fillna(0.0).mean()
    nb.agent_cost = at.loc[take].mean()
    nb.cost = nb.ret_cost + nb.agent_cost
    nb.q = new
    nb.text = {n: B.text[o] for n, o in zip(new, take)}
    nb.cache = {}
    for r, c in B.cache.items():
        c2 = dict(c)
        for k in ("queries", "results", "latency_s"):
            if k in c:
                c2[k] = {n: c[k][o] for n, o in zip(new, take) if o in c[k]}
        nb.cache[r] = c2
    nb.Yn = nb.Y.to_numpy(dtype=float)
    nb.Cn = nb.conf.reindex(index=nb.Y.index, columns=nb.Y.columns).to_numpy(dtype=float)
    nb.qrow = {q: i for i, q in enumerate(nb.Y.index)}
    nb._rows = {}
    return nb


def one_test(row):
    """Runs in a worker: all replicates of one test. Returns the replicate gains."""
    K = _load("inference27", "27_inference_refit.py")
    C24 = _load("review24", "24_review_checks.py")
    A, R = K.A, K.R
    method, rd, ps = row["method"], row["reader"], row["plan_set"]
    A.CONFIG["include_r3"] = ps == "with two hop"
    df, _ = A.load_reader(rd)
    dss = sorted(df.dataset.unique())
    base = {ds: A.Block(df, ds) for ds in dss}
    emb = {}
    if method in ("ar_emb", "plan_emb", "plan_knn"):
        for ds in dss:
            E = R.load_emb(ds).reindex([str(q) for q in base[ds].q])
            E.index = base[ds].q
            emb[ds] = E
    out = []
    t0 = time.time()
    for b in range(REPS):
        rng = np.random.default_rng(10_000 + b)
        for m_ in K.MODS:
            m_._CUR_SEED[0] = b
        gains = []
        for ds in dss:
            B0 = base[ds]
            take = rng.choice(B0.q, len(B0.q), replace=True)
            B = resample_full(B0, df[df.dataset == ds], take)
            E2 = emb[ds].loc[take].set_axis(B.q) if ds in emb else None
            F = A.decision_features(B) if method in ("planner_tsr", "adaptive_rag", "adaptive_k",
                                                     "gate_closed_book", "conf_cascade") else None
            mk = {"planner_tsr": lambda: A.m_planner_tsr(B, F), "adaptive_rag": lambda: A.m_adaptive_rag(B, F),
                  "adaptive_k": lambda: A.m_adaptive_k(B, F), "gate_closed_book": lambda: A.m_gate_closed_book(B, F),
                  "conf_cascade": lambda: A.m_conf_cascade(B, F), "ak_paper": lambda: R.m_ak_paper(B),
                  "ar_emb": lambda: R.m_ar_emb(B, E2), "plan_emb": lambda: R.m_plan_emb(B, E2),
                  "plan_knn": lambda: R.m_plan_knn(B, E2)}[method]
            m = mk()
            if m is None:
                continue
            gains.append(C24.heldout_grouped(B, m, b))
        out.append(float(np.concatenate(gains).mean()))
        if b % 10 == 9:
            print(f"    {method} {rd} {ps}: {b + 1} replicates ({time.time() - t0:.0f}s)", flush=True)
    for m_ in K.MODS:
        m_._CUR_SEED[0] = 0
    return out


def main():
    D = draw()
    print("Draw (seed %d):\n%s" % (SEED, D[["plan_set", "family", "reader", "method"]].to_string()), flush=True)
    if "--draw-only" in ARGS:
        return
    rows = D.to_dict("records")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(one_test, rows))
    reps, out = [], []
    for r, g in zip(rows, res):
        g = np.array(g)
        sd = float(g.std(ddof=1))
        out.append({**r, "reps": len(g), "refit_mean": float(g.mean()), "sd_refit": sd,
                    "ratio_nb": sd / r["se_nb"], "ratio_plain": sd / r["se_plain"]})
        reps += [{"plan_set": r["plan_set"], "reader": r["reader"], "method": r["method"], "rep": i, "gain": v}
                 for i, v in enumerate(g)]
    O = pd.DataFrame(out)
    O.to_csv(I / "refit_random.csv", index=False)
    pd.DataFrame(reps).to_csv(I / "refit_random_reps.csv", index=False)
    C = pd.read_csv(I / "calibration.csv")
    old = float((C.sd_refit / C.se_nb).max())
    new = max(old, float(O.ratio_nb.max()))
    L = [f"Refit bootstrap on {len(O)} randomly drawn tests ({REPS} replicates each; {time.time() - t0:.0f}s)",
         O[["plan_set", "reader", "method", "gain", "sd_refit", "ratio_plain", "ratio_nb"]].round(3).to_string(),
         f"ratio to the corrected error: {O.ratio_nb.min():.2f} to {O.ratio_nb.max():.2f} (calibrated factor {old:.2f})",
         f"ratio to the per question error: {O.ratio_plain.min():.2f} to {O.ratio_plain.max():.2f}",
         f"factor to use: {new:.2f}" + (" (unchanged)" if new == old else " (REPLACES the old factor)")]
    pd.DataFrame([{"old_factor": old, "max_random_ratio": float(O.ratio_nb.max()), "factor": new}]).to_csv(
        I / "calibration_random.csv", index=False)
    (I / "summary_refit_random.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
