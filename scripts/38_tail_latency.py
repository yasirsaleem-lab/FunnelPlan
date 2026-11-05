"""
The primary family against a hull matched on tail latency. CPU only. Plan: docs/ANALYSIS_PLAN_2026-11-05.md, CPU 3.

Concern: the hull matches the policy's MEAN latency. A random mix of a cheap and an expensive plan can have the same
mean as a policy and a different tail, so a service level on the 95th (or 99th) percentile may change the verdict.

For every primary test (the paper's folds, seeds and settings; 30_policy_checks.py recovers each question's plan and
its realized latency), on each held out fold:
  target   the 95th percentile of the policy's realized per question latency on the fold (and the 99th)
  hull_p   the best random mix of two fixed plans (or one plan) whose 95th (99th) percentile latency on the
           TRAINING questions is at most the target, chosen by training success; the mix's latency distribution is
           the weighted mixture of the two plans' per question latencies; weights in steps of 0.05; candidates are
           the plans on the training (percentile, success) frontier. If no mix meets the target, the plan with the
           lowest percentile is used (as the mean hull does below its cheapest vertex).
  gain     the policy's held out success minus the mix's held out success.
Tests: per question test on the 2,000 pooled gains (mean over seeds), corrected test on the fold gains, and the
corrected test with errors x the largest refit ratio; Holm over the 27 primary tests of each plan set; a policy beats
this hull only if it is positive in both plan sets under all three.

Usage:  python scripts/38_tail_latency.py [--quick]
Output: results_full/analysis/tail_latency/  tests.csv, folds.csv, per_query.csv, summary.txt
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
from scipy import stats

from config import settings

QUICK = "--quick" in sys.argv
OUT = settings.RESULTS_ROOT / "analysis" / ("tail_latency_quick" if QUICK else "tail_latency")
OUT.mkdir(parents=True, exist_ok=True)
PRIMARY = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade"]
QS = (95, 99)
LAMS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]] + (["--quick"] if QUICK else [])
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


P30 = _load("checks30", "30_policy_checks.py")
K, A = P30.K, P30.A
INFL = float((pd.read_csv(settings.RESULTS_ROOT / "analysis" / "inference" / "calibration.csv")
              .eval("sd_refit / se_nb")).max())


def mix_quantile(La, Lb, lam, q):
    """q-th percentile of the latency of a mix that runs a with probability lam and b otherwise."""
    xs = np.unique(np.concatenate([La, Lb]))
    Fa = np.searchsorted(np.sort(La), xs, side="right") / len(La)
    Fb = np.searchsorted(np.sort(Lb), xs, side="right") / len(Lb)
    F = lam * Fa + (1 - lam) * Fb
    return float(xs[np.argmax(F >= q / 100.0 - 1e-12)])


def tail_hull(C, Y, sel, test, target, q):
    """Best mix on the training questions sel with q-th percentile latency <= target; returns its success on test
    and its realized q-th percentile on test (plan drawn per question with the mix weight, seeded)."""
    Lsel = C.loc[sel]
    S = Y.loc[sel].mean()
    pq = Lsel.quantile(q / 100.0, interpolation="higher")
    order = pq.sort_values().index
    front, best = [], -1.0
    for p in order:                                   # plans on the (percentile, success) frontier
        if S[p] > best:
            front.append(p)
            best = S[p]
    cand = None
    for i, a in enumerate(front):
        for b in front[i:]:
            La, Lb = Lsel[a].to_numpy(float), Lsel[b].to_numpy(float)
            for lam in (LAMS if a != b else [1.0]):
                if mix_quantile(La, Lb, lam, q) <= target + 1e-12:
                    v = lam * S[a] + (1 - lam) * S[b]
                    if cand is None or v > cand[0] + 1e-12:
                        cand = (v, a, b, lam)
    if cand is None:
        a = b = front[0]
        lam = 1.0
    else:
        _, a, b, lam = cand
    yt = lam * Y.loc[test, a].to_numpy(float) + (1 - lam) * Y.loc[test, b].to_numpy(float)
    return float(yt.mean()), a, b, lam


def run_cell(tag, rd, ds, name, B, mk, M, swap_blocks, cols, rows, perq):
    k = A.CONFIG["k_folds"]
    seeds = A.CONFIG["seeds"][:1] if QUICK else A.CONFIG["seeds"]
    G = {q: [] for q in QS}
    for sd in seeds:
        for m_ in K.MODS:
            m_._CUR_SEED[0] = sd
        meth = mk()
        if meth is None:
            return
        fit, apply_ = meth
        rng = np.random.default_rng(sd)
        qq = B.q.copy()
        rng.shuffle(qq)
        folds = [qq[i::k] for i in range(k)]
        for i in range(k):
            test = folds[i]
            sel = np.concatenate([folds[j] for j in range(k) if j != i])
            cfg = fit(sel)
            s, c = apply_(cfg, test)
            s = np.asarray(s, float)
            pick = P30.chosen_index(swap_blocks, name, apply_, cfg, test)
            ql = list(test)
            if name == "reader_cascade":
                x, y, _, _ = cfg
                ra, rb = f"{rd.split('->')[0]}|{x}", f"{rd.split('->')[1]}|{y}"
                same = x.split("@")[0] == y.split("@")[0]
                pf = [cols[j] for j in pick]
                esc = np.array([p == rb for p in pf])
                cr = M["c"].loc[ql, ra].to_numpy() + np.where(
                    esc, M["at"].loc[ql, rb].to_numpy() + (0.0 if same else M["rt"].loc[ql, rb].to_numpy()), 0.0)
            else:
                cr = P30.realized_cost(name, cfg, cols, pick, test, M, B)
            hm = A.hull_at(B.Y, B.cost, sel, c, eval_idx=test)
            for q in QS:
                target = float(np.percentile(cr, q))
                h, a, b, lam = tail_hull(M["c"], B.Y, sel, test, target, q)
                rows.append({"plan_set": tag, "reader": rd, "dataset": ds, "method": name, "seed": sd, "fold": i,
                             "q": q, "target_s": target, "gain": s.mean() - h, "gain_mean_hull": s.mean() - hm,
                             "mix_a": a, "mix_b": b, "lam": lam, "n_test": len(test), "n_train": len(sel)})
                G[q].append(pd.Series(s - h, index=test, name=sd))
    for q in QS:
        g = pd.concat(G[q]).groupby(level=0).mean()
        perq.append(pd.DataFrame({"plan_set": tag, "reader": rd, "dataset": ds, "method": name, "q": q,
                                  "query_id": g.index.astype(str), "gain": g.to_numpy()}))
    for m_ in K.MODS:
        m_._CUR_SEED[0] = 0


def main():
    t0 = time.time()
    rows, perq = [], []
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        tag = "with two hop" if r3 else "single step"
        blocks, mts = {}, {}
        for rd in P30.readers():
            df, _ = A.load_reader(rd)
            t5 = P30.R.T5Router(rd)
            for ds in P30.datasets(df):
                B = A.Block(df, ds)
                blocks[(rd, ds)] = B
                M = mts[(rd, ds)] = P30.mats(df, B)
                mk = P30.makers_for(B, df, ds, t5)
                for name in PRIMARY:
                    run_cell(tag, rd, ds, name, B, mk[name], M, [(B, 0)], list(B.Y.columns), rows, perq)
                print(f"    {tag} {rd} {ds} done ({time.time() - t0:.0f}s)", flush=True)
        for a_, b_ in A.CONFIG["reader_pairs"]:
            for ds in sorted({d for (r, d) in blocks if r == a_} & {d for (r, d) in blocks if r == b_}):
                res0 = A.reader_cascade(blocks, a_, b_, ds)
                if res0 is None:
                    continue
                Bj = res0[0]
                a, b = blocks[(a_, ds)], blocks[(b_, ds)]
                Ma, Mb = mts[(a_, ds)], mts[(b_, ds)]
                M = {k_: pd.concat([Ma[k_].loc[Bj.q].add_prefix(a_ + "|"), Mb[k_].loc[Bj.q].add_prefix(b_ + "|")], axis=1)
                     for k_ in ("rt", "at", "J", "CL", "c")}
                run_cell(tag, f"{a_}->{b_}", ds, "reader_cascade", Bj, lambda: A.reader_cascade(blocks, a_, b_, ds)[1],
                         M, [(a, 0), (b, a.Y.shape[1])], list(Bj.Y.columns), rows, perq)
    A.CONFIG["include_r3"] = False
    F, PQ = pd.DataFrame(rows), pd.concat(perq)
    F.to_csv(OUT / "folds.csv", index=False)
    PQ.to_csv(OUT / "per_query.csv", index=False)
    T = []
    for (tag, rd, m, q), g in F.groupby(["plan_set", "reader", "method", "q"]):
        cells = [K.nb_cell(x.gain, x.n_test.mean(), x.n_train.mean()) for _, x in g.groupby("dataset")]
        mean, se, df_ = K.nb_pooled(cells)
        d = PQ[(PQ.plan_set == tag) & (PQ.reader == rd) & (PQ.method == m) & (PQ.q == q)].gain.to_numpy()
        n = len(d)
        sp = d.std(ddof=1) / np.sqrt(n)
        T.append({"plan_set": tag, "reader": rd, "method": m, "q": q, "gain": mean,
                  "gain_mean_hull": g.gain_mean_hull.mean(), "target_s": g.target_s.mean(),
                  "p_plain": float(A.p_two_sided(d)), "plain_low": d.mean() - stats.t.ppf(0.975, n - 1) * sp,
                  "plain_high": d.mean() + stats.t.ppf(0.975, n - 1) * sp,
                  "se_nb": se, "df_nb": df_, "p_nb": float(2 * stats.t.sf(abs(mean / se), df_)),
                  "nb_low": mean - stats.t.ppf(0.975, df_) * se, "nb_high": mean + stats.t.ppf(0.975, df_) * se,
                  "p_infl": float(2 * stats.t.sf(abs(mean / (se * INFL)), df_)),
                  "infl_low": mean - stats.t.ppf(0.975, df_) * se * INFL, "infl_high": mean + stats.t.ppf(0.975, df_) * se * INFL,
                  "ub95_plain": d.mean() + stats.t.ppf(0.95, n - 1) * sp})
    T = pd.DataFrame(T)
    for col in ("p_plain", "p_nb", "p_infl"):
        T[col + "_holm"] = np.nan
        for _, idx in T.groupby(["plan_set", "q"]).groups.items():
            T.loc[idx, col + "_holm"] = A.holm(T.loc[idx, col].to_numpy())
    T["positive_all"] = ((T.p_plain_holm < 0.05) & (T.plain_low > 0) & (T.p_nb_holm < 0.05) & (T.nb_low > 0)
                         & (T.p_infl_holm < 0.05) & (T.infl_low > 0))
    T["negative_corrected"] = (T.p_nb_holm < 0.05) & (T.nb_high < 0)
    T.to_csv(OUT / "tests.csv", index=False)
    for (tag, q), g in T.groupby(["plan_set", "q"]):
        say(f"{tag}, P{q} matched hull: {len(g)} tests; positive under all three {int(g.positive_all.sum())}; "
            f"negative (corrected) {int(g.negative_corrected.sum())}; gain {g.gain.min():+.3f} to {g.gain.max():+.3f} "
            f"(mean matched hull: {g.gain_mean_hull.min():+.3f} to {g.gain_mean_hull.max():+.3f}); "
            f"upper bound below 0.03: {int((g.ub95_plain < 0.03).sum())}")
    for q in QS:
        g = T[T.q == q]
        w = g[g.positive_all].groupby(["reader", "method"]).plan_set.nunique()
        say(f"P{q}: positive in both plan sets under all three tests: {list(w[w == 2].index) or 'none'}")
    say(f"({time.time() - t0:.0f}s)")
    (OUT / "summary.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
