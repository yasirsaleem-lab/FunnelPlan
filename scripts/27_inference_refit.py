"""
Inference that accounts for refitting. CPU only; no new model calls.

Concern: the paper's t test and bootstrap treat the fitted policies and hulls as fixed. Gains in one fold share one
fitted model, so they are correlated, and intervals that ignore this are too narrow (Nadeau and Bengio, Machine
Learning 52, 2003; Bates, Hastie and Tibshirani, JASA 2024). The refit bootstrap of 24_review_checks.py covers four
tests only.

Parts:
  nb      Every test of the primary family (27 per plan set), the secondary family (30) and the stronger family (10),
          in both plan sets, rerun with the paper's code (same seeds, folds and settings), keeping the gain of every
          fold. The corrected resampled t test (Nadeau and Bengio 2003; repeated k fold form of Bouckaert and Frank,
          PAKDD 2004) uses these 25 fold gains per reader and dataset:
              var = (1/J + n_test/n_train) * S^2,  J = 5 folds x 5 seeds,
          where S^2 is the variance of the fold gains. A pooled test averages the four datasets (equal weights), so
          its variance is the mean of the four variances divided by 4; degrees of freedom by Welch-Satterthwaite.
          Reported per test: gain, plain and corrected 95% intervals, one sided 95% upper bounds, p values, Holm
          within each family and plan set, and the non superiority test at 0.03. Check: the per question gains must
          equal the stored ones (analysis/final*, stronger_policies, metis*).
          Calibration: corrected SE against the refit bootstrap SD for the four tests 24 refitted.
          Also the PAIRED per question gain of Eq. 2, s(q, pi(q)) - [lambda s(q, a) + (1 - lambda) s(q, b)], next to
          the form 11 stores, s(q, pi(q)) minus the fold's mean hull success. Both have the same mean; the paired form
          removes question difficulty from the variance. Neither accounts for refitting; the corrected test does.
  mixed   The same corrected test for the mixed workload runs of 26_mixed_workload.py, from their fold files.
  own     The embedding routers (plan_emb, plan_knn) of the secondary family with their own cost (--overhead_ms,
          default 24.6), per dataset, with the corrected test; the secondary family is recomputed with them.
          Run it after nb:  python scripts/27_inference_refit.py --parts own   (about 15 minutes)
  mde     Table IX (protocol ablation): the minimum detectable effect of each decision rule (80% power), and the
          number of cells expected to win if every cell had a true gain of 0.01, 0.02 or 0.03.

Usage:  python scripts/27_inference_refit.py                     every part (about 1 to 2 hours, mostly nb)
        python scripts/27_inference_refit.py --parts mixed,mde   minutes
        python scripts/27_inference_refit.py --quick             small run to test the code (one reader and dataset)
Output: results_full/analysis/inference/  folds_nb.csv, tests_nb.csv, calibration.csv, tests_mixed_nb.csv,
        mde_ablation.csv, summary_<parts>.txt
"""

import hashlib
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

ARGS = sys.argv[1:]


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


QUICK = "--quick" in ARGS
PARTS = arg("--parts", "nb,mixed,mde").split(",")
ONLY = [m for m in arg("--methods", "").split(",") if m]      # testing: run only these policies
MARGIN = 0.03
AN = settings.RESULTS_ROOT / "analysis"
OUT = AN / ("inference_quick" if QUICK else "inference")
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


R = _load("routers12", "12_routers.py")
A = R.A
S20 = _load("stronger20", "20_stronger_policies.py")
try:
    M13 = _load("metis13", "13_metis_profiler.py")
    PROFILES = M13.load_profiles()
except Exception as e:
    M13, PROFILES = None, None
    print(f"METIS port skipped: {e}", flush=True)
MODS = [A, S20.A] + ([M13.A] if M13 is not None else [])

PRIMARY = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade", "reader_cascade"]
SECONDARY = ["ak_paper", "ar_emb", "ar_t5", "plan_emb", "plan_knn", "metis"]
STRONGER = ["clf_gate", "planner_sk"]
FAMILY = {**{m: "primary" for m in PRIMARY}, **{m: "secondary" for m in SECONDARY}, **{m: "stronger" for m in STRONGER}}
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


# FLAN-T5 predictions are cached per fold and seed by 12_routers.py; never fine-tune here
class T5Miss(Exception):
    pass


_t5_predict = R.T5Router.predict_all


def _t5_cached_only(self, B, sel, rungs, y, seed):
    key = hashlib.sha1(("|".join([B.ds, ",".join(rungs), str(seed), *sorted(map(str, sel))])).encode()).hexdigest()
    if key not in self.cache:
        raise T5Miss(f"{B.ds} seed {seed}")
    return _t5_predict(self, B, sel, rungs, y, seed)


R.T5Router.predict_all = _t5_cached_only


# ====================================================================== the paper's protocol, keeping fold gains
def hull_per_question(Y, cost, sel, c, test):
    """The hull of A.hull_at per question: lambda * s(q, a) + (1 - lambda) * s(q, b) with the vertices a, b and the
    weight chosen on sel at cost c (Eq. 2 of the paper). Its mean over test equals A.hull_at(..., eval_idx=test)."""
    H = A._hull_vertices(Y, cost, sel)
    xs = [h[0] for h in H]
    if c <= xs[0]:
        return Y.loc[test, H[0][2]].to_numpy(float)
    if c >= xs[-1]:
        return Y.loc[test, H[-1][2]].to_numpy(float)
    j = int(np.searchsorted(xs, c, side="right")) - 1
    t = (c - xs[j]) / (xs[j + 1] - xs[j])
    return (1 - t) * Y.loc[test, H[j][2]].to_numpy(float) + t * Y.loc[test, H[j + 1][2]].to_numpy(float)


def heldout_folds(B, method, seed):
    """A.heldout, returning the per question gains and the gain of each fold."""
    fit, apply_ = method
    k = A.CONFIG["k_folds"]
    rng = np.random.default_rng(seed)
    q = B.q.copy()
    rng.shuffle(q)
    folds = [q[i::k] for i in range(k)]
    diffs, frows = [], []
    for i in range(k):
        test = folds[i]
        sel = np.concatenate([folds[j] for j in range(k) if j != i])
        cfg = fit(sel)
        s, c = apply_(cfg, test)
        h = A.hull_at(B.Y, B.cost, sel, c, eval_idx=test)
        d = np.asarray(s, float) - h
        hq = hull_per_question(B.Y, B.cost, sel, c, test)
        assert abs(hq.mean() - h) < 1e-9, "per question hull does not match A.hull_at"
        diffs.append(pd.DataFrame({"gain": d, "gain_paired": np.asarray(s, float) - hq}, index=test))
        frows.append({"seed": seed, "fold": i, "gain": d.mean(), "n_test": len(test), "n_train": len(sel)})
    return pd.concat(diffs), frows


def repeated_folds(B, make):
    per, frows = [], []
    for sd in A.CONFIG["seeds"]:
        for m_ in MODS:
            m_._CUR_SEED[0] = sd
        m = make()
        if m is None:
            return None
        d, fr = heldout_folds(B, m, sd)
        per.append(d.sort_index())
        frows += fr
    for m_ in MODS:
        m_._CUR_SEED[0] = 0
    D = sum(per) / len(per)                       # per question gains averaged over seeds (both forms)
    return D, frows


# ====================================================================== corrected resampled t test
def nb_cell(g, n_test, n_train):
    """Mean and corrected variance of one reader and dataset from its J fold gains."""
    g = np.asarray(g, float)
    J = len(g)
    return g.mean(), (1.0 / J + n_test / n_train) * g.var(ddof=1), J - 1


def nb_pooled(cells):
    """cells: list of (mean, var, df) per dataset, equal weights. Returns mean, se, Welch-Satterthwaite df."""
    w = 1.0 / len(cells)
    m = sum(w * c[0] for c in cells)
    v = sum(w * w * c[1] for c in cells)
    df = v ** 2 / sum((w * w * c[1]) ** 2 / c[2] for c in cells) if v > 0 else 1.0
    return m, np.sqrt(v), df


def nb_row(m, se, df):
    t = m / se if se > 0 else 0.0
    return {"gain": m, "se_nb": se, "df_nb": df, "nb_low": m - stats.t.ppf(0.975, df) * se,
            "nb_high": m + stats.t.ppf(0.975, df) * se, "ub95_nb": m + stats.t.ppf(0.95, df) * se,
            "p_nb": 2 * stats.t.sf(abs(t), df), "p_nonsup_nb": stats.t.cdf((m - MARGIN) / se, df) if se > 0 else 1.0}


def plain_row(d):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    return {"se_plain": se, "plain_low": d.mean() - stats.t.ppf(0.975, n - 1) * se,
            "plain_high": d.mean() + stats.t.ppf(0.975, n - 1) * se, "ub95_plain": d.mean() + stats.t.ppf(0.95, n - 1) * se,
            "p_plain": A.p_two_sided(d), "p_nonsup_plain": stats.t.cdf((d.mean() - MARGIN) / se, n - 1)}


def add_holm(T, keys):
    for col in ("p_plain", "p_nb", "p_nonsup_plain", "p_nonsup_nb", "p_paired", "p_nonsup_paired"):
        if col in T:
            T[col + "_holm"] = np.nan
            for _, idx in T.groupby(keys).groups.items():
                T.loc[idx, col + "_holm"] = A.holm(T.loc[idx, col].to_numpy())
    return T


# ====================================================================== part nb
def stored_gains(r3):
    """Per question gains the paper's scripts stored, and METIS's per dataset gains."""
    sfx = "_r3" if r3 else ""
    P = pd.concat([pd.read_csv(AN / f"final{sfx}" / "per_query_gains.csv", dtype={"query_id": str}),
                   pd.read_csv(AN / f"final{sfx}" / "routers_per_query.csv", dtype={"query_id": str}),
                   pd.read_csv(AN / "stronger_policies" / f"per_query_gains{sfx}.csv", dtype={"query_id": str})])
    P = P.set_index(["reader", "dataset", "method", "query_id"]).gain
    Mt = pd.read_csv(AN / f"metis{sfx}" / "metis.csv")
    Mt = Mt[(Mt.method == "metis") & (Mt.dataset != "pooled")].set_index(["reader", "dataset"]).gain
    return P, Mt


def part_nb():
    say("\n== corrected resampled t test for every primary, secondary and stronger test (both plan sets)")
    folds, perq, checks = [], [], []
    t0 = time.time()
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        tag = "with two hop" if r3 else "single step"
        P_ref, M_ref = stored_gains(r3)
        readers = A.readers()[:1] if QUICK else A.readers()
        blocks = {}
        for rd in readers:
            df, _ = A.load_reader(rd)
            t5 = R.T5Router(rd)
            for ds in (sorted(df.dataset.unique())[:1] if QUICK else sorted(df.dataset.unique())):
                B = A.Block(df, ds)
                blocks[(rd, ds)] = B
                F = A.decision_features(B)
                X = S20.cb_features(df, ds, B.q)
                F2 = F.join(X.add_prefix("cb_"))
                E = R.load_emb(ds).reindex([str(q) for q in B.q])
                E.index = B.q
                makers = {"planner_tsr": lambda: A.m_planner_tsr(B, F), "adaptive_rag": lambda: A.m_adaptive_rag(B, F),
                          "adaptive_k": lambda: A.m_adaptive_k(B, F), "gate_closed_book": lambda: A.m_gate_closed_book(B, F),
                          "conf_cascade": lambda: A.m_conf_cascade(B, F),
                          "ak_paper": lambda: R.m_ak_paper(B), "ar_emb": lambda: R.m_ar_emb(B, E),
                          "ar_t5": lambda: R.m_ar_t5(B, t5), "plan_emb": lambda: R.m_plan_emb(B, E),
                          "plan_knn": lambda: R.m_plan_knn(B, E),
                          "clf_gate": lambda: S20.m_clf_gate(B, X), "planner_sk": lambda: S20.m_planner_sk(B, F2)}
                if M13 is not None:
                    makers["metis"] = lambda: M13.m_metis(B, PROFILES, False)
                for name, mk in makers.items():
                    if ONLY and name not in ONLY:
                        continue
                    try:
                        res = repeated_folds(B, mk)
                    except T5Miss as e:
                        say(f"    {tag} {rd} {ds} ar_t5: no cached FLAN-T5 prediction ({e}); skipped")
                        continue
                    if res is None:
                        continue
                    D, fr = res
                    folds += [{"plan_set": tag, "reader": rd, "dataset": ds, "method": name, **f} for f in fr]
                    perq.append(pd.DataFrame({"plan_set": tag, "reader": rd, "dataset": ds, "method": name,
                                              "query_id": D.index.astype(str), "gain": D.gain.to_numpy(),
                                              "gain_paired": D.gain_paired.to_numpy()}))
                    if name == "metis":
                        ref = M_ref.get((rd, ds), np.nan)
                        diff = abs(D.gain.mean() - ref)
                    else:
                        ref = P_ref.reindex(pd.MultiIndex.from_product([[rd], [ds], [name], D.index.astype(str)]))
                        diff = float(np.nanmax(np.abs(ref.to_numpy() - D.gain.to_numpy()))) if ref.notna().any() else np.nan
                    checks.append({"plan_set": tag, "reader": rd, "dataset": ds, "method": name, "max_abs_diff": diff})
                print(f"    {tag} {rd} {ds} done ({time.time() - t0:.0f}s)", flush=True)
        for a_, b_ in ([] if ONLY and "reader_cascade" not in ONLY else A.CONFIG["reader_pairs"]):
            for ds in sorted({d for (r, d) in blocks if r == a_} & {d for (r, d) in blocks if r == b_}):
                res0 = A.reader_cascade(blocks, a_, b_, ds)
                if res0 is None:
                    continue
                D, fr = repeated_folds(res0[0], lambda: A.reader_cascade(blocks, a_, b_, ds)[1])
                rd = f"{a_}->{b_}"
                folds += [{"plan_set": tag, "reader": rd, "dataset": ds, "method": "reader_cascade", **f} for f in fr]
                perq.append(pd.DataFrame({"plan_set": tag, "reader": rd, "dataset": ds, "method": "reader_cascade",
                                          "query_id": D.index.astype(str), "gain": D.gain.to_numpy(),
                                          "gain_paired": D.gain_paired.to_numpy()}))
                ref = P_ref.reindex(pd.MultiIndex.from_product([[rd], [ds], ["reader_cascade"], D.index.astype(str)]))
                checks.append({"plan_set": tag, "reader": rd, "dataset": ds, "method": "reader_cascade",
                               "max_abs_diff": float(np.nanmax(np.abs(ref.to_numpy() - D.gain.to_numpy())))})
    A.CONFIG["include_r3"] = False
    FO, PQ, CH = pd.DataFrame(folds), pd.concat(perq, ignore_index=True), pd.DataFrame(checks)
    FO.to_csv(OUT / "folds_nb.csv", index=False)
    CH.to_csv(OUT / "reproduction_check.csv", index=False)
    say(f"  reproduction check against the stored gains: largest difference {CH.max_abs_diff.max():.2e} "
        f"({int((CH.max_abs_diff > 1e-9).sum())} of {len(CH)} cells differ by more than 1e-9)")

    rows = []
    for (tag, rd, m), g in PQ.groupby(["plan_set", "reader", "method"]):
        f = FO[(FO.plan_set == tag) & (FO.reader == rd) & (FO.method == m)]
        cells = [nb_cell(x.gain, x.n_test.mean(), x.n_train.mean()) for _, x in f.groupby("dataset")]
        pp = plain_row(g.gain_paired)
        rows.append({"plan_set": tag, "family": FAMILY[m], "reader": rd, "method": m, "n": len(g),
                     **nb_row(*nb_pooled(cells)), **plain_row(g.gain),
                     **{k.replace("plain", "paired"): v for k, v in pp.items()}})
    T = add_holm(pd.DataFrame(rows), ["plan_set", "family"])
    T["se_ratio"] = T.se_nb / T.se_plain
    T["se_ratio_paired"] = T.se_nb / T.se_paired
    T.to_csv(OUT / "tests_nb.csv", index=False)

    for (tag, fam), g in T.groupby(["plan_set", "family"], sort=False):
        say(f"\n  {tag} | {fam}: {len(g)} tests; corrected/plain SE ratio median {g.se_ratio.median():.2f} "
            f"(range {g.se_ratio.min():.2f} to {g.se_ratio.max():.2f})")
        say(f"    corrected/paired SE ratio median {g.se_ratio_paired.median():.2f} (range {g.se_ratio_paired.min():.2f} "
            f"to {g.se_ratio_paired.max():.2f})")
        for lab, p, lo, hi, ub, pn in (("plain", "p_plain_holm", "plain_low", "plain_high", "ub95_plain", "p_nonsup_plain_holm"),
                                       ("paired", "p_paired_holm", "paired_low", "paired_high", "ub95_paired", "p_nonsup_paired_holm"),
                                       ("corrected", "p_nb_holm", "nb_low", "nb_high", "ub95_nb", "p_nonsup_nb_holm")):
            say(f"    {lab:9s} positive {int(((g[p] < 0.05) & (g[lo] > 0)).sum()):2d}  negative "
                f"{int(((g[p] < 0.05) & (g[hi] < 0)).sum()):2d}  upper bound below {MARGIN}: {int((g[ub] < MARGIN).sum()):2d}"
                f"  non superiority at {MARGIN} after Holm: {int((g[pn] < 0.05).sum()):2d}  largest upper bound "
                f"{g[ub].max():.3f} ({g.loc[g[ub].idxmax(), 'reader']}, {g.loc[g[ub].idxmax(), 'method']})")
    say("\n  tests whose verdict changes under the corrected test (Holm p < 0.05 with the interval on one side, or "
        "upper bound below 0.03):")
    ch = T[((T.p_plain_holm < 0.05) != (T.p_nb_holm < 0.05)) | ((T.ub95_plain < MARGIN) != (T.ub95_nb < MARGIN))]
    say(ch[["plan_set", "family", "reader", "method", "gain", "ub95_plain", "ub95_nb", "p_plain_holm", "p_nb_holm"]]
        .round(4).to_string(index=False) if len(ch) else "    none")

    # calibration against the refit bootstrap
    rb = AN / "review" / "refit_bootstrap.csv"
    if rb.exists() and not QUICK:
        Rb = pd.read_csv(rb)
        cal = []
        for r in Rb.itertuples():
            tag = "with two hop" if r.plan_set == "with two hop" else "single step"
            t = T[(T.plan_set == tag) & (T.reader == r.reader) & (T.method == r.method)]
            if len(t):
                t = t.iloc[0]
                cal.append({"plan_set": tag, "reader": r.reader, "method": r.method, "se_plain": t.se_plain,
                            "se_paired": t.se_paired, "se_nb": t.se_nb, "sd_refit": r.sd,
                            "nb_over_refit": t.se_nb / r.sd, "paired_over_refit": t.se_paired / r.sd,
                            "plain_over_refit": t.se_plain / r.sd})
        C = pd.DataFrame(cal)
        C.to_csv(OUT / "calibration.csv", index=False)
        say("\n  calibration: standard errors against the refit bootstrap SD (24_review_checks.py, 100 refits)")
        say(C.round(4).to_string(index=False))


# ====================================================================== part own
def part_own():
    """The embedding routers of the secondary family with their own cost (per dataset, as in the paper)."""
    ms = float(arg("--overhead_ms", "24.6"))
    say(f"\n== embedding routers with their own cost ({ms:.1f} ms unless the chosen plan embeds the query with BGE-M3 "
        f"anyway), per dataset; corrected test; secondary family recomputed with these in place of the uncharged ones")
    M26 = _load("mixed26", "26_mixed_workload.py")
    mods = MODS + [M26.A]
    ref_f = AN / "review" / "plan_emb_with_overhead_per_query.csv"
    REF = pd.read_csv(ref_f, dtype={"query_id": str}) if ref_f.exists() else None
    folds, perq, checks = [], [], []
    t0 = time.time()
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        M26.A.CONFIG["include_r3"] = r3
        tag = "with two hop" if r3 else "single step"
        for rd in (A.readers()[:1] if QUICK else A.readers()):
            df, _ = A.load_reader(rd)
            for ds in (sorted(df.dataset.unique())[:1] if QUICK else sorted(df.dataset.unique())):
                B = A.Block(df, ds)
                E = R.load_emb(ds).reindex([str(q) for q in B.q])
                E.index = B.q
                for name, kind in (("plan_emb_own", "emb"), ("plan_knn_own", "knn")):
                    per, fr = [], []
                    for sd in A.CONFIG["seeds"]:
                        for m_ in mods:
                            m_._CUR_SEED[0] = sd
                        d, f = heldout_folds(B, M26.m_plan_own(B, E, kind, ms / 1000.0), sd)
                        per.append(d.sort_index())
                        fr += f
                    for m_ in mods:
                        m_._CUR_SEED[0] = 0
                    D = sum(per) / len(per)
                    folds += [{"plan_set": tag, "reader": rd, "dataset": ds, "method": name, **f} for f in fr]
                    perq.append(pd.DataFrame({"plan_set": tag, "reader": rd, "dataset": ds, "method": name,
                                              "query_id": D.index.astype(str), "gain": D.gain.to_numpy()}))
                    if REF is not None and name == "plan_emb_own":
                        r_ = REF[(REF.variant == "dataset") & (REF.plan_set == tag) & (REF.reader == rd) &
                                 (REF.dataset == ds) & (np.isclose(REF.delta_ms, ms))].set_index("query_id").gain
                        if len(r_):
                            checks.append(float(np.nanmax(np.abs(r_.reindex(D.index.astype(str)).to_numpy() - D.gain.to_numpy()))))
                print(f"    own cost {tag} {rd} {ds} done ({time.time() - t0:.0f}s)", flush=True)
    A.CONFIG["include_r3"] = False
    FO, PQ = pd.DataFrame(folds), pd.concat(perq, ignore_index=True)
    FO.to_csv(OUT / "folds_own.csv", index=False)
    if checks:
        say(f"  check against 24_review_checks.py --parts overhead (per plan logistic router): largest difference "
            f"{max(checks):.2e} over {len(checks)} cells")
    rows = []
    for (tag, rd, m), g in PQ.groupby(["plan_set", "reader", "method"]):
        f = FO[(FO.plan_set == tag) & (FO.reader == rd) & (FO.method == m)]
        cells = [nb_cell(x.gain, x.n_test.mean(), x.n_train.mean()) for _, x in f.groupby("dataset")]
        rows.append({"plan_set": tag, "family": "secondary", "reader": rd, "method": m, "n": len(g),
                     **nb_row(*nb_pooled(cells)), **plain_row(g.gain)})
    O = pd.DataFrame(rows)
    T = pd.read_csv(OUT / "tests_nb.csv") if (OUT / "tests_nb.csv").exists() else None
    if T is None:
        say("  tests_nb.csv missing (run --parts nb first); the family is not recomputed")
        O.to_csv(OUT / "tests_own.csv", index=False)
        return
    S = T[(T.family == "secondary") & ~T.method.isin(["plan_emb", "plan_knn"])][O.columns.intersection(T.columns)]
    F = add_holm(pd.concat([S, O], ignore_index=True), ["plan_set"])
    F.to_csv(OUT / "tests_secondary_own_cost.csv", index=False)
    for tag, g in F.groupby("plan_set", sort=False):
        say(f"\n  {tag}: secondary family with the embedding routers charged ({len(g)} tests)")
        for lab, p, lo, hi, ub in (("plain", "p_plain_holm", "plain_low", "plain_high", "ub95_plain"),
                                   ("corrected", "p_nb_holm", "nb_low", "nb_high", "ub95_nb")):
            pos = g[(g[p] < 0.05) & (g[lo] > 0)]
            say(f"    {lab:9s} positive {len(pos)}" + (f" ({', '.join(f'{r.reader} {r.method} {r.gain:+.3f}' for r in pos.itertuples())})"
                                                       if len(pos) else "")
                + f"  negative {int(((g[p] < 0.05) & (g[hi] < 0)).sum())}  upper bound below {MARGIN}: {int((g[ub] < MARGIN).sum())}")
        say(g[g.method.str.endswith("_own")].sort_values("gain", ascending=False)
            [["reader", "method", "gain", "nb_low", "nb_high", "p_nb_holm", "plain_low", "plain_high"]].round(4).to_string(index=False))
    say("  ar_emb, ar_t5 and ak_paper stay uncharged: charging a decision cost can only lower their gains, and none is positive.")


# ====================================================================== part mixed
def part_mixed():
    say("\n== corrected resampled t test for the mixed workload (26_mixed_workload.py)")
    out = []
    for fo, tf in (("folds.csv", "tests.csv"), ("folds_extra.csv", "tests_extra.csv")):
        f, t = AN / "mixed" / fo, AN / "mixed" / tf
        if not (f.exists() and t.exists()):
            say(f"  {fo} missing; skipped")
            continue
        FO, T = pd.read_csv(f), pd.read_csv(t)
        nte = FO.groupby(["plan_set", "reader", "method"]).size()            # not used; fold sizes from the pool
        for (tag, rd, m), g in FO.groupby(["plan_set", "reader", "method"]):
            fam = T[(T.plan_set == tag) & (T.reader == rd) & (T.method == m)].family.iloc[0]
            n = int(T[(T.plan_set == tag) & (T.reader == rd) & (T.method == m)].n.iloc[0])
            n_te = n / A.CONFIG["k_folds"]
            for comp in ("global", "label"):
                mm, v, df = nb_cell(g[f"gain_{comp}"], n_te, n - n_te)
                out.append({"plan_set": tag, "family": fam, "comparator": comp, "reader": rd, "method": m,
                            **nb_row(mm, np.sqrt(v), df)})
        del nte
    if not out:
        return
    X = add_holm(pd.DataFrame(out), ["plan_set", "family", "comparator"])
    X.to_csv(OUT / "tests_mixed_nb.csv", index=False)
    X["positive"] = (X.p_nb_holm < 0.05) & (X.nb_low > 0)
    for (tag, fam, comp), g in X.groupby(["plan_set", "family", "comparator"], sort=False):
        pos = g[g.positive]
        say(f"  {tag:12s} {fam:12s} vs {comp:6s} hull: {len(g):2d} tests, positive {len(pos):2d}"
            + (f"  ({', '.join(f'{r.reader} {r.method} {r.gain:+.3f}' for r in pos.itertuples())})" if len(pos) else "")
            + f"; upper bound below {MARGIN}: {int((g.ub95_nb < MARGIN).sum())}")


# ====================================================================== part mde
def part_mde():
    say("\n== Table IX: minimum detectable effect (80% power) of each decision rule, and expected wins")
    P = pd.read_csv(AN / "final" / "per_query_gains.csv", dtype={"query_id": str})
    P = P[P.method.isin(PRIMARY[:5])]
    cell = P.groupby(["reader", "dataset", "method"]).gain.agg(["std", "count"])
    pooled = P.groupby(["reader", "method"]).gain.agg(["std", "count"])
    zb = stats.norm.ppf(0.8)
    rows = []

    def add(rule, sd, n, alpha, k_cells):
        z = stats.norm.ppf(1 - alpha / 2)
        se = sd / np.sqrt(n)
        mde = (z + zb) * se
        exp = {f"expected_wins_true_{d:.2f}": float(np.sum(stats.norm.sf(z - d / se))) for d in (0.01, 0.02, 0.03)}
        rows.append({"rule": rule, "tests": k_cells, "alpha_per_test": alpha, "mde_median": float(np.median(mde)),
                     "mde_min": float(np.min(mde)), "mde_max": float(np.max(mde)), **exp})

    add("per cell, 95% interval above 0 and p < 0.05", cell["std"].to_numpy(), cell["count"].to_numpy(), 0.05, len(cell))
    add("per cell, Holm over 100 (strictest step, Bonferroni bound)", cell["std"].to_numpy(), cell["count"].to_numpy(),
        0.05 / len(cell), len(cell))
    add("pooled over datasets, p < 0.05", pooled["std"].to_numpy(), pooled["count"].to_numpy(), 0.05, len(pooled))
    add("pooled, Holm over 25 (strictest step, Bonferroni bound)", pooled["std"].to_numpy(), pooled["count"].to_numpy(),
        0.05 / len(pooled), len(pooled))
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "mde_ablation.csv", index=False)
    say(M.round(3).to_string(index=False))
    say("  Expected wins: the number of tests that would reach the rule if every cell had that true gain (normal "
        "approximation, plain per question SE; the Holm rows use the strictest step, so they are a lower bound).")
    say("  Under a true gain of 0 in every cell, about half of the point estimates would be above 0 (steps 0 to 4 "
        "of Table IX count point estimates, not tests).")


def main():
    t0 = time.time()
    say(f"Inference with refitting; parts {PARTS}; seeds {A.CONFIG['seeds']}" + ("; QUICK TEST RUN" if QUICK else ""))
    for p in PARTS:
        {"nb": part_nb, "own": part_own, "mixed": part_mixed, "mde": part_mde}[p]()
        say(f"  ({p} done, {time.time() - t0:.0f}s)")
        (OUT / f"summary_{'_'.join(PARTS)}.txt").write_text("\n".join(LINES), encoding="utf-8")
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
