"""
Refit bootstrap where the variance estimates disagree. CPU only.

Concern (review of 7 November, M2 and M3): the per question test and the corrected test can be too narrow, and the
inflation factor 2.61 (the largest single seed refit ratio over 16 tests) may be too wide. Where a verdict depends on
which of them is right, measure the spread directly, with the estimator the paper uses (five seeds per replicate).

Part "bounds": the 12 primary tests whose bound under the x2.61 test exceeds 0.03 (tests_nb.csv), the two reader
cascades among them included.
Part "tail": the tail matched tests (38_tail_latency.py) that are positive under the per question test in at least
one plan set: planner_tsr for Qwen3 14B and Llama 3.1 8B, conf_cascade for Qwen3 14B (both plan sets), adaptive_k
for Qwen3 14B (single step). Each replicate gives the P95 and the P99 gain.

Each replicate: the questions of every dataset are resampled with replacement (copies of a question stay in one
fold), plan costs are recomputed on the resample, and for each of five seeds the fold split, the policy, the hull
and its weights are refitted (the tail hull is chosen on the training folds as in script 38). The replicate's gain
is the pooled gain averaged over the five seeds, as in the paper. The refit standard error is the SD over replicates.
It is read around the observed gain (a resample holds about 63% distinct questions, which shifts a learned policy's
mean down; see 24_review_checks.py).

Usage:  python scripts/43_refit_targeted.py [--parts bounds,tail] [--reps 100] [--workers 4] [--seeds 5]
                                           [--methods planner_tsr,conf_cascade]  (bounds part: only these methods)
        Each finished test is saved in OUT/done/; a rerun with the same settings skips it. Change --reps or
        --seeds only together with a new --out folder.
        python scripts/43_refit_targeted.py --quick     (2 replicates, 1 seed, first test of each part, to test code)
Output: results_full/analysis/refit_targeted/  bounds.csv, tail.csv, reps.csv, summary.txt
"""
import importlib.util
import json
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
from scipy import stats

from config import settings

ARGS = sys.argv[1:]
arg = lambda n, d=None: ARGS[ARGS.index(n) + 1] if n in ARGS else d
QUICK = "--quick" in ARGS
REPS = 2 if QUICK else int(arg("--reps", "100"))
WORKERS = int(arg("--workers", "4"))
NSEEDS = 1 if QUICK else int(arg("--seeds", "5"))
PARTS = arg("--parts", "bounds,tail").split(",")
ONLY_METHODS = [m for m in arg("--methods", "").split(",") if m]
TAIL_TARGET_TRAIN = "--tail-target=train" in ARGS     # as 38_tail_latency.py --target=train
TAIL_SOURCE = "tail_latency_train" if TAIL_TARGET_TRAIN else "tail_latency"       # e.g. --methods planner_tsr,conf_cascade
AN = settings.RESULTS_ROOT / "analysis"
OUT = Path(arg("--out", str(AN / "refit_targeted")))
INFL = 2.614042
TAIL_TESTS = [("planner_tsr", "qwen3-14b", "single step"), ("planner_tsr", "qwen3-14b", "with two hop"),
              ("planner_tsr", "llama3.1-8b", "single step"), ("planner_tsr", "llama3.1-8b", "with two hop"),
              ("conf_cascade", "qwen3-14b", "single step"), ("conf_cascade", "qwen3-14b", "with two hop"),
              ("adaptive_k", "qwen3-14b", "single step")]


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


def bound_tests():
    T = pd.read_csv(AN / "inference" / "tests_nb.csv")
    T = T[T.family == "primary"].copy()
    T["ub_infl"] = T.gain + stats.t.ppf(0.95, T.df_nb) * T.se_nb * INFL
    T = T[T.ub_infl > 0.03].sort_values(["plan_set", "reader", "method"]).reset_index(drop=True)
    return T[["plan_set", "reader", "method", "gain", "se_plain", "se_nb", "df_nb", "ub95_plain", "ub95_nb", "ub_infl"]]


def grouped_folds(q, k, seed):
    orig = np.array([x.split("#")[0] for x in q])
    uniq = np.unique(orig)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {u: i % k for i, u in enumerate(uniq)}
    fid = np.array([fold_of[o] for o in orig])
    return [q[fid == i] for i in range(k)]


def makers(A, B, method):
    F = A.decision_features(B)
    return {"planner_tsr": lambda: A.m_planner_tsr(B, F), "adaptive_rag": lambda: A.m_adaptive_rag(B, F),
            "adaptive_k": lambda: A.m_adaptive_k(B, F), "gate_closed_book": lambda: A.m_gate_closed_book(B, F),
            "conf_cascade": lambda: A.m_conf_cascade(B, F)}[method]


def one_bound(row):
    """All replicates of one primary test (mean matched hull). Returns the replicate gains."""
    R37 = _load("refit37", "37_refit_random.py")
    K = R37._load("inference27", "27_inference_refit.py")
    A = K.A
    method, rd, ps = row["method"], row["reader"], row["plan_set"]
    A.CONFIG["include_r3"] = ps == "with two hop"
    rds = rd.split("->")
    data = {r: A.load_reader(r)[0] for r in rds}
    dss = sorted(set.intersection(*[set(d.dataset.unique()) for d in data.values()]))
    if QUICK:
        dss = dss[:1]
    base = {(r, ds): A.Block(data[r], ds) for r in rds for ds in dss}
    k = A.CONFIG["k_folds"]
    out, t0 = [], time.time()
    for b in range(REPS):
        rng = np.random.default_rng(10_000 + b)
        res = {}
        for ds in dss:
            q0 = np.intersect1d(*[base[(r, ds)].q for r in rds]) if len(rds) > 1 else base[(rds[0], ds)].q
            take = rng.choice(q0, len(q0), replace=True)
            res[ds] = {(r, ds): R37.resample_full(base[(r, ds)], data[r][data[r].dataset == ds], take) for r in rds}
        per_seed = []
        for s in range(NSEEDS):
            for m_ in K.MODS:
                m_._CUR_SEED[0] = s
            gains = []
            for ds in dss:
                blocks = res[ds]
                if method == "reader_cascade":
                    Bt, (fit, apply_) = A.reader_cascade(blocks, rds[0], rds[1], ds)
                else:
                    Bt = blocks[(rds[0], ds)]
                    meth = makers(A, Bt, method)()
                    if meth is None:
                        continue
                    fit, apply_ = meth
                for test in grouped_folds(Bt.q, k, 100_000 * (b + 1) + s):
                    sel = np.setdiff1d(Bt.q, test)
                    cfg = fit(sel)
                    sv, c = apply_(cfg, test)
                    gains.append(np.asarray(sv, float) - A.hull_at(Bt.Y, Bt.cost, sel, c, eval_idx=test))
            per_seed.append(float(np.concatenate(gains).mean()))
        out.append(float(np.mean(per_seed)))
        if b % 10 == 9 or QUICK:
            print(f"    bounds {method} {rd} {ps}: {b + 1} replicates ({time.time() - t0:.0f}s)", flush=True)
    for m_ in K.MODS:
        m_._CUR_SEED[0] = 0
    return out


def one_tail(row):
    """All replicates of one tail matched test. Returns a list of (gain at P95, gain at P99)."""
    R37 = _load("refit37", "37_refit_random.py")
    T38 = _load("tail38", "38_tail_latency.py")
    if TAIL_TARGET_TRAIN:
        T38.LAMS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    P30, K, A = T38.P30, T38.K, T38.A
    method, rd, ps = row["method"], row["reader"], row["plan_set"]
    A.CONFIG["include_r3"] = ps == "with two hop"
    df, _ = A.load_reader(rd)
    dss = sorted(df.dataset.unique())[:1] if QUICK else sorted(df.dataset.unique())
    base = {ds: A.Block(df, ds) for ds in dss}
    mats = {ds: P30.mats(df, base[ds]) for ds in dss}
    k = A.CONFIG["k_folds"]
    out, t0 = [], time.time()
    for b in range(REPS):
        rng = np.random.default_rng(10_000 + b)
        res = {}
        for ds in dss:
            B0 = base[ds]
            take = rng.choice(B0.q, len(B0.q), replace=True)
            B = R37.resample_full(B0, df[df.dataset == ds], take)
            M = {kk: v.loc[take].set_axis(B.q) for kk, v in mats[ds].items()}
            res[ds] = (B, M)
        per_seed = {95: [], 99: []}
        for s in range(NSEEDS):
            for m_ in K.MODS:
                m_._CUR_SEED[0] = s
            g = {95: [], 99: []}
            for ds in dss:
                B, M = res[ds]
                meth = makers(A, B, method)()
                if meth is None:
                    continue
                fit, apply_ = meth
                cols = list(B.Y.columns)
                for test in grouped_folds(B.q, k, 100_000 * (b + 1) + s):
                    sel = np.setdiff1d(B.q, test)
                    cfg = fit(sel)
                    sv, _ = apply_(cfg, test)
                    sv = np.asarray(sv, float)
                    pick = P30.chosen_index([(B, 0)], method, apply_, cfg, test)
                    cr = P30.realized_cost(method, cfg, cols, pick, test, M, B)
                    if TAIL_TARGET_TRAIN:
                        pick_s = P30.chosen_index([(B, 0)], method, apply_, cfg, sel)
                        cr = P30.realized_cost(method, cfg, cols, pick_s, sel, M, B)
                    for q in (95, 99):
                        target = float(np.percentile(cr, q))
                        h, _, _, _ = T38.tail_hull(M["c"], B.Y, sel, test, target, q)
                        g[q].append(sv - h)
            for q in (95, 99):
                per_seed[q].append(float(np.concatenate(g[q]).mean()))
        out.append((float(np.mean(per_seed[95])), float(np.mean(per_seed[99]))))
        if b % 10 == 9 or QUICK:
            print(f"    tail {method} {rd} {ps}: {b + 1} replicates ({time.time() - t0:.0f}s)", flush=True)
    for m_ in K.MODS:
        m_._CUR_SEED[0] = 0
    return out


def _key(part, r):
    return f"{part}__{r['method']}__{r['reader'].replace('->', '_to_')}__{r['plan_set'].replace(' ', '_')}"


def run_part(part, fn, rows):
    """Runs the tests of one part in parallel; each finished test is saved at once (OUT/done/<key>.json), and tests
    saved by an earlier run are not run again, so an interrupted run loses only the tests in progress."""
    from concurrent.futures import as_completed
    done_dir = OUT / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    todo = []
    for r in rows:
        f = done_dir / f"{_key(part, r)}.json"
        if f.exists() and not QUICK:
            res[_key(part, r)] = json.loads(f.read_text())
            print(f"  {part}: {_key(part, r)} loaded from an earlier run", flush=True)
        else:
            todo.append(r)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        fut = {ex.submit(fn, r): r for r in todo}
        for f_ in as_completed(fut):
            r = fut[f_]
            out = f_.result()
            res[_key(part, r)] = out
            (done_dir / f"{_key(part, r)}.json").write_text(json.dumps(out))
            print(f"  {part}: {_key(part, r)} finished and saved", flush=True)
    return [res[_key(part, r)] for r in rows]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run, m = np.empty_like(p), 0.0, len(p)
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L, REP = [], []
    say = lambda s: (print(s, flush=True), L.append(s))
    say(f"Refit bootstrap where the tests disagree: {REPS} replicates, {NSEEDS} seeds per replicate; the refit "
        f"interval is the observed gain +/- z x the replicate SD.")
    t0 = time.time()
    if "bounds" in PARTS:
        D = bound_tests()
        if ONLY_METHODS:
            D = D[D.method.isin(ONLY_METHODS)].reset_index(drop=True)
        if QUICK:
            D = D.head(1)
        res = run_part("bounds", one_bound, D.to_dict("records"))
        rows = []
        for r, g in zip(D.to_dict("records"), res):
            g = np.array(g)
            sd = float(g.std(ddof=1))
            rows.append({**r, "reps": len(g), "refit_mean": float(g.mean()), "sd_refit": sd,
                         "ratio_nb": sd / r["se_nb"], "ratio_plain": sd / r["se_plain"],
                         "ub95_refit": r["gain"] + stats.norm.ppf(0.95) * sd,
                         "p_refit": float(2 * stats.norm.sf(abs(r["gain"] / sd)))})
            REP += [{"part": "bounds", "plan_set": r["plan_set"], "reader": r["reader"], "method": r["method"],
                     "q": "", "rep": i, "gain": v} for i, v in enumerate(g)]
        O = pd.DataFrame(rows)
        O.to_csv(OUT / "bounds.csv", index=False)
        say("\n== bounds: primary tests whose bound under x2.61 exceeds 0.03")
        say(O[["plan_set", "reader", "method", "gain", "ub95_plain", "ub95_nb", "ub_infl", "ub95_refit",
               "ratio_plain", "ratio_nb"]].round(4).to_string(index=False))
        say(f"   ratio of the 5 seed refit spread to the corrected error: {O.ratio_nb.min():.2f} to {O.ratio_nb.max():.2f}; "
            f"to the per question error: {O.ratio_plain.min():.2f} to {O.ratio_plain.max():.2f}")
        say(f"   refit bounds below 0.03: {int((O.ub95_refit < 0.03).sum())} of {len(O)}; below 0.04: "
            f"{int((O.ub95_refit < 0.04).sum())}; largest {O.ub95_refit.max():.4f}")
    if "tail" in PARTS and not (TAIL_TARGET_TRAIN and not (AN / TAIL_SOURCE / "tests.csv").exists()):
        T38t = pd.read_csv(AN / TAIL_SOURCE / "tests.csv")
        tests = TAIL_TESTS
        if TAIL_TARGET_TRAIN:      # the tests positive under the per question test in this version
            pos = T38t[(T38t.p_plain_holm < 0.05) & (T38t.plain_low > 0)]
            tests = sorted({(m, r, p) for m, r, p in zip(pos.method, pos.reader, pos.plan_set)})
            if not tests:
                print("  tail (target on training questions): no test is positive under the per question test")
        D = pd.DataFrame([{"method": m, "reader": r, "plan_set": p} for m, r, p in (tests[:1] if QUICK else tests)])
        res = run_part("tail", one_tail, D.to_dict("records")) if len(D) else []
        rows = []
        for r, g in zip(D.to_dict("records"), res):
            g = np.array(g)
            for j, q in enumerate((95, 99)):
                obs = T38t[(T38t.method == r["method"]) & (T38t.reader == r["reader"]) & (T38t.plan_set == r["plan_set"])
                           & (T38t.q == q)].iloc[0]
                sd = float(g[:, j].std(ddof=1))
                z = stats.norm.ppf(0.975)
                rows.append({**r, "q": q, "gain": obs.gain, "plain_low": obs.plain_low, "nb_low": obs.nb_low,
                             "reps": len(g), "refit_mean": float(g[:, j].mean()), "sd_refit": sd,
                             "refit_low": obs.gain - z * sd, "refit_high": obs.gain + z * sd,
                             "p_refit": float(2 * stats.norm.sf(abs(obs.gain / sd))),
                             "sd_ratio_nb": sd / obs.se_nb})
                REP += [{"part": "tail", "plan_set": r["plan_set"], "reader": r["reader"], "method": r["method"],
                         "q": q, "rep": i, "gain": v} for i, v in enumerate(g[:, j])]
        O = pd.DataFrame(rows)
        if O.empty:
            (OUT / "summary.txt").write_text("\n".join(L) + "\nno tail test to refit\n", encoding="utf-8")
            return
        # Holm over the 27 primary tests of each plan set and percentile: the other tests keep their per question p
        for (ps, q), idx in O.groupby(["plan_set", "q"]).groups.items():
            fam = T38t[(T38t.plan_set == ps) & (T38t.q == q)].copy()
            for i in idx:
                m = (fam.method == O.at[i, "method"]) & (fam.reader == O.at[i, "reader"])
                fam.loc[m, "p_use"] = O.at[i, "p_refit"]
            fam["p_use"] = fam.p_use.fillna(fam.p_plain)
            fam["p_use_holm"] = holm(fam.p_use.to_numpy())
            for i in idx:
                m = (fam.method == O.at[i, "method"]) & (fam.reader == O.at[i, "reader"])
                O.at[i, "p_refit_holm"] = float(fam.loc[m, "p_use_holm"].iloc[0])
        O["positive_refit"] = (O.p_refit_holm < 0.05) & (O.refit_low > 0)
        O.to_csv(OUT / "tail.csv", index=False)
        say("\n== tail: tests positive under the per question test at P95 or P99")
        say(O[["plan_set", "reader", "method", "q", "gain", "plain_low", "nb_low", "refit_low", "refit_high",
               "p_refit_holm", "positive_refit", "sd_ratio_nb"]].round(4).to_string(index=False))
        w = O[O.positive_refit].groupby(["reader", "method", "q"]).plan_set.nunique()
        say(f"   positive under the refit bootstrap in both plan sets: {list(w[w == 2].index) or 'none'}")
    pd.DataFrame(REP).to_csv(OUT / "reps.csv", index=False)
    say(f"({time.time() - t0:.0f}s)")
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
