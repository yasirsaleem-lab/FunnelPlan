"""
Per policy checks on the stored answers.
CPU only; no new model calls.

Parts:
  main      One rerun of every test (primary, secondary and stronger families, both plan sets, 5 seeds, the paper's
            folds and settings) that recovers, for every held out question, the plan whose answer the policy returns.
            From it:
            exposure  share of the policy's answers whose label_final came from the judge, and share of
                      long contained answers (the stratum where the rules disagree most), against the same shares of
                      the hull mix at the policy's cost; per reader and policy.
            realized  the policy's REALIZED cost, the mean of c(q, plan) over the held out questions
                      (with the probe, closed book or first cascade step it pays), instead of the mean plan cost
                      C(plan). The hull is read at that cost with its vertices' held out mean costs. Corrected test.
            p95       95th percentile of the per question latency of the policy and of its hull (the hull's
                      plan drawn per question with its weight), over the held out questions.
  trainonly every test rerun with plan costs C(p) computed on the training folds only (the paper
            computes them over all 500 questions, held out ones included). Corrected test. ar_t5 is left out: its
            ladder depends on the costs, so it would need new FLAN-T5 fine tunes (GPU); it loses in every test.
  sweep     (; --sweep-methods to choose, plan_emb_own = per plan router with its 24.6 ms own cost;
            --sweep-tag _own writes sweep_tests_own.csv) budget sweep: each tunable policy (TSR planner, closed book gate, confidence cascade, reader
            cascade, per plan router) is tuned on the training folds to the best training success at a mean cost
            within a budget, for 6 budgets from the cheapest plan to the most successful plan, and scored against
            the held out hull. Single step plans, 3 seeds (--seeds to change).
  agreement label_final against the human majority of audits 1 and 2, by reader.

Usage:  python scripts/30_policy_checks.py                         every part (about 3 hours)
        python scripts/30_policy_checks.py --parts main,agreement   (about 75 minutes)
        python scripts/30_policy_checks.py --parts trainonly        (about 60 minutes)
        python scripts/30_policy_checks.py --parts sweep            (about 50 minutes)
        --quick  one reader and dataset, to test the code
Output: results_full/analysis/policy_checks/
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
PARTS = arg("--parts", "main,agreement,trainonly,sweep").split(",")
SWEEP_SEEDS = [int(s) for s in arg("--seeds", "0,1,2").split(",")]
ONLY = [m for m in arg("--methods", "").split(",") if m]      # testing: run only these policies
OUT = settings.RESULTS_ROOT / "analysis" / ("policy_checks_quick" if QUICK else "policy_checks")
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


K = _load("inference27", "27_inference_refit.py")
A, R, S20, M13 = K.A, K.R, K.S20, K.M13
CB = "closed_book@0"
LINES = []
SINGLE = {"adaptive_rag", "adaptive_k", "ak_paper", "ar_emb", "ar_t5", "plan_emb", "plan_knn", "metis"}
Y_ONLY_N = {"plan_emb", "plan_knn"}          # their apply_ trains on B.Y, so only the numpy view is swapped


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def readers():
    want = [r for r in arg("--readers", "").split(",") if r]
    if want:
        return want
    return A.readers()[:1] if QUICK else A.readers()


def datasets(df):
    return sorted(df.dataset.unique())[:1] if QUICK else sorted(df.dataset.unique())


def mats(df, B):
    """Per question matrices on B's questions and plans: retrieval time, reader time, judge decided, long contained."""
    d = df[df.dataset == B.ds].copy()
    d["judge"] = (d.label_final_source == "judge").astype(float)
    em, ct = pd.to_numeric(d.em, errors="coerce"), pd.to_numeric(d.contains, errors="coerce")
    d["clong"] = ((em == 0) & (ct == 1) & (pd.to_numeric(d.n_answer_tokens, errors="coerce") >= 8)).astype(float)
    out = {}
    for c, nm in (("retrieval_time_s", "rt"), ("agent_time_s", "at"), ("judge", "J"), ("clong", "CL")):
        out[nm] = d.pivot_table(index="query_id", columns="plan", values=c).reindex(index=B.Y.index, columns=B.Y.columns)
    out["rt"] = out["rt"].fillna(0.0)
    out["c"] = out["rt"] + out["at"]
    return out


def makers_for(B, df, ds, t5):
    F = A.decision_features(B)
    X = S20.cb_features(df, ds, B.q)
    F2 = F.join(X.add_prefix("cb_"))
    E = R.load_emb(ds).reindex([str(q) for q in B.q])
    E.index = B.q
    mk = {"planner_tsr": lambda: A.m_planner_tsr(B, F), "adaptive_rag": lambda: A.m_adaptive_rag(B, F),
          "adaptive_k": lambda: A.m_adaptive_k(B, F), "gate_closed_book": lambda: A.m_gate_closed_book(B, F),
          "conf_cascade": lambda: A.m_conf_cascade(B, F),
          "ak_paper": lambda: R.m_ak_paper(B), "ar_emb": lambda: R.m_ar_emb(B, E),
          "ar_t5": lambda: R.m_ar_t5(B, t5), "plan_emb": lambda: R.m_plan_emb(B, E),
          "plan_knn": lambda: R.m_plan_knn(B, E),
          "clf_gate": lambda: S20.m_clf_gate(B, X), "planner_sk": lambda: S20.m_planner_sk(B, F2)}
    if M13 is not None:
        mk["metis"] = lambda: M13.m_metis(B, K.PROFILES, False)
    return {k: v for k, v in mk.items() if not ONLY or k in ONLY}


def chosen_index(blocks, name, apply_, cfg, test):
    """Column index (in the scoring frame's columns) of the plan whose answer the policy returns, per question:
    apply_ is rerun with the outcome matrix replaced by the column indices."""
    saved = []
    for j, (B, off) in enumerate(blocks):
        ncol = B.Y.shape[1]
        idx = np.tile(np.arange(off, off + ncol, dtype=float), (len(B.Y), 1))
        saved.append((B, B.Y, B.Yn))
        B.Yn = idx
        if name not in Y_ONLY_N:
            B.Y = pd.DataFrame(idx, index=B.Y.index, columns=B.Y.columns)
    try:
        s, _ = apply_(cfg, test)
    finally:
        for B, Y, Yn in saved:
            B.Y, B.Yn = Y, Yn
    return np.asarray(s).round().astype(int)


def hull_vertex_weight(Y, cost, sel, c):
    H = A._hull_vertices(Y, cost, sel)
    xs = [h[0] for h in H]
    if c <= xs[0]:
        return H[0][2], H[0][2], 0.0, H
    if c >= xs[-1]:
        return H[-1][2], H[-1][2], 0.0, H
    j = int(np.searchsorted(xs, c, side="right")) - 1
    return H[j][2], H[j + 1][2], (c - xs[j]) / (xs[j + 1] - xs[j]), H


# ====================================================================== part main
def realized_cost(name, cfg, cols, pick, test, M, B, extra=None):
    """Per question realized cost of the policy's choices (the paper's cost rules applied to per question times)."""
    q = list(test)
    c = M["c"]
    pf = [cols[k] for k in pick]
    base = np.array([c.at[qq, p] for qq, p in zip(q, pf)])
    if name in SINGLE:
        return base
    if name == "planner_tsr":
        probe = M["rt"].loc[q, "bm25@1"].to_numpy() if "bm25@1" in M["rt"].columns and "bm25" in B.cache else 0.0
        return base + np.where([p.split("@")[0] == "bm25" for p in pf], 0.0, probe)
    if name == "planner_sk":
        probe = M["rt"].loc[q, "bm25@1"].to_numpy() if "bm25@1" in M["rt"].columns and "bm25" in B.cache else 0.0
        cbc = c.loc[q, CB].to_numpy() if CB in c.columns else 0.0
        return base + np.where([p.split("@")[0] == "bm25" for p in pf], 0.0, probe) + \
            np.where([p == CB for p in pf], 0.0, cbc)
    if name in ("gate_closed_book", "clf_gate"):
        cbc = c.loc[q, CB].to_numpy()
        return cbc + np.where([p == CB for p in pf], 0.0, base)
    if name == "conf_cascade":
        a, b, _ = cfg
        ca = c.loc[q, a].to_numpy()
        same = a.split("@")[0] == b.split("@")[0]
        cb_ = M["at"].loc[q, b].to_numpy() + (0.0 if same else M["rt"].loc[q, b].to_numpy())
        return ca + np.where([p == b for p in pf], cb_, 0.0)
    raise ValueError(name)


def part_main():
    say("\n== main pass: chosen plans for every test; exposure, realized cost, p95 latency")
    rows_e, rows_f, rows_l = [], [], []
    t0 = time.time()
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        tag = "with two hop" if r3 else "single step"
        blocks, mts = {}, {}
        for rd in readers():
            df, _ = A.load_reader(rd)
            t5 = R.T5Router(rd)
            for ds in datasets(df):
                B = A.Block(df, ds)
                blocks[(rd, ds)] = B
                M = mts[(rd, ds)] = mats(df, B)
                for name, mk in makers_for(B, df, ds, t5).items():
                    try:
                        run_main(tag, rd, ds, name, B, mk, M, rows_e, rows_f, rows_l, [(B, 0)], list(B.Y.columns))
                    except K.T5Miss:
                        continue
                print(f"    main {tag} {rd} {ds} done ({time.time() - t0:.0f}s)", flush=True)
        for a_, b_ in ([] if ONLY and "reader_cascade" not in ONLY else A.CONFIG["reader_pairs"]):
            for ds in sorted({d for (r, d) in blocks if r == a_} & {d for (r, d) in blocks if r == b_}):
                res0 = A.reader_cascade(blocks, a_, b_, ds)
                if res0 is None:
                    continue
                Bj = res0[0]
                a, b = blocks[(a_, ds)], blocks[(b_, ds)]
                cols = list(Bj.Y.columns)
                Ma, Mb = mts[(a_, ds)], mts[(b_, ds)]
                M = {k: pd.concat([Ma[k].loc[Bj.q].add_prefix(a_ + "|"), Mb[k].loc[Bj.q].add_prefix(b_ + "|")], axis=1)
                     for k in ("rt", "at", "J", "CL", "c")}
                run_main(tag, f"{a_}->{b_}", ds, "reader_cascade", Bj,
                         lambda: A.reader_cascade(blocks, a_, b_, ds)[1], M, rows_e, rows_f, rows_l,
                         [(a, 0), (b, a.Y.shape[1])], cols)
    A.CONFIG["include_r3"] = False
    E, Fo, L = pd.DataFrame(rows_e), pd.DataFrame(rows_f), pd.DataFrame(rows_l)
    E.to_csv(OUT / "exposure_folds.csv", index=False)
    Fo.to_csv(OUT / "realized_folds.csv", index=False)
    L.to_csv(OUT / "latency_per_question.csv", index=False)

    say("\n-- exposure: share of answers graded by the judge / long contained answers, policy against hull mix "
        "(mean over folds, seeds and datasets)")
    Ex = E.groupby(["plan_set", "reader", "method"])[["policy_judge", "hull_judge", "policy_clong", "hull_clong"]].mean()
    Ex["judge_diff"] = Ex.policy_judge - Ex.hull_judge
    Ex["clong_diff"] = Ex.policy_clong - Ex.hull_clong
    Ex = Ex.reset_index()
    Ex.to_csv(OUT / "exposure.csv", index=False)
    for tag, g in Ex.groupby("plan_set", sort=False):
        say(f"  {tag}: judge share difference (policy minus hull) median {g.judge_diff.median():+.4f}, range "
            f"{g.judge_diff.min():+.4f} to {g.judge_diff.max():+.4f}; long contained difference range "
            f"{g.clong_diff.min():+.4f} to {g.clong_diff.max():+.4f} over {len(g)} tests")
        say(g.groupby("method")[["policy_judge", "hull_judge", "judge_diff", "clong_diff"]].mean().round(4).to_string())

    say("\n-- realized cost: gain with the policy's realized cost and the hull at its vertices' held out costs")
    rows = []
    for (tag, rd, m), g in Fo.groupby(["plan_set", "reader", "method"]):
        cells = [K.nb_cell(x.gain_realized, x.n_test.mean(), x.n_train.mean()) for _, x in g.groupby("dataset")]
        cells0 = [K.nb_cell(x.gain, x.n_test.mean(), x.n_train.mean()) for _, x in g.groupby("dataset")]
        r = K.nb_row(*K.nb_pooled(cells))
        rows.append({"plan_set": tag, "reader": rd, "method": m, "family": K.FAMILY[m], "gain_paper": K.nb_pooled(cells0)[0],
                     "cost_paper": g.cost.mean(), "cost_realized": g.cost_realized.mean(), **r})
    T = K.add_holm(pd.DataFrame(rows), ["plan_set", "family"])
    T.to_csv(OUT / "realized_tests.csv", index=False)
    for (tag, fam), g in T.groupby(["plan_set", "family"], sort=False):
        pos = g[(g.p_nb_holm < 0.05) & (g.nb_low > 0)]
        say(f"  {tag:12s} {fam:9s} {len(g):2d} tests: gain change (realized minus paper) median "
            f"{(g.gain - g.gain_paper).median():+.4f}, range {(g.gain - g.gain_paper).min():+.4f} to "
            f"{(g.gain - g.gain_paper).max():+.4f}; positive {len(pos)}"
            + (f" ({', '.join(f'{r.reader} {r.method} {r.gain:+.3f}' for r in pos.itertuples())})" if len(pos) else "")
            + f"; upper bound below 0.03: {int((g.ub95_nb < 0.03).sum())}")

    say("\n-- p95 latency of the policy and of its hull (per question latency over the held out questions of seed 0)")
    P = L.groupby(["plan_set", "reader", "method", "seed"]).agg(
        p95_policy=("policy_s", lambda x: np.percentile(x, 95)), p95_hull=("hull_s", lambda x: np.percentile(x, 95)),
        mean_policy=("policy_s", "mean"), mean_hull=("hull_s", "mean")).groupby(level=[0, 1, 2]).mean().reset_index()
    P["p95_ratio"] = P.p95_policy / P.p95_hull
    P.to_csv(OUT / "p95.csv", index=False)
    for tag, g in P.groupby("plan_set", sort=False):
        say(f"  {tag}: p95 policy / p95 hull median {g.p95_ratio.median():.2f}, range {g.p95_ratio.min():.2f} to "
            f"{g.p95_ratio.max():.2f} over {len(g)} tests")
        say(g.groupby("method")[["mean_policy", "mean_hull", "p95_policy", "p95_hull", "p95_ratio"]].mean().round(3).to_string())


def run_main(tag, rd, ds, name, B, mk, M, rows_e, rows_f, rows_l, swap_blocks, cols):
    k = A.CONFIG["k_folds"]
    for sd in A.CONFIG["seeds"]:
        for m_ in K.MODS:
            m_._CUR_SEED[0] = sd
        meth = mk()
        if meth is None:
            return
        fit, apply_ = meth
        rng = np.random.default_rng(sd)
        q = B.q.copy()
        rng.shuffle(q)
        folds = [q[i::k] for i in range(k)]
        coin = np.random.default_rng(5000 + sd)
        for i in range(k):
            test = folds[i]
            sel = np.concatenate([folds[j] for j in range(k) if j != i])
            cfg = fit(sel)
            s, c = apply_(cfg, test)
            s = np.asarray(s, float)
            pick = chosen_index(swap_blocks, name, apply_, cfg, test)
            pf = [cols[j] for j in pick]
            assert np.allclose([B.Y.at[qq, p] for qq, p in zip(test, pf)], s), f"plan recovery failed: {name}"
            h = A.hull_at(B.Y, B.cost, sel, c, eval_idx=test)
            pa, pb, t, H = hull_vertex_weight(B.Y, B.cost, sel, c)
            ql = list(test)
            rows_e.append({"plan_set": tag, "reader": rd, "dataset": ds, "method": name, "seed": sd, "fold": i,
                           "policy_judge": np.mean([M["J"].at[qq, p] for qq, p in zip(ql, pf)]),
                           "hull_judge": (1 - t) * M["J"].loc[ql, pa].mean() + t * M["J"].loc[ql, pb].mean(),
                           "policy_clong": np.mean([M["CL"].at[qq, p] for qq, p in zip(ql, pf)]),
                           "hull_clong": (1 - t) * M["CL"].loc[ql, pa].mean() + t * M["CL"].loc[ql, pb].mean()})
            # realized cost of the policy
            if name == "reader_cascade":
                x, y, _, _ = cfg
                ra = f"{rd.split('->')[0]}|{x}"
                rb = f"{rd.split('->')[1]}|{y}"
                same = x.split("@")[0] == y.split("@")[0]
                esc = np.array([p == rb for p in pf])
                cr = M["c"].loc[ql, ra].to_numpy() + np.where(
                    esc, M["at"].loc[ql, rb].to_numpy() + (0.0 if same else M["rt"].loc[ql, rb].to_numpy()), 0.0)
            else:
                cr = realized_cost(name, cfg, cols, pick, test, M, B)
            # hull at the realized cost, with its training vertices placed at their held out mean costs
            xs = np.array([M["c"].loc[ql, h_[2]].mean() for h_ in H])
            ys = np.array([B.Y.loc[ql, h_[2]].mean() for h_ in H])
            o = np.argsort(xs, kind="stable")
            hr = float(np.interp(cr.mean(), xs[o], ys[o]))
            rows_f.append({"plan_set": tag, "reader": rd, "dataset": ds, "method": name, "seed": sd, "fold": i,
                           "gain": s.mean() - h, "gain_realized": s.mean() - hr, "cost": c, "cost_realized": cr.mean(),
                           "n_test": len(test), "n_train": len(sel)})
            # per question latency of the hull: its plan drawn with the hull weight
            hp = np.where(coin.random(len(test)) < t, pb, pa)
            hs = np.array([M["c"].at[qq, p] for qq, p in zip(ql, hp)])
            if sd == 0:                                    # one seed is enough for the latency distribution
                rows_l += [{"plan_set": tag, "reader": rd, "method": name, "seed": sd, "dataset": ds, "policy_s": a1,
                            "hull_s": b1} for a1, b1 in zip(cr, hs)]
    for m_ in K.MODS:
        m_._CUR_SEED[0] = 0


# ====================================================================== part trainonly
def set_costs(B, M, sel):
    B.ret_cost = M["rt"].loc[sel].mean()
    B.agent_cost = M["at"].loc[sel].mean()
    B.cost = B.ret_cost + B.agent_cost             # new objects, so the hull memo cannot return stale vertices


def part_trainonly():
    say("\n== plan costs from the training folds only (policy and hull alike)")
    rows = []
    t0 = time.time()
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        tag = "with two hop" if r3 else "single step"
        blocks, mts = {}, {}
        for rd in readers():
            df, _ = A.load_reader(rd)
            t5 = R.T5Router(rd)
            for ds in datasets(df):
                B = A.Block(df, ds)
                blocks[(rd, ds)], mts[(rd, ds)] = B, mats(df, B)
                for name, mk in makers_for(B, df, ds, t5).items():
                    if name == "ar_t5":       # its rungs change with the training costs; FLAN-T5 is not refitted here
                        continue
                    try:
                        rows += run_trainonly(tag, rd, ds, name, B, mk, [(B, mts[(rd, ds)])])
                    except K.T5Miss:
                        say(f"    {tag} {rd} {ds} ar_t5: rungs change with training costs, no cached FLAN-T5 "
                            f"prediction; skipped")
                print(f"    trainonly {tag} {rd} {ds} done ({time.time() - t0:.0f}s)", flush=True)
        for a_, b_ in ([] if ONLY and "reader_cascade" not in ONLY else A.CONFIG["reader_pairs"]):
            for ds in sorted({d for (r, d) in blocks if r == a_} & {d for (r, d) in blocks if r == b_}):
                if A.reader_cascade(blocks, a_, b_, ds) is None:
                    continue
                rows += run_trainonly(tag, f"{a_}->{b_}", ds, "reader_cascade", None,
                                      lambda: A.reader_cascade(blocks, a_, b_, ds),
                                      [(blocks[(a_, ds)], mts[(a_, ds)]), (blocks[(b_, ds)], mts[(b_, ds)])], cascade=True)
    A.CONFIG["include_r3"] = False
    Fo = pd.DataFrame(rows)
    Fo.to_csv(OUT / "trainonly_folds.csv", index=False)
    ref = pd.read_csv(K.OUT / "tests_nb.csv") if (K.OUT / "tests_nb.csv").exists() else None
    out = []
    for (tag, rd, m), g in Fo.groupby(["plan_set", "reader", "method"]):
        cells = [K.nb_cell(x.gain, x.n_test.mean(), x.n_train.mean()) for _, x in g.groupby("dataset")]
        out.append({"plan_set": tag, "reader": rd, "method": m, "family": K.FAMILY[m], **K.nb_row(*K.nb_pooled(cells))})
    T = K.add_holm(pd.DataFrame(out), ["plan_set", "family"])
    if ref is not None:
        T = T.merge(ref[["plan_set", "reader", "method", "gain"]].rename(columns={"gain": "gain_paper"}),
                    on=["plan_set", "reader", "method"], how="left")
    T.to_csv(OUT / "trainonly_tests.csv", index=False)
    for (tag, fam), g in T.groupby(["plan_set", "family"], sort=False):
        pos = g[(g.p_nb_holm < 0.05) & (g.nb_low > 0)]
        d = (g.gain - g.gain_paper) if "gain_paper" in g else g.gain * np.nan
        say(f"  {tag:12s} {fam:9s} {len(g):2d} tests: gain change median {d.median():+.4f} (range {d.min():+.4f} to "
            f"{d.max():+.4f}); positive {len(pos)}"
            + (f" ({', '.join(f'{r.reader} {r.method} {r.gain:+.3f}' for r in pos.itertuples())})" if len(pos) else "")
            + f"; upper bound below 0.03: {int((g.ub95_nb < 0.03).sum())}")


def run_trainonly(tag, rd, ds, name, B, mk, bm, cascade=False):
    k = A.CONFIG["k_folds"]
    base = bm[0][0]
    orig = [(b, b.ret_cost, b.agent_cost, b.cost) for b, _ in bm]
    rows = []
    try:
        for sd in A.CONFIG["seeds"]:
            for m_ in K.MODS:
                m_._CUR_SEED[0] = sd
            rng = np.random.default_rng(sd)
            q = (np.intersect1d(bm[0][0].q, bm[1][0].q) if cascade else base.q).copy()
            rng.shuffle(q)
            folds = [q[i::k] for i in range(k)]
            for i in range(k):
                test = folds[i]
                sel = np.concatenate([folds[j] for j in range(k) if j != i])
                for b, M in bm:
                    set_costs(b, M, sel)
                if cascade:
                    Bs, meth = mk()
                else:
                    Bs, meth = B, mk()
                    if meth is None:
                        return []
                fit, apply_ = meth
                cfg = fit(sel)
                s, c = apply_(cfg, test)
                h = A.hull_at(Bs.Y, Bs.cost, sel, c, eval_idx=test)
                rows.append({"plan_set": tag, "reader": rd, "dataset": ds, "method": name, "seed": sd, "fold": i,
                             "gain": float(np.mean(s)) - h, "cost": c, "n_test": len(test), "n_train": len(sel)})
    finally:
        for b, rc, ac, cc in orig:
            b.ret_cost, b.agent_cost, b.cost = rc, ac, cc
        for m_ in K.MODS:
            m_._CUR_SEED[0] = 0
    return rows


# ====================================================================== part sweep
def part_sweep():
    say(f"\n== budget sweep (single step plans, seeds {SWEEP_SEEDS}): each policy tuned on the training folds to the "
        f"best training success within the budget, then scored against the held out hull at its realised mean cost")
    A.CONFIG["include_r3"] = False
    names = [m for m in arg("--sweep-methods", "planner_tsr,gate_closed_book,conf_cascade,plan_emb,reader_cascade")
             .split(",") if m]
    M26 = _load("mixed26", "26_mixed_workload.py") if "plan_emb_own" in names else None
    mods = K.MODS + ([M26.A] if M26 is not None else [])     # every copy of 11 whose hull_at or seed a policy uses
    orig_hull = [m_.hull_at for m_ in mods]
    BUDGET = [None]

    def penal(Y, cost, idx, c, eval_idx=None):
        return 1e3 * max(0.0, c - BUDGET[0])

    COSTS = []

    def tuned(meth):
        fit, apply_ = meth

        def apply2(cfg, idx):
            sres, c = apply_(cfg, idx)
            COSTS.append(c)
            return sres, c

        def fit2(sel):
            for m_ in mods:
                m_.hull_at = penal
            try:
                return fit(sel)
            finally:
                for m_, h_ in zip(mods, orig_hull):
                    m_.hull_at = h_
        return fit2, apply2

    rows = []
    t0 = time.time()
    blocks = {}
    for rd in readers():
        df, _ = A.load_reader(rd)
        for ds in datasets(df):
            B = A.Block(df, ds)
            blocks[(rd, ds)] = B
            mk = makers_for(B, df, ds, None)
            if M26 is not None:
                E26 = R.load_emb(ds).reindex([str(q) for q in B.q])
                E26.index = B.q
                mk["plan_emb_own"] = (lambda B=B, E26=E26: M26.m_plan_own(B, E26, "emb", 0.0246))
            acc = B.Y.mean()
            grid = np.linspace(float(B.cost.min()), float(B.cost[acc.idxmax()]), 6)
            Hfull = A._hull_vertices(B.Y, B.cost, B.q)
            for b in grid:
                BUDGET[0] = b
                xs = [h[0] for h in Hfull]
                j = min(max(int(np.searchsorted(xs, b, side="right")) - 1, 0), len(Hfull) - 2)
                slope = (Hfull[j + 1][1] - Hfull[j][1]) / (Hfull[j + 1][0] - Hfull[j][0]) if len(Hfull) > 1 else 0.0
                for name in [n for n in names if n != "reader_cascade"]:
                    for sd in SWEEP_SEEDS:
                        for m_ in mods:
                            m_._CUR_SEED[0] = sd
                        meth = mk[name]()
                        if meth is None:
                            continue
                        COSTS.clear()
                        d, fr = K.heldout_folds(B, tuned(meth), sd)
                        rows += [{"reader": rd, "dataset": ds, "method": name, "budget_s": b, "hull_slope": slope,
                                  "cost": cc, **f} for f, cc in zip(fr, COSTS)]
            print(f"    sweep {rd} {ds} done ({time.time() - t0:.0f}s)", flush=True)
    for a_, b_ in (A.CONFIG["reader_pairs"][:1] if "reader_cascade" in names else []):
        for ds in sorted({d for (r, d) in blocks if r == a_} & {d for (r, d) in blocks if r == b_}):
            res0 = A.reader_cascade(blocks, a_, b_, ds)
            if res0 is None:
                continue
            Bj = res0[0]
            grid = np.linspace(float(Bj.cost.min()), float(Bj.cost[Bj.Y.mean().idxmax()]), 6)
            for b in grid:
                BUDGET[0] = b
                for sd in SWEEP_SEEDS:
                    for m_ in mods:
                        m_._CUR_SEED[0] = sd
                    COSTS.clear()
                    d, fr = K.heldout_folds(Bj, tuned(A.reader_cascade(blocks, a_, b_, ds)[1]), sd)
                    rows += [{"reader": f"{a_}->{b_}", "dataset": ds, "method": "reader_cascade", "budget_s": b,
                              "hull_slope": np.nan, "cost": cc, **f} for f, cc in zip(fr, COSTS)]
    for m_ in mods:
        m_._CUR_SEED[0] = 0
    S = pd.DataFrame(rows)
    S["budget_step"] = S.groupby(["reader", "dataset", "method"]).budget_s.rank(method="dense").astype(int)
    sfx = arg("--sweep-tag", "")
    S.to_csv(OUT / f"sweep_folds{sfx}.csv", index=False)
    out = []
    for (rd, m, st), g in S.groupby(["reader", "method", "budget_step"]):
        cells = [K.nb_cell(x.gain, x.n_test.mean(), x.n_train.mean()) for _, x in g.groupby("dataset")]
        r = K.nb_row(*K.nb_pooled(cells))
        out.append({"reader": rd, "method": m, "budget_step": st, "budget_s": g.budget_s.mean(), "cost": g.cost.mean(),
                    "hull_slope": g.hull_slope.mean(), **r})
    T = pd.DataFrame(out)
    T["cost_over_budget"] = T.cost / T.budget_s
    T.to_csv(OUT / f"sweep_tests{sfx}.csv", index=False)
    say("  mean gain over readers by budget step (1 = cheapest plan, 6 = most successful plan); corrected 95% "
        "intervals per test in sweep_tests.csv")
    say(T.pivot_table(index="method", columns="budget_step", values="gain", aggfunc="mean").round(4).to_string())
    say("  tests with the corrected interval above 0 (no correction for the 6 budgets; descriptive):")
    pos = T[T.nb_low > 0]
    say(pos[["reader", "method", "budget_step", "budget_s", "gain", "nb_low", "nb_high"]].round(4).to_string(index=False)
        if len(pos) else "    none")


# ====================================================================== part agreement
def part_agreement():
    say("\n== label_final against the human majority of audits 1 and 2, by reader")
    G = _load("grading07", "07_grading_full.py")
    H = pd.read_csv(settings.PROCESSED_DATA_ROOT / "grading" / "audit_adjudicated.csv", dtype={"query_id": str},
                    keep_default_na=False)
    lab = {}
    for rd in A.readers():
        d = pd.read_csv(A.PROC / f"tsr_labels_{rd}_graded.csv", dtype={"query_id": str}, keep_default_na=False)
        d["pair"] = d.dataset + "::" + d.query_id + "::" + d.answer.map(G.normalize)
        lab[rd] = d.drop_duplicates("pair").set_index("pair").label_final.astype(int)
    rows = []
    for _, r in H.iterrows():
        for rd in str(r.reader).split(","):
            rd = rd.strip()
            if rd in lab and r.pair in lab[rd].index:
                rows.append({"audit": r.audit, "reader": rd, "human": int(r.human), "final": int(lab[rd][r.pair])})
    X = pd.DataFrame(rows)
    out = []
    for rd, g in X.groupby("reader"):
        y, f = g.human.to_numpy(), g.final.to_numpy()
        sens = ((y == 1) & (f == 1)).sum() / max(1, (y == 1).sum())
        spec = ((y == 0) & (f == 0)).sum() / max(1, (y == 0).sum())
        out.append({"reader": rd, "audit": ",".join(sorted(g.audit.unique())), "n": len(g), "agreement": (y == f).mean(),
                    "sensitivity": sens, "specificity": spec, "J": sens + spec - 1, "human_correct": y.mean(),
                    "final_correct": f.mean()})
    T = pd.DataFrame(out)
    T.to_csv(OUT / "agreement_by_reader.csv", index=False)
    say(T.round(3).to_string(index=False))
    say("  Note: the audits sampled strata of unique answers, not each reader at random; the Qwen3 judge and the "
        "Qwen3 readers can be compared here, but the sample per reader is small.")


def main():
    t0 = time.time()
    say(f"Policy checks; parts {PARTS}; seeds {A.CONFIG['seeds']}" + ("; QUICK TEST RUN" if QUICK else ""))
    for p in PARTS:
        {"main": part_main, "trainonly": part_trainonly, "sweep": part_sweep, "agreement": part_agreement}[p]()
        say(f"  ({p} done, {time.time() - t0:.0f}s)")
        (OUT / f"summary_{'_'.join(PARTS)}.txt").write_text("\n".join(LINES), encoding="utf-8")
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
