#!/usr/bin/env python
"""
Checks fixed in the amendment of 2026-10-13 20:45, items (b), (e) and (f). CPU only, no new answers.

  (b) sanity     Llama 3.1 8B "contains" accuracy (closed book, BM25 at k = 1, 3, 5, 10) next to the published
                 Never RAG and Always RAG In-Accuracy of Moskvoretskii et al. (ACL 2025, Llama 3.1 8B, BM25, 500
                 questions per dataset). Adaptive RAG against Always RAG from the R6 output (weak_protocol).
  (e) grading    label_final against the majority of three raters in each audit: sensitivity, specificity,
                 J = sens + spec - 1 (raw, and reweighted to the population strata); rater agreement (raw,
                 Cohen kappa, Gwet AC1, Krippendorff alpha). Per arm grading exposure: share of judge decided
                 answers and of the contains_long stratum, per plan (spread across plans bounds any H1 policy),
                 and for IRCoT against its hull plan (H3).
      headroom   Held out oracles (2 plans, 3 plans, all plans; chosen on the training folds as in 15) at their
                 own mean cost, against the hull at that SAME cost: oracle gap = oracle - hull(cost). Fraction of
                 that gap each policy closes (pooled gain / mean gap). All plan oracle under label_strict against
                 label_final (headroom the judge creates).
  (f) mechanism  planner_tsr and planner_sk (28): per plan AUROC, mean correlation of predicted success across
                 plans (difficulty), AUROC on which of the two bracketing hull plans wins where they disagree,
                 share of queries sent to the most used plan; per arm grading exposure for these two policies.
                 Seed 0 folds only (descriptive).

Usage:  python scripts/29_checks.py            (all parts; about 10 to 20 minutes)
        python scripts/29_checks.py sanity grading headroom mechanism   (any subset)
Output: results_full/analysis/checks/  *.csv and checks_summary.txt
"""

import importlib.util
import os
import re
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
from tsr.grader import normalize


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


A = _load("analysis10", "10_analysis_full.py")
PROC = settings.PROCESSED_DATA_ROOT
GDIR = PROC / "grading"
AN = settings.RESULTS_ROOT / "analysis"
OUT = AN / "checks"
L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def long_contains(d):
    return (d["em"].astype(int) == 0) & (d["contains"].astype(int) == 1) & (d["n_answer_tokens"].astype(float) >= 8)


def graded(reader):
    d = pd.read_csv(PROC / f"tsr_labels_{reader}_graded.csv", dtype={"query_id": str}, keep_default_na=False)
    d = d[d["repeat"].astype(int) == 0].copy()
    d["plan"] = d["retriever"] + "@" + d["depth"].astype(int).astype(str)
    return d


# ====================================================================== (b)
# Moskvoretskii et al., ACL 2025 (2025.acl-long.319), Llama 3.1 8B, BM25, In-Accuracy (answer contained).
# Values checked against Table 1 of the paper on 13 Oct 2026 (BM25, Karpukhin et al. Wikipedia; depth not stated).
PUBLISHED = {"nq": (0.446, 0.496), "hotpotqa": (0.286, 0.410), "2wikimultihopqa": (0.318, 0.374),
             "musique": (0.106, 0.100)}


def part_sanity():
    say("== (b) Sanity check: Llama 3.1 8B 'contains' accuracy against Moskvoretskii et al. (ACL 2025)")
    say("   published: Never RAG / Always RAG In-Accuracy (Llama 3.1 8B, BM25; their retrieval depth: check the paper)")
    d = graded("llama3.1-8b")
    rows = []
    for ds, (nv, al) in PUBLISHED.items():
        g = d[d.dataset == ds]
        r = {"dataset": ds, "published_never": nv, "published_always": al,
             "ours_closed_book": g[g.plan == "closed_book@0"]["contains"].astype(int).mean()}
        for k in (1, 3, 5, 10):
            r[f"ours_bm25@{k}"] = g[g.plan == f"bm25@{k}"]["contains"].astype(int).mean()
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "sanity_llama.csv", index=False)
    say(S.round(3).to_string(index=False))
    say("   A gap above about 0.05 needs an explanation (corpus, depth, prompt).")
    W = pd.read_csv(AN / "weak_protocol" / "weak_vs_ours.csv")
    a = W[W.method == "adaptive_rag"]
    say(f"\n   Adaptive RAG against Always RAG under the weak protocol (R6, strict EM, one seed): mean margin "
        f"{a.margin_vs_always.mean():+.3f}; {int((a.margin_vs_always > 0).sum())} of {len(a)} reader x dataset "
        f"cells above Always RAG, {int((a.margin_vs_always < 0).sum())} below")
    say(a.groupby("reader").margin_vs_always.mean().round(3).to_string())
    say("   Reading: if Adaptive RAG roughly ties Always RAG, its loss against the hull comes from the stronger "
        "comparator, not from a broken implementation.")


# ====================================================================== (e) grading
def agreement(x, y):
    x, y = np.asarray(x, int), np.asarray(y, int)
    n = len(x)
    pa = (x == y).mean()
    p1, p2 = x.mean(), y.mean()
    pe_k = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (pa - pe_k) / (1 - pe_k) if pe_k < 1 else np.nan
    pi = (p1 + p2) / 2
    pe_g = 2 * pi * (1 - pi)
    ac1 = (pa - pe_g) / (1 - pe_g)
    dis = int((x != y).sum())
    nv = 2 * n
    n1 = int(x.sum() + y.sum())
    n0 = nv - n1
    alpha = 1 - dis * (nv - 1) / (n0 * n1) if n0 and n1 else np.nan
    return {"n": n, "raw_agreement": pa, "cohen_kappa": kappa, "gwet_ac1": ac1, "krippendorff_alpha": alpha,
            "rate_rater1": p1, "rate_rater2": p2}


STRATA = {  # first matching stratum in this order (08 and 18 sampled in the same order)
    "audit1": [("f1_02_04", lambda d: (d.strict == 0) & (d.f1 >= 0.2) & (d.f1 < 0.4)),
               ("f1_04_07", lambda d: (d.strict == 0) & (d.f1 >= 0.4) & (d.f1 < 0.7)),
               ("f1_07_up", lambda d: (d.strict == 0) & (d.f1 >= 0.7)),
               ("contains_long", lambda d: (d.em == 0) & (d.contains == 1) & (d.n_answer_tokens >= 8)),
               ("strict_short", lambda d: (d.strict == 1) & (d.n_answer_tokens < 8)),
               ("f1_below_02", lambda d: (d.strict == 0) & (d.f1 < 0.2))],
    "audit2": [("long_low_f1", lambda d: (d.strict == 0) & (d.f1 < 0.2) & (d.n_answer_tokens >= 8)),
               ("f1_02_04", lambda d: (d.strict == 0) & (d.f1 >= 0.2) & (d.f1 < 0.4)),
               ("f1_04_07", lambda d: (d.strict == 0) & (d.f1 >= 0.4) & (d.f1 < 0.7)),
               ("f1_07_up", lambda d: (d.strict == 0) & (d.f1 >= 0.7)),
               ("contains_long", lambda d: (d.em == 0) & (d.contains == 1) & (d.n_answer_tokens >= 8)),
               ("strict_short", lambda d: (d.strict == 1) & (d.n_answer_tokens < 8)),
               ("short_low_f1", lambda d: (d.strict == 0) & (d.f1 < 0.2) & (d.n_answer_tokens < 8))]}
AUDIT_READERS = {"audit1": ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"], "audit2": ["qwen3-32b", "gemma3-27b"]}


def part_grading():
    say("\n== (e) Grading: label_final against the human majority, rater agreement, per arm exposure")
    H = pd.read_csv(GDIR / "audit_adjudicated.csv", dtype={"query_id": str}, keep_default_na=False)
    rows, agr = [], []
    for audit, readers in AUDIT_READERS.items():
        frames = []
        for r in readers:
            d = graded(r)
            d["pair"] = d.dataset + "::" + d.query_id + "::" + d.answer.map(normalize)
            frames.append(d)
        P = pd.concat(frames).drop_duplicates("pair")
        for c in ("strict", "em", "contains"):
            P[c] = P[c].astype(int)
        for c in ("f1", "n_answer_tokens"):
            P[c] = pd.to_numeric(P[c], errors="coerce")
        lab = P.set_index("pair")["label_final"].astype(int)
        h = H[H.audit == audit].copy()
        h["final"] = h["pair"].map(lab)
        h = h.dropna(subset=["final"])
        y, f = h["human"].astype(int).to_numpy(), h["final"].astype(int).to_numpy()
        tp, fn = int(((y == 1) & (f == 1)).sum()), int(((y == 1) & (f == 0)).sum())
        tn, fp = int(((y == 0) & (f == 0)).sum()), int(((y == 0) & (f == 1)).sum())
        sens, spec = tp / max(1, tp + fn), tn / max(1, tn + fp)
        # population weights: unique answers in each stratum (first match), over the same readers' grids
        P["stratum"] = None
        for name, cond in STRATA[audit]:
            m = P["stratum"].isna() & cond(P)
            P.loc[m, "stratum"] = name
        key = pd.read_csv(GDIR / ("audit_key.csv" if audit == "audit1" else "audit2_key.csv"))
        h = h.merge(key[["pair", "stratum"]], on="pair", how="left")
        pop = P["stratum"].value_counts()
        covered = P["stratum"].notna().mean()
        w = h["stratum"].map(lambda s: pop.get(s, 0) / max(1, (h["stratum"] == s).sum()))
        yw, fw = h["human"].astype(int), h["final"].astype(int)
        sens_w = (w * ((yw == 1) & (fw == 1))).sum() / max(1e-9, (w * (yw == 1)).sum())
        spec_w = (w * ((yw == 0) & (fw == 0))).sum() / max(1e-9, (w * (yw == 0)).sum())
        rows.append({"audit": audit, "n": len(h), "tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": sens,
                     "specificity": spec, "J": sens + spec - 1, "sensitivity_weighted": sens_w,
                     "specificity_weighted": spec_w, "J_weighted": sens_w + spec_w - 1,
                     "population_share_covered_by_strata": covered})
        a = H[H.audit == audit]
        agr.append({"audit": audit, **agreement(a["rater1"].astype(int), a["rater2"].astype(int))})
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "grading_J.csv", index=False)
    say(G.round(3).to_string(index=False))
    say("   J is the factor by which non differential grading error shrinks a true difference "
        "(observed = J x true). The weighted J reweights audit strata to the population of unique answers in "
        "those strata; answers outside every stratum are not covered (share shown).")
    Ag = pd.DataFrame(agr)
    Ag.to_csv(OUT / "rater_agreement.csv", index=False)
    say("\n   Rater 1 (rater 1) against rater 2 (rater 2), before adjudication:")
    say(Ag.round(3).to_string(index=False))

    # ---- per plan exposure (H1 bound)
    exp = []
    for r in A.readers():
        d = graded(r)
        d["judge"] = (d["label_final_source"] == "judge").astype(int)
        d["clong"] = long_contains(d).astype(int)
        d["ntok"] = pd.to_numeric(d["n_answer_tokens"], errors="coerce")
        g = d.groupby(["dataset", "plan"]).agg(judge_share=("judge", "mean"), contains_long_share=("clong", "mean"),
                                              mean_answer_tokens=("ntok", "mean")).reset_index()
        g["reader"] = r
        exp.append(g)
    E = pd.concat(exp, ignore_index=True)
    E.to_csv(OUT / "exposure_per_plan.csv", index=False)
    sp = E.groupby(["reader", "dataset"]).agg(judge_min=("judge_share", "min"), judge_max=("judge_share", "max"),
                                              clong_min=("contains_long_share", "min"),
                                              clong_max=("contains_long_share", "max")).reset_index()
    sp["judge_spread"] = sp.judge_max - sp.judge_min
    sp["clong_spread"] = sp.clong_max - sp.clong_min
    say("\n   Per plan exposure to the judge and to the contains_long stratum (all plans of a reader and dataset).")
    say("   Any H1 policy and its hull mix both use these plans, so the spread bounds the arm difference:")
    say(sp[["reader", "dataset", "judge_min", "judge_max", "clong_min", "clong_max"]].round(3).to_string(index=False))

    # ---- H3: IRCoT against the hull plan at IRCoT's cost (the best fixed plan; IRCoT costs more than any plan)
    h3 = pd.read_csv(AN / "h3_ircot" / "h3.csv")
    h3 = h3[h3.hull == "single_step"]
    rows3 = []
    for _, r in h3.iterrows():
        f = PROC / "r4_ircot" / f"labels_{r.reader}_graded.csv"
        if not f.exists():
            continue
        ir = pd.read_csv(f, dtype={"query_id": str}, keep_default_na=False)
        ir = ir[ir.dataset == r.dataset]
        base = graded(r.reader)
        base = base[(base.dataset == r.dataset) & (base.plan == r.best_fixed_plan)]
        rows3.append({"reader": r.reader, "dataset": r.dataset, "gain": r.gain, "p_holm": r.p_holm,
                      "ircot_judge": (ir.label_final_source == "judge").mean(),
                      "plan_judge": (base.label_final_source == "judge").mean(),
                      "ircot_contains_long": long_contains(ir).mean(), "plan_contains_long": long_contains(base).mean(),
                      "ircot_tokens": pd.to_numeric(ir.n_answer_tokens, errors="coerce").mean(),
                      "plan_tokens": pd.to_numeric(base.n_answer_tokens, errors="coerce").mean(),
                      "hull_plan": r.best_fixed_plan})
    H3 = pd.DataFrame(rows3)
    H3.to_csv(OUT / "exposure_h3.csv", index=False)
    say("\n   H3 arms: IRCoT against its hull plan (the best fixed plan). Differential exposure is the risk here:")
    say(H3.round(3).to_string(index=False))
    say("   If IRCoT sends more answers to the judge or writes more long answers, the strict judge penalises IRCoT, "
        "so a significant H3 gain is conservative; the reverse would inflate it.")


# ====================================================================== headroom
def greedy_union(Ytr, k):
    chosen, cover = [], np.zeros(len(Ytr), bool)
    for _ in range(k):
        gain = [(np.logical_or(cover, Ytr[:, j]).mean(), j) for j in range(Ytr.shape[1]) if j not in chosen]
        _, j = max(gain)
        chosen.append(j)
        cover |= Ytr[:, j].astype(bool)
    return chosen


def oracle_on(Yte, cost, idx):
    sub, c = Yte[:, idx], cost[idx]
    ok = sub.max(1) == 1
    cc = np.where(sub == 1, c[None, :], np.inf).min(1)
    return ok.mean(), float(np.where(ok, cc, c.min()).mean())


def part_headroom():
    say("\n== Headroom at matched cost (held out; oracle plans chosen on the training folds; 5 folds x 5 seeds)")
    rows = []
    for rd in A.readers():
        df, _ = A.load_reader(rd)
        strict = graded(rd)
        for ds in sorted(df.dataset.unique()):
            B = A.Block(df, ds)
            Yn, cost = B.Yn, B.cost[B.Y.columns].to_numpy()
            Ys = strict[strict.dataset == ds].pivot_table(index="query_id", columns="plan", values="label_strict",
                                                          aggfunc="first").reindex(index=B.Y.index,
                                                                                   columns=B.Y.columns).astype(float)
            acc = {k: [] for k in ("k2", "k3", "all")}
            gap = {k: [] for k in acc}
            for sd in A.CONFIG["seeds"]:
                q = B.q.copy()
                np.random.default_rng(sd).shuffle(q)
                folds = [q[i::A.CONFIG["k_folds"]] for i in range(A.CONFIG["k_folds"])]
                for i, te in enumerate(folds):
                    tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
                    ri, ti = B.rows(tr), B.rows(te)
                    for k, K in (("k2", 2), ("k3", 3), ("all", None)):
                        idx = list(range(Yn.shape[1])) if K is None else greedy_union(Yn[ri], K)
                        s, c = oracle_on(Yn[ti], cost, idx)
                        acc[k].append(s)
                        gap[k].append(s - A.hull_at(B.Y, B.cost, tr, c, eval_idx=te))
            row = {"reader": rd, "dataset": ds}
            for k in acc:
                row[f"oracle_{k}"] = np.mean(acc[k])
                row[f"gap_{k}"] = np.mean(gap[k])
            row["oracle_all_final"] = float((B.Yn.max(1) == 1).mean())
            row["oracle_all_strict"] = float((np.nan_to_num(Ys.to_numpy()).max(1) == 1).mean())
            rows.append(row)
    Hd = pd.DataFrame(rows)
    Hd.to_csv(OUT / "headroom_matched.csv", index=False)
    say(Hd.round(3).to_string(index=False))
    say(f"   Mean oracle gap over the hull at matched cost: 2 plans {Hd.gap_k2.mean():+.3f}, 3 plans "
        f"{Hd.gap_k3.mean():+.3f}, all plans {Hd.gap_all.mean():+.3f}")
    say(f"   All plan oracle (any plan correct), label_final {Hd.oracle_all_final.mean():.3f} against label_strict "
        f"{Hd.oracle_all_strict.mean():.3f}: the judge adds {Hd.oracle_all_final.mean() - Hd.oracle_all_strict.mean():+.3f}")
    M = pd.read_csv(AN / "final" / "methods.csv")
    M = M[M.dataset != "pooled"]
    pr = M.groupby(["reader", "method"]).gain.mean().reset_index()
    gp = Hd.groupby("reader")[["gap_k2", "gap_k3", "gap_all"]].mean()
    pr = pr[pr.reader.isin(gp.index)]
    for k in ("k2", "k3", "all"):
        pr[f"fraction_of_gap_{k}"] = pr.apply(lambda r: r.gain / gp.loc[r.reader, f"gap_{k}"], axis=1)
    pr.to_csv(OUT / "fraction_of_gap.csv", index=False)
    say("\n   Fraction of the matched cost oracle gap each policy closes (pooled gain / mean gap; negative = loses):")
    say(pr.pivot(index="method", columns="reader", values="fraction_of_gap_k2").round(2).to_string())


# ====================================================================== (f) mechanism
def part_mechanism():
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    S28 = _load("stronger28", "28_stronger_policies.py")
    say("\n== (f) Mechanism: what the planners learn (seed 0 folds, single step plans, descriptive)")
    A.CONFIG["include_r3"] = False
    rows, expo = [], []
    for rd in A.readers():
        df, _ = A.load_reader(rd)
        gd = graded(rd)
        gd["judge"] = (gd.label_final_source == "judge").astype(int)
        gd["clong"] = long_contains(gd).astype(int)
        for ds in sorted(df.dataset.unique()):
            B = A.Block(df, ds)
            F = A.decision_features(B)
            X = S28.cb_features(df, ds, B.q)
            plans = B.plans
            J = gd[gd.dataset == ds].pivot_table(index="query_id", columns="plan", values="judge", aggfunc="first")
            C = gd[gd.dataset == ds].pivot_table(index="query_id", columns="plan", values="clong", aggfunc="first")
            for name, FF, maker in (("planner_tsr", F, lambda: A.m_planner_tsr(B, F)),
                                    ("planner_sk", F.join(X.add_prefix("cb_")),
                                     lambda: S28.m_planner_sk(B, F.join(X.add_prefix("cb_"))))):
                A._CUR_SEED[0] = 0
                fit, apply_ = maker()
                q = B.q.copy()
                np.random.default_rng(0).shuffle(q)
                folds = [q[i::5] for i in range(5)]
                Pall, Yall, pick_all, pair_rows, arm = [], [], [], [], []
                for i, te in enumerate(folds):
                    tr = np.concatenate([folds[j] for j in range(5) if j != i])
                    bst, prior, lam = fit(tr)
                    base = np.tile(np.log(prior / (1 - prior)).to_numpy(), len(te))
                    Xq = FF.loc[te].to_numpy(np.float32)
                    des = np.concatenate([np.repeat(Xq, len(plans), 0), np.tile(np.eye(len(plans)), (len(te), 1))], 1)
                    p = bst.predict(xgb.DMatrix(des, base_margin=base)).reshape(len(te), len(plans))
                    pick = (p - lam * B.cost[plans].to_numpy()[None, :]).argmax(1)
                    Yte = B.Y.loc[te, plans].to_numpy()
                    Pall.append(p); Yall.append(Yte); pick_all.append(pick)
                    _, c = apply_((bst, prior, lam), te)
                    H = A._hull_vertices(B.Y, B.cost, tr)
                    xs = [h[0] for h in H]
                    j = int(np.clip(np.searchsorted(xs, c), 1, len(H) - 1)) if len(H) > 1 else 0
                    a_, b_ = (H[j - 1][2], H[j][2]) if len(H) > 1 else (H[0][2], H[0][2])
                    wb = 0.0 if len(H) == 1 else float(np.clip((c - xs[j - 1]) / max(1e-9, xs[j] - xs[j - 1]), 0, 1))
                    ia, ib = plans.index(a_), plans.index(b_)
                    dis = Yte[:, ia] != Yte[:, ib]
                    for t in np.where(dis)[0]:
                        pair_rows.append((p[t, ia] - p[t, ib], Yte[t, ia]))
                    chosen = [plans[k] for k in pick]
                    arm.append({"policy_judge": np.mean([J.at[qq, pl] for qq, pl in zip(te, chosen)]),
                                "hull_judge": (1 - wb) * J.loc[te, a_].mean() + wb * J.loc[te, b_].mean(),
                                "policy_clong": np.mean([C.at[qq, pl] for qq, pl in zip(te, chosen)]),
                                "hull_clong": (1 - wb) * C.loc[te, a_].mean() + wb * C.loc[te, b_].mean()})
                P_, Y_, pk = np.vstack(Pall), np.vstack(Yall), np.concatenate(pick_all)
                aucs = [roc_auc_score(Y_[:, k], P_[:, k]) for k in range(len(plans)) if 0 < Y_[:, k].mean() < 1]
                cor = np.corrcoef(P_.T)
                off = cor[~np.eye(len(plans), dtype=bool)]
                pr = np.array(pair_rows)
                pair_auc = roc_auc_score(pr[:, 1], pr[:, 0]) if len(pr) > 10 and 0 < pr[:, 1].mean() < 1 else np.nan
                rows.append({"reader": rd, "dataset": ds, "policy": name, "per_plan_auroc": np.mean(aucs),
                             "mean_cross_plan_corr": np.nanmean(off), "pair_auroc": pair_auc, "n_discordant": len(pr),
                             "modal_plan_share": np.bincount(pk).max() / len(pk)})
                ar = pd.DataFrame(arm).mean()
                expo.append({"reader": rd, "dataset": ds, "policy": name, **ar.to_dict()})
    Mx = pd.DataFrame(rows)
    Mx.to_csv(OUT / "mechanism.csv", index=False)
    Ex = pd.DataFrame(expo)
    Ex.to_csv(OUT / "exposure_policy_vs_hull.csv", index=False)
    say(Mx.groupby("policy")[["per_plan_auroc", "mean_cross_plan_corr", "pair_auroc", "modal_plan_share"]]
        .mean().round(3).to_string())
    say("   Reading: high per plan AUROC with high cross plan correlation and a pair AUROC near 0.5 means the "
        "planner learns which queries are hard, not which plan wins.")
    say("\n   Grading exposure, policy choices against the hull mix (mean over blocks):")
    say(Ex.groupby("policy")[["policy_judge", "hull_judge", "policy_clong", "hull_clong"]].mean().round(3).to_string())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [a for a in sys.argv[1:] if not a.startswith("-")] or ["sanity", "grading", "headroom", "mechanism"]
    fn = {"sanity": part_sanity, "grading": part_grading, "headroom": part_headroom, "mechanism": part_mechanism}
    t0 = time.time()
    say(f"Checks (amendment 13 Oct 20:45 (b), (e), (f)); label {A.CONFIG['label']}; parts {parts}")
    for p in parts:
        fn[p]()
    name = "checks_summary.txt" if len(parts) == 4 else f"checks_{'_'.join(parts)}.txt"
    (OUT / name).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.0f}s. Written {OUT / name}")


if __name__ == "__main__":
    main()
