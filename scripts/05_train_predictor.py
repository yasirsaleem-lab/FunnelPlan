#!/usr/bin/env python
"""
Train and evaluate the TSR predictor (v3).

Replaces the old single split / single dataset / rho only script. See
tsr/cv.py for the method. Writes:
  results/tsr/cv_predictions.csv   out of fold predictions per (query, plan)
  results/tsr/metrics.json         prediction metrics + tests against the prior
  results/tsr/budget_sweep.csv     planner success/cost per budget
  results/tsr/frontier.csv         utility frontier vs best fixed plan
  results/tsr/fixed_plans.csv      cost and success of every fixed plan
  results/tsr/cascade.csv          learned vs random vs ideal escalation
  results/tsr/summary.txt          readable summary
  tsr/models/funnel_predictor.joblib  final decision time model (all data)

Legacy mode: if tsr_labels_v3.csv / features_v3.csv do not exist yet, the
script runs on tsr_labels_v2.csv with its 21 stored features (mapped to p_ and
r_ columns) plus query lexical features. Use this only to reproduce the
diagnosis; the v2 reranker features were computed with the padding bug.
"""

import ast
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from tsr import cv
from tsr.features import feature_columns, query_lexical, standardize_by_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {
    "k_folds": 5,
    "seed": 0,
    "budgets_s": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    "lambdas": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 1.0],
    "cheap_plan": None,      # None -> cheapest plan by cost
    "strong_plan": None,     # None -> best plan by overall prior
    "datasets": ["nq", "hotpotqa"],
    # Which grading defines success. "adjudicated" = direct judgment for every
    # ambiguous answer (see 04d). For the robustness table, rerun with
    # "strict", "f1_07", "lenient".
    "label_rule": "adjudicated",
}

PROC = settings.PROCESSED_DATA_ROOT
OUT = settings.RESULTS_ROOT / "tsr"
MODEL_PATH = settings.PROJECT_ROOT / "tsr" / "models" / "funnel_predictor.joblib"
LEGACY_NAMES = ["p_legacy_log_depth", "p_legacy_num_results", "p_legacy_mean", "p_legacy_std",
                "p_legacy_min", "p_legacy_max", "p_legacy_median"] + \
               [f"_onehot_{i}" for i in range(8)] + \
               ["r_legacy_top1", "r_legacy_top5_mean", "r_legacy_top20_mean", "r_legacy_gap",
                "r_legacy_n_above", "r_legacy_rank_first"]


# ----------------------------------------------------------------------
def load_queries_jsonl(ds):
    path = PROC / ds / "queries.jsonl"
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            out[str(obj.get("id", i))] = obj.get("text", "")
    return out


def load_v3():
    graded = PROC / "tsr_labels_v3_graded.csv"
    if graded.exists():
        labels = pd.read_csv(graded, dtype={"query_id": str}, keep_default_na=False)
        rule = CONFIG["label_rule"]
        col = f"label_{rule}"
        if col not in labels.columns:
            fallback = next(c for c in ("label_adjudicated", "label_audited", "label_f1_07") if c in labels.columns)
            logger.warning(f"{col} not in {graded.name}; using {fallback}")
            col = fallback
        labels["label"] = labels[col].astype(int)
        logger.info(f"Labels: {graded.name}, column {col}")
    else:
        labels = pd.read_csv(PROC / settings.LABELS_FILE, dtype={"query_id": str})
        logger.warning("tsr_labels_v3_graded.csv not found; using the original lenient labels. "
                       "Run scripts/04d_regrade_labels.py first.")
    feats = pd.read_csv(PROC / settings.FEATURES_FILE, dtype={"query_id": str})
    df = cv.aggregate_repeats(labels).merge(feats, on=["dataset", "query_id", "retriever", "depth"], how="inner")
    logger.info(f"v3: {len(df)} rows after join ({labels['repeat'].nunique()} repeat(s))")
    return df


def load_legacy_v2():
    raw = pd.read_csv(PROC / "tsr_labels_v2.csv", dtype={"query_id": str})
    X = np.array([ast.literal_eval(s) for s in raw["features"]], dtype=float)
    feats = pd.DataFrame(X, columns=LEGACY_NAMES).drop(columns=[c for c in LEGACY_NAMES if c.startswith("_")])
    df = pd.concat([raw[["dataset", "query_id", "retriever", "depth", "label"]].reset_index(drop=True), feats], axis=1)
    df = cv.aggregate_repeats(df.assign(repeat=0)).merge(
        df.drop(columns=["label"]), on=["dataset", "query_id", "retriever", "depth"])
    df = standardize_by_retriever(df, [c for c in df.columns if c.startswith("p_legacy_")
                                       and c not in ("p_legacy_log_depth", "p_legacy_num_results")])
    df = df.drop(columns=["p_legacy_log_depth", "p_legacy_num_results"])
    qrows = []
    for ds in df["dataset"].unique():
        qt = load_queries_jsonl(ds)
        for qid in df.loc[df["dataset"] == ds, "query_id"].unique():
            if qid in qt:
                qrows.append({"dataset": ds, "query_id": qid, **query_lexical(qt[qid])})
    if qrows:
        df = df.merge(pd.DataFrame(qrows), on=["dataset", "query_id"], how="left")
    rets = sorted(df["retriever"].unique())
    for r in rets:
        df[f"plan_ret_{r}"] = (df["retriever"] == r).astype(float)
    df["plan_log_depth"] = np.log(df["depth"].astype(float))
    logger.warning("LEGACY MODE: v2 labels + v2 features (reranker features carry the padding bug).")
    return df


# ----------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    legacy = not ((PROC / settings.LABELS_FILE).exists() and (PROC / settings.FEATURES_FILE).exists())
    df = load_legacy_v2() if legacy else load_v3()
    df = cv.add_keys(df[df["dataset"].isin(CONFIG["datasets"])])
    df = cv.attach_cost(df)
    logger.info(f"{df['g'].nunique()} queries, {df['plan'].nunique()} plans, cost source: {df['cost_source'].iloc[0]}")

    cols = list(df.columns)
    feature_sets = {
        "decision": feature_columns(cols, ("q_", "plan_")),                 # usable by the planner
        "post": feature_columns(cols, ("q_", "plan_", "p_", "r_")),         # reference only
        "all": feature_columns(cols, ("q_", "plan_", "p_", "r_", "x_")),    # reference only
    }
    feature_sets = {k: v for k, v in feature_sets.items() if v}
    if not any(c.startswith("q_") for c in feature_sets.get("decision", [])):
        feature_sets.pop("decision", None)
    for k, v in feature_sets.items():
        logger.info(f"feature set {k}: {len(v)} columns")

    # ---------------- cross validation ----------------
    oof = cv.cross_validate(df, feature_sets, k=CONFIG["k_folds"], seed=CONFIG["seed"])
    pcols = ["p_prior"] + [f"p_{k}" for k in feature_sets]
    keep = ["dataset", "query_id", "retriever", "depth", "plan", "g", "fold", "label", "label_mean", "cost"] + pcols
    oof[keep].to_csv(OUT / "cv_predictions.csv", index=False)

    metrics = {"legacy_mode": legacy, "n_queries": int(df["g"].nunique()),
               "cost_source": df["cost_source"].iloc[0], "models": {}, "vs_prior": {}}
    for pc in pcols:
        metrics["models"][pc] = cv.prediction_metrics(oof, pc)
        if pc != "p_prior":
            metrics["vs_prior"][pc] = cv.paired_rho_test(oof, pc)
    per_ds = {}
    for ds, sub in oof.groupby("dataset"):
        per_ds[ds] = {pc: cv.prediction_metrics(sub, pc) for pc in pcols}
    metrics["per_dataset"] = per_ds

    # ---------------- planner simulation ----------------
    fixed = cv.fixed_plan_points(oof)
    fixed.to_csv(OUT / "fixed_plans.csv", index=False)
    sweep = cv.budget_sweep(oof, pcols, CONFIG["budgets_s"])
    sweep.to_csv(OUT / "budget_sweep.csv", index=False)
    frontier = pd.concat([cv.frontier_gain(cv.utility_frontier(oof, pc, CONFIG["lambdas"]), fixed)
                          for pc in pcols], ignore_index=True)
    frontier.to_csv(OUT / "frontier.csv", index=False)
    boot = []
    if "p_decision" in pcols:
        for B in CONFIG["budgets_s"]:
            if (oof["cost"] <= B).any():
                boot.append(cv.paired_bootstrap_gain(oof, "p_decision", "p_prior", B))
    metrics["decision_vs_best_fixed_by_budget"] = boot

    # ---------------- cascade ----------------
    prior_all = cv.fit_prior(oof)
    cheap = CONFIG["cheap_plan"] or fixed.iloc[0]["plan"]
    strong = CONFIG["strong_plan"] or max(prior_all, key=prior_all.get)
    casc_cols = feature_columns(cols, ("q_", "p_"))
    casc = cv.cascade_cv(oof, cheap, strong, casc_cols, k=CONFIG["k_folds"], seed=CONFIG["seed"])
    curve = cv.cascade_curve(casc)
    curve.to_csv(OUT / "cascade.csv", index=False)
    from sklearn.metrics import roc_auc_score
    yc = (casc["y_cheap"] >= 0.5).astype(int)
    metrics["cascade"] = {
        "cheap_plan": cheap, "strong_plan": strong, "n_features": len(casc_cols),
        "auc_p_cheap": float(roc_auc_score(yc, casc["p_cheap"])) if yc.nunique() > 1 else np.nan,
        "mean_gain_over_random": float((curve["success_learned"] - curve["success_random"]).iloc[1:-1].mean()),
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ---------------- final model ----------------
    if "decision" in feature_sets:
        prior = cv.fit_prior(df)
        booster = cv.fit_residual_xgb(df, feature_sets["decision"], prior, seed=CONFIG["seed"])
        a = df[df["plan"] == cheap]
        casc_model = cv.fit_residual_xgb(a, casc_cols, {cheap: float(a["label_mean"].mean())}) if casc_cols else None
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        pstats_path = PROC / "features_v3_pstats.json"
        pstats = json.loads(pstats_path.read_text()) if pstats_path.exists() else {}
        joblib.dump({
            "p_stats": pstats, "probe_depth": 100,
            "booster": booster, "feature_cols": feature_sets["decision"], "prior": prior,
            "plans": sorted(df["plan"].unique()), "retrievers": sorted(df["retriever"].unique()),
            "depths": sorted(df["depth"].unique()), "cascade": {"cheap": cheap, "strong": strong,
            "booster": casc_model, "feature_cols": casc_cols}, "legacy_mode": legacy, "version": "v3",
        }, MODEL_PATH)
        logger.info(f"Saved {MODEL_PATH}")

    write_summary(metrics, sweep, frontier, fixed, curve)


def write_summary(m, sweep, frontier, fixed, curve):
    L = []
    L.append(f"FunnelPlan TSR predictor, v3 evaluation  (legacy_mode={m['legacy_mode']})")
    L.append(f"{m['n_queries']} queries, 5 fold grouped CV, cost source: {m['cost_source']}")
    L.append("")
    L.append("Prediction (out of fold)")
    L.append(f"{'model':14s} {'rho':>7s} {'se':>6s} {'n':>4s} {'AUCmix':>7s} {'logloss':>8s}")
    for pc, v in m["models"].items():
        L.append(f"{pc:14s} {v['rho_mean']:7.3f} {v['rho_se']:6.3f} {v['rho_n_queries']:4d} "
                 f"{v['auc_mixed_queries']:7.3f} {v['logloss']:8.4f}")
    L.append("")
    L.append("Paired test against the plan prior (per query rho, Wilcoxon)")
    for pc, v in m["vs_prior"].items():
        L.append(f"  {pc:14s} diff={v['mean_diff']:+.3f}  p={v['p_wilcoxon']:.4f}  n={v['n']}")
    L.append("  p_post and p_all use features that need every plan to run first."
             " They are references, not deployable planners.")
    L.append("")
    L.append("Budget sweep (success = mean TSR of the chosen plan; p_prior = best fixed plan in budget)")
    piv = sweep.pivot_table(index="budget_s", columns="method", values="success")
    L.append(piv.round(3).to_string())
    L.append("")
    if m.get("decision_vs_best_fixed_by_budget"):
        L.append("Decision model minus best fixed plan, paired bootstrap 95% CI")
        for b in m["decision_vs_best_fixed_by_budget"]:
            L.append(f"  budget {b['budget_s']:>4}s: {b['gain']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]")
        L.append("")
    L.append("Utility frontier: mean gain over the best randomised mix of fixed plans at equal mean cost (hull)")
    L.append(frontier.groupby("method")["gain_vs_fixed"].mean().round(4).to_string())
    L.append("")
    c = m["cascade"]
    L.append(f"Cascade {c['cheap_plan']} -> {c['strong_plan']}  AUC(P cheap succeeds)={c['auc_p_cheap']:.3f}")
    L.append(curve[curve["escalation_rate"].round(2).isin([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])]
             .round(3).to_string(index=False))
    L.append(f"Mean gain of learned over random escalation: {c['mean_gain_over_random']:+.4f}")
    L.append("")
    L.append("Fixed plans (cost, success)")
    L.append(fixed.round(3).to_string(index=False))
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
