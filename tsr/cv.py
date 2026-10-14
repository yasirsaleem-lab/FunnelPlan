"""
Cross validated TSR prediction and planner evaluation (v3).

Design choices, and why:
  * Grouped K fold by query, stratified by dataset, on BOTH datasets. A single
    40 query split (old 05) gives a noisy estimate and wastes data.
  * The baseline is the plan prior: each plan's mean success on the training
    folds. It uses no features. On the v2 labels it beat every trained model
    (rho 0.16 to 0.21 against 0.10 to 0.14). Any model must beat it.
  * Models learn the RESIDUAL over the prior: the prior's logit is passed as
    XGBoost base_margin, so trees only model how a query differs from average.
  * Soft labels: if labels were repeated, the target is the mean over repeats.
  * Rho is reported, but the decision metrics are what count: success and cost
    of the chosen plan under a budget, against best fixed plan and oracle; and
    for a cascade, learned escalation against random escalation at the same rate.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

EPS = 1e-6


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def aggregate_repeats(labels: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "query_id", "retriever", "depth"]
    agg = {"label": "mean"}
    for c in ("agent_time_s", "prompt_tokens", "retrieval_time_s"):
        if c in labels.columns:
            agg[c] = "mean"
    out = labels.groupby(keys, as_index=False).agg(agg).rename(columns={"label": "label_mean"})
    out["label"] = (out["label_mean"] >= 0.5).astype(int)
    return out


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["query_id"] = df["query_id"].astype(str)
    df["depth"] = df["depth"].astype(int)
    df["g"] = df["dataset"] + "::" + df["query_id"]
    df["plan"] = df["retriever"] + "@" + df["depth"].astype(str)
    return df


def make_folds(df: pd.DataFrame, k: int = 5, seed: int = 0) -> pd.Series:
    """Fold id per row. Queries are shuffled within each dataset and dealt
    round robin, so every fold has the same dataset mix."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for ds, sub in df.groupby("dataset"):
        gs = np.array(sorted(sub["g"].unique()))
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            fold_of[g] = i % k
    return df["g"].map(fold_of)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def fit_prior(train: pd.DataFrame, alpha: float = 2.0) -> Dict[str, float]:
    glob = train["label_mean"].mean()
    s = train.groupby("plan")["label_mean"].agg(["sum", "count"])
    return ((s["sum"] + alpha * glob) / (s["count"] + alpha)).to_dict()


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "max_depth": 3, "eta": 0.03, "min_child_weight": 5.0,
    "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 5.0, "alpha": 0.5,
    "nthread": -1, "verbosity": 0,
}
XGB_ROUNDS = 300


def fit_residual_xgb(train: pd.DataFrame, cols: Sequence[str], prior: Dict[str, float],
                     params: Optional[dict] = None, rounds: int = XGB_ROUNDS, seed: int = 0):
    import xgboost as xgb
    p = dict(XGB_PARAMS, **(params or {}), seed=seed)
    d = xgb.DMatrix(train[list(cols)].to_numpy(np.float32), label=train["label_mean"].to_numpy(),
                    base_margin=logit(train["plan"].map(prior).to_numpy()))
    return xgb.train(p, d, num_boost_round=rounds)


def predict_residual_xgb(booster, test: pd.DataFrame, cols: Sequence[str], prior: Dict[str, float]):
    import xgboost as xgb
    d = xgb.DMatrix(test[list(cols)].to_numpy(np.float32),
                    base_margin=logit(test["plan"].map(prior).to_numpy()))
    return booster.predict(d)


def cross_validate(df: pd.DataFrame, feature_sets: Dict[str, List[str]], k: int = 5,
                   seed: int = 0) -> pd.DataFrame:
    """Out of fold predictions: column p_prior plus p_<name> per feature set."""
    df = df.copy()
    df["fold"] = make_folds(df, k, seed)
    df["p_prior"] = np.nan
    for name in feature_sets:
        df[f"p_{name}"] = np.nan
    for f in range(k):
        tr, te = df["fold"] != f, df["fold"] == f
        prior = fit_prior(df[tr])
        df.loc[te, "p_prior"] = df.loc[te, "plan"].map(prior).to_numpy()
        for name, cols in feature_sets.items():
            if not cols:
                continue
            b = fit_residual_xgb(df[tr], cols, prior, seed=seed)
            df.loc[te, f"p_{name}"] = predict_residual_xgb(b, df[te], cols, prior)
    return df


# ----------------------------------------------------------------------
# Prediction metrics
# ----------------------------------------------------------------------
def per_query_rho(df: pd.DataFrame, pcol: str) -> pd.Series:
    out = {}
    for g, sub in df.groupby("g"):
        if sub["label_mean"].nunique() < 2 or sub[pcol].nunique() < 2:
            continue
        out[g] = spearmanr(sub[pcol], sub["label_mean"])[0]
    return pd.Series(out, dtype=float)


def prediction_metrics(df: pd.DataFrame, pcol: str) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    y, p = df["label"].to_numpy(), np.clip(df[pcol].to_numpy(), EPS, 1 - EPS)
    rho = per_query_rho(df, pcol)
    # AUC within mixed queries only: does the model rank a query's plans?
    mixed = df[df.groupby("g")["label"].transform("nunique") > 1]
    return {
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc_all": float(roc_auc_score(y, p)),
        "auc_mixed_queries": float(roc_auc_score(mixed["label"], mixed[pcol])) if len(mixed) else np.nan,
        "rho_mean": float(rho.mean()),
        "rho_se": float(rho.std(ddof=1) / np.sqrt(len(rho))) if len(rho) > 1 else np.nan,
        "rho_n_queries": int(len(rho)),
    }


def paired_rho_test(df: pd.DataFrame, pcol: str, base: str = "p_prior") -> Dict[str, float]:
    a, b = per_query_rho(df, pcol), per_query_rho(df, base)
    idx = a.index.intersection(b.index)
    d = (a[idx] - b[idx]).dropna()
    if len(d) < 10 or np.allclose(d, 0):
        return {"mean_diff": float(d.mean()) if len(d) else np.nan, "p_wilcoxon": np.nan, "n": int(len(d))}
    return {"mean_diff": float(d.mean()), "p_wilcoxon": float(wilcoxon(d).pvalue), "n": int(len(d))}


# ----------------------------------------------------------------------
# Planner simulation
# ----------------------------------------------------------------------
def attach_cost(df: pd.DataFrame, cost_model=None, measured: bool = True) -> pd.DataFrame:
    """Expected plan cost (seconds). Uses measured per plan means when the
    labels carry timing columns; otherwise the CostModel."""
    df = df.copy()
    if measured and {"agent_time_s", "retrieval_time_s"}.issubset(df.columns) \
            and df[["agent_time_s", "retrieval_time_s"]].notna().all().all():
        # Median, not mean: a few queries hit GPU contention (e.g. NQ 0.6B
        # reranker: median 1.26 s, max 13.5 s), and the mean would overstate
        # the typical cost of that plan.
        per_row = df["agent_time_s"] + df["retrieval_time_s"]
        df["cost"] = per_row.groupby(df["plan"]).transform("median")
        df["cost_source"] = "measured"
    else:
        if cost_model is None:
            from cost_model import CostModel
            cost_model = CostModel()
        cache = {}
        for r, d in df[["retriever", "depth"]].drop_duplicates().itertuples(index=False):
            cache[(r, d)] = cost_model.plan_cost(r, int(d))
        df["cost"] = [cache[(r, d)] for r, d in zip(df["retriever"], df["depth"])]
        df["cost_source"] = "cost_model"
    return df


def _pick(df: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    idx = score.groupby(df["g"]).idxmax()
    return df.loc[idx.values]


def budget_sweep(df: pd.DataFrame, pcols: Sequence[str], budgets: Sequence[float]) -> pd.DataFrame:
    """For each budget, pick the plan with the highest predicted TSR among plans
    whose expected cost fits. Report realised success and cost. p_prior gives
    the best fixed plan under that budget. 'oracle' succeeds if any feasible
    plan succeeds."""
    rows = []
    for B in budgets:
        feas = df[df["cost"] <= B]
        if feas.empty:
            continue
        nq = df["g"].nunique()
        for pc in pcols:
            ch = _pick(feas, feas[pc])
            rows.append({"budget_s": B, "method": pc, "success": ch["label_mean"].sum() / nq,
                         "mean_cost_s": ch["cost"].mean(), "n_plans_feasible": feas["plan"].nunique()})
        orc = feas.groupby("g")["label"].max()
        rows.append({"budget_s": B, "method": "oracle", "success": orc.sum() / nq,
                     "mean_cost_s": np.nan, "n_plans_feasible": feas["plan"].nunique()})
    return pd.DataFrame(rows)


def utility_frontier(df: pd.DataFrame, pcol: str, lambdas: Sequence[float]) -> pd.DataFrame:
    """Pick argmax p - lambda * cost per query. Traces a cost/success curve."""
    rows = []
    for lam in lambdas:
        ch = _pick(df, df[pcol] - lam * df["cost"])
        rows.append({"lambda": lam, "method": pcol, "mean_cost_s": ch["cost"].mean(),
                     "success": ch["label_mean"].mean()})
    return pd.DataFrame(rows)


def fixed_plan_points(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("plan").agg(mean_cost_s=("cost", "mean"), success=("label_mean", "mean"))
              .reset_index().sort_values("mean_cost_s"))


def fixed_envelope(fixed: pd.DataFrame, cost: float) -> float:
    """Best success of any single fixed plan with mean cost <= cost."""
    ok = fixed[fixed["mean_cost_s"] <= cost + 1e-12]
    return float(ok["success"].max()) if len(ok) else np.nan


def fixed_hull(fixed: pd.DataFrame, cost: float) -> float:
    """Success of the best RANDOMISED mix of two fixed plans at mean cost <= cost
    (upper concave hull of the fixed plan points).

    This is the fair baseline. Any planner that mixes plans across queries
    reaches costs between two fixed plans; flipping a biased coin between those
    two plans reaches the same cost with linearly interpolated success, with no
    prediction at all. The step envelope (fixed_envelope) ignores this and
    credits a planner with gains that are pure interpolation.
    """
    pts = sorted(zip(fixed["mean_cost_s"], fixed["success"]))
    # Only the efficient part matters: points after the most successful plan are
    # dominated (more cost, less success). Beyond that plan the baseline is flat:
    # a larger budget can always still run the best plan.
    best = max(range(len(pts)), key=lambda i: (pts[i][1], -pts[i][0]))
    pts = pts[: best + 1]
    hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= \
                (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    xs, ys = zip(*hull)
    if cost < xs[0] - 1e-12:
        return np.nan
    return float(np.interp(cost, xs, ys, right=ys[-1]))


def frontier_gain(frontier: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    f = frontier.copy()
    f["best_fixed_at_cost"] = [fixed_envelope(fixed, c) for c in f["mean_cost_s"]]
    f["gain_vs_step"] = f["success"] - f["best_fixed_at_cost"]
    f["hull_at_cost"] = [fixed_hull(fixed, c) for c in f["mean_cost_s"]]
    # Main metric: gain over a randomised mix of fixed plans at equal cost.
    f["gain_vs_fixed"] = f["success"] - f["hull_at_cost"]
    return f


def paired_bootstrap_gain(df: pd.DataFrame, pcol: str, base: str, budget: float,
                          n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    feas = df[df["cost"] <= budget]
    a = _pick(feas, feas[pcol]).set_index("g")["label_mean"]
    b = _pick(feas, feas[base]).set_index("g")["label_mean"]
    d = (a - b.reindex(a.index)).to_numpy()
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"budget_s": budget, "gain": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": int(len(d))}


# ----------------------------------------------------------------------
# Cascade
# ----------------------------------------------------------------------
def cascade_cv(df: pd.DataFrame, cheap: str, strong: str, cols: Sequence[str], k: int = 5,
               seed: int = 0) -> pd.DataFrame:
    """One row per query: OOF P(cheap plan succeeds), labels of both plans,
    and the cost of each path. Cost of accept = retrieval(cheap) + agent(cheap).
    Cost of escalate = retrieval(cheap) + retrieval(strong) + agent(strong).
    """
    a = df[df["plan"] == cheap].set_index("g")
    b = df[df["plan"] == strong].set_index("g")
    q = a.index.intersection(b.index)
    a, b = a.loc[q].copy(), b.loc[q]
    a["fold"] = make_folds(a.reset_index(), k, seed).to_numpy()
    a["p_cheap"] = np.nan
    for f in range(k):
        tr, te = a["fold"] != f, a["fold"] == f
        prior = {cheap: float(a.loc[tr, "label_mean"].mean())}
        if cols:
            bst = fit_residual_xgb(a[tr], cols, prior, seed=seed)
            a.loc[te, "p_cheap"] = predict_residual_xgb(bst, a[te], cols, prior)
        else:
            a.loc[te, "p_cheap"] = prior[cheap]
    out = pd.DataFrame({
        "p_cheap": a["p_cheap"], "y_cheap": a["label_mean"], "y_strong": b["label_mean"],
        "cost_accept": a["cost"], "cost_escalate": a["cost"] - _agent_part(a) + b["cost"],
    })
    return out


def _agent_part(rows: pd.DataFrame) -> pd.Series:
    if "agent_time_s" in rows.columns and rows["agent_time_s"].notna().all():
        return rows.groupby("plan")["agent_time_s"].transform("median")
    from cost_model import CostModel
    cm = CostModel()
    return rows["depth"].map(lambda d: cm.agent_latency(int(d)))


def cascade_curve(c: pd.DataFrame, n_points: int = 21) -> pd.DataFrame:
    """Sweep the escalation rate. Compare learned escalation (lowest p_cheap
    first) with random escalation and with the ideal (escalate exactly the
    queries where the cheap plan fails)."""
    n = len(c)
    order = c["p_cheap"].sort_values(kind="mergesort").index
    fails_first = c.sort_values("y_cheap", kind="mergesort").index
    rows = []
    for rate in np.linspace(0, 1, n_points):
        m = int(round(rate * n))
        esc = set(order[:m])
        e = c.index.isin(list(esc))
        learned = np.where(e, c["y_strong"], c["y_cheap"]).mean()
        cost = np.where(e, c["cost_escalate"], c["cost_accept"]).mean()
        rand = (1 - m / n) * c["y_cheap"].mean() + (m / n) * c["y_strong"].mean()
        ie = c.index.isin(list(fails_first[:m]))
        ideal = np.where(ie, np.maximum(c["y_strong"], c["y_cheap"]), c["y_cheap"]).mean()
        rows.append({"escalation_rate": m / n, "mean_cost_s": cost, "success_learned": learned,
                     "success_random": rand, "success_ideal": ideal})
    return pd.DataFrame(rows)
