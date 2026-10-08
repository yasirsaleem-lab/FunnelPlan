"""Tests for the v3 planner, predictor and CV logic (no GPU, no model downloads)."""

import numpy as np
import pandas as pd
import pytest

from tsr import cv
from tsr.features import feature_columns, plan_identity, probe_features


def synthetic_table(n_q=120, seed=0):
    """Plans differ in base rate; plan 'b@10' succeeds more on long queries."""
    rng = np.random.default_rng(seed)
    rows = []
    plans = [("a", 10, 0.5, 1.0), ("a", 100, 0.6, 3.0), ("b", 10, 0.4, 1.5)]
    for q in range(n_q):
        ds = "nq" if q % 2 else "hotpotqa"
        long_q = q % 3 == 0
        for r, d, base, cost in plans:
            p = base + (0.4 if (r == "b" and long_q) else 0.0)
            rows.append({"dataset": ds, "query_id": str(q), "retriever": r, "depth": d,
                         "label_mean": float(rng.random() < p), "q_long": float(long_q),
                         "plan_ret_a": float(r == "a"), "plan_ret_b": float(r == "b"),
                         "plan_log_depth": np.log(d), "cost": cost})
    df = cv.add_keys(pd.DataFrame(rows))
    df["label"] = (df["label_mean"] >= 0.5).astype(int)
    return df


def test_folds_group_queries():
    df = synthetic_table()
    folds = cv.make_folds(df, 5, 0)
    assert (df.assign(f=folds).groupby("g")["f"].nunique() == 1).all()


def test_model_beats_prior_when_signal_exists():
    df = synthetic_table(n_q=600)
    cols = feature_columns(df.columns, ("q_", "plan_"))
    oof = cv.cross_validate(df, {"decision": cols}, k=5, seed=0)
    m_prior = cv.prediction_metrics(oof, "p_prior")
    m_dec = cv.prediction_metrics(oof, "p_decision")
    assert m_dec["logloss"] < m_prior["logloss"]


def test_budget_sweep_oracle_is_upper_bound():
    df = synthetic_table()
    df["p_prior"] = df["plan"].map(cv.fit_prior(df))
    s = cv.budget_sweep(df, ["p_prior"], [1.0, 2.0, 5.0]).pivot_table(
        index="budget_s", columns="method", values="success")
    assert (s["oracle"] >= s["p_prior"] - 1e-12).all()


def test_cascade_curve_endpoints():
    df = synthetic_table()
    c = cv.cascade_cv(df.assign(agent_time_s=0.5), "a@10", "a@100", ["q_long"], k=3)
    curve = cv.cascade_curve(c, 5)
    assert np.isclose(curve.iloc[0]["success_learned"], c["y_cheap"].mean())
    assert np.isclose(curve.iloc[-1]["success_learned"], c["y_strong"].mean())
    assert (curve["success_ideal"] >= curve["success_learned"] - 1e-12).all()


def test_feature_prefixes_exclude_diag():
    cols = ["q_x", "plan_ret_a", "p_top1", "r_max", "x_vote", "diag_gold_pos"]
    assert feature_columns(cols, ("q_", "plan_")) == ["q_x", "plan_ret_a"]
    assert "diag_gold_pos" not in feature_columns(cols, ("q_", "plan_", "p_", "r_", "x_", "diag_"))


def test_probe_and_plan_features_shape():
    f = probe_features("who wrote hamlet", [{"id": str(i), "score": 10 - i} for i in range(100)])
    assert all(k.startswith("q_") for k in f)
    p = plan_identity("a", 10, ["a", "b"])
    assert p["plan_ret_a"] == 1.0 and p["plan_ret_b"] == 0.0
