#!/usr/bin/env python
"""
Offline planner evaluation (v3), from the out of fold predictions in
results/tsr/cv_predictions.csv. No GPU, no retrieval reruns.

Change (2026-10-02). The old script judged the planner by nDCG / recall
against qrels, read a file that does not exist (tsr_labels_clean.csv), used a
model API that 05 no longer saved, and gave the planner no reranker plans.
The planner's job is task success under a cost budget, so that is what this
script measures, per dataset and pooled:
  * success and mean cost per budget: planner vs best fixed plan vs oracle
  * utility frontier vs the envelope of fixed plans
  * results/planner/frontier.png if matplotlib is available
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from tsr import cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BUDGETS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
LAMBDAS = list(np.round(np.concatenate([[0], np.geomspace(0.002, 1.0, 25)]), 5))
IN = settings.RESULTS_ROOT / "tsr" / "cv_predictions.csv"
OUT = settings.RESULTS_ROOT / "planner"


def evaluate(df: pd.DataFrame, tag: str):
    pcols = [c for c in df.columns if c.startswith("p_")]
    sweep = cv.budget_sweep(df, pcols, BUDGETS).assign(scope=tag)
    fixed = cv.fixed_plan_points(df).assign(scope=tag)
    fr = pd.concat([cv.frontier_gain(cv.utility_frontier(df, pc, LAMBDAS), fixed) for pc in pcols],
                   ignore_index=True).assign(scope=tag)
    return sweep, fr, fixed


def plot(fr: pd.DataFrame, fixed: pd.DataFrame, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not installed; skipping figure")
        return
    scopes = list(fr["scope"].unique())
    fig, axes = plt.subplots(1, len(scopes), figsize=(5 * len(scopes), 4), squeeze=False)
    for ax, s in zip(axes[0], scopes):
        fx = fixed[fixed["scope"] == s]
        ax.scatter(fx["mean_cost_s"], fx["success"], s=14, c="0.6", label="fixed plans")
        for m, sub in fr[fr["scope"] == s].groupby("method"):
            sub = sub.sort_values("mean_cost_s")
            ax.plot(sub["mean_cost_s"], sub["success"], marker=".", label=m)
        ax.set_title(s)
        ax.set_xlabel("mean cost per query (s)")
        ax.set_ylabel("task success rate")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=600)


def main():
    if not IN.exists():
        logger.error(f"{IN} not found. Run scripts/05_train_predictor.py first.")
        return
    df = pd.read_csv(IN, dtype={"query_id": str})
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [evaluate(df, "pooled")] + [evaluate(sub, ds) for ds, sub in df.groupby("dataset")]
    sweep = pd.concat([p[0] for p in parts]); fr = pd.concat([p[1] for p in parts])
    fixed = pd.concat([p[2] for p in parts])
    sweep.to_csv(OUT / "budget_sweep.csv", index=False)
    fr.to_csv(OUT / "frontier.csv", index=False)
    fixed.to_csv(OUT / "fixed_plans.csv", index=False)
    plot(fr, fixed, OUT / "frontier.png")
    for s in sweep["scope"].unique():
        piv = sweep[sweep["scope"] == s].pivot_table(index="budget_s", columns="method", values="success")
        print(f"\n== {s}: success by budget ==\n{piv.round(3).to_string()}")
    print("\nMean frontier gain over a randomised mix of fixed plans at equal cost (hull):")
    print(fr.groupby(["scope", "method"])["gain_vs_fixed"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
