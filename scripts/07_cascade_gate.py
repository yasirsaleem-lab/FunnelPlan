#!/usr/bin/env python
"""
2026-10-06: the held out hull is now chosen on the selection half and scored
on the test half (before: chosen on the test half, which inflated the hull by
selection and biased the gain down by about 2 to 6 points). Verdict unchanged:
f1_07 +0.017 [-0.027, +0.062], FAIL.

Route B gate: does a reader cascade (7B -> 14B, gated on the 7B's confidence)
beat the best fixed configuration at equal cost?

A "configuration" is (reader, retriever, depth); closed book counts as a
retriever. The baseline is the upper concave HULL of all fixed configurations
of both readers: any random mix of two fixed configurations reaches every cost
between them, so a cascade must beat the hull, not one fixed plan.

Cascade: answer with the 7B on plan p; if the 7B's mean answer log
probability is below a threshold, answer again with the 14B on plan p'.
Cost = 7B cost on p + (14B agent cost on p' + retrieval cost of p' if p' uses
a different retriever than p).

Selection bias: there are many (p, p', threshold) choices. They are chosen on
one random half of the queries and evaluated on the other half, then the
halves swap. The reported gain is held out. 95% CI by paired bootstrap over
held out queries.

Gate rule (decided before running): Route B only if the pooled held out gain
over the hull is >= 0.03 and its 95% CI excludes 0.

Usage: python scripts/07_cascade_gate.py      (grading rule set in CONFIG)
Output: results/tsr/cascade_gate.txt and cascade_gate.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from tsr.grader import apply_rule, grade_detail
from tsr.cv import fixed_hull

CONFIG = {
    "small": "qwen2.5-7b",
    "large": "qwen3-14b",
    "rule": "f1_07",                 # grading rule for success
    "rates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "n_boot": 2000,
    "seed": 0,
    "gate_gain": 0.03,
}
PROC = settings.PROCESSED_DATA_ROOT
OUT = settings.RESULTS_ROOT / "tsr"


def load(reader):
    df = pd.read_csv(PROC / f"tsr_labels_{reader}.csv", dtype={"query_id": str}, keep_default_na=False)
    df = df[df["repeat"] == 0].copy()
    gold = {ds: json.load(open(PROC / ds / "answers.json", encoding="utf-8")) for ds in df["dataset"].unique()}
    df["y"] = [apply_rule(grade_detail(a, gold[d].get(q, [])), CONFIG["rule"])
               for d, q, a in zip(df["dataset"], df["query_id"], df["answer"])]
    df["reader"] = reader
    df["plan"] = df["retriever"] + "@" + df["depth"].astype(str)
    df["cfg"] = reader + "|" + df["plan"]
    df["ret_cost"] = pd.to_numeric(df["retrieval_time_s"], errors="coerce").fillna(0.0)
    df["agent_cost"] = pd.to_numeric(df["agent_time_s"], errors="coerce")
    return df


def tables(df):
    """Per query matrices: success, cost and confidence for every configuration."""
    y = df.pivot_table(index="query_id", columns="cfg", values="y")
    # median cost per configuration: the planner pays the typical cost
    ret = df.groupby("cfg")["ret_cost"].median()
    agent = df.groupby("cfg")["agent_cost"].median()
    conf = df.pivot_table(index="query_id", columns="cfg", values="mean_logprob")
    keep = y.columns[y.notna().all()]
    return y[keep], ret[keep], agent[keep], conf.reindex(columns=keep)


def fixed_points(y, ret, agent, idx):
    return pd.DataFrame({"plan": y.columns, "mean_cost_s": (ret + agent).values,
                         "success": y.loc[idx].mean().values})


def cascade_outcome(y, ret, agent, conf, small_cfg, large_cfg, rate, idx, thr=None):
    """Success vector and mean cost of one cascade on queries idx."""
    c = conf.loc[idx, small_cfg]
    if thr is None:
        thr = c.quantile(rate)
    esc = (c <= thr).values
    p_small, p_large = small_cfg.split("|")[1], large_cfg.split("|")[1]
    same_ret = p_small.split("@")[0] == p_large.split("@")[0]
    cost_small = ret[small_cfg] + agent[small_cfg]
    cost_large = agent[large_cfg] + (0.0 if same_ret else ret[large_cfg])
    succ = np.where(esc, y.loc[idx, large_cfg].values, y.loc[idx, small_cfg].values)
    cost = cost_small + esc.mean() * cost_large
    return succ, float(cost), float(thr)


def select_and_test(y, ret, agent, conf, sel_idx, test_idx):
    small = [c for c in y.columns if c.startswith(CONFIG["small"] + "|") and not c.endswith("@0")]
    large = [c for c in y.columns if c.startswith(CONFIG["large"] + "|")]
    hull_sel = fixed_points(y, ret, agent, sel_idx)
    best = None
    for s in small:
        for l in large:
            for r in CONFIG["rates"]:
                succ, cost, thr = cascade_outcome(y, ret, agent, conf, s, l, r, sel_idx)
                h = fixed_hull(hull_sel, cost)
                if np.isnan(h):
                    continue
                gain = succ.mean() - h
                if best is None or gain > best[0]:
                    best = (gain, s, l, r, thr)
    _, s, l, r, thr = best
    succ, cost, _ = cascade_outcome(y, ret, agent, conf, s, l, r, test_idx, thr=thr)
    # fair hull: plans chosen on the selection half, success measured on the test half
    fs, ft = fixed_points(y, ret, agent, sel_idx), fixed_points(y, ret, agent, test_idx).set_index("plan")
    pts = sorted(zip(fs["mean_cost_s"], fs["success"], fs["plan"]))
    bi = max(range(len(pts)), key=lambda i: (pts[i][1], -pts[i][0])); H = []
    for p_ in pts[: bi + 1]:
        while len(H) >= 2 and (H[-1][1]-H[-2][1])*(p_[0]-H[-2][0]) <= (p_[1]-H[-2][1])*(H[-1][0]-H[-2][0]): H.pop()
        H.append(p_)
    xs = [a for a, _, _ in H]; ys = [ft.loc[n, "success"] for _, _, n in H]
    h = float(ys[0]) if cost <= xs[0] else float(np.interp(cost, xs, ys))
    return {"small_cfg": s, "large_cfg": l, "rate_sel": r, "threshold": thr, "cost": cost,
            "success": float(succ.mean()), "hull": h, "gain": float(succ.mean() - h),
            "selection_gain": best[0]}, succ, h


def main():
    a, b = load(CONFIG["small"]), load(CONFIG["large"])
    both = pd.concat([a, b], ignore_index=True)
    rng = np.random.default_rng(CONFIG["seed"])
    lines, rows, pooled = [], [], []
    for ds, sub in both.groupby("dataset"):
        y, ret, agent, conf = tables(sub)
        q = np.array(sorted(y.index))
        rng.shuffle(q)
        halves = [q[: len(q) // 2], q[len(q) // 2:]]
        fx = fixed_points(y, ret, agent, y.index).sort_values("success", ascending=False)
        lines.append(f"== {ds}: {len(q)} queries, {y.shape[1]} fixed configurations")
        lines.append("Top fixed configurations (success, median cost):")
        for r_ in fx.head(5).itertuples():
            lines.append(f"  {r_.plan:42s} {r_.success:.3f}  {r_.mean_cost_s:.2f}s")
        for k, (sel, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
            res, succ, h = select_and_test(y, ret, agent, conf, sel, test)
            res.update(dataset=ds, split=k)
            rows.append(res)
            pooled.append(succ - h)
            lines.append(f"  split {k}: {res['small_cfg']} -> {res['large_cfg']} at {res['rate_sel']:.0%} escalation | "
                         f"held out success {res['success']:.3f} at {res['cost']:.2f}s, hull {res['hull']:.3f}, "
                         f"gain {res['gain']:+.3f} (selection half: {res['selection_gain']:+.3f})")
    d = np.concatenate(pooled)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(CONFIG["n_boot"])]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    passed = d.mean() >= CONFIG["gate_gain"] and lo > 0
    lines.append("")
    lines.append(f"POOLED held out gain over the hull: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (n={len(d)})")
    lines.append(f"Gate (gain >= {CONFIG['gate_gain']:.2f} and CI > 0): {'PASS -> Route B is viable' if passed else 'FAIL -> stay on Route A'}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "cascade_gate.csv", index=False)
    (OUT / "cascade_gate.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
