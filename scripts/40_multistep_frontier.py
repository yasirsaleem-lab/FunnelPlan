"""
IRCoT against fixed step chain of thought controllers. CPU only. Plan: docs/ANALYSIS_PLAN_2026-11-05.md, E2.
Needs the graded F_T answers (34_fixed_step_controllers.py merge, then 07 regrade, judge, finalize) and the graded
IRCoT answers (including NQ).

  frontier  per reader and dataset, on all 500 questions: success and mean latency of every single step plan, the
            fixed two hop plans, F_1 to F_4 and IRCoT; and IRCoT minus F_4 on the same questions (the value of
            adaptive stopping), with bootstrap intervals.
  hull      IRCoT against the held out hull at its own cost (the protocol of 15_h3_ircot.py) when F_1 to F_4 join the
            fixed plans as plans: single step plans plus F_T, and single step plus two hop plus F_T. Per question
            test and refit bootstrap (100 resamples, hull refitted each time); Holm over the tests of each hull.
            The decision rule for IRCoT: Holm p below 0.05 and interval above zero under both.

Usage:  python scripts/40_multistep_frontier.py
Output: results_full/analysis/multistep/  frontier.csv, stopping.csv, tests.csv, summary.txt
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

OUT = settings.RESULTS_ROOT / "analysis" / "multistep"
OUT.mkdir(parents=True, exist_ok=True)
PROC = settings.PROCESSED_DATA_ROOT
READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "nq"]
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


M15 = _load("h3_15", "15_h3_ircot.py")
A = M15.A


def graded(path, col):
    d = pd.read_csv(path, dtype={"query_id": str}, keep_default_na=False)
    d = d[d["repeat"].astype(int) == 0].copy() if "repeat" in d else d
    d["y"] = d[col].astype(int)
    d["depth"] = d["depth"].astype(int)
    d["plan"] = d["retriever"] + "@" + d["depth"].astype(str)
    for c in ("retrieval_time_s", "agent_time_s", "mean_logprob"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["retrieval_time_s"] = d["retrieval_time_s"].fillna(0.0)
    return d


def ircot(reader, col):
    fr = [graded(f, col) for f in (PROC / "r4_ircot" / f"labels_{reader}_graded.csv",
                                   PROC / "r4_ircot" / f"labels_{reader}_nq_graded.csv") if f.exists()]
    d = pd.concat(fr)
    d["cost"] = d.agent_time_s + d.retrieval_time_s
    return d


def boot_mean(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)], [2.5, 97.5])


def main():
    t0 = time.time()
    fr_rows, st_rows, tests = [], [], []
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        tag = "single step + F_T" if not r3 else "single step + two hop + F_T"
        for rd in READERS:
            df, col = A.load_reader(rd)
            fx = graded(PROC / "r5_cotfixed" / f"labels_{rd}_graded.csv", col)
            ir = ircot(rd, col)
            cols = ["dataset", "query_id", "retriever", "depth", "repeat", "y", "plan", "retrieval_time_s",
                    "agent_time_s", "mean_logprob"]
            fx["repeat"] = 0
            dfa = pd.concat([df, fx[cols]], ignore_index=True)
            for ds in DATASETS:
                if ds not in set(fx.dataset) or ds not in set(ir.dataset):
                    continue
                B = A.Block(dfa, ds)
                irs = ir[ir.dataset == ds]
                if not r3:                                 # the frontier once, with every plan
                    succ, cost = B.Y.mean(), B.cost
                    for p in B.plans:
                        kind = "F_T" if p.startswith("cotfixed") else ("two hop" if A.is_multi(p) else "single step")
                        fr_rows.append({"reader": rd, "dataset": ds, "plan": p, "kind": kind, "success": succ[p],
                                        "latency_s": cost[p]})
                    x = irs.set_index("query_id").reindex(B.q)
                    fr_rows.append({"reader": rd, "dataset": ds, "plan": "ircot", "kind": "IRCoT",
                                    "success": x.y.mean(), "latency_s": x.cost.mean()})
                    f4 = B.Y["cotfixed4@4"]
                    d_s = x.y.to_numpy(float) - f4.to_numpy(float)
                    c4 = fx[(fx.dataset == ds) & (fx.plan == "cotfixed4@4")].set_index("query_id").reindex(B.q)
                    d_c = x.cost.to_numpy(float) - (c4.agent_time_s + c4.retrieval_time_s).to_numpy(float)
                    lo, hi = boot_mean(d_s)
                    st_rows.append({"reader": rd, "dataset": ds, "ircot_minus_f4_success": d_s.mean(), "low": lo,
                                    "high": hi, "ircot_minus_f4_latency_s": d_c.mean(),
                                    "f4_success": f4.mean(), "ircot_success": x.y.mean()})
                d, sm, folds, qs = M15.h3_block(B, irs, A.CONFIG["seeds"], A.CONFIG["k_folds"])
                reps = M15.h3_refit(B, irs, dfa[dfa.dataset == ds], A.CONFIG["k_folds"], M15.REFIT_REPS)
                sd = float(reps.std(ddof=1))
                lo, hi = A.boot_ci(d, A.CONFIG["n_boot"], A.CONFIG["seed"])
                tests.append({"hull": tag, "reader": rd, "dataset": ds, "gain": d.mean(), "ci_low": lo, "ci_high": hi,
                              "p": A.p_two_sided(d), "sd_refit": sd, "refit_low": d.mean() - 1.96 * sd,
                              "refit_high": d.mean() + 1.96 * sd,
                              "p_refit": float(2 * M15.stats.norm.sf(abs(d.mean()) / sd)),
                              "hull_at_ircot_cost": float(folds.hull.mean()),
                              "hull_top_plan": B.Y.mean().idxmax()})
                say(f"  {tag:28s} {rd:12s} {ds:16s} gain {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  refit sd {sd:.3f}  "
                    f"({time.time() - t0:.0f}s)")
    A.CONFIG["include_r3"] = False
    FR, ST, T = pd.DataFrame(fr_rows), pd.DataFrame(st_rows), pd.DataFrame(tests)
    for h, g in T.groupby("hull"):
        T.loc[g.index, "p_holm"] = A.holm(g.p.to_numpy())
        T.loc[g.index, "p_refit_holm"] = A.holm(g.p_refit.to_numpy())
    T["win"] = (T.p_holm < 0.05) & (T.ci_low > 0) & (T.p_refit_holm < 0.05) & (T.refit_low > 0)
    FR.to_csv(OUT / "frontier.csv", index=False)
    ST.to_csv(OUT / "stopping.csv", index=False)
    T.to_csv(OUT / "tests.csv", index=False)
    say("\nIRCoT minus F_4 on the same questions (value of adaptive stopping):")
    say(ST.round(3).to_string(index=False))
    say("\nF_T against IRCoT, success and latency (mean over readers):")
    say(FR[FR.kind.isin(["F_T", "IRCoT"])].groupby(["dataset", "plan"])[["success", "latency_s"]].mean().round(3).to_string())
    for h, g in T.groupby("hull"):
        w = g[g.win]
        say(f"\n{h}: IRCoT wins {len(w)} of {len(g)}: " + ", ".join(f"{r.reader} {r.dataset} {r.gain:+.3f}" for r in w.itertuples()))
    (OUT / "summary.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
