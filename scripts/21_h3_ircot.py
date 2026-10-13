#!/usr/bin/env python
"""
H3 (pre-registered): IRCoT against the held out fixed plan hull at IRCoT's own mean latency.

Per (reader, dataset) for Qwen2.5 7B, Llama 3.1 8B, Qwen3 14B on HotpotQA, 2WikiMultihopQA, MuSiQue (9 tests):
  the hull's plans are chosen on the training folds (5 folds x 5 seeds, exactly as in 10_analysis_full.py) and
  measured on the test fold at c = IRCoT's mean cost on that test fold. Per query gain = IRCoT success minus the
  hull value, averaged over seeds. One sample t test on the per query gains, 95% bootstrap CI, Holm over the 9.
Primary hull: single step plans (H3 as written). Secondary: single step plus the fixed two hop plans (R3).

Labels: label_final everywhere (IRCoT answers graded by 08 regrade, judge, finalize, like every other answer).
Costs are compared only within a reader (each reader's own runtime).

Usage
  python scripts/21_h3_ircot.py                 all readers and datasets with a graded IRCoT file
  python scripts/21_h3_ircot.py --partial       also run on readers/datasets with fewer than 500 IRCoT answers
                                                (interim look only; never reported)
Output: results_full/analysis/h3_ircot/h3.csv, summary.txt
"""

import importlib.util
import os
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings

_s = importlib.util.spec_from_file_location("a10", Path(__file__).parent / "10_analysis_full.py")
A = importlib.util.module_from_spec(_s)
_s.loader.exec_module(A)

READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
PROC = settings.PROCESSED_DATA_ROOT


def load_ircot(reader, col):
    f = PROC / "r4_ircot" / f"labels_{reader}_graded.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, dtype={"query_id": str}, keep_default_na=False)
    if col not in d.columns:
        sys.exit(f"{f.name} has no {col}; run 08 judge and finalize first.")
    d["y"] = d[col].astype(int)
    d["cost"] = pd.to_numeric(d["agent_time_s"], errors="coerce") + pd.to_numeric(d["retrieval_time_s"], errors="coerce")
    return d


def h3_block(B, ir, seeds, k):
    """Per query gain of IRCoT over the held out hull at IRCoT's mean test fold cost, averaged over seeds."""
    ir = ir.set_index("query_id")
    q_all = np.array([q for q in B.q if q in ir.index])
    gains = {q: [] for q in q_all}
    seed_means, rows = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        q = q_all.copy()
        rng.shuffle(q)
        folds = [q[i::k] for i in range(k)]
        g_seed = []
        for i in range(k):
            test = folds[i]
            sel = np.concatenate([folds[j] for j in range(k) if j != i])
            c = float(ir.loc[test, "cost"].mean())
            h = A.hull_at(B.Y, B.cost, sel, c, eval_idx=test)
            s = ir.loc[test, "y"].to_numpy(float)
            for qq, v in zip(test, s - h):
                gains[qq].append(v)
            g_seed.append(s.mean() - h)
            rows.append({"seed": seed, "fold": i, "ircot": s.mean(), "hull": h, "cost": c})
        seed_means.append(np.mean(g_seed))
    d = np.array([np.mean(gains[q]) for q in q_all])
    return d, np.array(seed_means), pd.DataFrame(rows), q_all


def main():
    partial = "--partial" in sys.argv
    lab = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--label=")), None)
    if lab:
        A.CONFIG["label"] = lab                   # sensitivity run; the main run (label_final) is untouched
    out = settings.RESULTS_ROOT / "analysis" / ("h3_ircot" + (f"_{lab}" if lab else ""))
    out.mkdir(parents=True, exist_ok=True)
    L, res = [], []
    say = lambda s: (print(s, flush=True), L.append(s))
    for with_r3 in (False, True):
        A.CONFIG["include_r3"] = with_r3
        say(f"===== hull: {'single step + fixed two hop plans (R3)' if with_r3 else 'single step plans (primary, H3 as written)'}")
        for rd in READERS:
            df, col = A.load_reader(rd)
            ir_all = load_ircot(rd, col)
            if ir_all is None:
                say(f"  {rd}: no graded IRCoT file yet")
                continue
            for ds in DATASETS:
                ir = ir_all[ir_all.dataset == ds]
                if len(ir) < 500 and not partial:
                    say(f"  {rd} {ds}: {len(ir)} IRCoT answers (need 500)")
                    continue
                if ir.empty:
                    continue
                B = A.Block(df, ds)
                d, sm, folds, qs = h3_block(B, ir, A.CONFIG["seeds"], A.CONFIG["k_folds"])
                lo, hi = A.boot_ci(d, A.CONFIG["n_boot"], A.CONFIG["seed"])
                acc = B.Y.loc[qs].mean()
                best = acc.idxmax()
                r = {"hull": "with_r3" if with_r3 else "single_step", "reader": rd, "dataset": ds, "n": len(qs),
                     "ircot_success": float(ir.set_index("query_id").loc[qs, "y"].mean()),
                     "ircot_cost_s": float(ir.set_index("query_id").loc[qs, "cost"].mean()),
                     "hull_at_ircot_cost": float(folds.hull.mean()), "gain": d.mean(), "ci_low": lo, "ci_high": hi,
                     "p": A.p_two_sided(d), "seed_min": sm.min(), "seed_max": sm.max(),
                     "best_fixed_plan": best, "best_fixed_success": float(acc[best]), "best_fixed_cost_s": float(B.cost[best]),
                     "lm_calls": float(ir.lm_calls.mean()), "retrieval_calls": float(ir.retrieval_calls.mean())}
                res.append(r)
                say(f"  {rd:12s} {ds:16s} n={len(qs)}  IRCoT {r['ircot_success']:.3f} at {r['ircot_cost_s']:.2f}s "
                    f"({r['ircot_cost_s'] / r['best_fixed_cost_s']:.1f}x the best plan's cost)  hull {r['hull_at_ircot_cost']:.3f}  "
                    f"gain {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  seeds {sm.min():+.3f}..{sm.max():+.3f}  "
                    f"best plan {best} {acc[best]:.3f}")
    R = pd.DataFrame(res)
    if len(R):
        for h, g in R.groupby("hull"):
            R.loc[g.index, "p_holm"] = A.holm(g["p"].to_numpy())
        say("")
        say("Holm over the tests of each hull (pre-registered family: 9 tests, single step hull):")
        for _, r in R.iterrows():
            sig = "SIGNIFICANT" if (r["p_holm"] < 0.05 and r["ci_low"] > 0) else ("significantly BELOW" if (r["p_holm"] < 0.05 and r["ci_high"] < 0) else "n.s.")
            say(f"  {r['hull']:12s} {r['reader']:12s} {r['dataset']:16s} gain {r['gain']:+.3f}  p {r['p']:.4f}  Holm p {r['p_holm']:.4f}  {sig}")
        if partial and (R["n"] < 500).any():
            say("PARTIAL: some cells have fewer than 500 questions. Interim look only; do not report.")
        R.to_csv(out / "h3.csv", index=False)
    (out / "summary.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
