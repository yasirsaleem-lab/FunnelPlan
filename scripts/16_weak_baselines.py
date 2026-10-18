"""
R6: how much of an "adaptive retrieval wins" result is produced by the evaluation protocol?
No new GPU runs: the same answers, scored under the weak protocol common in the literature.

Weak protocol (what many adaptive RAG papers report):
  * baselines: Always RAG (one retriever, k = 5; BM25 as in Adaptive-RAG) and Never RAG (closed book),
    compared on accuracy only, with NO cost matching;
  * one random split (seed 0), no repeats;
  * strict exact match grading (label_strict).
Our protocol: the held out fixed plan hull at equal mean latency, 5 folds x 5 seeds, label_final, Holm.

For every (policy, reader, dataset) cell this reports the policy's held out success under the weak protocol, its
margin over Always RAG and over Never RAG, its cost relative to Always RAG, and its gain under our protocol. The
headline is the share of cells that "win" under each protocol.

Runs 11_analysis_full.py itself for the weak setting (label strict, seed 0) if that output is missing, then reads
the main output (analysis/final/). Usage:
  python scripts/16_weak_baselines.py
Output: results_full/analysis/weak_protocol/weak_vs_ours.csv, summary.txt
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

ALWAYS = "bm25@5"
NEVER = "closed_book@0"
WEAK_SUFFIX = "_weakseed0"


def run_weak_if_missing():
    out = settings.RESULTS_ROOT / "analysis" / ("strict" + WEAK_SUFFIX)
    if (out / "methods.csv").exists() and (out / "fixed_plans.csv").exists():
        return out
    s = importlib.util.spec_from_file_location("a10", Path(__file__).parent / "11_analysis_full.py")
    A = importlib.util.module_from_spec(s)
    s.loader.exec_module(A)
    A.CONFIG.update(label="strict", seeds=[0], out_suffix=WEAK_SUFFIX, reader_pairs=[])
    sys_argv = sys.argv
    sys.argv = [sys.argv[0]]                 # never --r3 for the weak protocol
    A.main()
    sys.argv = sys_argv
    return out


def main():
    weak_dir = run_weak_if_missing()
    main_dir = settings.RESULTS_ROOT / "analysis" / "final"
    if not (main_dir / "methods.csv").exists():
        sys.exit(f"Run 11_analysis_full.py first (main protocol): {main_dir / 'methods.csv'} is missing.")
    W = pd.read_csv(weak_dir / "methods.csv")
    F = pd.read_csv(weak_dir / "fixed_plans.csv")
    M = pd.read_csv(main_dir / "methods.csv")
    W = W[(W.dataset != "pooled") & (W.method != "reader_cascade")]
    M = M[(M.dataset != "pooled") & (M.method != "reader_cascade")]
    base = F[F.plan.isin([ALWAYS, NEVER])].pivot_table(index=["reader", "dataset"], columns="plan",
                                                       values=["success", "cost_s"])
    rows = []
    for _, w in W.iterrows():
        k = (w.reader, w.dataset)
        if k not in base.index:
            continue
        b = base.loc[k]
        m = M[(M.reader == w.reader) & (M.dataset == w.dataset) & (M.method == w.method)]
        rows.append({"reader": w.reader, "dataset": w.dataset, "method": w.method,
                     "weak_success": w.success, "always_rag": b[("success", ALWAYS)], "never_rag": b[("success", NEVER)],
                     "margin_vs_always": w.success - b[("success", ALWAYS)],
                     "margin_vs_never": w.success - b[("success", NEVER)],
                     "cost_vs_always": w.cost_s / b[("cost_s", ALWAYS)],
                     "ours_gain": float(m.gain.iloc[0]) if len(m) else np.nan,
                     "ours_ci_low": float(m.ci_low.iloc[0]) if len(m) else np.nan,
                     "ours_p_holm": float(m.p_holm.iloc[0]) if len(m) and "p_holm" in m else np.nan})
    R = pd.DataFrame(rows)
    out = settings.RESULTS_ROOT / "analysis" / "weak_protocol"
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "weak_vs_ours.csv", index=False)
    L = [f"Weak protocol: strict EM, seed 0 only, baselines Always RAG ({ALWAYS}) and Never RAG ({NEVER}), no cost matching.",
         f"Our protocol: held out hull at equal mean latency, 5 x 5 folds and seeds, label_final, Holm.", ""]
    win_weak = (R.margin_vs_always > 0) & (R.margin_vs_never > 0)
    win_ours = (R.ours_ci_low > 0) & (R.ours_p_holm < 0.05)
    L.append(f"cells (method x reader x dataset): {len(R)}")
    L.append(f"'beats both Always RAG and Never RAG' under the weak protocol: {int(win_weak.sum())} ({win_weak.mean():.0%})")
    L.append(f"  of those, cheaper than Always RAG: {int((win_weak & (R.cost_vs_always < 1)).sum())}")
    L.append(f"beats the hull under our protocol (CI above 0 and Holm p < 0.05): {int(win_ours.sum())} ({win_ours.mean():.0%})")
    L.append("")
    L.append("by method (mean over readers and datasets):")
    L.append(R.groupby("method")[["margin_vs_always", "margin_vs_never", "cost_vs_always", "ours_gain"]].mean()
             .round(3).to_string())
    L.append("")
    L.append("by reader:")
    L.append(R.assign(win_weak=win_weak, win_ours=win_ours).groupby("reader")[["win_weak", "win_ours"]].mean().round(2).to_string())
    (out / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
