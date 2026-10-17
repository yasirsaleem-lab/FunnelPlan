"""
Cost axes (amendment 2026-10-13 20:45, item (d)). Concern: latency comes from one GPU, so the verdict may be
an artifact of that machine. Refits the hull and retests the primary family (27 pooled tests, Holm) with four costs
in place of latency, in both plan sets:
  tokens     reader prompt plus generated tokens, per 1,000
  tokens_w4  prompt plus 4 x generated tokens, per 1,000 (output priced at 4 x input)
  flops      per 1e13: 2 x reader parameters x tokens, plus reranker and dense query encoder (ANN search not counted)
  calls      LLM calls plus retrieval, fusion and rerank calls
Definitions: add_cost_axes() in 11_analysis_full.py. The verdict is called independent of the cost axis only if no
positive survivor appears under any axis in both plan sets.

Usage:  python scripts/19_cost_axes.py              (CPU, 8 runs of 10, about 1.5 to 2.5 hours)
        python scripts/19_cost_axes.py --skip-run   (only rebuild the summary)
Output: results_full/analysis/final_cost<axis>[_r3]/, cost_axes_primary.csv, cost_axes_summary.txt
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import settings

_s = importlib.util.spec_from_file_location("s24", Path(__file__).parent / "17_label_sensitivity.py")
S = importlib.util.module_from_spec(_s)
_s.loader.exec_module(S)
AN = settings.RESULTS_ROOT / "analysis"
AXES = ["tokens", "tokens_w4", "flops", "calls"]


def main():
    if "--skip-run" not in sys.argv:
        for ax in AXES:
            for r3 in (False, True):
                cmd = [sys.executable, "scripts/11_analysis_full.py", f"--cost={ax}"] + (["--r3"] if r3 else [])
                print(">>", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True, cwd=ROOT)
    L, rows = [], []
    say = lambda s: (print(s, flush=True), L.append(s))
    say("Primary family (27 pooled tests, Holm) under each cost axis, label_final. Latency is the main axis.")
    say(f"{'axis':10s} {'plan set':10s} {'positive':>8s} {'negative':>8s} {'largest gain':>13s}  largest gain at")
    for ax in ["latency"] + AXES:
        for r3 in (False, True):
            folder = AN / (("final" if ax == "latency" else f"final_cost{ax}") + ("_r3" if r3 else ""))
            if not (folder / "per_query_gains.csv").exists():
                say(f"{ax:10s} {'two hop' if r3 else 'single':10s} missing {folder.name}")
                continue
            F = S.primary_family(folder)
            F["axis"], F["plan_set"] = ax, ("two_hop" if r3 else "single_step")
            rows.append(F)
            pos = F[(F.p_holm < 0.05) & (F.ci_low > 0)]
            neg = F[(F.p_holm < 0.05) & (F.ci_high < 0)]
            top = F.loc[F.gain.idxmax()]
            say(f"{ax:10s} {'two hop' if r3 else 'single':10s} {len(pos):8d} {len(neg):8d} {top.gain:+13.3f}  "
                f"{top.method} {top.reader}")
            for _, r in pos.iterrows():
                say(f"    POSITIVE {r.reader} {r.method} {r.gain:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] Holm p {r.p_holm:.4f}")
    if rows:
        A = pd.concat(rows, ignore_index=True)
        A.to_csv(AN / "cost_axes_primary.csv", index=False)
        both = (A[(A.p_holm < 0.05) & (A.ci_low > 0)].groupby(["axis", "reader", "method"]).plan_set.nunique() == 2)
        say("")
        say(f"Positive in BOTH plan sets under some axis: {int(both.sum())} "
            + (", ".join(f'{a} {r} {m}' for (a, r, m), v in both.items() if v) if both.any() else ""))
        say("Verdict independent of the cost axis: " + ("YES" if not both.any() else "NO, see above"))
    (AN / "cost_axes_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWritten {AN / 'cost_axes_summary.txt'}")


if __name__ == "__main__":
    main()
