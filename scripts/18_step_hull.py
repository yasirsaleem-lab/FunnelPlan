"""
Baseline without mixing: the fixed plan hull assumes a system randomises between two plans, so this compares
against the best SINGLE fixed plan at or below the policy's cost instead.

Runs 11_analysis_full.py with --hull=step (both plan sets) and puts the primary family side by side:
  gain_mix   policy minus the hull (main protocol: best randomised mix of fixed plans at the policy's mean cost)
  gain_step  policy minus the best single fixed plan whose mean cost is at most the policy's (no mixing)
  mix_value  gain_step minus gain_mix = what a NON adaptive coin flip between two fixed plans already gains over the
             step baseline at that cost. A policy beats non adaptive mixing only if gain_step exceeds mix_value,
             which is exactly gain_mix > 0.
Holm over the 27 tests is computed for gain_step as well.

Usage:  python scripts/18_step_hull.py            (CPU, about 20 to 30 minutes)
        python scripts/18_step_hull.py --skip-run  (only rebuild the summary)
Output: results_full/analysis/final_stephull[_r3]/ and results_full/analysis/step_hull_summary.txt
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


def main():
    if "--skip-run" not in sys.argv:
        for r3 in (False, True):
            cmd = [sys.executable, "scripts/11_analysis_full.py", "--hull=step"] + (["--r3"] if r3 else [])
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT)
    L = []
    say = lambda s: (print(s, flush=True), L.append(s))
    say("Primary family (27 pooled tests) against two baselines, label_final.")
    say("mix_value = gain of a non adaptive coin flip between two fixed plans over the best single plan at that cost.")
    for r3 in (False, True):
        suf = "_r3" if r3 else ""
        M = S.primary_family(AN / f"final{suf}").set_index(["reader", "method"])
        T = S.primary_family(AN / f"final_stephull{suf}").set_index(["reader", "method"])
        J = M[["gain", "p_holm"]].add_suffix("_mix").join(T[["gain", "ci_low", "ci_high", "p_holm"]].add_suffix("_step"))
        J["mix_value"] = J["gain_step"] - J["gain_mix"]
        pos = J[(J.p_holm_step < 0.05) & (J.ci_low_step > 0)]
        say(f"\n== {'with two hop plans' if r3 else 'single step plans'}: positive against the step baseline "
            f"{len(pos)} of {len(J)}; positive against the hull {int(((J.p_holm_mix < 0.05) & (J.gain_mix > 0)).sum())}")
        say(f"   mean mix_value {J.mix_value.mean():+.3f} (range {J.mix_value.min():+.3f} to {J.mix_value.max():+.3f})")
        say(J.round(3).sort_values("gain_step", ascending=False).to_string())
        J.to_csv(AN / f"step_hull_primary{suf}.csv")
    (AN / "step_hull_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWritten {AN / 'step_hull_summary.txt'}")


if __name__ == "__main__":
    main()
