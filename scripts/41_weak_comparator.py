"""
The comparator effect at a fixed test. Table IV changes the comparator and the test in steps; this script holds the
test fixed (the pooled per question test with Holm over the 27 primary tests, as in Table V) and changes only the
comparator: the hull at the policy's cost, the best single plan at the same cost, and the weak comparator of prior
work (Always RAG and Never RAG, accuracy only, no cost matching).

The policies are the same in every column: each is tuned against the hull on the training folds, as in the main
protocol; only the held out comparator changes (11_analysis_full.py --eval=always / --eval=never).

Usage:  python scripts/41_weak_comparator.py            (CPU; four runs of 11_analysis_full.py, about 35 minutes)
        python scripts/41_weak_comparator.py --skip-run (only rebuild the summary)
Output: results_full/analysis/final_evalalways[_r3]/, final_evalnever[_r3]/, and
        results_full/analysis/weak_comparator/summary.txt, counts.csv, tests.csv
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import settings

_s = importlib.util.spec_from_file_location("s17", Path(__file__).parent / "17_label_sensitivity.py")
S = importlib.util.module_from_spec(_s)
_s.loader.exec_module(S)
AN = settings.RESULTS_ROOT / "analysis"
OUT = AN / "weak_comparator"


def main():
    if "--skip-run" not in sys.argv:
        for ev in ("always", "never"):
            for r3 in (False, True):
                cmd = [sys.executable, "scripts/11_analysis_full.py", f"--eval={ev}"] + (["--r3"] if r3 else [])
                print(">>", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True, cwd=ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    L, rows, tests = [], [], []
    say = lambda s: (print(s, flush=True), L.append(s))
    say("Primary family (27 pooled tests per plan set), label_final, per question test, Holm over the 27.")
    say("Same policies in every column (tuned against the hull); only the held out comparator changes.")
    for r3 in (False, True):
        suf = "_r3" if r3 else ""
        tag = "with two hop" if r3 else "single step"
        F = {"hull": S.primary_family(AN / f"final{suf}"),
             "best single plan": S.primary_family(AN / f"final_stephull{suf}"),
             "Always RAG": S.primary_family(AN / f"final_evalalways{suf}"),
             "Never RAG": S.primary_family(AN / f"final_evalnever{suf}")}
        pos = {}
        for name, T in F.items():
            T = T.set_index(["reader", "method"])
            pos[name] = (T.p_holm < 0.05) & (T.ci_low > 0)
            tests.append(T.reset_index().assign(plan_set=tag, comparator=name))
        both = pos["Always RAG"] & pos["Never RAG"]
        r = {"plan_set": tag, "tests": len(both), "vs hull": int(pos["hull"].sum()),
             "vs best single plan": int(pos["best single plan"].sum()), "vs Always RAG": int(pos["Always RAG"].sum()),
             "vs Never RAG": int(pos["Never RAG"].sum()), "vs Always and Never RAG": int(both.sum())}
        rows.append(r)
        say(f"\n== {tag}: positive (Holm p < 0.05, interval above 0) against")
        for k, v in r.items():
            if k.startswith("vs"):
                say(f"   {k:24s} {v} of {r['tests']}")
        g = F["Always RAG"]
        say(f"   gain over Always RAG {g.gain.min():+.3f} to {g.gain.max():+.3f}; over Never RAG "
            f"{F['Never RAG'].gain.min():+.3f} to {F['Never RAG'].gain.max():+.3f}")
    pd.DataFrame(rows).to_csv(OUT / "counts.csv", index=False)
    pd.concat(tests).to_csv(OUT / "tests.csv", index=False)
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
