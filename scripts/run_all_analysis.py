"""
Rebuild every analysis, table and figure of the paper from the released answer files. CPU only.

Inputs (all in the artifact): the graded answer files (data/processed_full/tsr_labels_*_graded.csv, r3_2hop/,
r4_ircot/), the retrieval caches (data/cache_full/retrievals_v3/*.pkl, used for query text and BM25 score features),
the router inputs (data/processed_full/routers/emb_*.npz and t5_cache_*.json), the METIS profiles
(data/processed_full/metis/profiles.jsonl) and the human audit files (data/processed_full/grading/).
No GPU, no model weights and no corpus are needed. Generating the answers themselves is a separate GPU step
(scripts 04, 17, 19, 20 and the judge stage of 08; see ARTIFACT.md).

Usage:  python scripts/run_all_analysis.py            (3.7 hours on a Ryzen 7 5700X, 8 cores)
        python scripts/run_all_analysis.py --from 15  (resume at step 15 of the list; see --list)
        python scripts/run_all_analysis.py --list     (print the steps only)
Output: results_full/analysis/, results_full/tables/, results_full/tables_final/ (tables.md, verdicts.txt),
        results_full/figures_final/ (every PNG at 600 dpi, every PDF vector), and results_full/run_all_log.txt
Check:  results_full/tables_final/verdicts.txt must match the verdicts reported in the paper.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = sys.executable

# (script number, command, what it produces); order respects every dependency
STEPS = [
    (10, ["scripts/10_analysis_full.py"], "H1 primary family, single step plans -> analysis/final/"),
    (10, ["scripts/10_analysis_full.py", "--r3"], "H1 primary family, with two hop plans -> analysis/final_r3/"),
    (13, ["scripts/13_routers.py", "eval", "ar_tfidf,ar_emb,plan_emb,plan_knn,ak_paper,ar_t5"], "secondary routers, single step"),
    (13, ["scripts/13_routers.py", "eval", "ar_tfidf,ar_emb,plan_emb,plan_knn,ak_paper,ar_t5", "--r3"], "secondary routers, two hop"),
    (15, ["scripts/15_power_headroom.py"], "power, held out oracles, label stability -> tables/"),
    (21, ["scripts/21_h3_ircot.py"], "H3, IRCoT against the hull -> analysis/h3_ircot/"),
    (22, ["scripts/22_weak_baselines.py"], "R6, weak protocol -> analysis/weak_protocol/"),
    (23, ["scripts/23_metis_profiler.py", "eval"], "METIS port, single step -> analysis/metis/"),
    (23, ["scripts/23_metis_profiler.py", "eval", "--r3"], "METIS port, two hop -> analysis/metis_r3/"),
    (24, ["scripts/24_label_sensitivity.py"], "grading sensitivity (lenient, strict)"),
    (25, ["scripts/25_step_hull.py"], "step baseline"),
    (27, ["scripts/27_cost_axes.py"], "cost axes: tokens, tokens x4, FLOPs, calls"),
    (28, ["scripts/28_stronger_policies.py"], "classifier gate and planner with self knowledge"),
    (29, ["scripts/29_checks.py"], "sanity table, grading factor, matched cost headroom, mechanism"),
    (12, ["scripts/12_tables_final.py"], "final tables and verdicts -> tables_final/"),
    (26, ["scripts/26_figures.py"], "final figures -> figures_final/"),
    ("F1", ["figures_src/fig_framework.py"], "Figure 1, benchmark layers -> figures_final/fig_framework"),
    ("F2", ["figures_src/fig_policies.py"], "Figure 2, policy flow charts -> figures_final/fig_policies"),
]


def main():
    if "--list" in sys.argv:
        for i, (n, cmd, what) in enumerate(STEPS, 1):
            print(f"{i:2d}. [{n}] python {' '.join(cmd):75s} {what}")
        return
    start = 0
    if "--from" in sys.argv:
        start = int(sys.argv[sys.argv.index("--from") + 1]) - 1
    log = ROOT / "results_full" / "run_all_log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== run_all_analysis started {time.strftime('%Y-%m-%d %H:%M:%S')} (from step {start + 1})\n")
        for i, (n, cmd, what) in enumerate(STEPS[start:], start + 1):
            t0 = time.time()
            line = f"[{i}/{len(STEPS)}] python {' '.join(cmd)}  ({what})"
            print(line, flush=True)
            fh.write(line + "\n"); fh.flush()
            r = subprocess.run([PY] + cmd, cwd=ROOT)
            msg = f"    exit {r.returncode} after {time.time() - t0:.0f}s"
            print(msg, flush=True)
            fh.write(msg + "\n"); fh.flush()
            if r.returncode != 0:
                fh.write(f"Stopped. Fix the error and resume with --from {i}\n")
                sys.exit(f"Step {i} failed. Resume with: python scripts/run_all_analysis.py --from {i}")
        fh.write(f"=== done in {(time.time() - t_all) / 3600:.2f} h\n")
    print(f"\nAll steps done in {(time.time() - t_all) / 3600:.2f} h. Compare results_full/tables_final/verdicts.txt "
          f"with the paper.")


if __name__ == "__main__":
    main()
