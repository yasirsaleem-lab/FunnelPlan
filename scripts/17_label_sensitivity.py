"""
Grading sensitivity (reviewer point: label_final is conservative; the adjudicated human audit gives it 0.80 and
0.84 accuracy, with false negatives far above false positives). Reruns H1 (primary family, both plan sets) and H3
with other label rules and checks whether any conclusion changes.

For each label rule in RULES:
  1. 11_analysis_full.py with that label, single step plans and --r3   -> analysis/<label>/ and analysis/<label>_r3/
  2. the primary family (27 pooled tests) with Holm, from per_query_gains.csv
  3. 15_h3_ircot.py with that label                                       -> analysis/h3_ircot_<label>/
Nothing in analysis/final* or analysis/h3_ircot/ is touched.

Usage:  python scripts/17_label_sensitivity.py                (lenient and strict; CPU, about 40 to 60 minutes)
        python scripts/17_label_sensitivity.py lenient
Output: results_full/analysis/label_sensitivity.txt
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

import numpy as np
import pandas as pd

from config import settings

RULES = [a for a in sys.argv[1:] if not a.startswith("-")] or ["lenient", "strict"]
PRIMARY = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade", "reader_cascade"]
AN = settings.RESULTS_ROOT / "analysis"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty_like(p); run = 0.0; m = len(p)
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); a[i] = min(1.0, run)
    return a


def primary_family(folder):
    from scipy import stats
    pq = pd.read_csv(folder / "per_query_gains.csv", dtype={"query_id": str})
    pq = pq[pq.method.isin(PRIMARY)]
    rows = []
    for (rd, m), g in pq.groupby(["reader", "method"]):
        d = g.gain.to_numpy(float)
        p = 1.0 if np.allclose(d, d[0]) else float(stats.ttest_1samp(d, 0.0).pvalue)
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        rows.append({"reader": rd, "method": m, "gain": d.mean(), "ci_low": np.percentile(bs, 2.5),
                     "ci_high": np.percentile(bs, 97.5), "p": p, "n": len(d)})
    F = pd.DataFrame(rows)
    F["p_holm"] = holm(F.p.values)
    return F


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    L = []
    say = lambda s: (print(s, flush=True), L.append(s))
    py = sys.executable
    for lab in RULES:
        for r3 in (False, True):
            run([py, "scripts/11_analysis_full.py", f"--label={lab}"] + (["--r3"] if r3 else []))
        run([py, "scripts/15_h3_ircot.py", f"--label={lab}"])
        say(f"\n===== label rule: {lab}")
        for r3 in (False, True):
            F = primary_family(AN / (lab + ("_r3" if r3 else "")))
            pos = F[(F.p_holm < 0.05) & (F.ci_low > 0)]; neg = F[(F.p_holm < 0.05) & (F.ci_high < 0)]
            say(f"  H1 primary family, {'with two hop plans' if r3 else 'single step plans'}: {len(F)} tests; "
                f"positive survivors {len(pos)}; negative survivors {len(neg)}; largest gain {F.gain.max():+.3f} "
                f"({F.loc[F.gain.idxmax(), 'method']}, {F.loc[F.gain.idxmax(), 'reader']})")
            for _, r in pd.concat([pos, neg]).iterrows():
                say(f"    {'POS' if r.gain > 0 else 'neg'} {r.reader:24s} {r.method:16s} {r.gain:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] Holm p {r.p_holm:.4f}")
        H = pd.read_csv(AN / f"h3_ircot_{lab}" / "h3.csv")
        for hull, g in H.groupby("hull"):
            sig = g[(g.p_holm < 0.05) & (g.ci_low > 0)]
            say(f"  H3, hull {hull}: {len(sig)} of {len(g)} significant: "
                + ", ".join(f"{r.reader} {r.dataset} {r.gain:+.3f}" for r in sig.itertuples()))
    out = AN / "label_sensitivity.txt"
    prev = out.read_text(encoding="utf-8") if out.exists() and len(RULES) < 2 else ""
    out.write_text(prev + "\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
