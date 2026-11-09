"""
Grading and mechanism checks on the stored answers. CPU only. Plan: docs/ANALYSIS_PLAN_2026-11-05.md, CPU 4 to 6.

Parts:
  grading    (a) label_final against the human majority of audits 1 and 2, by region: the ambiguous band (where the
                 judge decides) and outside it (where the strict rule decides). The audits oversample the band, so the
                 accuracy over all answers is the two regions' accuracies weighted by their share of all answers.
                 Bootstrap interval over audited items within each region.
             (b) the same agreement by dataset and by reader.
             (c) sensitivity of the margin claim to grading: every primary upper bound shifted by the largest
                 audited shift of label_final against the policy (and by the end of its interval); counts below
                 0.02 to 0.05.
  mechanism  how often the two hull plans disagree (exactly one answers), per reader and dataset (seed 0), and how
             often two single step plans of different retrievers at the same depth disagree.
  splithalf  the primary family on two disjoint halves of every dataset (question ids split by a seeded hash,
             seed 20261105), with the paper's protocol on each half; per question test and corrected test.

Usage:  python scripts/39_grading_mechanism.py [--parts grading,mechanism,splithalf] [--quick]
Output: results_full/analysis/grading_mechanism/
"""
import hashlib
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
from scipy import stats

from config import settings

ARGS = sys.argv[1:]
arg = lambda n, d=None: ARGS[ARGS.index(n) + 1] if n in ARGS else d
PARTS = arg("--parts", "grading,mechanism,splithalf").split(",")
QUICK = "--quick" in ARGS
PROC = settings.PROCESSED_DATA_ROOT
AN = settings.RESULTS_ROOT / "analysis"
OUT = AN / ("grading_mechanism_quick" if QUICK else "grading_mechanism")
OUT.mkdir(parents=True, exist_ok=True)
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


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)], [2.5, 97.5])


# ====================================================================== grading
def part_grading():
    G = _load("grading07", "07_grading_full.py")
    say("\n== grading: label_final against the human majority of audits 1 and 2, by region")
    au = pd.read_csv(PROC / "grading" / "audit_adjudicated.csv", dtype={"query_id": str})
    rows, pop = [], []
    for f in sorted(PROC.glob("tsr_labels_*_graded.csv")):
        d = pd.read_csv(f, dtype={"query_id": str}, keep_default_na=False,
                        usecols=["dataset", "query_id", "answer", "reader", "ambiguous", "label_final"])
        d["pair"] = d.dataset + "::" + d.query_id + "::" + d.answer.map(G.normalize)
        d["amb"] = d.ambiguous.astype(str).isin(["1", "True", "1.0"])
        pop.append(d[["reader", "dataset", "amb"]])
        rows.append(d.drop_duplicates(["reader", "pair"])[["reader", "pair", "amb", "label_final"]])
    P = pd.concat(pop)
    L = pd.concat(rows).set_index(["reader", "pair"])
    # an answer given by two readers is one audited item listed as "reader_a,reader_b": either reader's label is it
    ex = au.assign(reader=au.reader.str.split(",")).explode("reader").join(L, on=["reader", "pair"])
    lab = ex.dropna(subset=["label_final"]).groupby(["audit", "audit_id"]).agg(label_final=("label_final", "first"),
                                                                              amb=("amb", "first"))
    au = au.join(lab, on=["audit", "audit_id"])
    au["reader"] = au.reader.str.split(",").str[0]
    au["label_final"] = pd.to_numeric(au.label_final, errors="coerce")
    au["ok"] = (au.label_final == au.human).astype(float)
    miss = int(au.label_final.isna().sum())
    au = au.dropna(subset=["label_final"])
    share = float(P.amb.mean())
    out = []
    for audit, g in [("audit1", au[au.audit == "audit1"]), ("audit2", au[au.audit == "audit2"]), ("both", au)]:
        a_in, a_out = g[g.amb.astype(bool)].ok, g[~g.amb.astype(bool)].ok
        rng = np.random.default_rng(1)
        bs = [share * a_in.to_numpy()[rng.integers(0, len(a_in), len(a_in))].mean()
              + (1 - share) * a_out.to_numpy()[rng.integers(0, len(a_out), len(a_out))].mean() for _ in range(2000)]
        r = {"audit": audit, "items": len(g), "items_in_band": len(a_in), "share_in_band_audited": len(a_in) / len(g),
             "acc_in_band": a_in.mean(), "acc_outside": a_out.mean(), "acc_audited": g.ok.mean(),
             "share_in_band_all_answers": share, "acc_weighted": share * a_in.mean() + (1 - share) * a_out.mean(),
             "acc_weighted_low": np.percentile(bs, 2.5), "acc_weighted_high": np.percentile(bs, 97.5)}
        out.append(r)
        say(f"  {audit}: {len(g)} items, {len(a_in)} in the ambiguous band ({len(a_in) / len(g):.0%}; all answers "
            f"{share:.1%}); accuracy in band {a_in.mean():.3f}, outside {a_out.mean():.3f}, on the audited sample "
            f"{g.ok.mean():.3f}; weighted to all answers {r['acc_weighted']:.3f} [{r['acc_weighted_low']:.3f}, "
            f"{r['acc_weighted_high']:.3f}]")
    if miss:
        say(f"  ({miss} audited items not found in the graded files; left out)")
    pd.DataFrame(out).to_csv(OUT / "grading_weighted.csv", index=False)
    by = []
    for key in ("dataset", "reader"):
        for v, g in au.groupby(key):
            lo, hi = boot(g.ok)
            by.append({"by": key, "value": v, "items": len(g), "accuracy": g.ok.mean(), "low": lo, "high": hi})
    B = pd.DataFrame(by)
    B.to_csv(OUT / "grading_by_dataset_reader.csv", index=False)
    say(B.round(3).to_string(index=False))

    say("\n-- the margin claim under the largest audited grading shift")
    I = AN / "inference"
    T = pd.read_csv(I / "tests_nb.csv")
    C = pd.read_csv(I / "calibration.csv")
    _r = I / "calibration_random.csv"
    infl = max(float((C.sd_refit / C.se_nb).max()), float(pd.read_csv(_r).factor.iloc[0]) if _r.exists() else 0.0)
    T = T[T.family == "primary"].copy()
    T["ub_infl"] = T.gain + stats.t.ppf(0.95, T.df_nb) * T.se_nb * infl
    au3 = pd.read_csv(AN / "audit3" / "audit3_results.csv")
    shifts = {"largest point shift": float(-au3.bias_label_final_on_gain.min()),
              "end of its interval": float(-au3.bias_label_final_on_gain_low.min())}
    rows = []
    for nm, sh in [("none", 0.0)] + list(shifts.items()):
        for m in (0.02, 0.03, 0.04, 0.05):
            rows.append({"shift": nm, "shift_value": sh, "margin": m,
                         **{f"below_{c}": int((T[c] + sh < m).sum()) for c in ("ub95_plain", "ub95_nb", "ub_infl")}})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "margin_under_grading_shift.csv", index=False)
    say(S.to_string(index=False))


# ====================================================================== mechanism
def part_mechanism():
    say("\n== mechanism: how often two plans disagree (exactly one answers)")
    M = pd.read_csv(AN / "robustness" / "mechanism_pairs.csv", dtype={"query_id": str})
    M = M[M.seed == 0]
    A = _load("analysis11", "11_analysis_full.py")
    rows = []
    for rd in A.readers():
        df, _ = A.load_reader(rd)
        n_q = df[df.retriever == "bm25"].groupby("dataset").query_id.nunique()
        for (pol, ds), g in M[M.reader == rd].groupby(["policy", "dataset"]):
            rows.append({"what": f"hull pair of {pol}", "reader": rd, "dataset": ds, "share": len(g) / n_q[ds]})
        d5 = df[(df.depth.astype(int) == 5) & (df.repeat.astype(int) == 0)]
        Y = d5.pivot_table(index=["dataset", "query_id"], columns="retriever", values="y")
        rs = list(Y.columns)
        for ds, g in Y.groupby(level=0):
            sh = [float(((g[a] + g[b]) == 1).mean()) for i, a in enumerate(rs) for b in rs[i + 1:]]
            rows.append({"what": "two retrievers at depth 5 (mean over pairs)", "reader": rd, "dataset": ds,
                         "share": float(np.mean(sh))})
    R_ = pd.DataFrame(rows)
    R_.to_csv(OUT / "disagreement.csv", index=False)
    say(R_.groupby("what").share.agg(["mean", "min", "max"]).round(3).to_string())


# ====================================================================== split half
def part_splithalf():
    say("\n== split half: the primary family on two disjoint halves of every dataset")
    K = _load("inference27", "27_inference_refit.py")
    R37 = _load("refit37", "37_refit_random.py")
    A = K.A
    half = lambda q: int(hashlib.sha1(f"20261105|{q}".encode()).hexdigest(), 16) % 2
    folds, perq = [], []
    t0 = time.time()
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        tag = "with two hop" if r3 else "single step"
        for rd in (A.readers()[:1] if QUICK else A.readers()):
            df, _ = A.load_reader(rd)
            for ds in (sorted(df.dataset.unique())[:1] if QUICK else sorted(df.dataset.unique())):
                B0 = A.Block(df, ds)
                for h in (0, 1):
                    take = np.array([q for q in B0.q if half(q) == h])
                    B = R37.resample_full(B0, df[df.dataset == ds], take)
                    F = A.decision_features(B)
                    mk = {"planner_tsr": lambda: A.m_planner_tsr(B, F), "adaptive_rag": lambda: A.m_adaptive_rag(B, F),
                          "adaptive_k": lambda: A.m_adaptive_k(B, F), "gate_closed_book": lambda: A.m_gate_closed_book(B, F),
                          "conf_cascade": lambda: A.m_conf_cascade(B, F)}
                    for name, m in mk.items():
                        res = K.repeated_folds(B, m)
                        if res is None:
                            continue
                        D, fr = res
                        folds += [{"plan_set": tag, "half": h, "reader": rd, "dataset": ds, "method": name, **f} for f in fr]
                        perq.append(pd.DataFrame({"plan_set": tag, "half": h, "reader": rd, "dataset": ds, "method": name,
                                                  "gain": D.gain.to_numpy()}))
                print(f"    {tag} {rd} {ds} ({time.time() - t0:.0f}s)", flush=True)
    A.CONFIG["include_r3"] = False
    FO, PQ = pd.DataFrame(folds), pd.concat(perq)
    FO.to_csv(OUT / "splithalf_folds.csv", index=False)
    rows = []
    for (tag, h, rd, m), g in FO.groupby(["plan_set", "half", "reader", "method"]):
        cells = [K.nb_cell(x.gain, x.n_test.mean(), x.n_train.mean()) for _, x in g.groupby("dataset")]
        r = K.nb_row(*K.nb_pooled(cells))
        d = PQ[(PQ.plan_set == tag) & (PQ.half == h) & (PQ.reader == rd) & (PQ.method == m)].gain.to_numpy()
        rows.append({"plan_set": tag, "half": h, "reader": rd, "method": m, "n": len(d), **r, **K.plain_row(d)})
    T = pd.DataFrame(rows)
    T = K.add_holm(T, ["plan_set", "half"])
    T.to_csv(OUT / "splithalf_tests.csv", index=False)
    for (tag, h), g in T.groupby(["plan_set", "half"]):
        say(f"  {tag}, half {h}: {len(g)} tests; positive per question {int(((g.p_plain_holm < 0.05) & (g.plain_low > 0)).sum())}, "
            f"corrected {int(((g.p_nb_holm < 0.05) & (g.nb_low > 0)).sum())}; largest gain {g.gain.max():+.3f}; "
            f"bounds below 0.03 / 0.05: {int((g.ub95_plain < 0.03).sum())} / {int((g.ub95_plain < 0.05).sum())}")


def main():
    t0 = time.time()
    for p in PARTS:
        {"grading": part_grading, "mechanism": part_mechanism, "splithalf": part_splithalf}[p]()
    say(f"({time.time() - t0:.0f}s)")
    (OUT / f"summary_{'_'.join(PARTS)}.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
