"""
Final tables for the paper (5 readers, both plan sets). CPU only, reads results; never reruns an analysis.

Writes results_full/tables_final/:
  tables.md          every table in readable form (the source for the paper text)
  tab_*.tex          booktabs LaTeX tables
  verdicts.txt       the pre-registered decisions, computed from the files (H1, secondary family, H3, robustness)
  tab_summary.tex    every family and robustness check in one table (the paper's verdict table)
  tab_rater.tex      grading rules against each human reference (majority of three, each rater alone)
  tab_p_*.tex        compact tables for the paper body (retrievers, plans, gains, IRCoT, weak protocol, mechanism, raters)

Inputs (all produced by earlier scripts): analysis/final[_r3]/ (10, 13), analysis/metis[_r3]/ (23),
analysis/h3_ircot[_lenient|_strict]/ (21, 24), analysis/weak_protocol/ (22), analysis/lenient*|strict* (24),
analysis/step_hull_primary*.csv (25, optional), analysis/cost_axes_primary.csv (27), analysis/stronger_policies (28),
analysis/checks (29), grading/audit_adjudicated.csv (audits), r4_ircot labels (20).

Usage: python scripts/12_tables_final.py               (writes results_full/tables_final/)
       python scripts/12_tables_final.py --out DIR     (writes to DIR instead; for testing)
"""

import re
import importlib.util
import os
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import settings

AN = settings.RESULTS_ROOT / "analysis"
PROC = settings.PROCESSED_DATA_ROOT
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else settings.RESULTS_ROOT / "tables_final"
READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b", "gemma3-27b", "qwen3-32b"]
RN = {"qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B", "qwen3-14b": "Qwen3 14B",
      "gemma3-27b": "Gemma 3 27B", "qwen3-32b": "Qwen3 32B",
      "qwen2.5-7b->qwen3-14b": "7B to Qwen3 14B", "qwen2.5-7b->llama3.1-8b": "7B to Llama 3.1 8B"}
DN = {"nq": "NQ", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki", "musique": "MuSiQue"}
PN = {"planner_tsr": "TSR planner", "adaptive_rag": "Adaptive RAG router", "adaptive_k": "Adaptive k (gap)",
      "gate_closed_book": "Closed book gate", "conf_cascade": "Confidence cascade", "reader_cascade": "Reader cascade",
      "ak_paper": "Adaptive k (published rule)", "ar_emb": "Adaptive RAG, BGE M3 LR", "ar_t5": "Adaptive RAG, FLAN T5",
      "plan_emb": "Per plan LR", "plan_knn": "Per plan kNN", "metis": "METIS port"}
PRIMARY = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade", "reader_cascade"]
SECONDARY = ["ak_paper", "ar_emb", "ar_t5", "plan_emb", "plan_knn"]

_s = importlib.util.spec_from_file_location("s24", Path(__file__).parent / "24_label_sensitivity.py")
S = importlib.util.module_from_spec(_s)
_s.loader.exec_module(S)
holm, primary_family = S.holm, S.primary_family

MD, TEX, VERD = [], {}, []
NONINF, AUD = [], {}     # filled by t_noninf and t_audit, read by t_summary and t_rater


def sg(x, d=3):
    return "" if pd.isna(x) else f"{x:+.{d}f}"


def pv(p):
    return "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def md_table(title, df, note=""):
    MD.append(f"\n## {title}\n")
    MD.append("| " + " | ".join(map(str, df.columns)) + " |")
    MD.append("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False):
        MD.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    if note:
        MD.append(f"\n{note}")


def tex_esc(v):
    """Escape a cell for LaTeX: underscores, percent, comparison signs, and a true minus before numbers."""
    s = str(v)
    if "$" in s:
        return s
    s = s.replace("_", r"\_").replace("%", r"\%").replace(">=", r"$\geq$").replace("<=", r"$\leq$")
    s = s.replace("<", r"$<$").replace(">", r"$>$").replace("\u2020", r"$^\dagger$")
    return re.sub(r"(?<![A-Za-z0-9])-(?=\d)", r"$-$", s)


def tex_table(name, df, caption, align=None):
    esc = tex_esc
    align = align or "l" * len(df.columns)
    L = [r"\begin{table}[t]", r"\centering\small", f"\\caption{{{esc(caption)}}}", f"\\label{{tab:{name}}}",
         f"\\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    for r in df.itertuples(index=False):
        L.append(" & ".join(esc(v) for v in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TEX[name] = "\n".join(L)


# ====================================================================== H1 primary
def t_h1():
    A, B = primary_family(AN / "final"), primary_family(AN / "final_r3")
    rows = []
    for m in PRIMARY:
        for rd in sorted(A[A.method == m].reader, key=lambda r: READERS.index(r) if r in READERS else 9):
            a = A[(A.method == m) & (A.reader == rd)].iloc[0]; b = B[(B.method == m) & (B.reader == rd)].iloc[0]
            rows.append({"Policy": PN[m], "Reader": RN.get(rd, rd),
                         "Gain, single step [95% CI]": f"{sg(a.gain)} [{sg(a.ci_low)}, {sg(a.ci_high)}]", "Holm p": pv(a.p_holm),
                         "Gain, with two hop [95% CI]": f"{sg(b.gain)} [{sg(b.ci_low)}, {sg(b.ci_high)}]", "Holm p ": pv(b.p_holm)})
    T = pd.DataFrame(rows)
    md_table("H1, primary family (27 pooled tests per plan set, label_final)", T)
    tex_table("h1", T, "H1, primary family: gain over the held out fixed plan hull at equal mean latency (27 pooled tests, 2,000 questions each).", "llrrrr")
    for nm, F in (("single step plans", A), ("with fixed two hop plans", B)):
        pos = F[(F.p_holm < 0.05) & (F.ci_low > 0)]; neg = F[(F.p_holm < 0.05) & (F.ci_high < 0)]
        VERD.append(f"H1 primary family, {nm}: {len(F)} tests, positive survivors {len(pos)}, negative survivors {len(neg)} "
                    f"({', '.join(sorted(set(neg.method)))}); largest gain {F.gain.max():+.3f}")
    return A, B


# ====================================================================== secondary family
def secondary(tag, mt):
    r = pd.read_csv(AN / tag / "routers_pooled.csv"); r = r[r.method.isin(SECONDARY)][["reader", "method", "gain", "ci_low", "ci_high", "p"]]
    m = pd.read_csv(AN / mt / "metis.csv"); m = m[(m.dataset == "pooled") & (m.method == "metis")][["reader", "method", "gain", "ci_low", "ci_high", "p"]]
    F = pd.concat([r, m], ignore_index=True)
    F["p_holm"] = holm(F.p.values)
    return F


def t_secondary():
    A, B = secondary("final", "metis"), secondary("final_r3", "metis_r3")
    n_ok = len(A) == 30 and len(B) == 30
    rows = []
    for m in SECONDARY + ["metis"]:
        for rd in READERS:
            a = A[(A.method == m) & (A.reader == rd)]; b = B[(B.method == m) & (B.reader == rd)]
            a = a.iloc[0] if len(a) else None; b = b.iloc[0] if len(b) else None
            rows.append({"Method": PN[m], "Reader": RN[rd],
                         "Gain, single step": "" if a is None else f"{sg(a.gain)} [{sg(a.ci_low)}, {sg(a.ci_high)}]",
                         "Holm p": "" if a is None else pv(a.p_holm),
                         "Gain, with two hop": "" if b is None else f"{sg(b.gain)} [{sg(b.ci_low)}, {sg(b.ci_high)}]",
                         "Holm p ": "" if b is None else pv(b.p_holm)})
    T = pd.DataFrame(rows)
    md_table("Secondary family (30 tests per plan set; ar_tfidf excluded, appendix only)", T,
             "Decision rule (amendment 2026-10-12 20:30): significant only if Holm p < 0.05 and CI above 0 in BOTH plan sets.")
    tex_table("secondary", T, "Secondary family: stronger routers and the METIS port (30 tests per plan set).", "llrrrr")
    sa = (A.p_holm < 0.05) & (A.ci_low > 0); sb = (B.p_holm < 0.05) & (B.ci_low > 0)
    ka = set(map(tuple, A[sa][["reader", "method"]].values)); kb = set(map(tuple, B[sb][["reader", "method"]].values))
    VERD.append(f"Secondary family complete in both plan sets: {n_ok} ({len(A)} and {len(B)} tests)")
    VERD.append(f"Secondary: positive in single step set {sorted(ka)}; in two hop set {sorted(kb)}; in BOTH (counts) {sorted(ka & kb)}")
    VERD.append(f"Secondary: negative survivors single step {int(((A.p_holm < 0.05) & (A.ci_high < 0)).sum())}, two hop {int(((B.p_holm < 0.05) & (B.ci_high < 0)).sum())}")


# ====================================================================== H3 and sensitivity
def t_h3():
    rows, verd = [], {}
    H = {lab: pd.read_csv(AN / f"h3_ircot{suf}" / "h3.csv") for lab, suf in
         (("final", ""), ("lenient", "_lenient"), ("strict", "_strict")) if (AN / f"h3_ircot{suf}" / "h3.csv").exists()}
    F = H["final"]
    for _, a in F[F.hull == "single_step"].iterrows():
        b = F[(F.hull == "with_r3") & (F.reader == a.reader) & (F.dataset == a.dataset)].iloc[0]
        sig = lambda lab, hull: (lambda x: "yes" if len(x) and x.p_holm.iloc[0] < 0.05 and x.ci_low.iloc[0] > 0 else "no")(
            H[lab][(H[lab].hull == hull) & (H[lab].reader == a.reader) & (H[lab].dataset == a.dataset)]) if lab in H else ""
        rows.append({"Reader": RN[a.reader], "Dataset": DN[a.dataset], "IRCoT success": f"{a.ircot_success:.3f}",
                     "Latency vs best plan": f"{a.ircot_cost_s / a.best_fixed_cost_s:.0f}x",
                     "Gain, single step hull": f"{sg(a.gain)} [{sg(a.ci_low)}, {sg(a.ci_high)}]", "Holm p": pv(a.p_holm),
                     "Gain, with two hop": sg(b.gain), "Holm p ": pv(b.p_holm),
                     "Sig. final/lenient/strict": "/".join(sig(l, "single_step") for l in ("final", "lenient", "strict"))})
    T = pd.DataFrame(rows)
    md_table("H3, IRCoT against the hull at IRCoT's own latency (9 pre-registered tests)", T)
    tex_table("h3", T, "H3: IRCoT against the held out fixed plan hull at IRCoT's own mean latency.", "llrrrrrrl")
    for lab, D in H.items():
        for hull, g in D.groupby("hull"):
            k = g[(g.p_holm < 0.05) & (g.ci_low > 0)]
            VERD.append(f"H3 [{lab}, {hull}]: {len(k)} of {len(g)} significant: " + ", ".join(f"{RN[r.reader]} {DN[r.dataset]} {r.gain:+.3f}" for r in k.itertuples()))
    # robust: significant under all three rules with the single step hull
    if len(H) == 3:
        keys = None
        for lab, D in H.items():
            g = D[D.hull == "single_step"]; k = set(map(tuple, g[(g.p_holm < 0.05) & (g.ci_low > 0)][["reader", "dataset"]].values))
            keys = k if keys is None else keys & k
        VERD.append(f"H3 robust under all three label rules (single step hull): {sorted(keys)}")


def t_sensitivity():
    rows = []
    for lab in ("final", "lenient", "strict"):
        for r3 in (False, True):
            f = AN / (lab + ("_r3" if r3 else ""))
            if not (f / "per_query_gains.csv").exists():
                continue
            F = primary_family(f)
            rows.append({"Label rule": lab, "Plan set": "with two hop" if r3 else "single step",
                         "Positive survivors": int(((F.p_holm < 0.05) & (F.ci_low > 0)).sum()),
                         "Negative survivors": int(((F.p_holm < 0.05) & (F.ci_high < 0)).sum()),
                         "Largest gain": sg(F.gain.max())})
    T = pd.DataFrame(rows)
    md_table("Grading sensitivity: H1 primary family under three label rules", T)
    tex_table("sensitivity", T, "H1 under three grading rules (27 tests per row).", "llrrr")


def t_step():
    rows = []
    for suf, nm in (("", "single step"), ("_r3", "with two hop")):
        f = AN / f"step_hull_primary{suf}.csv"
        if not f.exists():
            continue
        J = pd.read_csv(f)
        rows.append({"Plan set": nm, "Positive vs hull": int(((J.p_holm_mix < 0.05) & (J.gain_mix > 0)).sum()),
                     "Positive vs best single plan": int(((J.p_holm_step < 0.05) & (J.ci_low_step > 0)).sum()),
                     "Mean value of non adaptive mixing": sg(J.mix_value.mean()),
                     "Largest gain vs best single plan": sg(J.gain_step.max())})
    if rows:
        T = pd.DataFrame(rows)
        md_table("Baseline without randomised mixing (step baseline, 27 primary tests)", T,
                 "Value of non adaptive mixing = what a coin flip between two fixed plans gains over the best single plan at the same cost.")
        tex_table("step", T, "Primary family against the best single fixed plan at or below the policy's cost.", "lrrrr")
    else:
        VERD.append("Step baseline: not run yet (scripts/25_step_hull.py)")


# ====================================================================== R3, R6, IRCoT behaviour, headroom
def t_r3():
    F = pd.read_csv(AN / "final_r3" / "fixed_plans.csv"); rows = []
    for rd in READERS:
        for ds in DN:
            g = F[(F.reader == rd) & (F.dataset == ds)]
            s = g[~g.plan.str.startswith("bm25_2hop")]; t = g[g.plan.str.startswith("bm25_2hop")]
            if not len(t):
                continue
            bs, bt = s.loc[s.success.idxmax()], t.loc[t.success.idxmax()]
            rows.append({"Reader": RN[rd], "Dataset": DN[ds], "Best single step": f"{bs.success:.3f} ({bs.plan}, {bs.cost_s:.2f} s)",
                         "Best two hop": f"{bt.success:.3f} ({bt.plan}, {bt.cost_s:.2f} s)", "Difference": sg(bt.success - bs.success)})
    T = pd.DataFrame(rows)
    md_table("R3, best fixed two hop plan against the best single step plan (in sample, label_final)", T)
    tex_table("r3", T, "Fixed two hop plans against the best single step plan (in sample).", "llllr")


def t_r6():
    W = pd.read_csv(AN / "weak_protocol" / "weak_vs_ours.csv")
    win_w = (W.margin_vs_always > 0) & (W.margin_vs_never > 0); win_o = (W.ours_ci_low > 0) & (W.ours_p_holm < 0.05)
    g = W.groupby("method")[["margin_vs_always", "margin_vs_never", "cost_vs_always", "ours_gain"]].mean()
    T = pd.DataFrame([{"Policy": PN[m], "Margin vs Always RAG": sg(r.margin_vs_always), "Margin vs Never RAG": sg(r.margin_vs_never),
                       "Cost vs Always RAG": f"{r.cost_vs_always:.2f}x", "Gain, our protocol": sg(r.ours_gain)} for m, r in g.iterrows()])
    md_table("R6, weak protocol against ours (same answers)", T,
             f"Cells winning under the weak protocol: {int(win_w.sum())} of {len(W)} ({int((win_w & (W.cost_vs_always < 1)).sum())} also cheaper than Always RAG); under ours: {int(win_o.sum())}.")
    tex_table("r6", T, "Weak protocol (strict EM, one seed, no cost matching) against ours.", "lrrrr")
    VERD.append(f"R6: weak protocol wins {int(win_w.sum())} of {len(W)}; our protocol {int(win_o.sum())}")


def t_ircot():
    rows = []
    for rd in ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]:
        d = pd.read_csv(PROC / "r4_ircot" / f"labels_{rd}_graded.csv", keep_default_na=False)
        for ds, g in d.groupby("dataset"):
            rows.append({"Reader": RN[rd], "Dataset": DN[ds], "label_final": f"{g.label_final.astype(int).mean():.3f}",
                         "strict": f"{g.strict.astype(int).mean():.3f}", "LM calls": f"{pd.to_numeric(g.lm_calls).mean():.2f}",
                         "Retrieval calls": f"{pd.to_numeric(g.retrieval_calls).mean():.2f}",
                         "Stops on answer": f"{(g.stopped_on_answer.astype(str) == 'True').mean():.3f}",
                         "Seconds per question": f"{(pd.to_numeric(g.agent_time_s) + pd.to_numeric(g.retrieval_time_s)).mean():.1f}",
                         "Demos dropped (questions)": int((pd.to_numeric(g.get('demos_dropped_max', 0), errors='coerce').fillna(0) > 0).sum())})
    T = pd.DataFrame(rows)
    md_table("R4, IRCoT behaviour", T)
    tex_table("ircot", T, "IRCoT runs: accuracy, calls, stopping and latency.", "llrrrrrrr")


def t_headroom():
    rows = []
    for rd in READERS:
        d = pd.read_csv(PROC / f"tsr_labels_{rd}_graded.csv", dtype={"query_id": str}, keep_default_na=False,
                        usecols=["dataset", "query_id", "retriever", "depth", "repeat", "label_final"])
        d = d[d.repeat.astype(int) == 0]; d["depth"] = d.depth.astype(int); d = d[d.depth <= 10]
        d["plan"] = d.retriever + "@" + d.depth.astype(str); d["y"] = d.label_final.astype(int)
        for ds, g in d.groupby("dataset"):
            Y = g.pivot_table(index="query_id", columns="plan", values="y"); acc = Y.mean()
            rows.append({"Reader": RN[rd], "Dataset": DN[ds], "Closed book": f"{acc.get('closed_book@0', np.nan):.3f}",
                         "Best fixed plan": f"{acc.max():.3f}", "Oracle": f"{Y.max(axis=1).mean():.3f}",
                         "Headroom": f"{Y.max(axis=1).mean() - acc.max():.3f}", "Unsolved": f"{(Y.max(axis=1) == 0).mean():.3f}"})
    T = pd.DataFrame(rows)
    md_table("H2, headroom (in sample, plans with depth <= 10 for every reader)", T)
    tex_table("headroom", T, "Headroom per reader and dataset (in sample, 25 plans).", "llrrrrr")


# ====================================================================== amendment 13 Oct 20:45
def t_noninf():
    """(a) Non superiority at the preregistered 0.03 margin: one sided t test per primary test, Holm over 27."""
    from scipy import stats
    rows = []
    for tag, nm in (("final", "single step"), ("final_r3", "with two hop")):
        pq = pd.read_csv(AN / tag / "per_query_gains.csv", dtype={"query_id": str})
        pq = pq[pq.method.isin(PRIMARY)]
        R = []
        for (rd, m), g in pq.groupby(["reader", "method"]):
            d = g.gain.to_numpy(float); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
            R.append({"reader": rd, "method": m, "gain": d.mean(), "ub95": d.mean() + stats.t.ppf(0.95, n - 1) * se,
                      "p": stats.t.cdf((d.mean() - 0.03) / se, n - 1)})
        R = pd.DataFrame(R); R["p_holm"] = holm(R.p.values)
        top = R.loc[R.ub95.idxmax()]
        rows.append({"Plan set": nm, "Tests": len(R), "Upper bound below 0.03": int((R.ub95 < 0.03).sum()),
                     "Rejected at 0.03 after Holm": int((R.p_holm < 0.05).sum()),
                     "Largest upper bound": f"{top.ub95:.3f} ({PN[top.method]}, {RN.get(top.reader, top.reader)})"})
    T = pd.DataFrame(rows)
    md_table("Non superiority at the preregistered margin of 0.03 (one sided 95% upper bounds)", T,
             "Secondary analysis (amendment 13 Oct 20:45 (a)); the margin was preregistered on 8 Oct, the test was added later and its result was seen before the amendment.")
    tex_table("noninf", T, "Non superiority at the preregistered 0.03 margin (27 primary tests per plan set).", "lrrrl")
    NONINF.extend(rows)
    for r in rows:
        VERD.append(f"Non superiority at 0.03, {r['Plan set']}: {r['Rejected at 0.03 after Holm']} of {r['Tests']} after Holm; largest bound {r['Largest upper bound']}")


def t_cost_axes():
    """(d) The primary family under other cost axes."""
    A_ = pd.read_csv(AN / "cost_axes_primary.csv")
    AXN = {"latency": "Latency (main)", "tokens": "Reader tokens", "tokens_w4": "Tokens, output x4", "flops": "FLOPs", "calls": "Calls"}
    rows = []
    for ax in AXN:
        r = {"Cost axis": AXN[ax]}
        for ps, nm in (("single_step", "single step"), ("two_hop", "two hop")):
            g = A_[(A_.axis == ax) & (A_.plan_set == ps)]
            r[f"Positive, {nm}"] = int(((g.p_holm < 0.05) & (g.ci_low > 0)).sum())
            r[f"Negative, {nm}"] = int(((g.p_holm < 0.05) & (g.ci_high < 0)).sum())
            r[f"Largest gain, {nm}"] = sg(g.gain.max())
        rows.append(r)
    T = pd.DataFrame(rows)
    md_table("H1 primary family under five cost axes (27 tests per cell)", T,
             "Calls give every depth of a retriever the same cost, so this axis cannot price depth; Adaptive k is negative there for that reason.")
    tex_table("cost", T, "Primary family under five cost axes: positive and negative Holm survivors.", "lrrrrrr")
    VERD.append("Cost axes: positive survivors under latency, tokens, tokens x4, FLOPs, calls = "
                + ", ".join(f"{r['Positive, single step']}/{r['Positive, two hop']}" for r in rows))


def t_stronger():
    """(c) Classifier gate and planner with self knowledge (28)."""
    F = pd.read_csv(AN / "stronger_policies" / "family.csv")
    nmap = {"clf_gate": "Classifier gate (c1)", "planner_sk": "Planner with self knowledge (c2)"}
    rows = []
    for m in ("clf_gate", "planner_sk"):
        for rd in READERS:
            a = F[(F.method == m) & (F.reader == rd) & (F.plan_set == "single step")]
            b = F[(F.method == m) & (F.reader == rd) & (F.plan_set == "two hop")]
            if not len(a) or not len(b):
                continue
            a, b = a.iloc[0], b.iloc[0]
            rows.append({"Policy": nmap[m], "Reader": RN[rd],
                         "Gain, single step [95% CI]": f"{sg(a.gain)} [{sg(a.ci_low)}, {sg(a.ci_high)}]", "Holm p": pv(a.p_holm),
                         "Gain, with two hop [95% CI]": f"{sg(b.gain)} [{sg(b.ci_low)}, {sg(b.ci_high)}]", "Holm p ": pv(b.p_holm)})
    T = pd.DataFrame(rows)
    md_table("Stronger learned policies (secondary family of 10 tests per plan set)", T)
    tex_table("stronger", T, "Classifier gate and planner with closed book confidence features against the hull.", "llrrrr")
    pos = F[(F.p_holm < 0.05) & (F.ci_low > 0)].groupby(["reader", "method"]).plan_set.nunique()
    VERD.append(f"Stronger policies: {int((pos == 2).sum())} of 10 positive in both plan sets; range of gains {F.gain.min():+.3f} to {F.gain.max():+.3f}")
    f = AN / "stronger_policies" / "learning_curve.csv"
    if f.exists():
        LC = pd.read_csv(f)
        g = LC.groupby(["reader", "frac"])[["sum_gain", "n"]].sum()      # pooled gain per question = sum / count
        pts = (g["sum_gain"] / g["n"]).unstack()
        T2 = pts.reset_index().rename(columns={"reader": "Reader"})
        T2["Reader"] = T2["Reader"].map(RN)
        T2.columns = ["Reader"] + [f"{int(c * 100)}% of training" for c in pts.columns]
        for c in T2.columns[1:]:
            T2[c] = T2[c].map(sg)
        md_table("Learning curve of the planner with self knowledge (pooled gain over the hull, single step plans)", T2)
        tex_table("curve", T2, "Gain over the hull as the training share grows (descriptive).", "l" + "r" * (len(T2.columns) - 1))


def t_checks():
    """(b), (e), (f): sanity table, grading factor, agreement, matched cost headroom, mechanism (29)."""
    C = AN / "checks"
    S_ = pd.read_csv(C / "sanity_llama.csv")
    T = pd.DataFrame([{"Dataset": DN[r.dataset], "Published Never RAG": f"{r.published_never:.3f}", "Ours, closed book": f"{r.ours_closed_book:.3f}",
                       "Published Always RAG": f"{r.published_always:.3f}", "Ours, BM25 k=5": f"{r['ours_bm25@5']:.3f}",
                       "Ours, BM25 k=10": f"{r['ours_bm25@10']:.3f}"} for _, r in S_.iterrows()])
    md_table("Sanity check against Moskvoretskii et al. (ACL 2025), Llama 3.1 8B, answer contained accuracy", T,
             "Published values from Table 1 of Moskvoretskii et al. (checked 13 Oct 2026); their corpus is the Karpukhin et al. Wikipedia and their retrieval depth is not stated.")
    tex_table("sanity", T, "Llama 3.1 8B against published Never and Always RAG (answer contained accuracy).", "lrrrrr")
    W = pd.read_csv(AN / "weak_protocol" / "weak_vs_ours.csv"); a = W[W.method == "adaptive_rag"]
    VERD.append(f"Adaptive RAG vs Always RAG (weak protocol): mean margin {a.margin_vs_always.mean():+.3f}, above in "
                f"{int((a.margin_vs_always > 0).sum())} of {len(a)} cells (implementation behaves as published)")
    G = pd.read_csv(C / "grading_J.csv"); Ag = pd.read_csv(C / "rater_agreement.csv")
    T = pd.DataFrame([{"Audit": r.audit, "Sensitivity": f"{r.sensitivity:.3f}", "Specificity": f"{r.specificity:.3f}",
                       "J": f"{r.J:.3f}", "J, reweighted": f"{r.J_weighted:.3f}",
                       "Raters: raw agreement": f"{g.raw_agreement:.3f}", "Cohen kappa": f"{g.cohen_kappa:.3f}",
                       "Gwet AC1": f"{g.gwet_ac1:.3f}", "Krippendorff alpha": f"{g.krippendorff_alpha:.3f}"}
                      for (_, r), (_, g) in zip(G.iterrows(), Ag.iterrows())])
    md_table("Grading error and rater agreement", T,
             "Observed differences equal J times true differences when grading error is the same in both arms.")
    tex_table("grading", T, "label_final against the human majority (J) and agreement of the two raters before adjudication.", "lrrrrrrrr")
    H = pd.read_csv(C / "headroom_matched.csv")
    T = H.groupby("reader")[["oracle_k2", "gap_k2", "gap_k3", "gap_all", "oracle_all_final", "oracle_all_strict"]].mean().reindex(READERS).reset_index()
    T = pd.DataFrame([{"Reader": RN[r.reader], "Oracle gap, 2 plans": sg(r.gap_k2), "3 plans": sg(r.gap_k3), "All plans": sg(r.gap_all),
                       "Any plan correct, label_final": f"{r.oracle_all_final:.3f}", "Any plan correct, strict": f"{r.oracle_all_strict:.3f}"}
                      for _, r in T.iterrows()])
    Fg = pd.read_csv(C / "fraction_of_gap.csv")
    best = Fg.groupby("reader").fraction_of_gap_k2.max().reindex(READERS)
    T["Largest share of the 2 plan gap closed by a policy"] = [f"{v:.2f}" for v in best.values]
    md_table("Headroom at matched cost (held out oracles against the hull at the same mean cost)", T)
    tex_table("headroom_matched", T, "Held out oracle gap over the hull at matched cost, and the share policies close.", "lrrrrrr")
    VERD.append(f"Headroom at matched cost: 2 plan oracle gap {H.gap_k2.mean():+.3f}; best policy closes at most {Fg.fraction_of_gap_k2.max():.2f} of it")
    M = pd.read_csv(C / "mechanism.csv"); E = pd.read_csv(C / "exposure_policy_vs_hull.csv")
    g = M.groupby("policy")[["per_plan_auroc", "mean_cross_plan_corr", "pair_auroc", "modal_plan_share"]].mean()
    e = E.groupby("policy")[["policy_judge", "hull_judge", "policy_clong", "hull_clong"]].mean()
    T = pd.DataFrame([{"Policy": {"planner_tsr": "TSR planner", "planner_sk": "Planner with self knowledge"}[k],
                       "Per plan AUROC": f"{r.per_plan_auroc:.3f}", "Which hull plan wins, AUROC": f"{r.pair_auroc:.3f}",
                       "Most used plan share": f"{r.modal_plan_share:.3f}",
                       "Judge share, policy / hull": f"{e.loc[k, 'policy_judge']:.3f} / {e.loc[k, 'hull_judge']:.3f}"}
                      for k, r in g.iterrows()])
    md_table("Mechanism: planners predict difficulty, not which plan wins (seed 0 folds, single step plans)", T)
    tex_table("mechanism", T, "What the planners learn, and grading exposure of policy and hull.", "lrrrr")
    VERD.append("Mechanism: pair AUROC " + ", ".join(f"{k} {r.pair_auroc:.3f}" for k, r in g.iterrows()))


# ====================================================================== audits
def t_audit():
    s8 = importlib.util.spec_from_file_location("g8", Path(__file__).parent / "08_grading_full.py")
    G = importlib.util.module_from_spec(s8); s8.loader.exec_module(G)
    adj = pd.read_csv(G.GDIR / "audit_adjudicated.csv")
    j = pd.read_csv(G.JUDGE); jmap = dict(zip(j["pair"], (j["p_yes"] >= 0.5).astype(int)))
    graded = {}
    for f in G.label_files(main_only=True):
        g = pd.read_csv(f.with_name(f.stem + "_graded.csv"), dtype={"query_id": str}, keep_default_na=False)
        g["pair"] = g["dataset"] + "::" + g["query_id"] + "::" + g["answer"].map(G.normalize)
        graded.update(g.drop_duplicates("pair").set_index("pair")[["strict", "label_f1_07", "label_lenient", "ambiguous"]].to_dict("index"))
    rows = []
    for aud, name in (("audit1", "Audit 1 (7B, Llama, Qwen3 14B)"), ("audit2", "Audit 2 (Gemma, Qwen3 32B)")):
        a = adj[adj.audit == aud].copy()
        a = a[a.pair.isin(graded)]
        for c in ["strict", "label_f1_07", "label_lenient", "ambiguous"]:
            a[c] = [graded[p][c] for p in a.pair]
        a["final"] = [jmap.get(p, s_) if amb == 1 else s_ for p, s_, amb in zip(a.pair, a.strict, a.ambiguous)]
        AUD[name] = a
        kap = G.kappa(a.rater1.astype(int), a.rater2.astype(int))
        for c, nm in (("strict", "strict EM"), ("label_f1_07", "F1 >= 0.7"), ("label_lenient", "lenient"), ("final", "label_final")):
            ok = (a[c].astype(int) == a.human.astype(int)).values
            rng = np.random.default_rng(0); bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(2000)]
            rows.append({"Audit": name, "Rater kappa (two raters)": f"{kap:.3f}", "Rule": nm, "n": len(a),
                         "Accuracy vs majority of three": f"{ok.mean():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]",
                         "Kappa": f"{G.kappa(a[c].astype(int), a.human.astype(int)):.3f}",
                         "FP": int(((a[c].astype(int) == 1) & (a.human == 0)).sum()), "FN": int(((a[c].astype(int) == 0) & (a.human == 1)).sum())})
    T = pd.DataFrame(rows)
    num = T.assign(acc=T["Accuracy vs majority of three"].str.split(" ").str[0].astype(float),
                   lo=T["Accuracy vs majority of three"].str.extract(r"\[([0-9.]+),")[0].astype(float),
                   hi=T["Accuracy vs majority of three"].str.extract(r", ([0-9.]+)\]")[0].astype(float))
    num[["Audit", "Rule", "n", "acc", "lo", "hi", "FP", "FN"]].to_csv(OUT / "audit_rules.csv", index=False)
    md_table("Human audits: grading rules against the majority of three raters", T)
    tex_table("audit", T, "Grading rules against the human reference (majority of three raters).", "lrlrrrrr")


# ====================================================================== rater reference (amendment 14 Oct 2026)
def t_rater():
    """Accuracy of each grading rule against three human references: the majority of three raters (main), rater 2
    alone (not an author) and rater 1 alone (an author of the paper). Checks that the grading validation does not
    depend on the author's labels."""
    rules = (("strict", "strict EM"), ("label_f1_07", "F1 >= 0.7"), ("label_lenient", "lenient"), ("final", "label_final"))
    rows = []
    for name, a in AUD.items():
        for c, nm in rules:
            pred = a[c].astype(int).values
            row = {"Audit": name, "Rule": nm, "n": len(a)}
            for ref, col in (("Majority of three", "human"), ("Rater 2 alone (not an author)", "rater2"), ("Rater 1 alone (author)", "rater1")):
                row[ref] = f"{(pred == a[col].astype(int).values).mean():.3f}"
            rows.append(row)
    T = pd.DataFrame(rows)
    md_table("Grading rules against each human reference (rater 1 is a co-author; rater 2 is not)", T)
    tex_table("rater", T, "Grading rules against each human reference. Rater 1 is a co-author who labelled blind; rater 2 is not an author.", "llrrrr")
    for name in AUD:
        g = T[(T.Audit == name) & (T.Rule == "label_final")].iloc[0]
        VERD.append(f"Rater reference, {name}: label_final accuracy vs majority {g['Majority of three']}, "
                    f"vs rater 2 alone {g['Rater 2 alone (not an author)']}, vs rater 1 alone {g['Rater 1 alone (author)']}")


# ====================================================================== verdict summary (the paper's main table)
def t_summary():
    """Every family and robustness check in one table: Holm survivors with the CI above (Pos) or below (Neg) zero, and
    the largest pooled gain, for both plan sets."""
    def cnt(F, gain="gain", lo="ci_low", hi="ci_high", p="p_holm"):
        return (int(((F[p] < 0.05) & (F[lo] > 0)).sum()), int(((F[p] < 0.05) & (F[hi] < 0)).sum()), float(F[gain].max()), len(F))
    rows = []
    def add(name, A, B, **kw):
        a, b = cnt(A, **kw), cnt(B, **kw)
        rows.append({"Test family": name, "Tests": a[3], "Pos/Neg, single step": f"{a[0]}/{a[1]}", "Max, single step": sg(a[2]),
                     "Pos/Neg, with two hop": f"{b[0]}/{b[1]}", "Max, with two hop": sg(b[2])})
    add("Primary: latency (main)", primary_family(AN / "final"), primary_family(AN / "final_r3"))
    C = pd.read_csv(AN / "cost_axes_primary.csv")
    for ax, nm in (("tokens", "Primary: tokens"), ("tokens_w4", "Primary: tokens, output x4"),
                   ("flops", "Primary: FLOPs"), ("calls", "Primary: calls")):
        add(nm, C[(C.axis == ax) & (C.plan_set == "single_step")], C[(C.axis == ax) & (C.plan_set == "two_hop")])
    add("Primary: best single plan\u2020", pd.read_csv(AN / "step_hull_primary.csv"), pd.read_csv(AN / "step_hull_primary_r3.csv"),
        gain="gain_step", lo="ci_low_step", hi="ci_high_step", p="p_holm_step")
    for lab in ("strict", "lenient"):
        add(f"Primary: {lab} grading", primary_family(AN / lab), primary_family(AN / f"{lab}_r3"))
    add("Secondary routers, METIS", secondary("final", "metis"), secondary("final_r3", "metis_r3"))
    St = pd.read_csv(AN / "stronger_policies" / "family.csv")
    add("Stronger policies", St[St.plan_set == "single step"], St[St.plan_set == "two hop"])
    T = pd.DataFrame(rows)
    note = ""
    if len(NONINF) == 2:
        note = (f"Margin test at 0.03 (upper bound below the margin after Holm): {NONINF[0]['Rejected at 0.03 after Holm']} of "
                f"{NONINF[0]['Tests']} single step, {NONINF[1]['Rejected at 0.03 after Holm']} of {NONINF[1]['Tests']} with two hop.")
    md_table("Verdict summary: every family and robustness check (Pos/Neg = Holm survivors with CI above/below 0)", T, note)
    esc = tex_esc
    L = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Every test family and robustness check. Pos and Neg count Holm survivors whose 95\% CI lies above or below zero; Max is the largest pooled gain. Every row compares with the fixed plan hull except the row marked $\dagger$, which compares with the best single fixed plan at the same cost (no mixing).}",
         r"\label{tab:summary}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Single step plans} & \multicolumn{2}{c}{With two hop plans} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-6}", r"Test family & Tests & Pos/Neg & Max & Pos/Neg & Max \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(esc(v) for v in r.values()) + r" \\")
    if note:
        L += [r"\midrule", r"\multicolumn{6}{@{}p{0.95\columnwidth}@{}}{" + note + "}" + r" \\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TEX["summary"] = "\n".join(L)
    hull_rows = [r for r in rows if "best single plan" not in r["Test family"]]
    step = [r for r in rows if "best single plan" in r["Test family"]]
    VERD.append("Summary: rows against the hull with positive survivors in both plan sets: "
                + str(sum(int(r["Pos/Neg, single step"].split("/")[0]) > 0 and int(r["Pos/Neg, with two hop"].split("/")[0]) > 0 for r in hull_rows))
                + f" of {len(hull_rows)}")
    if step:
        VERD.append(f"Against the best single plan (no mixing): positive {step[0]['Pos/Neg, single step'].split('/')[0]} single step, "
                    f"{step[0]['Pos/Neg, with two hop'].split('/')[0]} with two hop; largest gain {step[0]['Max, single step']} and "
                    f"{step[0]['Max, with two hop']}. These wins vanish against the hull, so they come from mixing, not from question features.")


# ====================================================================== compact tables for the paper body
def _n(x, bold=False, d=3, sign=True):
    """A number as a LaTeX cell: true minus sign, optional bold."""
    if pd.isna(x):
        return ""
    x = 0.0 if round(float(x), d) == 0 else float(x)
    t = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    t = t.replace("-", "$-$")
    return r"\textbf{" + t + "}" if bold else t


def _tex(name, caption, colspec, head, body, star=False, size=r"\footnotesize", sep="3pt"):
    env = "table*" if star else "table"
    L = [rf"\begin{{{env}}}[t]", r"\centering" + size, rf"\setlength{{\tabcolsep}}{{{sep}}}", rf"\caption{{{caption}}}",
         rf"\label{{tab:{name}}}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"] + head + [r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    TEX[name] = "\n".join(L)


RET = [("closed_book", "Closed book"), ("bm25", "BM25"), ("splade_v3", "SPLADE v3"), ("bge_m3", "BGE M3 (dense)"),
       ("stella_1_5b", "Stella 1.5B"), ("bge_bm25_rrf", "BGE M3 + BM25 (RRF)"), ("qwen3_reranker_0_6b", "Qwen3 Reranker"),
       ("bm25_2hop", "BM25, two hop")]


def t_p_retrievers():
    """Success of each retriever at depth 5 (two hop: 5 per hop), mean over the five readers, and its retrieval time."""
    F = pd.read_csv(AN / "final_r3" / "fixed_plans.csv")
    F["ret"] = F.plan.str.split("@").str[0]; F["k"] = F.plan.str.split("@").str[1].astype(int)
    body = []
    for key, nm in RET:
        g = F[(F.ret == key) & (F.k == (0 if key == "closed_book" else 5))]
        cells = []
        for ds in DN:
            v = g[g.dataset == ds].groupby("reader").success.mean()
            cells.append(_n(v.mean(), sign=False))
        rt = g.retrieval_s.mean()
        body.append(f"{nm} & " + " & ".join(cells) + f" & {rt:.3f}" + r" \\")
    best = {}
    for ds in DN:
        g = F[(F.dataset == ds) & (F.k == 5)].groupby(["ret"]).success.mean()
        best[ds] = g.idxmax()
    _tex("p_retrievers", r"Fixed plans at depth 5: success averaged over the five readers (in sample, \texttt{label\_final}), and mean retrieval time per question. Two hop: BM25 with 5 passages per hop.",
         "@{}lrrrrr@{}", [r"Plan & NQ & HotpotQA & 2Wiki & MuSiQue & Retr.\ (s) \\"], body)
    md_table("Fixed plans at depth 5, mean success over readers", pd.DataFrame({"best plan at k=5": best}, index=list(best)).reset_index())


def t_p_plans():
    """Per reader and dataset: closed book, best single step plan, per question oracle over all single step plans of
    the reader, and the best two hop plan minus the best single step plan (all in sample)."""
    R3 = pd.read_csv(AN / "final_r3" / "fixed_plans.csv")
    body = []
    for rd in READERS:
        d = pd.read_csv(PROC / f"tsr_labels_{rd}_graded.csv", dtype={"query_id": str}, keep_default_na=False,
                        usecols=["dataset", "query_id", "retriever", "depth", "repeat", "label_final"])
        d = d[d.repeat.astype(int) == 0]
        d["plan"] = d.retriever + "@" + d.depth.astype(str); d["y"] = d.label_final.astype(int)
        cells = []
        for ds in DN:
            Y = d[d.dataset == ds].pivot_table(index="query_id", columns="plan", values="y")
            acc = Y.mean()
            g = R3[(R3.reader == rd) & (R3.dataset == ds)]
            two = g[g.plan.str.startswith("bm25_2hop")].success.max()
            one = g[~g.plan.str.startswith("bm25_2hop")].success.max()
            cells += [_n(acc.get("closed_book@0", np.nan), sign=False), _n(acc.max(), sign=False),
                      _n(Y.max(axis=1).mean(), sign=False), _n(two - one)]
        body.append(f"{RN[rd]} & " + " & ".join(cells) + r" \\")
    head = [r" & \multicolumn{4}{c}{NQ} & \multicolumn{4}{c}{HotpotQA} & \multicolumn{4}{c}{2Wiki} & \multicolumn{4}{c}{MuSiQue} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(l){14-17}",
            r"Reader" + r" & CB & Best & Orc. & 2h" * 4 + r" \\"]
    _tex("p_plans", r"Fixed plans per reader and dataset (in sample, \texttt{label\_final}). CB: closed book. Best: best single step plan. Orc.: share of questions that at least one single step plan answers. 2h: best fixed two hop plan minus the best single step plan.",
         "@{}l" + "rrrr" * 4 + "@{}", head, body, star=True, sep="2.6pt")


def t_p_gains():
    """Gain over the held out hull for every policy and reader in both plan sets; bold = Holm survivor."""
    P = (primary_family(AN / "final"), primary_family(AN / "final_r3"))
    Sx = (secondary("final", "metis"), secondary("final_r3", "metis_r3"))
    St = pd.read_csv(AN / "stronger_policies" / "family.csv")
    St = (St[St.plan_set == "single step"], St[St.plan_set == "two hop"])
    rows = [("Primary", P, "planner_tsr", "TSR planner"), ("Primary", P, "adaptive_rag", "Adaptive RAG"),
            ("Primary", P, "adaptive_k", "Adaptive k"), ("Primary", P, "gate_closed_book", "Closed book gate"),
            ("Primary", P, "conf_cascade", "Confidence cascade"),
            ("Primary", P, ("reader_cascade", "qwen2.5-7b->llama3.1-8b"), "Reader cascade to Llama 3.1 8B"),
            ("Primary", P, ("reader_cascade", "qwen2.5-7b->qwen3-14b"), "Reader cascade to Qwen3 14B"),
            ("Secondary", Sx, "ak_paper", "Adaptive k, published rule"), ("Secondary", Sx, "ar_emb", "Adaptive RAG, BGE M3 classifier"),
            ("Secondary", Sx, "ar_t5", "Adaptive RAG, FLAN-T5 classifier"), ("Secondary", Sx, "plan_emb", "Per plan logistic regression"),
            ("Secondary", Sx, "plan_knn", "Per plan kNN"), ("Secondary", Sx, "metis", "METIS port"),
            ("Stronger", St, "clf_gate", "Classifier gate"), ("Stronger", St, "planner_sk", "Planner, self knowledge")]
    body, last = [], None
    for fam, D, key, nm in rows:
        if last and fam != last:
            body.append(r"\midrule")
        last = fam
        cells = []
        for rd in READERS:
            for F in D:
                if isinstance(key, tuple):
                    r = F[(F.method == key[0]) & (F.reader == key[1])] if rd == "qwen2.5-7b" else F.iloc[0:0]
                else:
                    r = F[(F.method == key) & (F.reader == rd)]
                if not len(r):
                    cells.append(""); continue
                r = r.iloc[0]
                cells.append(_n(r.gain, bold=bool(r.p_holm < 0.05 and (r.ci_low > 0 or r.ci_high < 0))))
        body.append(f"{nm} & " + " & ".join(cells) + r" \\")
    head = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{RN[r]}}}" for r in READERS) + r" \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}",
            r"Policy" + r" & S & T" * 5 + r" \\"]
    _tex("p_gains", r"Gain over the held out fixed plan hull at equal mean latency, pooled over 2,000 questions, for every policy and reader. S: single step plans. T: with the fixed two hop plans. Bold: Holm survivor within its family (27, 30 or 10 tests per plan set) with the 95\% CI above or below zero. Reader cascades start from Qwen2.5 7B.",
         "@{}l" + "rr" * 5 + "@{}", head, body, star=True, sep="4pt")


def t_p_ircot():
    H = pd.read_csv(AN / "h3_ircot" / "h3.csv"); H = H[H.hull == "single_step"]
    body = []
    for rd in ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]:
        d = pd.read_csv(PROC / "r4_ircot" / f"labels_{rd}_graded.csv", keep_default_na=False)
        for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
            g = d[d.dataset == ds]; h = H[(H.reader == rd) & (H.dataset == ds)].iloc[0]
            sec = (pd.to_numeric(g.agent_time_s) + pd.to_numeric(g.retrieval_time_s)).mean()
            body.append(f"{RN[rd]} & {DN[ds]} & {g.label_final.astype(int).mean():.3f} & {h.best_fixed_success:.3f} & "
                        f"{pd.to_numeric(g.lm_calls).mean():.1f} & {pd.to_numeric(g.retrieval_calls).mean():.1f} & {sec:.1f} & "
                        f"{h.ircot_cost_s / h.best_fixed_cost_s:.0f}" + r"$\times$ \\")
    _tex("p_ircot", r"IRCoT runs. Succ.: IRCoT success; Best: best fixed plan (in sample). LM and Ret.: reader and retrieval calls per question. Sec.: seconds per question. Lat.: latency relative to the best fixed plan.",
         "@{}llrrrrrr@{}", [r"Reader & Data & Succ. & Best & LM & Ret. & Sec. & Lat. \\"], body)


def t_p_weak():
    W = pd.read_csv(AN / "weak_protocol" / "weak_vs_ours.csv")
    W["win"] = (W.margin_vs_always > 0) & (W.margin_vs_never > 0)
    order = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade"]
    nm = {"planner_tsr": "TSR planner", "adaptive_rag": "Adaptive RAG", "adaptive_k": "Adaptive k",
          "gate_closed_book": "Closed book gate", "conf_cascade": "Confidence cascade"}
    body = []
    for m in order:
        g = W[W.method == m]
        body.append(f"{nm[m]} & {_n(g.margin_vs_always.mean())} & {_n(g.margin_vs_never.mean())} & {g.cost_vs_always.mean():.2f}"
                    + r"$\times$" + f" & {int(g.win.sum())}/{len(g)} & {_n(g.ours_gain.mean())}" + r" \\")
    _tex("p_weak", r"The same answers under two protocols, mean over 20 reader and dataset cells. Weak protocol: accuracy margin over Always RAG (BM25, $k=5$) and Never RAG, cost relative to Always RAG, and cells that beat both (strict grading, one seed). Ours: gain over the held out hull at equal latency.",
         "@{}lrrrrr@{}", [r" & \multicolumn{4}{c}{Weak protocol} & Ours \\", r"\cmidrule(lr){2-5}\cmidrule(l){6-6}",
                          r"Policy & vs Always & vs Never & Cost & Wins & Gain \\"], body)


def t_p_mechanism():
    M = pd.read_csv(AN / "checks" / "mechanism.csv"); E = pd.read_csv(AN / "checks" / "exposure_policy_vs_hull.csv")
    g = M.groupby("policy")[["per_plan_auroc", "mean_cross_plan_corr", "pair_auroc", "modal_plan_share"]].mean()
    e = E.groupby("policy")[["policy_judge", "hull_judge"]].mean()
    body = []
    for k, nm in (("planner_tsr", "TSR planner"), ("planner_sk", "Self knowledge")):
        r = g.loc[k]
        body.append(f"{nm} & {r.per_plan_auroc:.2f} & {r.mean_cross_plan_corr:.2f} & {r.pair_auroc:.2f} & {r.modal_plan_share:.2f} & "
                    f"{e.loc[k, 'policy_judge']:.3f} / {e.loc[k, 'hull_judge']:.3f}" + r" \\")
    _tex("p_mechanism", r"What the planners learn (held out, seed 0 folds, single step plans, mean over 20 cells). Plan AUC: AUROC of predicted success per plan. Corr.: correlation of predictions across plans. Pair AUC: AUROC for which hull plan answers when exactly one does. Top plan: share of questions sent to the most used plan. Judge: share of answers graded by the judge, policy / hull. Self knowledge: the planner with closed book confidence features.",
         "@{}lrrrrr@{}", [r"Policy & Plan AUC & Corr. & Pair AUC & Top plan & Judge \\"], body, sep="2.4pt")


def t_p_rater():
    rules = (("strict", "Strict EM"), ("label_f1_07", r"F1 $\geq$ 0.7"), ("label_lenient", "Lenient"), ("final", r"\texttt{label\_final}"))
    body = []
    names = list(AUD)
    for c, nm in rules:
        cells = []
        for name in names:
            a = AUD[name]; pred = a[c].astype(int).values
            for col in ("human", "rater1", "rater2"):
                cells.append(f"{(pred == a[col].astype(int).values).mean():.3f}")
        body.append(f"{nm} & " + " & ".join(cells) + r" \\")
    rate = " & ".join(f"{AUD[n][col].astype(int).mean():.2f}" for n in names for col in ("human", "rater1", "rater2"))
    body += [r"\midrule", r"Share marked correct & " + rate + r" \\"]
    _tex("p_rater", r"Accuracy of each grading rule against three human references on the two audits of 200 answers. Maj.: majority of three raters (the reference used in the paper). R1: rater 1 alone (a coauthor, labelled blind). R2: rater 2 alone (not an author). Audit 1 covers Qwen2.5 7B, Llama 3.1 8B and Qwen3 14B; audit 2 covers Gemma 3 27B and Qwen3 32B.",
         "@{}lrrrrrr@{}", [r" & \multicolumn{3}{c}{Audit 1} & \multicolumn{3}{c}{Audit 2} \\",
                          r"\cmidrule(lr){2-4}\cmidrule(l){5-7}", r"Rule & Maj. & R1 & R2 & Maj. & R1 & R2 \\"], body)


def t_p_datasets():
    """Primary policies per dataset (held out gain, mean over the five readers), pooled, and the same gain measured on
    the training folds, where the policy and the hull were fitted (in sample)."""
    M = pd.read_csv(AN / "final" / "methods.csv"); M = M[~M.reader.str.contains("->")]
    nm = {"planner_tsr": "TSR planner", "adaptive_rag": "Adaptive RAG", "adaptive_k": "Adaptive k",
          "gate_closed_book": "Closed book gate", "conf_cascade": "Confidence cascade"}
    body = []
    for m in nm:
        g = M[M.method == m]
        per = g.groupby("dataset").gain.mean()
        pooled = g[g.dataset == "pooled"]
        cells = [_n(per.get(ds, np.nan)) for ds in DN] + [_n(pooled.gain.mean()), _n(g[g.dataset != "pooled"].gain_selection_half.mean())]
        body.append(f"{nm[m]} & " + " & ".join(cells) + r" \\")
    _tex("p_datasets", r"Primary policies per dataset: gain over the hull on held out questions with single step plans, mean over the five readers (descriptive, no test). Train: the same gain measured on the training folds, where the policy and the hull were fitted (mean over datasets and readers).",
         "@{}lrrrrrr@{}", [r" & \multicolumn{5}{c}{Held out} & \\", r"\cmidrule(lr){2-6}",
                          r"Policy & NQ & Hotpot & 2Wiki & MuSiQue & Pooled & Train \\"], body, sep="2.6pt")


def t_p_curve():
    LC = pd.read_csv(AN / "stronger_policies" / "learning_curve.csv")
    g = LC.groupby(["reader", "frac"])[["sum_gain", "n"]].sum()
    pts = (g["sum_gain"] / g["n"]).unstack()
    body = [f"{RN[rd]} & " + " & ".join(_n(pts.loc[rd, c]) for c in pts.columns) + r" \\" for rd in READERS if rd in pts.index]
    head = [r"Reader & " + " & ".join(f"{int(round(c * 100))}\\%" for c in pts.columns) + r" \\"]
    _tex("p_curve", r"Learning curve of the planner with self knowledge features: pooled gain over the hull (single step plans) when it is trained on a share of the training folds.",
         "@{}l" + "r" * len(pts.columns) + "@{}", head, body)


def t_p_sanity():
    S_ = pd.read_csv(AN / "checks" / "sanity_llama.csv")
    body = [f"{DN[r.dataset]} & {r.published_never:.3f} & {r.ours_closed_book:.3f} & {r.published_always:.3f} & "
            f"{r['ours_bm25@5']:.3f} & {r['ours_bm25@10']:.3f}" + r" \\" for _, r in S_.iterrows()]
    _tex("p_sanity", r"Llama 3.1 8B against the values published by Moskvoretskii et al.\ (their Table 1), answer contained accuracy. Pub.: published. Their corpus and prompt differ from ours.",
         "@{}lrrrrr@{}", [r" & \multicolumn{2}{c}{Never RAG} & \multicolumn{3}{c}{Always RAG (BM25)} \\",
                          r"\cmidrule(lr){2-3}\cmidrule(l){4-6}", r"Dataset & Pub. & Ours & Pub. & Ours $k$=5 & Ours $k$=10 \\"], body)


def t_p_headroom():
    C = AN / "checks"
    H = pd.read_csv(C / "headroom_matched.csv")
    T = H.groupby("reader")[["gap_k2", "gap_k3", "gap_all", "oracle_all_final", "oracle_all_strict"]].mean().reindex(READERS)
    best = pd.read_csv(C / "fraction_of_gap.csv").groupby("reader").fraction_of_gap_k2.max().reindex(READERS)
    body = [f"{RN[rd]} & {_n(r.gap_k2)} & {_n(r.gap_k3)} & {_n(r.gap_all)} & {r.oracle_all_final:.3f} & {r.oracle_all_strict:.3f} & {best[rd]:.2f}" + r" \\"
            for rd, r in T.iterrows()]
    _tex("p_headroom", r"Held out headroom at matched cost, mean over datasets. Gap: oracle over 2, 3 or all fixed plans minus the hull at the oracle's mean latency. Any: share of questions that some plan answers, under \texttt{label\_final} and the strict rule. Closed: largest share of the 2 plan gap closed by a primary policy.",
         "@{}lrrrrrr@{}", [r" & \multicolumn{3}{c}{Gap} & \multicolumn{2}{c}{Any} & \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
                          r"Reader & 2 & 3 & All & Final & Strict & Closed \\"], body, sep="2.8pt")


def t_p_h3rules():
    H = {lab: pd.read_csv(AN / f"h3_ircot{suf}" / "h3.csv") for lab, suf in (("final", ""), ("lenient", "_lenient"), ("strict", "_strict"))}
    body = []
    F = H["final"]
    for _, a in F[F.hull == "single_step"].iterrows():
        cells = []
        for lab, hull in (("final", "single_step"), ("lenient", "single_step"), ("strict", "single_step"), ("final", "with_r3")):
            r = H[lab][(H[lab].hull == hull) & (H[lab].reader == a.reader) & (H[lab].dataset == a.dataset)].iloc[0]
            cells.append(_n(r.gain, bold=bool(r.p_holm < 0.05 and r.ci_low > 0)))
        body.append(f"{RN[a.reader]} & {DN[a.dataset]} & " + " & ".join(cells) + r" \\")
    _tex("p_h3rules", r"IRCoT gain over the held out hull at its own latency under three grading rules. Bold: Holm survivor over the 9 tests with the 95\% CI above zero. Last column: hull with the fixed two hop plans, \texttt{label\_final}.",
         "@{}llrrrr@{}", [r" & & \multicolumn{3}{c}{Single step hull} & Two hop \\", r"\cmidrule(lr){3-5}\cmidrule(l){6-6}",
                          r"Reader & Data & Final & Lenient & Strict & Final \\"], body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MD.append("# FunnelPlan final tables\n\nGenerated by scripts/12_tables_final.py from the result files. Gains are differences in the share of questions answered correctly.")
    for fn in (t_h1, t_noninf, t_secondary, t_h3, t_sensitivity, t_step, t_cost_axes, t_stronger, t_r3, t_r6,
               t_ircot, t_headroom, t_checks, t_audit, t_rater, t_summary,
               t_p_retrievers, t_p_plans, t_p_gains, t_p_ircot, t_p_weak, t_p_mechanism, t_p_rater,
               t_p_datasets, t_p_curve, t_p_sanity, t_p_headroom, t_p_h3rules):
        try:
            fn()
        except FileNotFoundError as e:
            VERD.append(f"{fn.__name__}: input missing ({e.filename})")
    (OUT / "tables.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
    for k, v in TEX.items():
        (OUT / f"tab_{k}.tex").write_text(v + "\n", encoding="utf-8")
    (OUT / "verdicts.txt").write_text("\n".join(VERD) + "\n", encoding="utf-8")
    print("\n".join(VERD))
    print(f"\nWritten {OUT}: tables.md, verdicts.txt and {len(TEX)} LaTeX tables")


if __name__ == "__main__":
    main()
