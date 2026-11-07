"""
Final tables for the paper (5 readers, both plan sets). CPU only, reads results; never reruns an analysis.

Writes results_full/tables_final/:
  tables.md          every table in readable form (the source for the paper text)
  tab_*.tex          booktabs LaTeX tables
  verdicts.txt       the decisions of every test family, computed from the files (H1, secondary family, H2, robustness)
  tab_summary.tex    every family and robustness check in one table (the paper's verdict table)
  tab_rater.tex      grading rules against each human reference (majority of three, each rater alone)
  tab_p_*.tex        compact tables for the paper body (retrievers, plans, gains, IRCoT, weak protocol, mechanism, raters)
  tab_review_*.tex   review checks (scripts 24 and 25): inference, refit, mechanism, headroom, overhead, curve, latency, ablation
  tab_rev2_*.tex     further checks (scripts 26 to 33): inference, bounds, mixed workload, decomposition, curve,
                     difference routers, checks, exposure, agreement, p95, sweep, closed book prompt, Adaptive RAG with
                     IRCoT, MDE, and (when run) outside judge and audit 3

Inputs (all produced by earlier scripts): analysis/final[_r3]/ (10, 13), analysis/metis[_r3]/ (23),
analysis/h3_ircot[_lenient|_strict]/ (21, 24), analysis/weak_protocol/ (22), analysis/lenient*|strict* (24),
analysis/step_hull_primary*.csv (25, optional), analysis/cost_axes_primary.csv (27), analysis/stronger_policies (28),
analysis/checks (29), grading/audit_adjudicated.csv (audits), r4_ircot labels (20).

Usage: python scripts/22_tables_final.py               (writes results_full/tables_final/)
       python scripts/22_tables_final.py --out DIR     (writes to DIR instead; for testing)
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
      "plan_emb": "Per plan LR", "plan_knn": "Per plan kNN", "metis": "METIS style router"}
PRIMARY = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade", "reader_cascade"]
SECONDARY = ["ak_paper", "ar_emb", "ar_t5", "plan_emb", "plan_knn"]

_s = importlib.util.spec_from_file_location("s24", Path(__file__).parent / "17_label_sensitivity.py")
S = importlib.util.module_from_spec(_s)
_s.loader.exec_module(S)
holm, primary_family = S.holm, S.primary_family

MD, TEX, VERD = [], {}, []
NONINF, AUD = [], {}     # filled by t_noninf and t_audit, read by t_summary and t_rater


def sg(x, d=3):
    if pd.isna(x):
        return ""
    s = f"{x:+.{d}f}"
    return f"{0:.{d}f}" if float(s) == 0 else s      # no "-0.000" for values that round to zero


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
    s = s.replace("<", r"$<$").replace(">", r"$>$").replace("\u2020", r"$^\dagger$").replace("\u2021", r"$^\ddagger$")
    return re.sub(r"(?<![A-Za-z0-9])-(?=\d)", r"$-$", s)


def _hdr(c, a, width=14):
    """A header cell; long headers wrap onto several lines (greedy, about `width` characters each)."""
    lines, words = [], []
    for w in str(c).split(" "):  # a long word with slashes may break after each slash
        words += re.findall(r"[^/]+/?", w) if len(w) > width and "/" in w else [w]
    for w in words:
        if lines and len(lines[-1]) + 1 + len(w) <= width:
            lines[-1] += ("" if lines[-1].endswith("/") else " ") + w
        else:
            lines.append(w)
    if len(lines) == 1:
        return tex_esc(c)
    al = a if a in "lrc" else "c"
    cells = [("{}" if x.startswith("[") else "") + tex_esc(x) for x in lines]   # "\\ [" would read as a length
    return rf"\begin{{tabular}}[b]{{@{{}}{al}@{{}}}}" + r" \\ ".join(cells) + r"\end{tabular}"



# IEEE style: hyphenate compound modifiers in the LaTeX tables, the same rules as the paper text.
_PQ = r"(test|tests|gain|gains|latency|costs?|standard|oracle|interval|intervals|labels?|success|bootstrap|answers?)"
_HYPH = [(r"\bheld out\b(?=\s+[a-zA-Z$\\])", "held-out"), (r"\bHeld out\b(?=\s+[a-zA-Z$\\])", "Held-out"),
         (r"\bsingle step\b", "single-step"), (r"\bSingle step\b", "Single-step"), (r"\btwo hop\b", "two-hop"),
         (r"\bTwo hop\b", "Two-hop"), (r"\bmulti step\b", "multi-step"), (r"\bmulti hop\b", "multi-hop"),
         (r"\bclosed book\b", "closed-book"), (r"\bClosed book\b", "Closed-book"), (r"\bone sided\b", "one-sided"),
         (r"\bnon superiority\b", "non-superiority"), (r"\bhardware free\b", "hardware-free"),
         (r"\bper question\b(?=\s+" + _PQ + r"\b)", "per-question"), (r"\bPer question\b(?=\s+" + _PQ + r"\b)", "Per-question"),
         (r"\bper plan\b(?=\s+(logistic|router|routers|success|AUROC|models?|kNN|LR))", "per-plan"),
         (r"\bPer plan\b(?=\s+(logistic|router|routers|success|AUROC|models?|kNN|LR))", "Per-plan"),
         (r"\bend to end\b", "end-to-end"), (r"\bfew shot\b", "few-shot"), (r"\bchain of thought\b(?=\s+prompt)", "chain-of-thought"),
         (r"\bcross fitted\b", "cross-fitted"), (r"\bself knowledge\b", "self-knowledge"), (r"\bIn sample\b(?=\s+headroom)", "In-sample"),
         (r"\bin sample\b(?=\s+(oracle|headroom))", "in-sample"), (r"\bgradient boosted\b", "gradient-boosted")]



def refit_factor(I):
    """Largest ratio of the refit bootstrap spread to the corrected standard error: the four calibration tests
    (27/24) and, once run, the 12 randomly drawn tests (37), as the analysis plan requires."""
    import pandas as _pd
    f = float((_pd.read_csv(I / "calibration.csv").eval("sd_refit / se_nb")).max())
    r = I / "calibration_random.csv"
    return max(f, float(_pd.read_csv(r).factor.iloc[0])) if r.exists() else f


def ieee_hyphens(t):
    keep = []
    t = re.sub(r"\\(label|ref|cite)\{[^}]*\}", lambda m: keep.append(m.group(0)) or f"\x00{len(keep) - 1}\x00", t)
    for a, b in _HYPH:
        t = re.sub(a, b, t)
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], t)

def tex_table(name, df, caption, align=None, star=False):
    esc = tex_esc
    align = align or "l" * len(df.columns)
    head = " & ".join(_hdr(c, a) for c, a in zip(df.columns, align))
    env = "table*" if star else "table"
    L = [rf"\begin{{{env}}}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}", f"\\caption{{{esc(caption)}}}", f"\\label{{tab:{name}}}",
         f"\\begin{{tabular*}}{{\\linewidth}}{{{_fullwidth(align)}}}", r"\toprule", head + r" \\", r"\midrule"]
    for r in df.itertuples(index=False):
        L.append(" & ".join(esc(v) for v in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", rf"\end{{{env}}}"]
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
             "Decision rule: significant only if Holm p < 0.05 and CI above 0 in BOTH plan sets.")
    tex_table("secondary", T, "Secondary family: stronger routers and the METIS style router (30 tests per plan set).", "llrrrr")
    sa = (A.p_holm < 0.05) & (A.ci_low > 0); sb = (B.p_holm < 0.05) & (B.ci_low > 0)
    ka = set(map(tuple, A[sa][["reader", "method"]].values)); kb = set(map(tuple, B[sb][["reader", "method"]].values))
    VERD.append(f"Secondary family complete in both plan sets: {n_ok} ({len(A)} and {len(B)} tests)")
    VERD.append(f"Secondary: positive in single step set {sorted(ka)}; in two hop set {sorted(kb)}; in BOTH (counts) {sorted(ka & kb)}")
    VERD.append(f"Secondary: negative survivors single step {int(((A.p_holm < 0.05) & (A.ci_high < 0)).sum())}, two hop {int(((B.p_holm < 0.05) & (B.ci_high < 0)).sum())}")


# ====================================================================== H2 (IRCoT) and sensitivity
def t_h3():
    rows, verd = [], {}
    H = {lab: pd.read_csv(AN / f"h3_ircot{suf}" / "h3.csv") for lab, suf in
         (("final", ""), ("lenient", "_lenient"), ("strict", "_strict")) if (AN / f"h3_ircot{suf}" / "h3.csv").exists()}
    F = H["final"]
    for _, a in F[F.hull == "single_step"].iterrows():
        b = F[(F.hull == "with_r3") & (F.reader == a.reader) & (F.dataset == a.dataset)].iloc[0]
        sig = lambda lab, hull: (lambda x: "y" if len(x) and x.p_holm.iloc[0] < 0.05 and x.ci_low.iloc[0] > 0 else "n")(
            H[lab][(H[lab].hull == hull) & (H[lab].reader == a.reader) & (H[lab].dataset == a.dataset)]) if lab in H else ""
        rows.append({"Reader": RN[a.reader], "Dataset": DN[a.dataset], "IRCoT success": f"{a.ircot_success:.3f}",
                     "Latency vs best plan": f"{a.ircot_cost_s / a.best_fixed_cost_s:.0f}x",
                     "Gain, single step hull": sg(a.gain), "Holm p": pv(a.p_holm),
                     "Holm p, refit": pv(a.p_refit_holm) if "p_refit_holm" in a else "",
                     "Gain, with two hop": sg(b.gain), "Holm p ": pv(b.p_holm),
                     "Holm p, refit ": pv(b.p_refit_holm) if "p_refit_holm" in b else "",
                     "Sig. final/len./strict (y/n)": "/".join(sig(l, "single_step") for l in ("final", "lenient", "strict"))})
    T = pd.DataFrame(rows)
    md_table("H2, IRCoT against the hull at IRCoT's own latency (9 tests)", T)
    tex_table("h3", T, "H2: IRCoT against the held out fixed plan hull at IRCoT's own mean latency. Holm p: per question test. Holm p, refit: refit bootstrap (100 resamples of the questions, hull refitted each time).", "llrrrrrrrrl")
    TEX["h3"] = TEX["h3"].replace("\\setlength{\\tabcolsep}{4pt}", "\\setlength{\\tabcolsep}{2pt}")   # 11 columns
    if "p_refit_holm" in F:
        for hull, g in F.groupby("hull"):
            k = g[(g.p_holm < 0.05) & (g.ci_low > 0) & (g.p_refit_holm < 0.05) & (g.refit_low > 0)]
            VERD.append(f"H2 [final, {hull}] wins that also pass the refit bootstrap: {len(k)} of {len(g)}: "
                        + ", ".join(f"{RN[r.reader]} {DN[r.dataset]} {r.gain:+.3f}" for r in k.itertuples())
                        + f" (spread / per question SE {(g.sd_refit / g.se_plain).min():.2f} to {(g.sd_refit / g.se_plain).max():.2f})")
    for lab, D in H.items():
        for hull, g in D.groupby("hull"):
            k = g[(g.p_holm < 0.05) & (g.ci_low > 0)]
            VERD.append(f"H2 [{lab}, {hull}]: {len(k)} of {len(g)} significant: " + ", ".join(f"{RN[r.reader]} {DN[r.dataset]} {r.gain:+.3f}" for r in k.itertuples()))
    # robust: significant under all three rules with the single step hull
    if len(H) == 3:
        keys = None
        for lab, D in H.items():
            g = D[D.hull == "single_step"]; k = set(map(tuple, g[(g.p_holm < 0.05) & (g.ci_low > 0)][["reader", "dataset"]].values))
            keys = k if keys is None else keys & k
        VERD.append(f"H2 robust under all three label rules (single step hull): {sorted(keys)}")


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
        VERD.append("Step baseline: not run yet (scripts/18_step_hull.py)")


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
    tex_table("r6", T, "Weak protocol (strict rule, one seed, no cost matching) against ours.", "lrrrr")
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


# ====================================================================== margin test, cost axes, stronger policies, checks
def t_noninf():
    """(a) Non superiority at the 0.03 margin: one sided t test per primary test, Holm over 27."""
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
    md_table("Non superiority at a margin of 0.03 (one sided 95% upper bounds)", T,
             "Secondary analysis, added after the main results were seen.")
    tex_table("noninf", T, "Non superiority at the 0.03 margin (27 primary tests per plan set).", "lrrrl")
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
    s8 = importlib.util.spec_from_file_location("g8", Path(__file__).parent / "07_grading_full.py")
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
        for c, nm in (("strict", "strict"), ("label_f1_07", "F1 >= 0.7"), ("label_lenient", "lenient"), ("final", "label_final")):
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


# ====================================================================== rater reference
def t_rater():
    """Accuracy of each grading rule against three human references: the majority of three raters (main), rater 2
    alone (not an author) and rater 1 alone (an author of the paper). Checks that the grading validation does not
    depend on the author's labels."""
    rules = (("strict", "strict"), ("label_f1_07", "F1 >= 0.7"), ("label_lenient", "lenient"), ("final", "label_final"))
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
    add("Secondary family", secondary("final", "metis"), secondary("final_r3", "metis_r3"))
    St = pd.read_csv(AN / "stronger_policies" / "family.csv")
    add("Stronger policies", St[St.plan_set == "single step"], St[St.plan_set == "two hop"])
    note = ""
    I = AN / "inference"
    if (I / "tests_nb.csv").exists():
        from scipy import stats
        Tn = pd.read_csv(I / "tests_nb.csv")
        infl = refit_factor(I)
        nb = dict(gain="gain", lo="nb_low", hi="nb_high", p="p_nb_holm")
        pr = Tn[Tn.family == "primary"]
        A_, B_ = pr[pr.plan_set == "single step"], pr[pr.plan_set == "with two hop"]
        add("Primary: corrected test\u2021", A_, B_, **nb)
        def inflate(g):
            se = g.se_nb * infl
            g = g.assign(nb_low=g.gain - stats.t.ppf(0.975, g.df_nb) * se, nb_high=g.gain + stats.t.ppf(0.975, g.df_nb) * se,
                         ub_i=g.gain + stats.t.ppf(0.95, g.df_nb) * se)
            return g.assign(p_nb_holm=holm(np.asarray(2 * stats.t.sf((g.gain / se).abs(), g.df_nb), float)))
        Ai, Bi = inflate(A_), inflate(B_)
        add(f"Primary: corrected, x{infl:.2f}\u2021", Ai, Bi, **nb)
        OJ = AN / "outside_judge" / "tests_mistral_vs_qwen.csv"
        if OJ.exists():
            Oj = pd.read_csv(OJ); Oj = Oj[Oj.family == "primary"]
            add("Primary: outside judge\u2021", Oj[Oj.plan_set == "single step"], Oj[Oj.plan_set == "with two hop"], **nb)
        Ow = I / "tests_secondary_own_cost.csv"
        if Ow.exists():
            Ow = pd.read_csv(Ow)
            add("Secondary, charged\u2021", Ow[Ow.plan_set == "single step"], Ow[Ow.plan_set == "with two hop"], **nb)
        note = (f"Primary upper bounds below 0.03: per question {int((A_.ub95_plain < 0.03).sum())} and "
                f"{int((B_.ub95_plain < 0.03).sum())} of 27; corrected {int((A_.ub95_nb < 0.03).sum())} and {int((B_.ub95_nb < 0.03).sum())}; "
                f"corrected, errors x{infl:.2f}: {int((Ai.ub_i < 0.03).sum())} and {int((Bi.ub_i < 0.03).sum())} (single step and with two hop).")
    T = pd.DataFrame(rows)
    md_table("Verdict summary: every family and robustness check (Pos/Neg = Holm survivors with CI above/below 0)", T, note)
    esc = tex_esc
    L = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Every test family and robustness check. Pos and Neg count Holm survivors whose 95\% interval lies above or below zero; Max is the largest pooled gain. Rows marked $\ddagger$ use the corrected test, the others the per question test. Every row compares with the fixed plan hull except the row marked $\dagger$, which compares with the best single fixed plan at the same cost (no mixing).}",
         r"\label{tab:summary}", r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Single step plans} & \multicolumn{2}{c}{With two hop plans} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-6}", r"Test family & Tests & Pos/Neg & Max & Pos/Neg & Max \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(esc(v).replace(" x1.", " $\\times$1.") for v in r.values()) + r" \\")
    if note:
        L += [r"\midrule", r"\multicolumn{6}{@{}p{0.95\columnwidth}@{}}{" + note.replace(" x1.", " $\\times$1.") + "}" + r" \\"]
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
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


def _fullwidth(colspec):
    """Column spec for tabular* at the full column (or page) width: extra space goes between columns."""
    return (r"@{\extracolsep{\fill}}" + colspec[3:]) if colspec.startswith("@{}") else (r"@{\extracolsep{\fill}}" + colspec)


def _tex(name, caption, colspec, head, body, star=False, size=r"\footnotesize", sep="3pt"):
    env = "table*" if star else "table"
    L = [rf"\begin{{{env}}}[t]", r"\centering" + size, rf"\setlength{{\tabcolsep}}{{{sep}}}", rf"\caption{{{caption}}}",
         rf"\label{{tab:{name}}}", rf"\begin{{tabular*}}{{\linewidth}}{{{_fullwidth(colspec)}}}", r"\toprule"] + head + [r"\midrule"] + body + [r"\bottomrule", r"\end{tabular*}", rf"\end{{{env}}}"]
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
            ("Secondary", Sx, "plan_knn", "Per plan kNN"), ("Secondary", Sx, "metis", "METIS style router"),
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
    rules = (("strict", "Strict"), ("label_f1_07", r"F1 $\geq$ 0.7"), ("label_lenient", "Lenient"), ("final", r"\texttt{label\_final}"))
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


# ====================================================================== review checks (scripts 24 and 25)
def t_review():
    """Tables from the review checks: refit bootstrap, mechanism, same budget headroom, overhead, curve, latency, ablation."""
    R = AN / "review"
    if not R.exists():
        raise FileNotFoundError(str(R))
    from sklearn.metrics import roc_auc_score
    PS = {"single step": "single step", "with two hop": "with two hop"}
    # --- inference: plain against stratified bootstrap
    I = pd.read_csv(R / "inference_stratified.csv")
    I["w"] = (I.strat_high - I.strat_low) / (I.plain_high - I.plain_low)
    rows = []
    for (ps, fam), g in I.groupby(["plan_set", "family"], sort=False):
        rows.append({"Plan set": ps, "Family": fam, "Tests": len(g),
                     "Width ratio (median, range)": f"{g.w.median():.2f} ({g.w.min():.2f} to {g.w.max():.2f})",
                     "Interval above 0, plain / stratified": f"{int((g.plain_low > 0).sum())} / {int((g.strat_low > 0).sum())}",
                     "Upper bound below 0.03, plain / stratified": f"{int((g.plain_high < 0.03).sum())} / {int((g.strat_high < 0.03).sum())}"})
    T = pd.DataFrame(rows)
    md_table("Checks: plain against dataset stratified bootstrap", T)
    tex_table("review_inference", T, "Pooled tests under the plain bootstrap and a bootstrap stratified by dataset. Width ratio: stratified interval width over plain width.", "llrrrr")
    # --- refit bootstrap
    F = pd.read_csv(R / "refit_bootstrap.csv")
    plain = {}
    for ps, d in (("single step", "final"), ("with two hop", "final_r3")):
        for f in ("methods.csv", "routers_pooled.csv"):
            M = pd.read_csv(AN / d / f)
            if "dataset" in M:
                M = M[M.dataset == "pooled"]
            for r in M.itertuples():
                plain[(r.method, r.reader, ps)] = (r.ci_low, r.ci_high, r.gain)
    from scipy import stats
    rows = []
    for r in F.itertuples():
        lo, hi, g = plain.get((r.method, r.reader, r.plan_set), (np.nan, np.nan, np.nan))
        rows.append({"Policy": PN.get(r.method, r.method), "Reader": RN.get(r.reader, r.reader), "Plan set": r.plan_set,
                     "Gain": sg(g), "Plain 95% CI": f"[{sg(lo)}, {sg(hi)}]",
                     "Gain +/- 1.96 refit SD": f"[{sg(g - 1.96 * r.sd)}, {sg(g + 1.96 * r.sd)}]",
                     "p, refit SD": pv(2 * stats.norm.sf(abs(g / r.sd))), "Refit mean shift": sg(r.mean - g)})
    T = pd.DataFrame(rows)
    md_table("Checks: refit bootstrap (100 resamples; folds, policy, hull and weights refitted)", T)
    tex_table("review_refit", T, "Refit bootstrap for the tests closest to a decision: 100 resamples of the questions within each dataset, each rerunning folds, policy fitting, hull and weights (one seed per resample). The interval and p value use the refit spread (standard deviation) around the observed gain. Each resample holds only about 63% distinct questions, so a learned policy trains on less information and its refit mean shifts down (last column); the cascades, which learn little, do not shift.", "lllrrrrr")
    # --- mechanism
    M = pd.read_csv(R / "mechanism_all_seeds.csv")
    P = pd.read_csv(R / "mechanism_pairs.csv")
    rng = np.random.default_rng(0)
    rows = []
    for pol in ("planner_tsr", "planner_sk", "plan_emb"):
        g, p0 = M[M.policy == pol], P[(P.policy == pol) & (P.seed == P.seed.min())]
        blocks = [(x.score_diff.to_numpy(), x.first_wins.to_numpy()) for _, x in p0.groupby(["reader", "dataset"])]
        stat = lambda bl: np.mean([roc_auc_score(y, s) for s, y in bl if 0 < y.mean() < 1])
        bs = []
        for _ in range(500):
            bb = []
            for s, y in blocks:
                i = rng.integers(0, len(s), len(s))
                bb.append((s[i], y[i]))
            bs.append(stat(bb))
        ps = g.groupby("seed").pair_auroc.mean()
        rows.append({"Policy": {"planner_tsr": "TSR planner", "planner_sk": "Planner, self knowledge", "plan_emb": "Per plan LR"}[pol],
                     "Plan AUROC": f"{g.per_plan_auroc.mean():.3f}", "Cross plan corr.": f"{g.cross_plan_corr.mean():.3f}",
                     "Pair AUROC": f"{ps.mean():.3f}", "Seed range": f"{ps.min():.3f} to {ps.max():.3f}",
                     "Seed 0 95% CI": f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]",
                     "Pair questions per seed": f"{len(p0):,}", "Top plan share": f"{g.modal_plan_share.mean():.2f}"})
    T = pd.DataFrame(rows)
    md_table("Checks: what the policies learn (all 5 seeds; pair AUROC is the mean over reader and dataset blocks)", T)
    tex_table("review_mechanism", T, "What the policies learn, over all five seeds (single step plans, held out). Pair AUROC: for questions that exactly one of the two hull plans answers, AUROC for which one, averaged over the 20 reader and dataset blocks; interval: bootstrap over questions within blocks, seed 0.", "lrrrrrrr")
    # --- headroom at the policy's own budget
    H = pd.read_csv(R / "headroom_same_budget_pooled.csv")
    rows = []
    for m in ("planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade", "plan_emb"):
        row = {"Policy": PN.get(m, m)}
        for rd in READERS:
            x = H[(H.method == m) & (H.reader == rd)]
            row[RN[rd]] = f"{x.gap_pairs.iloc[0]:.3f} / {x.closed.iloc[0]:+.2f}" if len(x) else ""
        rows.append(row)
    T = pd.DataFrame(rows)
    md_table("Checks: oracle headroom at each policy's own cost (gap / fraction closed)", T)
    tex_table("review_headroom", T, "Oracle headroom at each policy's own mean cost: gap of the pair oracle frontier over the hull (pairs chosen on the training folds), and the fraction of that gap the policy closes (pooled gain over gap), mean over datasets. Per plan LR without its embedding cost.", "lrrrrr")
    # --- plan_emb with its own cost charged, and trained on all four datasets (exploratory)
    O = pd.read_csv(R / "plan_emb_with_overhead.csv")
    ms = O.delta_ms.max()
    Pf = R / "plan_emb_pooled_training.csv"
    Pt = pd.read_csv(Pf) if Pf.exists() else None
    rows = []
    for ps in ("single step", "with two hop"):
        Sec = secondary("final", "metis") if ps == "single step" else secondary("final_r3", "metis_r3")
        for rd in READERS:
            b = Sec[(Sec.method == "plan_emb") & (Sec.reader == rd)].iloc[0]
            o = O[(O.plan_set == ps) & (O.reader == rd) & (O.delta_ms == ms)].iloc[0]
            row = {"Reader": RN[rd], "Plan set": ps,
                   "Per dataset, replay [95% CI]": f"{sg(b.gain)} [{sg(b.ci_low)}, {sg(b.ci_high)}]", "Holm p": pv(b.p_holm),
                   "Per dataset, own cost [95% CI]": f"{sg(o.gain)} [{sg(o.ci_low)}, {sg(o.ci_high)}]", "p": pv(o.p)}
            if Pt is not None:
                for dl, lab in ((0.0, "Pooled, replay"), (ms, "Pooled, own cost")):
                    x = Pt[(Pt.plan_set == ps) & (Pt.reader == rd) & (np.isclose(Pt.delta_ms, dl))].iloc[0]
                    row[lab] = sg(x.gain)
                    row["p" + (" " if dl == 0 else "  ")] = pv(x.p)
            rows.append(row)
    T = pd.DataFrame(rows)
    md_table(f"Checks: per plan LR without and with its own cost ({ms:.1f} ms when the plan does not embed with BGE-M3), "
             "trained per dataset and on all four datasets", T)
    tex_table("review_overhead", T, f"Per plan logistic regression without and with its own cost: {ms:.1f} ms for the query embedding and classifiers, charged when the chosen plan does not embed the query with BGE-M3 anyway; lambda is tuned with these costs. Per dataset: trained on 400 questions per dataset (Holm p: secondary family of 30 tests). Pooled training (exploratory): one model on the training folds of all four datasets. p: unadjusted.",
              "ll" + "rr" * ((len(T.columns) - 2) // 2))
    TEX["review_overhead"] = TEX["review_overhead"].replace(r"\centering\small", r"\centering\footnotesize").replace(
        r"\setlength{\tabcolsep}{4pt}", r"\setlength{\tabcolsep}{2pt}")     # ten columns: smaller type to fit the width
    # --- learning curve of plan_emb
    LC = pd.read_csv(R / "learning_curve_plan_emb.csv")
    g = LC.groupby(["reader", "frac"])[["sum_gain", "n"]].sum()
    C = (g.sum_gain / g.n).unstack()
    rows = [{"Reader": RN[rd], **{f"{int(f * 100)}%": sg(C.loc[rd, f]) for f in C.columns}} for rd in READERS if rd in C.index]
    allr = LC.groupby("frac")[["sum_gain", "n"]].sum()
    rows.append({"Reader": "Pooled", **{f"{int(f * 100)}%": sg(allr.loc[f, "sum_gain"] / allr.loc[f, "n"]) for f in allr.index}})
    T = pd.DataFrame(rows)
    md_table("Checks: learning curve of per plan LR (gain over the hull, single step plans, no embedding cost)", T)
    tex_table("review_curve", T, "Learning curve of per plan logistic regression: pooled gain over the hull when trained on a share of the training folds (single step plans, without its embedding cost).", "lrrrrr")
    # --- live latency check
    L = AN.parent / "latency_e2e"
    rows, prow = [], []
    for ds in ("musique", "2wikimultihopqa"):
        fc = L / f"qwen2.5-7b_{ds}_cells.csv"
        if not fc.exists():
            continue
        Cc = pd.read_csv(fc)
        gc_ = Cc.groupby(["query_id", "plan"]).agg(live=("live_total_s", "median"), sr=("stored_retrieval_s", "first"),
                                                   sa=("stored_agent_s", "first"), lr=("live_retrieval_s", "median"),
                                                   dev=("device", "first")).reset_index()
        gc_["retr"] = gc_.plan.str.split("@").str[0]
        for rt_, x in gc_.groupby("retr"):
            dev = x.dev.iloc[0] if isinstance(x.dev.iloc[0], str) else ""
            rows.append({"Data": DN[ds], "Retriever": rt_.replace("_", " "), "Cells": len(x),
                         "Live s": f"{x.live.mean():.3f}", "Stored s": f"{(x.sr + x.sa).mean():.3f}",
                         "Live / stored": f"{x.live.mean() / (x.sr + x.sa).mean():.3f}",
                         "Retrieval live / stored s": f"{x.lr.mean():.3f} / {x.sr.mean():.3f}",
                         "Index": {"gpu": "GPU", "cpu": "CPU"}.get(dev, "no dense index" if rt_ != "closed_book" else "")})
        Pp = pd.read_csv(L / f"qwen2.5-7b_{ds}_policies.csv")
        for pol, x in Pp.groupby("policy"):
            prow.append({"Data": DN[ds], "Policy": PN.get(pol, pol), "Questions": len(x), "Live s": f"{x.live_s.mean():.3f}",
                         "Replay s": f"{x.replay_s.mean():.3f}", "Live / replay": f"{x.live_s.mean() / x.replay_s.mean():.3f}",
                         "Same decision": f"{(x.live_escalated == x.stored_escalated).mean():.2f}"})
    if rows:
        T = pd.DataFrame(rows)
        md_table("Checks: live end to end latency against the replay, Qwen2.5 7B (cells: median of 3 live runs)", T)
        tex_table("review_latency", T, "Live end to end time against the stored time of the same cells, Qwen2.5 7B (median of three live runs per cell). Index: where the dense index sat with the reader loaded. Stella and the reranker did not fit next to the reader.", "llrrrrrl")
        T = pd.DataFrame(prow)
        md_table("Checks: live end to end policy cost against the replayed cost, Qwen2.5 7B", T)
        tex_table("review_latency_policies", T, "Policies run live end to end against their replayed cost, Qwen2.5 7B (questions whose retrievers fit next to the reader). Per plan LR includes its 24.6 ms embedding in the live time only.", "llrrrrr")
    # --- protocol ablation
    A = pd.read_csv(R / "protocol_ablation.csv")
    T = pd.DataFrame({"Step": A.step, "Wins": [f"{w} of {o}" for w, o in zip(A.wins, A.of)]})
    md_table("Checks: the weak protocol turned into ours one factor at a time", T)
    tex_table("review_ablation", T, "The weak protocol turned into ours one factor at a time on the same 100 cells (5 primary policies, 5 readers, 4 datasets); the last two rows pool over datasets (25 tests).", "lr")


# ====================================================================== further checks (scripts 26 to 33)
def _pos(g, p="p_nb_holm", lo="nb_low"):
    return int(((g[p] < 0.05) & (g[lo] > 0)).sum())


def _neg(g, p="p_nb_holm", hi="nb_high"):
    return int(((g[p] < 0.05) & (g[hi] < 0)).sum())


def _rng(x):
    return f"{sg(x.min())} to {sg(x.max())}" if len(x) else ""


def t_review2():
    """Tables for the further checks: mixed workload (26), corrected inference (27), difference routers (28),
    policy checks (30), closed book prompt (31), outside judge (32, if run), Adaptive RAG with IRCoT (33), audit 3 (29,
    if finalised). Every number is read from the result files."""
    from scipy import stats
    I = AN / "inference"
    if not (I / "tests_nb.csv").exists():
        raise FileNotFoundError(str(I / "tests_nb.csv"))
    PSN = ["single step", "with two hop"]

    # ---------------------------------------------------------------- corrected inference, all families
    T = pd.read_csv(I / "tests_nb.csv")
    C = pd.read_csv(I / "calibration.csv")
    infl = refit_factor(I)
    own = pd.read_csv(I / "tests_secondary_own_cost.csv") if (I / "tests_secondary_own_cost.csv").exists() else None

    def inflated(g):
        se = g.se_nb * infl
        lo = g.gain - stats.t.ppf(0.975, g.df_nb) * se
        ub = g.gain + stats.t.ppf(0.95, g.df_nb) * se
        p = 2 * stats.t.sf((g.gain / se).abs(), g.df_nb)
        return lo, ub, pd.Series(holm(np.asarray(p, float)), index=g.index)

    rows = []
    fams = [("Primary", T[T.family == "primary"]), ("Secondary, routers uncharged", T[T.family == "secondary"])]
    if own is not None:
        fams.append(("Secondary, routers charged", own))
    fams.append(("Stronger", T[T.family == "stronger"]))
    for fam, F in fams:
        for ps in PSN:
            g = F[F.plan_set == ps]
            if g.empty:
                continue
            lo_i, ub_i, ph_i = inflated(g)
            rows.append({"Family": fam, "Plan set": ps, "Tests": len(g),
                         f"Positive: per question / corrected / x{infl:.2f}": f"{_pos(g, 'p_plain_holm', 'plain_low')} / {_pos(g)} / {int(((ph_i < 0.05) & (lo_i > 0)).sum())}",
                         f"Bound below 0.03: per question / corrected / x{infl:.2f}": f"{int((g.ub95_plain < 0.03).sum())} / {int((g.ub95_nb < 0.03).sum())} / {int((ub_i < 0.03).sum())}",
                         "Largest bound, corrected": f"{g.ub95_nb.max():.3f}"})
    R = pd.DataFrame(rows)
    md_table("Checks: inference with refitting (corrected resampled t test; xF = corrected SE times the largest refit ratio F)", R)
    xt = f"$\\times${infl:.2f}"
    tex_table("rev2_inference", R.rename(columns=lambda c: c.replace(f"x{infl:.2f}", xt)), f"Inference that accounts for refitting. Per question: $t$ test on the 2,000 pooled gains, with the fitted policies and hulls fixed. Corrected: resampled t test on the 25 fold gains per reader and dataset (Nadeau and Bengio). {xt}: corrected standard errors inflated by the largest ratio of the refit bootstrap spread to the corrected error. Positive: Holm p below 0.05 and interval above 0. Bound: one sided 95\\% upper bound.", "llrrrr", star=True)
    VERD.append(f"Check: inference: primary positives (per question/corrected/x{infl:.2f}) " +
                "; ".join(f"{r['Plan set']}: {r[f'Positive: per question / corrected / x{infl:.2f}']}" for _, r in R[R.Family == "Primary"].iterrows()))

    # every upper bound of the primary family, so that readers can apply their own margin
    P = T[T.family == "primary"]
    rows = []
    for m in PRIMARY:
        for rd in [r for r in READERS + [k for k in RN if "->" in k] if r in set(P[P.method == m].reader)]:
            a = P[(P.method == m) & (P.reader == rd) & (P.plan_set == "single step")]
            b = P[(P.method == m) & (P.reader == rd) & (P.plan_set == "with two hop")]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            rows.append({"Policy": PN[m], "Reader": RN.get(rd, rd), "Gain, single": sg(a.gain),
                         "Bound, per question": f"{a.ub95_plain:.3f}", "Bound, corrected": f"{a.ub95_nb:.3f}",
                         "Gain, two hop": sg(b.gain), "Bound, per question ": f"{b.ub95_plain:.3f}", "Bound, corrected ": f"{b.ub95_nb:.3f}"})
    B = pd.DataFrame(rows)
    md_table("Checks: every one sided 95% upper bound of the primary family", B)
    tex_table("rev2_bounds", B, "Every one sided 95% upper bound of the primary family, under the per question test and under the corrected test, single step plans and with two hop plans; readers can apply their own margin.", "llrrrrrr", star=True)

    # ---------------------------------------------------------------- mixed workload
    MX = AN / "mixed"
    if (I / "tests_mixed_nb.csv").exists():
        X = pd.read_csv(I / "tests_mixed_nb.csv")
        key = ["plan_set", "family", "comparator", "reader", "method"]
        pqf = [MX / f for f in ("tests.csv", "tests_extra.csv") if (MX / f).exists()]
        if pqf:                                   # per question test of 26_mixed_workload.py
            PQ = pd.concat([pd.read_csv(f) for f in pqf])[key + ["positive"]].rename(columns={"positive": "pos_pq"})
            X = X.merge(PQ, on=key, how="left")
        else:
            X["pos_pq"] = False
        parts = []
        for _, g in X.groupby(["family", "comparator", "plan_set"]):
            lo_i, ub_i, ph_i = inflated(g)
            parts.append(g.assign(pos_i=((ph_i < 0.05) & (lo_i > 0)).to_numpy()))
        X = pd.concat(parts)
        X["win_all"] = X.pos_pq.fillna(False).astype(bool) & (X.p_nb_holm < 0.05) & (X.nb_low > 0) & X.pos_i
        groups = [("Primary policies", "primary"), ("Adaptive RAG, silver labels", "ar_pub"),
                  ("Secondary routers, no own cost", "secondary"), ("Embedding routers, own cost", "own_cost"),
                  ("Dataset guess, own cost", "ds_label_own")]
        rows = []
        for name, fam in groups:
            for ps in PSN:
                g = X[(X.family == fam) & (X.plan_set == ps)]
                if g.empty:
                    continue
                gg, gl = g[g.comparator == "global"], g[g.comparator == "label"]
                rows.append({"Policies": name, "Plan set": ps, "Tests": len(gg),
                             f"Pos., global: per q. / corr. / x{infl:.2f}": f"{int(gg.pos_pq.fillna(False).astype(bool).sum())} / {_pos(gg)} / {int(gg.pos_i.sum())}",
                             "Neg., global": _neg(gg), "Gain, global hull": _rng(gg.gain),
                             "Pos./neg., label hull": f"{_pos(gl)} / {_neg(gl)}", "Gain, label hull": _rng(gl.gain)})
        M = pd.DataFrame(rows)
        both = []
        for (fam, comp), g in X.groupby(["family", "comparator"]):
            w = g[g.win_all].groupby(["reader", "method"]).plan_set.nunique()
            for (rd, m) in w[w == 2].index:
                both.append(f"{fam} vs {comp}: {RN.get(rd, rd)} {m}")
        md_table("Checks: mixed workload (2,000 pooled questions per reader; Holm within family; "
                 f"Pos. = per question / corrected / corrected x{infl:.2f})", M,
                 "Positive in both plan sets under all three tests: " + ("; ".join(both) if both else "none"))
        xt = f"$\\times${infl:.2f}"
        tex_table("rev2_mixed", M.rename(columns=lambda c: c.replace(f"x{infl:.2f}", xt)), "Mixed workload: the 2,000 questions of each reader pooled, one hull and each policy fitted on the pooled training folds, no dataset indicator. Global hull: best mix of fixed plans for the pool; this is the baseline a deployed system can use. Label hull: best mix of fixed plans per dataset under one budget; it knows each question's dataset and is a reference only. Holm within each family and comparator. Pos.: tests with the interval above 0 after Holm under the per question test, the corrected test, and the corrected test with errors " + xt + ". Neg. and the label hull columns use the corrected test.", "llrrrrrr", star=True)
        VERD.append("Check: mixed workload, positive in both plan sets under all three tests: " + ("; ".join(both) if both else "none"))
        VERD.append("Check: mixed workload (pos/neg): " + "; ".join(f"{r['Policies']} {r['Plan set']}: {r[f'Pos., global: per q. / corr. / x{infl:.2f}']} pos and {r['Neg., global']} neg global, {r['Pos./neg., label hull']} label"
                                                          for _, r in M.iterrows()))
    if (MX / "tests.csv").exists():
        W = pd.concat([pd.read_csv(MX / f) for f in ("tests.csv", "tests_extra.csv") if (MX / f).exists()])
        W = W.pivot_table(index=["plan_set", "reader", "method"], columns="comparator", values="gain").reset_index()
        W["workload"] = W["global"] - W["label"]
        names = {"planner_tsr": "TSR planner", "reader_cascade": "Reader cascade", "plan_emb": "Per plan LR",
                 "plan_knn": "Per plan kNN", "plan_emb_own": "Per plan LR, own cost", "plan_knn_own": "Per plan kNN, own cost",
                 "ds_label": "Dataset guess + label hull", "ds_label_own": "Dataset guess, own cost"}
        rows = []
        for m, nm in names.items():
            row = {"Policy": nm}
            for ps in PSN:
                g = W[(W.method == m) & (W.plan_set == ps)]
                tag = "single" if ps == "single step" else "two hop"
                row[f"Global, {tag}"] = sg(g["global"].mean()) if len(g) else ""
                row[f"Workload, {tag}"] = sg(g.workload.mean()) if len(g) else ""
                row[f"Label, {tag}"] = sg(g.label.mean()) if len(g) else ""
            rows.append(row)
        D = pd.DataFrame(rows)
        md_table("Checks: mixed workload decomposition (mean over readers): gain over global hull = workload part + gain over label hull", D)
        tex_table("rev2_decomp", D, "Where the gain on mixed traffic comes from (mean over readers): gain over the global hull = workload part (label hull minus global hull at the policy's cost) + gain over the label hull.", "lrrrrrr", star=True)
    if (MX / "headroom.csv").exists():
        Hm = pd.read_csv(MX / "headroom.csv")
        h = Hm.groupby(["plan_set", "reader"]).label_minus_global.max().reset_index()
        VERD.append("Check: label hull minus global hull (held out, best budget): " +
                    "; ".join(f"{ps}: {_rng(g.label_minus_global)}" for ps, g in h.groupby("plan_set")))
    if (MX / "curve.csv").exists():
        Cu = pd.read_csv(MX / "curve.csv")
        rows = []
        for (ps, m), g in Cu[Cu.comparator == "label"].groupby(["plan_set", "method"]):
            row = {"Plan set": ps, "Policy": {"plan_emb": "Per plan LR", "plan_knn": "Per plan kNN", "plan_emb_own": "Per plan LR, own cost",
                                              "plan_knn_own": "Per plan kNN, own cost"}.get(m, m)}
            for n, x in g.groupby("n_train"):
                row[f"{n:,}"] = sg(x.gain.mean())
            rows.append(row)
        Cv = pd.DataFrame(rows)
        md_table("Checks: learning curve on the mixed workload, gain over the label hull (per question skill), mean over readers, by training questions", Cv)
        tex_table("rev2_curve", Cv, "Learning curve on the mixed workload: gain over the label hull, which removes the workload part, mean over readers, by the number of pooled training questions.", "ll" + "r" * (len(Cv.columns) - 2))

    # ---------------------------------------------------------------- difference routers
    DR = AN / "difference_router"
    if (DR / "tests.csv").exists():
        Dt, Da = pd.read_csv(DR / "tests.csv"), pd.read_csv(DR / "pair_auroc.csv")
        rows = []
        for m, nm in (("diff_clf", "Difference classifier"), ("diff_reg", "Difference regression")):
            for ps in PSN:
                a0 = Dt[(Dt.method == m) & (Dt.plan_set == ps)]
                a1 = Dt[(Dt.method == m + "_own") & (Dt.plan_set == ps)]
                au = Da[(Da.method == m) & (Da.plan_set == ps)]
                rows.append({"Router": nm, "Plan set": ps, "Gain, no own cost": _rng(a0.gain), "Positive": _pos(a0),
                             "Gain, own cost": _rng(a1.gain), "Positive ": _pos(a1),
                             "Pair AUROC": f"{au.pair_auroc.mean():.3f}", "Per plan LR, same pairs": f"{au.pair_auroc_plan_emb.mean():.3f}"})
        Dd = pd.DataFrame(rows)
        md_table("Checks: routers trained on the plan difference (corrected test, Holm over 10 per plan set and cost)", Dd)
        tex_table("rev2_diff", Dd, "Routers trained on the difference between two plans (BGE M3 features, pair and threshold chosen on the training folds), without and with their own 24.6 ms cost. Pair AUROC: held out questions where the two chosen plans disagree, against the per plan router's score on the same questions and pairs.", "llrrrrrr", star=True)

    # ---------------------------------------------------------------- policy checks (exposure, realized, train only, p95, sweep, agreement)
    PC = AN / "policy_checks"
    if (PC / "realized_tests.csv").exists():
        rows = []
        prim = lambda D: D[D.family == "primary"]
        rows.append(("Paper protocol, corrected test", prim(T)))
        rows.append(("Realized per question costs", prim(pd.read_csv(PC / "realized_tests.csv"))))
        if (PC / "trainonly_tests.csv").exists():
            rows.append(("Plan costs from training folds only", prim(pd.read_csv(PC / "trainonly_tests.csv"))))
        out = []
        for nm, D in rows:
            r = {"Check (primary family)": nm}
            for ps in PSN:
                g = D[D.plan_set == ps]
                r[f"Positive, {ps}"] = f"{_pos(g)} of {len(g)}"
                r[f"Largest gain, {ps}"] = sg(g.gain.max())
            out.append(r)
        CB = AN / "closed_book_prompt"
        if (CB / "replay_tests.csv").exists():
            Rp = pd.read_csv(CB / "replay_tests.csv")
            for v, nm in (("new_time", "Closed book prompt of~\\cite{moskvoretskii2025adaptive}, measured time (Llama, 7 policies)"),
                          ("old_time", "Closed book prompt of~\\cite{moskvoretskii2025adaptive}, old time (Llama, 7 policies)")):
                r = {"Check (primary family)": nm}
                for ps in PSN:
                    g = Rp[(Rp.variant == v) & (Rp.plan_set == ps)]
                    g = g.assign(p_nb_holm=holm(np.asarray(g.p_nb, float)))
                    r[f"Positive, {ps}"] = f"{_pos(g)} of {len(g)}"
                    r[f"Largest gain, {ps}"] = sg(g.gain.max())
                out.append(r)
        OJ = AN / "outside_judge"
        if (OJ / "tests_mistral_vs_qwen.csv").exists():
            Oj = prim(pd.read_csv(OJ / "tests_mistral_vs_qwen.csv"))
            r = {"Check (primary family)": "Labels from the outside judge"}
            for ps in PSN:
                g = Oj[Oj.plan_set == ps]
                r[f"Positive, {ps}"] = f"{_pos(g)} of {len(g)}"
                r[f"Largest gain, {ps}"] = sg(g.gain.max())
            out.append(r)
        Q = pd.DataFrame(out)
        rp_note = ""
        if (CB / "replay_tests.csv").exists():
            ms = set(Rp.method)
            rp_note = (f" The closed book rows replay {len(ms)} policies: {sum(m in PRIMARY for m in ms)} primary and "
                       f"{sum(m not in PRIMARY for m in ms)} secondary.")
        md_table("Checks: the primary verdict under each check (corrected test, Holm within plan set)", Q)
        tex_table("rev2_checks", Q, "The primary family under each check (corrected test; Holm within each plan set). Closed book prompt: Llama 3.1 8B rerun with the closed book answers of the few shot chain of thought prompt that Moskvoretskii et al.~\\cite{moskvoretskii2025adaptive} use, at their measured time and at the old time (Holm over the policies replayed)." + rp_note, "lrrrr", star=True)
    if (PC / "exposure.csv").exists():
        Ex = pd.read_csv(PC / "exposure.csv")
        rows = []
        for ps in PSN:
            g = Ex[Ex.plan_set == ps]
            rows.append({"Plan set": ps, "Tests": len(g), "Judge share, policy minus hull: median": sg(g.judge_diff.median(), 4),
                         "Range": _rng(g.judge_diff), "Long contained, policy minus hull: range": _rng(g.clong_diff)})
        md_table("Checks: grading exposure, policy against hull mix", pd.DataFrame(rows))
        tex_table("rev2_exposure", pd.DataFrame(rows), "Grading exposure of every policy against its hull mix (mean over folds, seeds and datasets): share of answers whose label came from the judge, and share of long contained answers.", "lrrrr")
    if (PC / "agreement_by_reader.csv").exists():
        Ag = pd.read_csv(PC / "agreement_by_reader.csv")
        Ag = Ag.assign(Reader=Ag.reader.map(RN), Audit=Ag.audit.astype(str).str.replace("audit", "", regex=False))
        Ag = Ag[["Reader", "Audit", "n", "agreement", "sensitivity", "specificity", "J"]]
        Ag.columns = ["Reader", "Audit", "Answers", "Agreement", "Sensitivity", "Specificity", "J"]
        for c in ("Agreement", "Sensitivity", "Specificity", "J"):
            Ag[c] = Ag[c].map(lambda x: f"{x:.3f}")
        md_table("Checks: label_final against the human majority, by reader (audits 1 and 2)", Ag)
        tex_table("rev2_agreement", Ag, "label_final against the majority of three human raters, by the reader that wrote the answer. label_final is the string rules, with the Qwen3 14B judge on the ambiguous band. Each reader was audited in one audit only. The Qwen3 14B row is graded by a judge of the same model.", "lcrrrrr")
    if (PC / "p95.csv").exists():
        P9 = pd.read_csv(PC / "p95.csv")
        rows = []
        for m in PRIMARY + ["plan_emb"]:
            row = {"Policy": PN.get(m, m)}
            for ps in PSN:
                g = P9[(P9.method == m) & (P9.plan_set == ps)]
                row[f"p95 ratio, {ps}"] = f"{g.p95_ratio.median():.2f}" if len(g) else ""
            rows.append(row)
        P95 = pd.DataFrame(rows)
        md_table("Checks: 95th percentile latency, policy over its hull (median over readers)", P95)
        tex_table("rev2_p95", P95, "Tail latency at matched mean cost: 95th percentile of the per question latency of each policy over that of its hull (the hull's plan drawn per question with its weight), median over readers, seed 0.", "lrr")
        VERD.append(f"Check: p95 ratio (policy/hull): single step median {P9[P9.plan_set == 'single step'].p95_ratio.median():.2f}, "
                    f"two hop {P9[P9.plan_set == 'with two hop'].p95_ratio.median():.2f}; max {P9.p95_ratio.max():.2f}")
    if (PC / "sweep_tests.csv").exists():
        Sw = pd.read_csv(PC / "sweep_tests.csv")
        if (PC / "sweep_tests_own.csv").exists():
            Sw = pd.concat([Sw, pd.read_csv(PC / "sweep_tests_own.csv")])
        nm = {"planner_tsr": "TSR planner", "gate_closed_book": "Closed book gate", "conf_cascade": "Confidence cascade",
              "reader_cascade": "Reader cascade (7B to Qwen3 14B)", "plan_emb": "Per plan LR, no own cost",
              "plan_emb_own": "Per plan LR, own cost"}
        rows = []
        for m, n_ in nm.items():
            g = Sw[Sw.method == m]
            if g.empty:
                continue
            row = {"Policy": n_}
            for st, x in g.groupby("budget_step"):
                row[str(st)] = sg(x.gain.mean())
            row["Intervals above 0"] = int((g.nb_low > 0).sum())
            rows.append(row)
        SW = pd.DataFrame(rows)
        b1 = Sw[(Sw.budget_step == 1) & Sw.method.isin(list(nm))]
        ov = (b1.cost / b1.budget_s)[(b1.cost / b1.budget_s) > 1.005]
        sw_note = (f" At budget 1, {len(ov)} of {len(b1)} policy and reader runs have no setting within the budget and cost "
                   f"{ov.min():.2f} to {ov.max():.2f} times the budget; they are still scored against the hull at their own cost.") if len(ov) else ""
        md_table("Checks: budget sweep, mean gain over readers by budget (1 = cheapest plan, 6 = most successful plan)", SW)
        tex_table("rev2_sweep", SW, "Budget sweep (single step plans, three seeds): each policy tuned on the training folds to its best success within the budget, scored against the held out hull at its mean cost. Budgets from the cheapest plan (1) to the most successful plan (6); mean gain over readers. Last column: reader and budget tests with the corrected interval above 0 (no correction for the six budgets)." + sw_note, "l" + "r" * (len(SW.columns) - 1))

    # ---------------------------------------------------------------- closed book prompt of [10]
    CB = AN / "closed_book_prompt"
    if (CB / "accuracy.csv").exists():
        Ac = pd.read_csv(CB / "accuracy.csv")
        runs = [("their_answers_their_q", "Their stored answers, their questions"), ("their_prompt_their_q", "Their prompt, our run, their questions"),
                ("our_prompt_their_q", "Our prompt, their questions"), ("our_prompt_our_q (paper)", "Our prompt, our questions (paper)"),
                ("their_prompt_our_q", "Their prompt, our questions")]
        rows = []
        for key, n_ in runs:
            g = Ac[Ac.run == key].set_index("dataset")
            row = {"Run": n_}
            for ds in ("nq", "hotpotqa", "2wikimultihopqa", "musique"):
                row[DN[ds]] = f"{g.contains_acc.get(ds, np.nan):.3f}"
            row["Abstain"] = f"{g.abstain.mean():.3f}"
            row["Time (s)"] = "" if g.time_s.isna().all() else f"{g.time_s.mean():.2f}"
            rows.append(row)
        rows.append({"Run": "Their Table 1, Never RAG", **{DN[d]: f"{v:.3f}" for d, v in
                                                          Ac.drop_duplicates("dataset").set_index("dataset").their_table1_never_rag.items()},
                     "Abstain": "", "Time (s)": ""})
        AcT = pd.DataFrame(rows)[["Run", "NQ", "HotpotQA", "2Wiki", "MuSiQue", "Abstain", "Time (s)"]]
        md_table("Checks: closed book accuracy of Llama 3.1 8B (answer contained, the metric of [10])", AcT)
        tex_table("rev2_closedbook", AcT, "Closed book accuracy of Llama 3.1 8B under answer contained accuracy, the metric of Moskvoretskii et al.~\\cite{moskvoretskii2025adaptive}: their prompt (the few shot chain of thought prompt of Adaptive RAG) against ours, on their 500 test questions per dataset and on ours. Abstain: share of refusals; time per answer. Their stored answers give their Table 1 values on the three multi hop datasets but not on NQ.", "lrrrrrr", star=True)

    # ---------------------------------------------------------------- Adaptive RAG with IRCoT on top
    AI = AN / "adaptive_rag_ircot"
    if (AI / "tests.csv").exists():
        At0 = pd.read_csv(AI / "tests.csv")
        Ac9 = At0[At0.level == "dataset"]
        At = At0[At0.level == "pooled"]
        rows = []
        for m, n_ in (("ar_ircot_tfidf", "TF IDF LR"), ("ar_ircot_emb", "BGE M3 LR")):
            for ps in PSN:
                g = At[(At.method == m) & (At.plan_set == ps)]
                f_, w_ = g[g.comparator == "fixed"], g[g.comparator == "with_ircot"]
                c9 = Ac9[(Ac9.method == m) & (Ac9.plan_set == ps)]
                cf, cw = c9[c9.comparator == "fixed"], c9[c9.comparator == "with_ircot"]
                rows.append({"Classifier": n_, "Plan set": ps, "IRCoT share": f"{f_.ircot_share.min():.2f} to {f_.ircot_share.max():.2f}",
                             "Gain, fixed hull": _rng(f_.gain), "Pooled pos./neg.": f"{_pos(f_)} / {_neg(f_)}",
                             "Cells pos./neg.": f"{_pos(cf)} / {_neg(cf)}",
                             "Gain, hull with IRCoT": _rng(w_.gain), "Pooled pos./neg. ": f"{_pos(w_)} / {_neg(w_)}",
                             "Cells pos./neg. ": f"{_pos(cw)} / {_neg(cw)}"})
        AT = pd.DataFrame(rows)
        md_table("Checks: Adaptive RAG with IRCoT as its top rung (pooled per reader, 3 readers)", AT)
        tex_table("rev2_arircot", AT, "Adaptive RAG with the published ladder (closed book, single step plan, IRCoT) on the three multi hop datasets, pooled per reader (Qwen2.5 7B, Llama 3.1 8B, Qwen3 14B). Fixed hull: mix of fixed plans. Hull with IRCoT: IRCoT added as one more plan. Corrected test. Pooled: Holm over the three readers. Cells: the nine reader and dataset tests, Holm over nine.", "llrrrrrrr", star=True)

    # ---------------------------------------------------------------- MDE for Table IX
    if (I / "mde_ablation.csv").exists():
        Md = pd.read_csv(I / "mde_ablation.csv")
        Md = pd.DataFrame({"Decision rule": Md.rule, "Tests": Md.tests, "MDE (median)": Md.mde_median.map(lambda x: f"{x:.3f}"),
                           "Expected wins at 0.02": Md["expected_wins_true_0.02"].map(lambda x: f"{x:.1f}"),
                           "Expected wins at 0.03": Md["expected_wins_true_0.03"].map(lambda x: f"{x:.1f}")})
        md_table("Checks: power of the Table IX decision rules (80% power)", Md)
        tex_table("rev2_mde", Md, "Power of the decision rules of the protocol ablation: minimum detectable effect at 80% power (median over tests), and the number of tests expected to win if every cell had a true gain of 0.02 or 0.03 (normal approximation; Holm rows use the strictest step).", "lrrrr")

    # ---------------------------------------------------------------- outside judge and audit 3 (when run)
    OJ = AN / "outside_judge"
    if (OJ / "judges_vs_humans.csv").exists():
        Jh = pd.read_csv(OJ / "judges_vs_humans.csv")
        Jh = pd.DataFrame({"Judge": Jh.judge.map({"qwen": "Qwen3 14B", "mistral": "Mistral Small 3.1 24B"}), "Answers": Jh.n,
                           "Agreement": Jh.agreement.map(lambda x: f"{x:.3f}"), "Sensitivity": Jh.sensitivity.map(lambda x: f"{x:.3f}"),
                           "Specificity": Jh.specificity.map(lambda x: f"{x:.3f}"), "J": Jh.J.map(lambda x: f"{x:.3f}")})
        md_table("Checks: both judges against the human majority on the audited answers in the ambiguous band", Jh)
        tex_table("rev2_judges", Jh, "The judge in our pipeline (Qwen3 14B) and a judge from outside every reader family (Mistral Small 3.1 24B, about 4 bits) against the human majority, on the audited answers of the ambiguous band.", "lrrrrr")
    A3 = AN / "audit3" / "audit3_results.csv"
    if A3.exists():
        Au = pd.read_csv(A3)
        tname = lambda t: ", ".join([PN.get(t.split("|")[0], t.split("|")[0]), RN.get(t.split("|")[1], t.split("|")[1]), t.split("|")[2]])
        Au = pd.DataFrame({"Test": Au.test.map(tname), "Questions": Au.questions, "Discordant share": Au.discordant_share.map(lambda x: f"{x:.2f}"),
                           "label_final error, policy": Au.policy_label_final_error.map(lambda x: f"{x:.3f}"),
                           "label_final error, hull": Au.hull_label_final_error.map(lambda x: f"{x:.3f}"),
                           "Bias on the gain [95% CI]": [f"{sg(a)} [{sg(b)}, {sg(c)}]" for a, b, c in
                                                         zip(Au.bias_label_final_on_gain, Au.bias_label_final_on_gain_low, Au.bias_label_final_on_gain_high)]})
        md_table("Checks: audit of the answers where the policy and the hull disagree", Au)
        tex_table("rev2_audit3", Au, "Human audit of the answers on which a policy and its hull disagree (two raters who are not authors, adjudicated): error of label_final in each arm and the bias it puts on the gain.", "lrrrrr")


def t_multistep():
    """Fixed step chains F_T against IRCoT (40_multistep_frontier.py): success and latency, mean over the three
    IRCoT readers, and IRCoT minus F_4 on the same questions."""
    D = AN / "multistep"
    FR = pd.read_csv(D / "frontier.csv")
    ST = pd.read_csv(D / "stopping.csv")
    T = pd.read_csv(D / "tests.csv")
    F = FR[FR.kind.isin(["F_T", "IRCoT"])].groupby(["dataset", "plan"])[["success", "latency_s"]].mean()
    rows = []
    for ds in ["nq", "hotpotqa", "2wikimultihopqa", "musique"]:
        if ds not in F.index.get_level_values(0):
            continue
        r = {"Dataset": DN[ds]}
        for t in (1, 2, 3, 4):
            x = F.loc[(ds, f"cotfixed{t}@4")]
            r[f"F{t}"] = f"{x.success:.3f} ({x.latency_s:.1f})"
        x = F.loc[(ds, "ircot")]
        r["IRCoT"] = f"{x.success:.3f} ({x.latency_s:.1f})"
        g = ST[ST.dataset == ds]
        r["IRCoT minus F4"] = f"{sg(g.ircot_minus_f4_success.min())} to {sg(g.ircot_minus_f4_success.max())}"
        rows.append(r)
    M = pd.DataFrame(rows)
    md_table("Fixed step chains against IRCoT (success, mean latency in s; mean over three readers)", M)
    tex_table("multistep", M, "Fixed step chains $F_T$ (IRCoT's reasoning for exactly $T$ retrievals, stopping off) "
              "against IRCoT: success and mean latency in seconds, mean over the three IRCoT readers. Last column: "
              "IRCoT minus $F_4$ on the same questions, range over readers.", "lrrrrrr", star=True)
    for h, g in T.groupby("hull"):
        k = g[(g.p_holm < 0.05) & (g.ci_low > 0) & (g.p_refit_holm < 0.05) & (g.refit_low > 0)]
        VERD.append(f"IRCoT against the hull with F_T [{h}]: {len(k)} of {len(g)} wins")
    VERD.append(f"IRCoT minus F4: success {ST.ircot_minus_f4_success.min():+.3f} to {ST.ircot_minus_f4_success.max():+.3f}; "
                f"intervals containing 0: {int(((ST.low <= 0) & (ST.high >= 0)).sum())} of {len(ST)}; latency saved "
                f"{ST.ircot_minus_f4_latency_s.min():.1f} to {ST.ircot_minus_f4_latency_s.max():.1f} s")


def t_arpub():
    """Adaptive RAG as published on the four dataset mix (36_adaptive_rag_published.py)."""
    D = AN / "adaptive_rag_published"
    T = pd.read_csv(D / "tests.csv")
    U = pd.read_csv(D / "usage.csv")
    T = T[T.plan_set == "single step"]
    rows = []
    for rd in ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]:
        g = T[T.reader == rd].set_index("comparator")
        u = U[(U.reader == rd) & (U.plan_set == "single step")]
        rows.append({"Reader": RN[rd], "vs Always RAG": sg(g.loc["always_rag", "gain"]),
                     "vs hull": sg(g.loc["hull_fixed", "gain"]), "vs label hull": sg(g.loc["hull_label", "gain"]),
                     "vs hull with IRCoT": sg(g.loc["hull_with_ircot", "gain"]),
                     "Latency (s)": f"{u.cost.mean():.1f}", "Always RAG (s)": f"{u.always_cost.mean():.2f}",
                     "Share to IRCoT": f"{u['share_ircot@0'].mean():.2f}"})
    M = pd.DataFrame(rows)
    md_table("Adaptive RAG as published (t5-large classifier, ladder up to IRCoT, four dataset mix)", M)
    tex_table("arpub", M, "Adaptive RAG as published: t5-large classifier trained on the four dataset mix, ladder of no "
              "retrieval, BM25 with 5 passages and IRCoT; gains on held-out folds, single-step plan set.", "lrrrrrrr", star=True)
    A_ = pd.read_csv(D / "tests.csv")
    for comp, g in A_.groupby("comparator"):
        VERD.append(f"Adaptive RAG as published vs {comp}: gain {g.gain.min():+.3f} to {g.gain.max():+.3f}; "
                    f"positive under all three in both plan sets: {int(g[g.positive_all_three].groupby('reader').plan_set.nunique().eq(2).sum())}; "
                    f"negative (corrected) {int(g.negative_corrected.sum())} of {len(g)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MD.append("# FunnelPlan final tables\n\nGenerated by scripts/22_tables_final.py from the result files. Gains are differences in the share of questions answered correctly.")
    for fn in (t_h1, t_noninf, t_secondary, t_h3, t_sensitivity, t_step, t_cost_axes, t_stronger, t_r3, t_r6,
               t_ircot, t_headroom, t_checks, t_audit, t_rater, t_summary,
               t_p_retrievers, t_p_plans, t_p_gains, t_p_ircot, t_p_weak, t_p_mechanism, t_p_rater,
               t_p_datasets, t_p_curve, t_p_sanity, t_p_headroom, t_p_h3rules, t_review, t_review2, t_multistep, t_arpub):
        try:
            fn()
        except FileNotFoundError as e:
            VERD.append(f"{fn.__name__}: input missing ({e.filename})")
    (OUT / "tables.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
    for k, v in TEX.items():
        (OUT / f"tab_{k}.tex").write_text(ieee_hyphens(v) + "\n", encoding="utf-8")
    (OUT / "verdicts.txt").write_text("\n".join(VERD) + "\n", encoding="utf-8")
    print("\n".join(VERD))
    print(f"\nWritten {OUT}: tables.md, verdicts.txt and {len(TEX)} LaTeX tables")


if __name__ == "__main__":
    main()
