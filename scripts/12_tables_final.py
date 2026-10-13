#!/usr/bin/env python
"""
Final tables for the paper (5 readers, both plan sets). CPU only, reads results; never reruns an analysis.

Writes results_full/tables_final/:
  tables.md          every table in readable form (the source for the paper text)
  tab_*.tex          booktabs LaTeX tables
  verdicts.txt       the pre-registered decisions, computed from the files (H1, secondary family, H3, robustness)

Inputs (all produced by earlier scripts): analysis/final[_r3]/ (10, 13), analysis/metis[_r3]/ (23),
analysis/h3_ircot[_lenient|_strict]/ (21, 24), analysis/weak_protocol/ (22), analysis/lenient*|strict* (24),
analysis/step_hull_primary*.csv (25, optional), grading/audit_adjudicated.csv (audits), r4_ircot labels (20).

Usage: python scripts/12_tables_final.py
"""

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
OUT = settings.RESULTS_ROOT / "tables_final"
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


def tex_table(name, df, caption, align=None):
    esc = lambda v: str(v).replace("_", r"\_").replace("%", r"\%").replace("<", r"$<$")
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
    tex_table("audit", T, "Grading rules against the human reference (majority of three independent raters).", "lrlrrrrr")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MD.append("# FunnelPlan final tables\n\nGenerated by scripts/12_tables_final.py from the result files. Gains are differences in the share of questions answered correctly.")
    for fn in (t_h1, t_secondary, t_h3, t_sensitivity, t_step, t_r3, t_r6, t_ircot, t_headroom, t_audit):
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
