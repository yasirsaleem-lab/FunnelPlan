#!/usr/bin/env python
"""
Paper tables (LaTeX, booktabs) plus a plain text copy of every table.

  tab_main.tex        pooled gain vs the held out hull: method x reader, 95% CI, Holm over the pooled family
  tab_per_dataset.tex gain per method x reader x dataset; marks nominal p < 0.05 and Holm survivors
  tab_headroom.tex    closed book, best fixed plan, per query oracle: how much a perfect router could gain
  tab_grading.tex     every grading rule against the human labels (rater 1, all 200 audit items)
  tab_cost.tex        cost model parameters and its error against the measured plan latency
  tab_ablation.tex    prompt ablation (strict vs soft instruction), from 09
  tables.txt          all of the above as text

Reads results_full/analysis/<label>/, data/processed_full/ (graded labels, grading/, cost_model_params.json,
ablation_prompt/). CPU only, under a minute. Run after 10 (and 08, 09, fit_cost_model).
"""

import json
import os
import re
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config import settings

CONFIG = {"label": "final", "n_boot": 2000, "seed": 0}

ANA = settings.RESULTS_ROOT / "analysis" / CONFIG["label"]
OUT = settings.RESULTS_ROOT / "tables"
PROC = settings.PROCESSED_DATA_ROOT
GDIR = PROC / "grading"

READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
R_TEX = {"qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B", "qwen3-14b": "Qwen3 14B"}
DATASETS = ["nq", "hotpotqa", "2wikimultihopqa", "musique"]
D_TEX = {"nq": "NQ", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki", "musique": "MuSiQue"}
METHODS = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade"]
M_TEX = {"planner_tsr": "TSR planner (ours)", "adaptive_rag": "Adaptive RAG router", "adaptive_k": "Adaptive $k$",
         "gate_closed_book": "Closed book gate", "conf_cascade": "Confidence cascade",
         "reader_cascade": "Reader cascade"}
ROUTERS = ["ak_paper", "ar_emb", "ar_t5", "plan_emb", "plan_knn"]      # from 13 (ar_tfidf is in the Holm family, not shown)
M_TEX.update({"ar_emb": "Adaptive RAG, BGE-M3 classifier", "ar_t5": "Adaptive RAG, FLAN-T5 classifier",
              "ar_tfidf": "Adaptive RAG, paper's label rule", "plan_emb": "Per plan, BGE-M3 + LR",
              "plan_knn": "Per plan, BGE-M3 $k$NN", "ak_paper": "Adaptive $k$, published rule (+5 buffer)",
              "adaptive_k": "Adaptive $k$, largest gap only"})
FAMILY = {}
N_LAB = 0
CASCADES = {"qwen2.5-7b->llama3.1-8b": "7B $\\rightarrow$ Llama 8B", "qwen2.5-7b->qwen3-14b": "7B $\\rightarrow$ 14B"}

TXT = []


def emit(name, tex, txt):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
    TXT.append(f"===== {name}\n{txt}\n")
    print(f"wrote {OUT / name}.tex")


def boot_ci(d, n=CONFIG["n_boot"], seed=CONFIG["seed"]):
    rng = np.random.default_rng(seed)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(p):
    p = np.asarray(p, float)
    order, m, run, adj = np.argsort(p), len(p), 0.0, np.empty(len(p))
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def f3(x):
    return f"{x:+.3f}".replace("-", "$-$") if np.isfinite(x) else "--"


def load_labels(reader):
    df = pd.read_csv(PROC / f"tsr_labels_{reader}_graded.csv", dtype={"query_id": str}, keep_default_na=False)
    df = df[df["repeat"].astype(int) == 0].copy()
    df["depth"] = df["depth"].astype(int)
    df["plan"] = df["retriever"] + "@" + df["depth"].astype(str)
    return df


# ======================================================================
def tab_main(M, PQ):
    pooled = []
    for (rd, m), g in PQ.groupby(["reader", "method"], sort=False):
        d = g["gain"].to_numpy()
        lo, hi = boot_ci(d)
        pooled.append({"reader": rd, "method": m, "gain": d.mean(), "lo": lo, "hi": hi,
                       "p": float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else 1.0, "n": len(d)})
    P = pd.DataFrame(pooled)
    # PRIMARY family (declared in docs/DECISIONS.md): 10's five methods + the two reader cascades, pooled.
    # SECONDARY family: 13's faithful and stronger baselines, pooled. Holm within each family.
    P["family"] = np.where(P.method.isin(METHODS + ["reader_cascade"]), "primary", "secondary")
    P["p_holm"] = np.nan
    for fam, g in P.groupby("family"):
        P.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    lvl = M[M.dataset != "pooled"]

    def cell(rd, m):
        r = P[(P.reader == rd) & (P.method == m)]
        if r.empty:
            return "--", "--"
        r = r.iloc[0]
        g = f3(r.gain)
        if r.p_holm < 0.05:
            g = f"\\textbf{{{g}}}"
        return g, f"[{f3(r.lo)}, {f3(r.hi)}]"

    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Gain over the fixed plan hull at equal mean latency, pooled over four datasets "
             "(2{,}000 queries per cell; held out 5 fold, 5 seeds). 95\\% bootstrap CI in brackets. Bold: Holm "
             f"adjusted $p<0.05$ within its family (upper block: primary, {int((P.family == 'primary').sum())} pooled tests; "
             f"lower block: secondary, {int((P.family == 'secondary').sum())}). Last column (exploratory): dataset level tests (of 12) with Holm "
             f"adjusted $p<0.05$ over all {FAMILY.get('n', len(lvl))}. Lower block: the published Adaptive $k$ rule and "
             "stronger question only routers (Section~\\ref{sec:routers}).}",
             "\\label{tab:main}", "\\small",
             "\\begin{tabular}{l" + "rr" * len(READERS) + "c}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{R_TEX[r]}}}" for r in READERS) + " & Holm \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(READERS))),
             "Method & " + " & ".join(["gain & 95\\% CI"] * len(READERS)) + " & sig. \\\\", "\\midrule"]
    txt = [f"{'method':22s} " + " ".join(f"{r:>30s}" for r in READERS)]
    blocks = [METHODS] + ([[m for m in ROUTERS if m in set(P.method)]] if set(ROUTERS) & set(P.method) else [])
    for bi, block in enumerate(blocks):
      if bi:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{2 * len(READERS) + 2}}}{{l}}{{\\emph{{Faithful and stronger baselines (13)}}}} \\\\")
      for m in block:
        cells = [cell(r, m) for r in READERS]
        nsig = int(((lvl.method == m) & (lvl.p_holm < 0.05)).sum())
        nneg = int(((lvl.method == m) & (lvl.p_holm < 0.05) & (lvl.gain < 0)).sum())
        sig = f"{nsig}" + (" (neg.)" if nsig and nneg == nsig else "")
        lines.append(f"{M_TEX[m]} & " + " & ".join(f"{a} & {b}" for a, b in cells) + f" & {sig} \\\\")
        txt.append(f"{m:22s} " + " ".join(f"{a.replace('$', '').replace(chr(92) + 'textbf', ''):>9s} "
                                          f"{b.replace('$', ''):>20s}" for a, b in cells) + f"  holm_sig={sig}")
    lines.append("\\midrule")
    for cs, lab in CASCADES.items():
        r = P[(P.reader == cs) & (P.method == "reader_cascade")]
        if r.empty:
            continue
        r = r.iloc[0]
        g = f"\\textbf{{{f3(r.gain)}}}" if r.p_holm < 0.05 else f3(r.gain)
        span = 2 * len(READERS)
        lines.append(f"Reader cascade {lab} & \\multicolumn{{{span}}}{{l}}{{{g} \\; [{f3(r.lo)}, {f3(r.hi)}] "
                     f"(hull over both readers' plans)}} & 0 \\\\")
        txt.append(f"reader_cascade {cs:28s} {r.gain:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] p_holm {r.p_holm:.3f}")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    txt.append("\npooled p values (t test on per query gains), Holm within family (primary / secondary):")
    txt += [f"  {r.family:9s} {r.reader:24s} {r.method:17s} {r.gain:+.3f} p {r.p:.2e} p_holm {r.p_holm:.2e}"
            for r in P.sort_values(["family", "method"]).itertuples()]
    emit("tab_main", "\n".join(lines), "\n".join(txt))
    return P


def tab_per_dataset(M):
    lvl = M[M.dataset != "pooled"]
    cols = [(r, d) for r in READERS for d in DATASETS]
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Gain over the held out hull per reader and dataset (500 queries per cell, mean of 5 seeds). "
             f"$^\\dagger$ nominal $p<0.05$; $^{{*}}$ Holm adjusted $p<0.05$ over all {FAMILY.get('n', len(lvl))} dataset level tests.}}",
             "\\label{tab:per_dataset}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{R_TEX[r]}}}" for r in READERS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(READERS))),
             "Method & " + " & ".join(D_TEX[d] for _, d in cols) + " \\\\", "\\midrule"]
    txt = [f"{'method':18s} " + " ".join(f"{r[:6] + '/' + D_TEX[d]:>16s}" for r, d in cols)]
    for m in METHODS + [x for x in ROUTERS if x in set(lvl.method)]:
        cells, tcells = [], []
        for r, d in cols:
            q = lvl[(lvl.reader == r) & (lvl.dataset == d) & (lvl.method == m)]
            if q.empty:
                cells.append("--"); tcells.append("--"); continue
            q = q.iloc[0]
            mark = "$^{*}$" if q.p_holm < 0.05 else ("$^\\dagger$" if q.p < 0.05 else "")
            cells.append(f3(q.gain) + mark)
            tcells.append(f"{q.gain:+.3f}" + ("*" if q.p_holm < 0.05 else ("+" if q.p < 0.05 else "")))
        lines.append(f"{M_TEX[m]} & " + " & ".join(cells) + " \\\\")
        txt.append(f"{m:18s} " + " ".join(f"{c:>16s}" for c in tcells))
    lines += ["\\midrule"]
    for cs, lab in CASCADES.items():
        cells = []
        for d in DATASETS:
            q = lvl[(lvl.reader == cs) & (lvl.dataset == d)]
            cells.append("--" if q.empty else f3(q.iloc[0].gain) + ("$^\\dagger$" if q.iloc[0].p < 0.05 else ""))
        lines.append(f"Reader cascade {lab} & \\multicolumn{{4}}{{l}}{{" + " \\; ".join(
            f"{D_TEX[d]} {c}" for d, c in zip(DATASETS, cells)) + "}" + " & " * 8 + " \\\\")
        txt.append(f"cascade {cs}: " + "  ".join(f"{D_TEX[d]} {c.replace('$', '')}" for d, c in zip(DATASETS, cells)))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    emit("tab_per_dataset", "\n".join(lines), "\n".join(txt))


# ======================================================================
def tab_headroom(FP, M):
    rows = []
    for rd in READERS:
        df = load_labels(rd)
        lab = "label_final" if "label_final" in df.columns else "label_f1_07"
        for ds in DATASETS:
            d = df[df.dataset == ds]
            Y = d.pivot_table(index="query_id", columns="plan", values=lab)
            Y = Y[Y.columns[Y.notna().all()]]
            fp = FP[(FP.reader == rd) & (FP.dataset == ds)].set_index("plan")
            cost = fp.loc[Y.columns, "cost_s"]
            acc = Y.mean()
            best = acc.idxmax()
            cb = acc.get("closed_book@0", np.nan)
            Yn = Y.to_numpy()
            solved = Yn.max(1) == 1
            c = cost.to_numpy()
            # oracle: per query, the cheapest plan that answers correctly; unsolved queries pay the cheapest plan
            big = np.where(Yn == 1, c[None, :], np.inf)
            oc = np.where(solved, big.min(1), c.min())
            # oracle restricted to {closed book, best fixed plan}: the value of a perfect retrieve or not gate
            if "closed_book@0" in Y.columns:
                yc, yb = Y["closed_book@0"].to_numpy(), Y[best].to_numpy()
                gate_s = np.maximum(yc, yb).mean()
                gate_c = np.where(yc == 1, cost["closed_book@0"], cost["closed_book@0"] + cost[best]).mean()
            else:
                gate_s = gate_c = np.nan
            meth = M[(M.reader == rd) & (M.dataset == ds) & (M.method.isin(METHODS))]
            rows.append({"reader": rd, "dataset": ds, "closed_book": cb, "best_plan": best, "best": acc[best],
                         "best_cost": cost[best], "oracle": solved.mean(), "oracle_cost": oc.mean(),
                         "headroom": solved.mean() - acc[best], "gate_oracle": gate_s, "gate_oracle_cost": gate_c,
                         "best_method_gain": meth.gain.max() if len(meth) else np.nan,
                         "n_plans": Y.shape[1]})
    H = pd.DataFrame(rows)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Headroom. Success (final label) and mean latency (s). Oracle: per query, the cheapest plan "
             "that answers correctly. Gate oracle: a perfect choice between closed book and the best fixed plan. "
             "Best gain: the largest held out gain of any adaptive method (Table~\\ref{tab:main}).}",
             "\\label{tab:headroom}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{llrrrrrrr}", "\\toprule",
             "Reader & Data & CB & \\multicolumn{2}{c}{Best fixed} & \\multicolumn{2}{c}{Oracle} & Gate or. & Best gain \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             " & & succ. & succ. & s & succ. & s & succ. & \\\\", "\\midrule"]
    txt = [f"{'reader':12s} {'data':8s} {'CB':>6s} {'best':>6s} {'cost':>6s} {'oracle':>7s} {'cost':>6s} "
           f"{'headrm':>7s} {'gateOr':>7s} {'bestGain':>8s}  best plan"]
    for i, r in H.iterrows():
        lines.append(f"{R_TEX[r.reader] if r.dataset == DATASETS[0] else ''} & {D_TEX[r.dataset]} & {r.closed_book:.3f} & "
                     f"{r.best:.3f} & {r.best_cost:.2f} & {r.oracle:.3f} & {r.oracle_cost:.2f} & {r.gate_oracle:.3f} & "
                     f"{f3(r.best_method_gain)} \\\\")
        txt.append(f"{r.reader:12s} {D_TEX[r.dataset]:8s} {r.closed_book:6.3f} {r.best:6.3f} {r.best_cost:6.2f} "
                   f"{r.oracle:7.3f} {r.oracle_cost:6.2f} {r.headroom:+7.3f} {r.gate_oracle:7.3f} "
                   f"{r.best_method_gain:+8.3f}  {r.best_plan}")
        if r.dataset == DATASETS[-1] and r.reader != READERS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    txt.append(f"\nmean headroom (oracle minus best fixed): {H.headroom.mean():+.3f}; "
               f"oracle cost / best fixed cost: {(H.oracle_cost / H.best_cost).median():.2f} (median); "
               f"gate oracle minus best fixed: {(H.gate_oracle - H.best).mean():+.3f}")
    emit("tab_headroom", "\n".join(lines), "\n".join(txt))
    return H


# ======================================================================
def tab_grading():
    from tsr.grader import normalize
    key = pd.read_csv(GDIR / "audit_key.csv", dtype={"query_id": str}, keep_default_na=False)
    xl = GDIR / "audit_rater1.xlsx"
    r1 = pd.read_excel(xl, sheet_name="audit") if xl.exists() else \
        pd.read_csv(GDIR / "audit_rater1.csv", encoding="utf-8-sig", keep_default_na=False)
    key["human"] = key["audit_id"].map(dict(zip(r1["audit_id"], pd.to_numeric(r1["label_1_correct_0_wrong"]))))
    jd = pd.read_csv(GDIR / "judge_scores.csv")
    jmap = dict(zip(jd["pair"], (jd["p_yes"] >= 0.5).astype(int)))
    rules = ["label_strict", "label_f1_07", "label_lenient", "label_proposed", "label_final"]
    graded, rates = {}, {r: [] for r in rules}
    for rd in READERS:
        g = pd.read_csv(PROC / f"tsr_labels_{rd}_graded.csv", dtype={"query_id": str}, keep_default_na=False)
        g = g[g["repeat"].astype(int) == 0]
        for r in rules:
            rates[r].append(g[r].astype(int).to_numpy())
        g["pair"] = g["dataset"] + "::" + g["query_id"] + "::" + g["answer"].map(normalize)
        graded.update(g.drop_duplicates("pair").set_index("pair")[rules + ["ambiguous"]].to_dict("index"))
    global N_LAB
    N_LAB = int(sum(len(x) for x in rates["label_strict"]))
    ev = key[key["pair"].isin(graded) & key["human"].notna()].copy()
    for r in rules + ["ambiguous"]:
        ev[r] = [int(graded[p][r]) for p in ev["pair"]]
    ev["judge"] = [jmap.get(p, np.nan) for p in ev["pair"]]
    ev["human"] = ev["human"].astype(int)
    names = {"label_strict": "Strict (EM or contains)", "label_f1_07": "Strict or F1 $\\geq$ 0.7",
             "label_lenient": "Strict or F1 $\\geq$ 0.4", "label_proposed": "Token rule (multi token gold)",
             "label_final": "Final: judge on ambiguous, else strict", "judge": "Judge alone (ambiguous items)"}
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Grading rules against human labels on a stratified audit of 200 answers, labelled independently by "
             "two human raters (agreement below). Acc.: agreement with rater 1, 95\\% bootstrap CI. FP: rule says "
             "correct, human says wrong. Success: share of all " + f"{N_LAB:,}".replace(",", "{,}") +
             " labelled answers the rule marks correct.}",
             "\\label{tab:grading}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrcrrrr}", "\\toprule",
             "Rule & Acc. & 95\\% CI & $\\kappa$ & FP & FN & Success \\\\", "\\midrule"]
    txt = [f"audit items with human label: {len(ev)}; human correct rate {ev.human.mean():.3f}"]
    for r in rules + ["judge"]:
        e = ev.dropna(subset=[r])
        y, h = e[r].astype(int).to_numpy(), e["human"].to_numpy()
        ok = (y == h).astype(float)
        lo, hi = boot_ci(ok)
        succ = np.concatenate(rates[r]).mean() if r in rates else np.nan
        fp, fn = int(((y == 1) & (h == 0)).sum()), int(((y == 0) & (h == 1)).sum())
        lines.append(f"{names[r]} & {ok.mean():.3f} & [{lo:.3f}, {hi:.3f}] & {kappa(y, h):.3f} & {fp} & {fn} & "
                     f"{'--' if np.isnan(succ) else f'{succ:.3f}'} \\\\")
        txt.append(f"{r:16s} n={len(e):3d} acc {ok.mean():.3f} [{lo:.3f}, {hi:.3f}] kappa {kappa(y, h):.3f} "
                   f"FP {fp:2d} FN {fn:2d} success {succ:.3f}")
    # inter rater agreement: second HUMAN (audit_rater2.xlsx, once complete) and the LLM rater (disclosed)
    def other(src):
        if not src.exists():
            return None
        o = pd.read_excel(src, sheet_name="audit") if src.suffix == ".xlsx" else \
            pd.read_csv(src, encoding="utf-8-sig", keep_default_na=False)
        m = dict(zip(o["audit_id"], pd.to_numeric(o["label_1_correct_0_wrong"], errors="coerce")))
        b = ev.assign(o=ev["audit_id"].map(m))
        return b if b["o"].notna().all() else len(b.dropna(subset=["o"]))
    lines.append("\\midrule")
    for src, lab in ((GDIR / "audit_rater2.xlsx", "Second human rater"),
                     (GDIR / "audit_llm.csv", "LLM rater (not human)")):
        b = other(src)
        if b is None:
            continue
        if not isinstance(b, pd.DataFrame):
            txt.append(f"{lab}: {b} of {len(ev)} labelled, not complete (not reported)")
            continue
        k2, ag = kappa(b["human"], b["o"].astype(int)), (b["human"] == b["o"]).mean()
        txt.append(f"rater 1 (human) vs {lab}: kappa {k2:.3f}, agreement {ag:.3f}, n={len(b)}")
        lines.append(f"\\multicolumn{{7}}{{l}}{{Rater 1 vs {lab}: $\\kappa$ = {k2:.3f}, agreement {ag:.3f}}} \\\\")
        if src.suffix == ".xlsx":
            fin = b["label_final"].to_numpy() == b["o"].astype(int).to_numpy()
            txt.append(f"final label vs second human: accuracy {fin.mean():.3f}, "
                       f"kappa {kappa(b['label_final'], b['o'].astype(int)):.3f}")
    st = ev.groupby("stratum").apply(lambda e: pd.Series({"n": len(e), "human": e.human.mean(),
                                                          "final_acc": (e.label_final == e.human).mean()}))
    txt.append("by stratum:\n" + st.round(3).to_string())
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    emit("tab_grading", "\n".join(lines), "\n".join(txt))


# ======================================================================
def tab_cost(FP):
    from cost_model import CostModel
    pf = PROC / "cost_model_params.json"
    raw = json.load(open(pf))["by_dataset"]
    rows, err = [], []
    for ds in DATASETS:
        for rd in READERS:
            ag = raw[ds]["agent"].get(rd)
            if ag is None:
                continue
            cm = CostModel(params_file=pf, dataset=ds, reader=rd)
            fp = FP[(FP.reader == rd) & (FP.dataset == ds)]
            for p in fp.itertuples():
                r, d = p.plan.split("@")
                d = int(d)
                if d == 0 or r not in cm.retrieval:
                    continue
                pred = cm.plan_cost(r, d)
                err.append({"dataset": ds, "reader": rd, "plan": p.plan, "measured": p.cost_s, "pred": pred,
                            "ape": abs(pred - p.cost_s) / p.cost_s})
            rows.append({"dataset": ds, "reader": rd, "a1_ms_per_100tok": ag["a1"] * 1e5, "a2_ms_per_tok": ag["a2"] * 1e3,
                         "gen_mean": ag["gen_mean"], "r2": ag.get("r2", np.nan)})
    A, E = pd.DataFrame(rows), pd.DataFrame(err)
    ret = pd.DataFrame([{"retriever": k, "dataset": ds, "c_ms": v["c"] * 1e3, "a_us": v["a"] * 1e6}
                        for ds in DATASETS for k, v in raw[ds]["retrieval"].items()])
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Plan cost model, $\\mathrm{cost}(r,k) = c_r + a_r k + a_0 + a_1 \\cdot \\mathrm{prompt}(r,k) + "
             "a_2 \\cdot \\overline{\\mathrm{gen}}$, fitted per dataset and reader. Values are ranges over the four "
             "datasets. APE: absolute percentage error of the predicted against the measured median plan latency "
             "(all retrieval plans).}",
             "\\label{tab:cost}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Reader & $a_1$ (ms/100 tok) & $a_2$ (ms/tok) & $R^2$ & med. APE & max APE \\\\", "\\midrule"]
    txt = ["agent cost per reader (range over datasets) and plan cost error:"]

    def rng(s, f="{:.1f}"):
        return f.format(s.min()) if np.isclose(s.min(), s.max(), rtol=0.05) else f"{f.format(s.min())}--{f.format(s.max())}"

    for rd in READERS:
        a, e = A[A.reader == rd], E[E.reader == rd]
        lines.append(f"{R_TEX[rd]} & {rng(a.a1_ms_per_100tok, '{:.1f}')} & {rng(a.a2_ms_per_tok, '{:.0f}')} & "
                     f"{rng(a.r2, '{:.2f}')} & {100 * e.ape.median():.1f}\\% & {100 * e.ape.max():.1f}\\% \\\\")
        txt.append(f"  {rd:12s} a1 {a.a1_ms_per_100tok.min():.2f}..{a.a1_ms_per_100tok.max():.2f} ms/100tok  "
                   f"a2 {a.a2_ms_per_tok.min():.1f}..{a.a2_ms_per_tok.max():.1f} ms/tok  R2 {a.r2.min():.3f}..{a.r2.max():.3f}  "
                   f"gen {a.gen_mean.min():.1f}..{a.gen_mean.max():.1f} tok  APE median {100 * e.ape.median():.1f}% "
                   f"max {100 * e.ape.max():.1f}% (n={len(e)})")
    lines += ["\\midrule", "Retriever & \\multicolumn{2}{r}{$c_r$ (ms)} & \\multicolumn{3}{r}{$a_r$ ($\\mu$s/doc)} \\\\",
              "\\midrule"]
    for r, g in ret.groupby("retriever", sort=False):
        rname = r.replace("_", "\\_")
        lines.append(f"{rname} & \\multicolumn{{2}}{{r}}{{{rng(g.c_ms, '{:.0f}')}}} & "
                     f"\\multicolumn{{3}}{{r}}{{{rng(g.a_us, '{:.0f}')}}} \\\\")
        txt.append(f"  {r:22s} c {g.c_ms.min():.1f}..{g.c_ms.max():.1f} ms   a {g.a_us.min():.1f}..{g.a_us.max():.1f} us/doc")
    worst = E.sort_values("ape", ascending=False).head(5)
    txt.append("worst plans:\n" + worst.round(3).to_string(index=False))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    emit("tab_cost", "\n".join(lines), "\n".join(txt))


# ======================================================================
def plan_tex(p):
    return p.replace("_", "\\_")


def tab_ablation():
    pat = re.compile(r"^\s*(\S+)@(\d+): strict prompt ([\d.]+) \| soft prompt ([\d.]+) \| soft minus strict ([+-][\d.]+) "
                     r"\[([+-][\d.]+), ([+-][\d.]+)\] \| known answers lost[^:]*: strict (\d+), soft (\d+) \| "
                     r"vs closed book: strict ([+-][\d.]+), soft ([+-][\d.]+)")
    rows = []
    for f in sorted((PROC / "ablation_prompt").glob("report_*.txt")):
        ds = f.stem.replace("report_", "")
        reader = None
        for line in f.read_text(encoding="utf-8").splitlines():
            m0 = re.match(r"== (\S+): closed book", line)
            if m0:
                reader = m0.group(1)
            m = pat.match(line)
            if m and reader:
                g = m.groups()
                rows.append({"dataset": ds, "reader": reader, "plan": f"{g[0]}@{g[1]}", "strict": float(g[2]),
                             "soft": float(g[3]), "diff": float(g[4]), "lo": float(g[5]), "hi": float(g[6]),
                             "lost_strict": int(g[7]), "lost_soft": int(g[8]), "vs_cb_strict": float(g[9]),
                             "vs_cb_soft": float(g[10])})
    if not rows:
        return
    T = pd.DataFrame(rows)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Prompt ablation (strict label). Strict instruction: answer only from the passages (main "
             "runs). Soft: use passages if they help, else own knowledge. Same queries and passages. Lost: queries "
             "answered correctly closed book but wrongly with retrieval.}",
             "\\label{tab:ablation}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lllrrrrr}", "\\toprule",
             "Data & Reader & Plan & Strict & Soft & Soft$-$strict [95\\% CI] & Lost (str./soft) & vs CB (str./soft) \\\\",
             "\\midrule"]
    for r in T.itertuples():
        lines.append(f"{D_TEX.get(r.dataset, r.dataset)} & {R_TEX.get(r.reader, r.reader)} & "
                     f"{plan_tex(r.plan)} & {r.strict:.3f} & {r.soft:.3f} & {f3(r.diff)} [{f3(r.lo)}, {f3(r.hi)}] & "
                     f"{r.lost_strict}/{r.lost_soft} & {f3(r.vs_cb_strict)}/{f3(r.vs_cb_soft)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    emit("tab_ablation", "\n".join(lines), T.round(3).to_string(index=False))


def main():
    M = pd.read_csv(ANA / "methods.csv")
    FP = pd.read_csv(ANA / "fixed_plans.csv")
    PQ = pd.read_csv(ANA / "per_query_gains.csv", dtype={"query_id": str})
    if (ANA / "routers.csv").exists():
        # one family for the dataset level Holm correction: every test run in 10 and 13
        R = pd.read_csv(ANA / "routers.csv")
        lvl = pd.concat([M[M.dataset != "pooled"], R], ignore_index=True)
        lvl["p_holm"] = holm(lvl["p"].to_numpy())
        M = pd.concat([lvl, M[M.dataset == "pooled"]], ignore_index=True)
        PQ = pd.concat([PQ, pd.read_csv(ANA / "routers_per_query.csv", dtype={"query_id": str})], ignore_index=True)
        PQ = PQ[PQ.method != "ar_tfidf"]          # near duplicate of adaptive_rag; kept in the Holm family only
        FAMILY["n"] = len(lvl)
    tab_main(M, PQ)
    tab_per_dataset(M)
    tab_headroom(FP, M)
    tab_grading()
    tab_cost(FP)
    tab_ablation()
    (OUT / "tables.txt").write_text("\n".join(TXT), encoding="utf-8")
    print("\n".join(TXT))


if __name__ == "__main__":
    main()
