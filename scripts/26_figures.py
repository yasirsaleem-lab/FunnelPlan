"""
Final data figures (5 readers). CPU only; reads results, never reruns an analysis.
Needs scripts/12_tables_final.py to have run first (it writes tables_final/audit_rules.csv).

Paper figures follow the IEEE conference template: exact print widths (7.16 in for two columns, 3.5 in for one),
8 pt text, no figure titles (the LaTeX caption carries the title), colours from figures_src/palette.py.
  f2_h1_forest           H1, 27 primary tests in both plan sets              two columns
  f4_frontier            two example frontiers with the policies             two columns
  f6_h3                  IRCoT against the hull at its own latency            two columns
  f7_r6                  the same answers under the weak protocol and ours    one column
  f9_grading             grading rules against the human majority            one column
  f10_headroom_matched   held out headroom and what planners learn           two columns
Repository figures (titled, same style): f3_headroom, f5_r3, f8_secondary, f11_learning_curve.

Usage:  python scripts/26_figures.py              (writes results_full/figures_final/)
        python scripts/26_figures.py --out DIR    (writes to DIR instead; for testing)
Output: .pdf (vector, TrueType fonts embedded) and .png at 600 dpi
"""
import importlib.util, os, sys
from pathlib import Path
os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import settings
from figures_src.palette import (CATEGORICAL, DATA_AMBER as AMB, DATA_GREY as GREY, DATA_SLATE as SLATE, DATA_TEAL as TEAL,
                     INK, MUTED, TEAL_RAMP)

_s = importlib.util.spec_from_file_location("t12", Path(__file__).parent / "12_tables_final.py")
T12 = importlib.util.module_from_spec(_s); _s.loader.exec_module(T12)
AN = settings.RESULTS_ROOT / "analysis"
ANS = str(AN) + "/"
PROC = settings.PROCESSED_DATA_ROOT
OUTD = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else settings.RESULTS_ROOT / "figures_final"
OUTD.mkdir(parents=True, exist_ok=True)

DPI = 600                  # PNG resolution; the PDF is vector
W2, W1 = 7.16, 3.5         # IEEE two column and one column figure widths (inches)
GRID, BAND = "#E6E8EB", "#F1F2F4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.titlesize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03})
RN = {"qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B", "qwen3-14b": "Qwen3 14B", "gemma3-27b": "Gemma 3 27B",
      "qwen3-32b": "Qwen3 32B"}
READERS = list(RN)
DN = {"nq": "NQ", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki", "musique": "MuSiQue"}
PN = {"planner_tsr": "TSR planner", "adaptive_rag": "Adaptive RAG", "adaptive_k": "Adaptive k",
      "gate_closed_book": "Closed book gate", "conf_cascade": "Confidence cascade", "reader_cascade": "Reader cascade"}


def _save(fig, name, exact=True):
    """exact=True keeps the figure size (paper figures: the print width is the IEEE width); False crops the margins."""
    base = OUTD / name.replace(".png", "")
    kw = {} if exact else {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(str(base) + ".png", dpi=DPI, **kw); fig.savefig(str(base) + ".pdf", **kw)
    plt.close(fig)


def hgrid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


def ci_marks(ax, rows, y, off, col, mk):
    for yi, r in zip(y, rows):
        if r is None:
            continue
        sig = r["p_holm"] < 0.05
        ax.plot([r["ci_low"], r["ci_high"]], [yi + off] * 2, color=col, lw=1.3, solid_capstyle="round", zorder=2)
        ax.plot(r["gain"], yi + off, mk, ms=4.6 if mk == "o" else 4.0, color=col if sig else "white", mec=col, mew=1.1, zorder=3)


# ---------------------------------------------------------------- F2 H1, six panels (paper)
def f2():
    A, B = T12.primary_family(AN / "final"), T12.primary_family(AN / "final_r3")
    def get(F, m, r):
        x = F[(F.method == m) & (F.reader == r)].iloc[0]
        return dict(gain=x.gain, ci_low=x.ci_low, ci_high=x.ci_high, p_holm=x.p_holm)
    panels = [(m, [(m, r) for r in READERS], [RN[r] for r in READERS])
              for m in ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade"]]
    panels.append(("reader_cascade", [("reader_cascade", "qwen2.5-7b->qwen3-14b"), ("reader_cascade", "qwen2.5-7b->llama3.1-8b")],
                   ["7B to Qwen3 14B", "7B to Llama 3.1 8B"]))
    fig, axes = plt.subplots(2, 3, figsize=(W2, 3.2), sharex=True)
    for ax, (m, keys, labels) in zip(axes.flat, panels):
        n = len(keys); y = np.arange(n)[::-1].astype(float)
        ax.axvspan(-0.027, 0.027, color=BAND, zorder=0, lw=0)
        ax.axvline(0, color=SLATE, lw=0.8, zorder=1)
        ci_marks(ax, [get(A, *k) for k in keys], y, 0.16, TEAL, "o")
        ci_marks(ax, [get(B, *k) for k in keys], y, -0.16, AMB, "D")
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_ylim(-0.7, 4.7 if n == 5 else 2.2)
        ax.set_title(PN[m], loc="left", pad=3); hgrid(ax)
        ax.set_xlim(-0.10, 0.06); ax.set_xticks([-0.08, -0.04, 0, 0.04])
    for ax in axes[1]:
        ax.set_xlabel("Gain over the hull")
    h = [plt.Line2D([], [], marker="o", color=TEAL, lw=1.3, ms=4.6, label="single step plans"),
         plt.Line2D([], [], marker="D", color=AMB, lw=1.3, ms=4.0, label="with fixed two hop plans"),
         plt.Line2D([], [], marker="o", color=MUTED, mfc="white", lw=0, ms=4.6, label="open: Holm p of 0.05 or more"),
         plt.Rectangle((0, 0), 1, 1, color=BAND, label="smaller than the detectable effect (0.027)")]
    fig.legend(handles=h, loc="outside upper center", ncol=4, handletextpad=0.4, columnspacing=1.2)
    _save(fig, "f2_h1_forest.png")


# ---------------------------------------------------------------- F3 headroom (repository)
def f3():
    rows = []
    for rd in READERS:
        d = pd.read_csv(PROC / f"tsr_labels_{rd}_graded.csv", dtype={"query_id": str}, keep_default_na=False,
                        usecols=["dataset", "query_id", "retriever", "depth", "repeat", "label_final"])
        d = d[(d.repeat.astype(int) == 0) & (d.depth.astype(int) <= 10)]
        d["plan"] = d.retriever + "@" + d.depth.astype(str); d["y"] = d.label_final.astype(int)
        for ds, gg in d.groupby("dataset"):
            Y = gg.pivot_table(index="query_id", columns="plan", values="y"); acc = Y.mean()
            rows.append(dict(reader=rd, closed_book=acc.get("closed_book@0"), best=acc.max(), oracle=Y.max(axis=1).mean()))
    g = pd.DataFrame(rows).groupby("reader")[["closed_book", "best", "oracle"]].mean().loc[READERS]
    fig, ax = plt.subplots(figsize=(W2, 2.6)); x = np.arange(len(READERS)); w = 0.26
    for i, (c, col, lab) in enumerate((("closed_book", GREY, "closed book"), ("best", TEAL, "best single fixed plan"),
                                       ("oracle", AMB, "per question oracle over 25 plans"))):
        bars = ax.bar(x + (i - 1) * (w + 0.02), g[c], w, color=col, label=lab, zorder=2)
        for bx, v in zip(bars, g[c]):
            ax.text(bx.get_x() + bx.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", color=INK)
    ax.set_xticks(x); ax.set_xticklabels([RN[r] for r in READERS]); ax.set_ylim(0, 0.9)
    ax.set_ylabel("Success (label_final)"); hgrid(ax, "y")
    ax.set_title("Headroom by reader (mean over 4 datasets, in sample, same 25 plans)", loc="left")
    ax.legend(ncol=3, loc="upper left")
    _save(fig, "f3_headroom.png")


# ---------------------------------------------------------------- F4 frontier examples (paper)
def f4():
    F = pd.read_csv(ANS + "final/fixed_plans.csv"); M = pd.read_csv(ANS + "final/methods.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.45))
    cols = dict(zip(["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade"], CATEGORICAL))
    for ax, (rd, ds) in zip(axes, [("qwen3-32b", "hotpotqa"), ("qwen2.5-7b", "musique")]):
        f = F[(F.reader == rd) & (F.dataset == ds)].sort_values("cost_s")
        ax.scatter(f.cost_s, f.success, s=11, color=GREY, zorder=2, label="fixed plan")
        pts = f[["cost_s", "success"]].values; pts = pts[np.argsort(pts[:, 0])]
        par, best = [], -1
        for pnt in pts:
            if pnt[1] > best:
                par.append(pnt); best = pnt[1]
        hl = []
        for pnt in par:
            while len(hl) >= 2 and (hl[-1][1] - hl[-2][1]) * (pnt[0] - hl[-2][0]) <= (pnt[1] - hl[-2][1]) * (hl[-1][0] - hl[-2][0]):
                hl.pop()
            hl.append(pnt)
        hl = np.array(hl); ax.plot(hl[:, 0], hl[:, 1], color=SLATE, lw=1.1, zorder=3, label="fixed plan hull")
        m = M[(M.reader == rd) & (M.dataset == ds)]
        for meth, col in cols.items():
            r = m[m.method == meth]
            if len(r):
                ax.scatter(r.cost_s, r.success, s=30, color=col, edgecolor="white", lw=0.7, zorder=4, label=PN[meth])
        ax.set_xscale("log"); ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([0.2, 0.5, 1, 2, 3])
        ax.set_xlabel("Mean latency per question (s, log scale)"); ax.set_ylabel("Success")
        ax.set_title(f"{RN[rd]}, {DN[ds]}", loc="left", pad=3); hgrid(ax, "both")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="outside right center", ncol=1, handletextpad=0.3)
    _save(fig, "f4_frontier.png")


# ---------------------------------------------------------------- F5 R3 heatmap (repository)
def f5():
    F = pd.read_csv(ANS + "final_r3/fixed_plans.csv"); rows = []
    for (rd, ds), g in F.groupby(["reader", "dataset"]):
        s1 = g[~g.plan.str.startswith("bm25_2hop")].success.max(); s2 = g[g.plan.str.startswith("bm25_2hop")].success.max()
        rows.append(dict(reader=rd, dataset=ds, diff=s2 - s1))
    P = pd.DataFrame(rows).pivot(index="reader", columns="dataset", values="diff").loc[READERS, list(DN)]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [AMB, "#F4F4F2", TEAL])
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    im = ax.imshow(P.values, cmap=cmap, norm=TwoSlopeNorm(0, -0.2, 0.2), aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.values[i, j]; ax.text(j, i, f"{v:+.3f}".replace("-", "\u2212"), ha="center", va="center", color=INK if abs(v) < 0.13 else "white")
    ax.set_xticks(range(4)); ax.set_xticklabels([DN[d] for d in P.columns]); ax.set_yticks(range(5)); ax.set_yticklabels([RN[r] for r in P.index])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Best fixed two hop plan minus best single step plan (in sample)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
    _save(fig, "f5_r3.png", exact=False)


# ---------------------------------------------------------------- F6 H3 (paper)
def f6():
    H = pd.read_csv(ANS + "h3_ircot/h3.csv"); a = H[H.hull == "single_step"].set_index(["reader", "dataset"])
    b = H[H.hull == "with_r3"].set_index(["reader", "dataset"])
    cells = [(r, d) for r in ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"] for d in ["hotpotqa", "2wikimultihopqa", "musique"]]
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [4.2, 1]}, sharey=True)
    y = np.arange(len(cells))[::-1]
    for yi, c in zip(y, cells):
        s, t = a.loc[c], b.loc[c]
        lo = min(s.hull_at_ircot_cost, t.hull_at_ircot_cost, s.ircot_success); hi = max(s.hull_at_ircot_cost, t.hull_at_ircot_cost, s.ircot_success)
        ax.plot([lo, hi], [yi, yi], color=GRID, lw=2.0, zorder=1)
        ax.plot(s.hull_at_ircot_cost, yi, "o", color=SLATE, ms=5, zorder=3)
        ax.plot(t.hull_at_ircot_cost, yi, "D", color=AMB, ms=4.4, zorder=3)
        sig = s.p_holm < 0.05 and s.ci_low > 0
        ax.plot(s.ircot_success, yi, "o", color=TEAL if sig else "white", mec=TEAL, mew=1.2, ms=5.6, zorder=4)
        tx.text(0.0, yi, f"{s.ircot_cost_s / s.best_fixed_cost_s:.0f}x", va="center", ha="left", color=INK)
        tx.text(1.0, yi, f"{s.gain:+.3f}".replace("-", "\u2212") + ("*" if sig else ""), va="center", ha="right", color=INK)
    ax.set_yticks(y); ax.set_yticklabels([f"{RN[r]}, {DN[d]}" for r, d in cells])
    ax.set_xlim(0.18, 0.64); ax.set_xlabel("Success on held out questions (label_final)"); hgrid(ax)
    tx.set_xlim(0, 1); tx.axis("off")
    tx.text(0.0, len(cells) - 0.35, "Latency\nvs best plan", va="bottom", ha="left", color=MUTED)
    tx.text(1.0, len(cells) - 0.35, "Gain", va="bottom", ha="right", color=MUTED)
    h = [plt.Line2D([], [], marker="o", color=SLATE, lw=0, ms=5, label="single step hull at IRCoT latency"),
         plt.Line2D([], [], marker="D", color=AMB, lw=0, ms=4.4, label="hull with fixed two hop plans"),
         plt.Line2D([], [], marker="o", color=TEAL, lw=0, ms=5.6, label="IRCoT (filled: Holm significant, *)")]
    fig.legend(handles=h, loc="outside upper center", ncol=3)
    _save(fig, "f6_h3.png")


# ---------------------------------------------------------------- F7 R6 (paper, one column)
def f7():
    W = pd.read_csv(ANS + "weak_protocol/weak_vs_ours.csv")
    W["weak"] = (W.margin_vs_always > 0) & (W.margin_vs_never > 0); W["ours"] = (W.ours_ci_low > 0) & (W.ours_p_holm < 0.05)
    g = W.groupby("reader")[["weak", "ours"]].mean().loc[READERS]
    fig, ax = plt.subplots(figsize=(W1, 2.15)); x = np.arange(5); w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, g.weak, w, color=AMB, label="weak protocol: beats Always RAG and Never RAG", zorder=2)
    b2 = ax.bar(x + w / 2 + 0.01, g.ours + 0.006, w, color=TEAL, label="our protocol: beats the hull", zorder=2)
    for bs, vals in ((b1, g.weak), (b2, g.ours)):
        for bx, v in zip(bs, vals):
            ax.text(bx.get_x() + bx.get_width() / 2, v + 0.025, f"{v:.0%}", ha="center", color=INK)
    ax.set_xticks(x); ax.set_xticklabels([RN[r].replace(" ", "\n", 1) for r in READERS]); ax.set_ylim(0, 1.12)
    ax.set_ylabel("Policy and dataset cells won")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); hgrid(ax, "y")
    fig.legend(loc="outside upper center", ncol=1)
    _save(fig, "f7_r6.png")


# ---------------------------------------------------------------- F8 secondary (repository)
def f8():
    def load(tag, mt):
        p = pd.read_csv(ANS + f"{tag}/routers_pooled.csv"); p = p[p.method != "ar_tfidf"]
        m = pd.read_csv(ANS + f"{mt}/metis.csv"); m = m[(m.dataset == "pooled") & (m.method == "metis")]
        return pd.concat([p[["reader", "method", "gain", "ci_low", "ci_high", "p"]], m[["reader", "method", "gain", "ci_low", "ci_high", "p"]]])
    A, B = load("final", "metis"), load("final_r3", "metis_r3")
    for D in (A, B):
        D["p_holm"] = T12.holm(D.p.values)
    names = {"ak_paper": "Adaptive k (published rule)", "ar_emb": "Adaptive RAG, BGE M3 LR", "ar_t5": "Adaptive RAG, FLAN T5",
             "plan_emb": "Per plan LR (BGE M3)", "plan_knn": "Per plan kNN", "metis": "METIS port"}
    order = [(m, r) for m in names for r in READERS]
    def rows(D):
        out = []
        for m, r in order:
            x = D[(D.method == m) & (D.reader == r)]
            out.append(None if not len(x) else dict(gain=x.gain.iloc[0], ci_low=x.ci_low.iloc[0], ci_high=x.ci_high.iloc[0], p_holm=x.p_holm.iloc[0]))
        return out
    n = len(order); y = np.arange(n)[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(W2, 6.4))
    ax.axvspan(-0.027, 0.027, color=BAND, zorder=0, lw=0); ax.axvline(0, color=SLATE, lw=0.8)
    ci_marks(ax, rows(A), y, 0.17, TEAL, "o"); ci_marks(ax, rows(B), y, -0.17, AMB, "D")
    ax.set_yticks(y); ax.set_yticklabels([f"{names[m]}, {RN[r]}" for m, r in order]); ax.set_xlim(-0.10, 0.07)
    ax.set_xlabel("Gain over the held out fixed plan hull at equal mean latency"); hgrid(ax)
    ax.set_title("Secondary family (30 tests per plan set, Holm within each; filled = Holm p < 0.05)", loc="left")
    ax.plot([], [], "o", color=TEAL, label="single step plans"); ax.plot([], [], "D", color=AMB, label="with fixed two hop plans")
    ax.legend(loc="lower left")
    _save(fig, "f8_secondary.png", exact=False)


# ---------------------------------------------------------------- F9 grading (paper, one column)
def f9():
    A = pd.read_csv(settings.RESULTS_ROOT / "tables_final" / "audit_rules.csv")
    rules = ["strict EM", "F1 >= 0.7", "lenient", "label_final"]
    shown = {"strict EM": "Strict EM", "F1 >= 0.7": "F1 of 0.7 or more", "lenient": "Lenient", "label_final": "label_final"}
    fig, ax = plt.subplots(figsize=(W1, 2.3)); y = np.arange(len(rules))[::-1]; h = 0.36
    for k, (aud, col, lab) in enumerate(((A.Audit.unique()[0], TEAL, "audit 1: Qwen2.5 7B, Llama 3.1 8B, Qwen3 14B"),
                                         (A.Audit.unique()[1], AMB, "audit 2: Gemma 3 27B, Qwen3 32B"))):
        g = A[A.Audit == aud].set_index("Rule").loc[rules]
        yy = y + (0.19 if k == 0 else -0.19)
        ax.barh(yy, g.acc, height=h, color=col, label=lab, zorder=2)
        ax.errorbar(g.acc, yy, xerr=[g.acc - g.lo, g.hi - g.acc], fmt="none", ecolor=INK, lw=0.7, capsize=1.6, zorder=3)
        for yi, v, hi in zip(yy, g.acc, g.hi):
            ax.text(hi + 0.008, yi, f"{v:.2f}", va="center", color=INK)
    ax.set_yticks(y); ax.set_yticklabels([shown[r] for r in rules]); ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Accuracy against the majority of three raters")
    hgrid(ax); fig.legend(loc="outside upper center", ncol=1)
    _save(fig, "f9_grading.png")


# ---------------------------------------------------------------- F10 held out headroom and what planners learn (paper)
def f10():
    C = AN / "checks"
    H = pd.read_csv(C / "headroom_matched.csv").groupby("reader")[["gap_k2", "gap_k3", "gap_all"]].mean().loc[READERS]
    P = T12.primary_family(AN / "final")
    P = P[P.reader.isin(READERS)].groupby("reader").gain.max().reindex(READERS)
    M = pd.read_csv(C / "mechanism.csv").groupby("policy")[["per_plan_auroc", "pair_auroc"]].mean()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 2.35), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(READERS)); w = 0.19
    for i, (c, col, lab) in enumerate((("gap_k2", TEAL_RAMP[0], "oracle over 2 plans"), ("gap_k3", TEAL_RAMP[1], "oracle over 3 plans"),
                                       ("gap_all", TEAL_RAMP[2], "oracle over all plans"))):
        ax.bar(x + (i - 1.5) * (w + 0.012), H[c], w, color=col, label=lab, zorder=2)
    ax.bar(x + 1.5 * (w + 0.012), P.values, w, color=AMB, label=f"best policy (at most {P.max():+.3f})", zorder=2)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([RN[r].replace(" ", "\n", 1) for r in READERS])
    ax.set_ylabel("Gain over the hull, same cost"); hgrid(ax, "y"); ax.set_ylim(-0.01, 0.5)
    ax.legend(ncol=2, loc="upper left", handlelength=1.2, columnspacing=0.8)
    names = {"planner_tsr": "TSR planner", "planner_sk": "Planner with\nself knowledge"}
    yb = np.arange(len(M)); h = 0.34
    bx.barh(yb + h / 2, M.per_plan_auroc, h, color=TEAL, label="Can the question be answered?")
    bx.barh(yb - h / 2, M.pair_auroc, h, color=AMB, label="Which hull plan wins?")
    for yi, a_, b_ in zip(yb, M.per_plan_auroc, M.pair_auroc):
        bx.text(max(a_, 0.5) + 0.008, yi + h / 2, f"{a_:.2f}", va="center", color=INK)
        bx.text(max(b_, 0.5) + 0.008, yi - h / 2, f"{b_:.2f}", va="center", color=INK)
    bx.axvline(0.5, color=SLATE, lw=0.8, ls="--")
    bx.set_yticks(yb); bx.set_yticklabels([names.get(i, i) for i in M.index]); bx.set_xlim(0.4, 0.76)
    bx.set_xticks([0.4, 0.5, 0.6, 0.7])
    bx.set_xlabel("Held out AUROC (dashed: chance)"); hgrid(bx); bx.set_ylim(-0.6, len(M) + 0.5)
    bx.legend(loc="upper center", ncol=1, handlelength=1.0, borderaxespad=0.0)
    _save(fig, "f10_headroom_matched.png")


# ---------------------------------------------------------------- F11 learning curve (repository)
def f11():
    LC = pd.read_csv(AN / "stronger_policies" / "learning_curve.csv")
    g = LC.groupby(["reader", "frac"])[["sum_gain", "n"]].sum()      # pooled gain per question = sum / count
    pts = (g["sum_gain"] / g["n"]).unstack()
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for rd, col in zip(READERS, CATEGORICAL):
        if rd in pts.index:
            ax.plot(pts.columns * 100, pts.loc[rd], marker="o", ms=3.2, lw=1.3, color=col, label=RN[rd])
    ax.axhline(0, color=SLATE, lw=0.9); ax.text(21, 0.002, "hull", color=SLATE)
    ax.set_xlabel("Share of the training folds used (%)"); ax.set_ylabel("Gain over the hull")
    ax.set_xticks([20, 40, 60, 80, 100]); hgrid(ax, "y")
    ax.set_title("Planner with self knowledge: more data approaches the hull", loc="left")
    ax.legend(ncol=2, loc="lower right")
    _save(fig, "f11_learning_curve.png", exact=False)


if __name__ == "__main__":
    for f in (f2, f3, f4, f5, f6, f7, f8, f9, f10, f11):
        f()
    print(f"Written {OUTD}")
