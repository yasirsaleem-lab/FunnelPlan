"""
Final figures for the paper (5 readers). CPU only; reads results, never reruns an analysis.
Needs scripts/12_tables_final.py to have run first (it writes tables_final/audit_rules.csv).
Usage:  python scripts/26_figures.py
Output: results_full/figures_final/f1..f11 (.pdf vector with embedded fonts, .png at 600 dpi)
"""
import importlib.util, os, sys
from pathlib import Path
os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from config import settings

_s = importlib.util.spec_from_file_location("t12", Path(__file__).parent / "12_tables_final.py")
T12 = importlib.util.module_from_spec(_s); _s.loader.exec_module(T12)
AN = settings.RESULTS_ROOT / "analysis"
ANS = str(AN) + "/"
PROC = settings.PROCESSED_DATA_ROOT
OUTD = settings.RESULTS_ROOT / "figures_final"; OUTD.mkdir(parents=True, exist_ok=True)


DPI = 600     # PNG resolution; the PDF is vector


def _save(fig, name):
    base = OUTD / name.replace(".png", "")
    fig.savefig(str(base) + ".png", dpi=DPI); fig.savefig(str(base) + ".pdf")


BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.grid": False,
                     "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight", "axes.unicode_minus": True,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
RN = {"qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B", "qwen3-14b": "Qwen3 14B", "gemma3-27b": "Gemma 3 27B",
      "qwen3-32b": "Qwen3 32B"}
READERS = list(RN)
DN = {"nq": "NQ", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki", "musique": "MuSiQue"}
PN = {"planner_tsr": "TSR planner", "adaptive_rag": "Adaptive RAG router", "adaptive_k": "Adaptive k (gap)",
      "gate_closed_book": "Closed book gate", "conf_cascade": "Confidence cascade", "reader_cascade": "Reader cascade"}


def hgrid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


# ---------------------------------------------------------------- F1 framework
def f1():
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    def box(x, y, w, h, title, body, fc="#f4f8fd", ec=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.2, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=6.2, color=INK2, linespacing=1.35)
    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))
    box(0.5, 25, 22.5, 19, "1. Plan space", "closed book, plus\n6 retrievers x depth\n{1,3,5,10,20}\n+ fixed two hop plans (R3)")
    box(26, 25, 22.5, 19, "2. Plan matrix", "every query x every plan\nx every reader\n4 datasets x 500 queries\n5 readers, one call at a time")
    box(51.5, 25, 22.5, 19, "3. Labels and cost", "label_final: strict rule,\nlocal judge on ambiguous\nanswers; human audit\ncost = mean measured latency")
    box(77, 25, 22.5, 19, "4. Fixed plan hull", "best randomised mix of\nfixed plans at each\nmean latency, chosen on\ntraining folds only")
    for x in (23.4, 48.9, 74.4):
        arr(x, 34.5, x + 2.3, 34.5)
    box(1, 1.5, 30, 18, "Policies under test", "primary: TSR planner, Adaptive RAG,\nAdaptive k, closed book gate,\nconfidence cascade, reader cascades\nsecondary: 5 routers + METIS port", fc="#fdf3ee", ec=ORANGE)
    box(35, 1.5, 30, 18, "Held out protocol", "5 folds x 5 seeds; hull and every\npolicy fitted on 4 folds, scored on 1\ngain = policy success minus hull\nat the policy's own mean latency", fc="#f6f6f4", ec=MUTED)
    box(69, 1.5, 30, 18, "Statistics", "per query gains, t test,\n95% bootstrap CI, Holm within a\npre-registered family; IRCoT (R4)\ncompared at its own latency (H3)", fc="#f6f6f4", ec=MUTED)
    arr(31.6, 10.5, 34.4, 10.5); arr(65.6, 10.5, 68.4, 10.5); arr(89, 24.6, 82, 20.2)
    _save(fig, "f1_framework.png"); plt.close(fig)


# ---------------------------------------------------------------- F2 H1 forest
def forest(rows_a, rows_b, labels, fname, title, mde=0.027, height=5.4, xlim=(-0.10, 0.06)):
    n = len(labels); y = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(7.0, height))
    ax.axvspan(-mde, mde, color="#f3f2ef", zorder=0, lw=0)
    ax.axvline(0, color=INK2, lw=0.8)
    for rows, off, col, mk, lab in ((rows_a, 0.17, BLUE, "o", "single step plans"), (rows_b, -0.17, ORANGE, "D", "with fixed two hop plans")):
        for yi, r in zip(y, rows):
            if r is None:
                continue
            sig = r["p_holm"] < 0.05
            ax.plot([r["ci_low"], r["ci_high"]], [yi + off] * 2, color=col, lw=1.6, solid_capstyle="round")
            ax.plot(r["gain"], yi + off, mk, ms=5.2 if mk == "o" else 4.6, color=col if sig else "white", mec=col, mew=1.3, zorder=3)
        ax.plot([], [], mk, color=col, mec=col, label=lab + " (filled = Holm p < 0.05)" if off > 0 else lab)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlim(*xlim); ax.set_ylim(-0.8, n - 0.2); ax.set_xlabel("gain over the held out fixed plan hull at equal mean latency (share of questions)")
    hgrid(ax); ax.set_title(title, loc="left", pad=20)
    ax.legend(loc="lower left", fontsize=7, bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _save(fig, fname); plt.close(fig)


def f2():
    A, B = T12.primary_family(AN / "final"), T12.primary_family(AN / "final_r3")
    order = [(m, r) for m in ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade"] for r in READERS]
    order += [("reader_cascade", "qwen2.5-7b->qwen3-14b"), ("reader_cascade", "qwen2.5-7b->llama3.1-8b")]
    def get(F, m, r):
        x = F[(F.method == m) & (F.reader == r)].iloc[0]
        return dict(gain=x.gain, ci_low=x.ci_low, ci_high=x.ci_high, p_holm=x.p_holm)
    a = [get(A, m, r) for m, r in order]; b = [get(B, m, r) for m, r in order]
    labels = [f"{PN[m]}, {RN.get(r, r.replace('qwen2.5-7b->', '7B to ').replace('qwen3-14b', 'Qwen3 14B').replace('llama3.1-8b', 'Llama 8B'))}" for m, r in order]
    forest(a, b, labels, "f2_h1_forest.png", "H1 primary family (27 pooled tests, 2,000 queries each)", height=6.2)


# ---------------------------------------------------------------- F3 headroom
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
    fig, ax = plt.subplots(figsize=(7.0, 2.7)); x = np.arange(len(READERS)); w = 0.26
    for i, (c, col, lab) in enumerate((("closed_book", MUTED, "closed book"), ("best", BLUE, "best single fixed plan"), ("oracle", ORANGE, "per query oracle over 25 plans"))):
        bars = ax.bar(x + (i - 1) * (w + 0.02), g[c], w, color=col, label=lab, zorder=2)
        for bx, v in zip(bars, g[c]):
            ax.text(bx.get_x() + bx.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.6, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([RN[r] for r in READERS]); ax.set_ylim(0, 0.9)
    ax.set_ylabel("success (label_final)"); hgrid(ax, "y")
    ax.set_title("Headroom by reader (mean over 4 datasets, in sample, same 25 plans)", loc="left")
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    _save(fig, "f3_headroom.png"); plt.close(fig)


# ---------------------------------------------------------------- F4 frontier example
def f4():
    F = pd.read_csv(ANS + "final/fixed_plans.csv"); M = pd.read_csv(ANS + "final/methods.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, (rd, ds) in zip(axes, [("qwen3-32b", "hotpotqa"), ("qwen2.5-7b", "musique")]):
        f = F[(F.reader == rd) & (F.dataset == ds)].sort_values("cost_s")
        ax.scatter(f.cost_s, f.success, s=13, color="#b9b8b3", zorder=2, label="fixed plans (in sample)")
        pts = f[["cost_s", "success"]].values; pts = pts[np.argsort(pts[:, 0])]
        par, best = [], -1
        for p in pts:
            if p[1] > best:
                par.append(p); best = p[1]
        hl = []
        for p in par:
            while len(hl) >= 2 and (hl[-1][1] - hl[-2][1]) * (p[0] - hl[-2][0]) <= (p[1] - hl[-2][1]) * (hl[-1][0] - hl[-2][0]):
                hl.pop()
            hl.append(p)
        hl = np.array(hl); ax.plot(hl[:, 0], hl[:, 1], color=INK2, lw=1.2, zorder=3, label="fixed plan hull (in sample)")
        m = M[(M.reader == rd) & (M.dataset == ds)]
        cols = {"planner_tsr": BLUE, "adaptive_rag": ORANGE, "adaptive_k": AQUA, "gate_closed_book": YELLOW, "conf_cascade": MAGENTA}
        for meth, col in cols.items():
            r = m[m.method == meth]
            if len(r):
                ax.scatter(r.cost_s, r.success, s=34, color=col, edgecolor="white", lw=0.8, zorder=4, label=PN[meth])
        ax.set_xscale("log"); ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([0.2, 0.5, 1, 2, 3]); ax.set_xlabel("mean latency per query (s, log scale)"); ax.set_ylabel("success")
        ax.set_title(f"{RN[rd]}, {DN[ds]}", loc="left"); hgrid(ax, "both")
    axes[1].legend(fontsize=6.3, loc="lower right")
    fig.tight_layout(); _save(fig, "f4_frontier.png"); plt.close(fig)


# ---------------------------------------------------------------- F5 R3 heatmap
def f5():
    F = pd.read_csv(ANS + "final_r3/fixed_plans.csv"); rows = []
    for (rd, ds), g in F.groupby(["reader", "dataset"]):
        s1 = g[~g.plan.str.startswith("bm25_2hop")].success.max(); s2 = g[g.plan.str.startswith("bm25_2hop")].success.max()
        rows.append(dict(reader=rd, dataset=ds, diff=s2 - s1))
    P = pd.DataFrame(rows).pivot(index="reader", columns="dataset", values="diff").loc[READERS, list(DN)]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    im = ax.imshow(P.values, cmap=cmap, norm=TwoSlopeNorm(0, -0.2, 0.2), aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.values[i, j]; ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7.6, color=INK if abs(v) < 0.13 else "white")
    ax.set_xticks(range(4)); ax.set_xticklabels([DN[d] for d in P.columns]); ax.set_yticks(range(5)); ax.set_yticklabels([RN[r] for r in P.index])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Best fixed two hop plan minus best single step plan (in sample)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
    _save(fig, "f5_r3.png"); plt.close(fig)


# ---------------------------------------------------------------- F6 H3
def f6():
    H = pd.read_csv(ANS + "h3_ircot/h3.csv"); a = H[H.hull == "single_step"].set_index(["reader", "dataset"]); b = H[H.hull == "with_r3"].set_index(["reader", "dataset"])
    cells = [(r, d) for r in ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"] for d in ["hotpotqa", "2wikimultihopqa", "musique"]]
    fig, ax = plt.subplots(figsize=(7.0, 3.6)); y = np.arange(len(cells))[::-1]
    for yi, c in zip(y, cells):
        s, t = a.loc[c], b.loc[c]
        lo = min(s.hull_at_ircot_cost, t.hull_at_ircot_cost, s.ircot_success); hi = max(s.hull_at_ircot_cost, t.hull_at_ircot_cost, s.ircot_success)
        ax.plot([lo, hi], [yi, yi], color=GRID, lw=2.2, zorder=1)
        ax.plot(s.hull_at_ircot_cost, yi, "o", color=MUTED, ms=6, zorder=3)
        ax.plot(t.hull_at_ircot_cost, yi, "D", color=ORANGE, ms=5, zorder=3)
        sig = s.p_holm < 0.05 and s.ci_low > 0
        ax.plot(s.ircot_success, yi, "o", color=BLUE if sig else "white", mec=BLUE, mew=1.4, ms=6.5, zorder=4)
        ax.text(1.02, yi, f"{s.ircot_cost_s / s.best_fixed_cost_s:.0f}x latency   {s.gain:+.3f}{'*' if sig else ''}", va="center", fontsize=7, color=INK2, transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData), clip_on=False)
    ax.set_yticks(y); ax.set_yticklabels([f"{RN[r]}, {DN[d]}" for r, d in cells], fontsize=7.3)
    ax.set_xlim(0.18, 0.66); ax.set_xlabel("success (label_final), held out"); hgrid(ax)
    ax.plot([], [], "o", color=MUTED, label="single step hull at IRCoT latency"); ax.plot([], [], "D", color=ORANGE, label="hull with fixed two hop plans")
    ax.plot([], [], "o", color=BLUE, label="IRCoT (filled = Holm significant)")
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3)
    ax.text(1.02, 1.01, "IRCoT latency / best plan; gain", fontsize=6.6, color=MUTED, transform=ax.transAxes, va="bottom")
    ax.set_title("H3: IRCoT against the hull at IRCoT's own latency", loc="left", pad=10)
    _save(fig, "f6_h3.png"); plt.close(fig)


# ---------------------------------------------------------------- F7 R6
def f7():
    W = pd.read_csv(ANS + "weak_protocol/weak_vs_ours.csv")
    W["weak"] = (W.margin_vs_always > 0) & (W.margin_vs_never > 0); W["ours"] = (W.ours_ci_low > 0) & (W.ours_p_holm < 0.05)
    g = W.groupby("reader")[["weak", "ours"]].mean().loc[READERS]
    fig, ax = plt.subplots(figsize=(7.0, 2.4)); x = np.arange(5); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, g.weak, w, color=ORANGE, label="weak protocol: beats Always RAG and Never RAG", zorder=2)
    b2 = ax.bar(x + w / 2 + 0.01, g.ours + 0.004, w, color=BLUE, label="our protocol: beats the hull (CI > 0, Holm p < 0.05)", zorder=2)
    for bs, vals in ((b1, g.weak), (b2, g.ours)):
        for bx, v in zip(bs, vals):
            ax.text(bx.get_x() + bx.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([RN[r] for r in READERS]); ax.set_ylim(0, 1.05); ax.set_ylabel("share of policy x dataset cells")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); hgrid(ax, "y"); ax.legend(fontsize=7, loc="upper right", ncol=1)
    ax.set_title("R6: the same answers, scored two ways (100 cells)", loc="left")
    _save(fig, "f7_r6.png"); plt.close(fig)


# ---------------------------------------------------------------- F8 secondary
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
    labels = [f"{names[m]}, {RN[r]}" for m, r in order]
    ra, rb = rows(A), rows(B)
    forest(ra, rb, labels, "f8_secondary.png", "Secondary family (30 tests per plan set, Holm within each)", height=6.6, xlim=(-0.10, 0.07))


# ---------------------------------------------------------------- F9 grading
def f9():
    A = pd.read_csv(settings.RESULTS_ROOT / "tables_final" / "audit_rules.csv")
    rules = ["strict EM", "F1 >= 0.7", "lenient", "label_final"]
    fig, ax = plt.subplots(figsize=(7.0, 2.4)); y = np.arange(len(rules))[::-1]; h = 0.36
    for k, (aud, col, lab) in enumerate(((A.Audit.unique()[0], BLUE, "audit 1 (7B, Llama, Qwen3 14B)"), (A.Audit.unique()[1], ORANGE, "audit 2 (Gemma, Qwen3 32B)"))):
        g = A[A.Audit == aud].set_index("Rule").loc[rules]
        yy = y + (0.2 if k == 0 else -0.2)
        ax.barh(yy, g.acc, height=h, color=col, label=lab, zorder=2)
        ax.errorbar(g.acc, yy, xerr=[g.acc - g.lo, g.hi - g.acc], fmt="none", ecolor=INK2, lw=0.8, capsize=2, zorder=3)
        for yi, v, hi in zip(yy, g.acc, g.hi):
            ax.text(hi + 0.006, yi, f"{v:.3f}", va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(rules); ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("accuracy against the majority of three independent raters (200 items per audit)")
    hgrid(ax); ax.legend(fontsize=7, loc="upper right"); ax.set_title("Grading rules against human labels", loc="left")
    _save(fig, "f9_grading.png"); plt.close(fig)


# ---------------------------------------------------------------- F10 headroom at matched cost and what policies reach
def f10():
    C = AN / "checks"
    H = pd.read_csv(C / "headroom_matched.csv").groupby("reader")[["gap_k2", "gap_k3", "gap_all"]].mean().loc[READERS]
    P = T12.primary_family(AN / "final")
    P = P[P.reader.isin(READERS)].groupby("reader").gain.max().reindex(READERS)
    M = pd.read_csv(C / "mechanism.csv").groupby("policy")[["per_plan_auroc", "pair_auroc"]].mean()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [2.3, 1]})
    x = np.arange(len(READERS)); w = 0.2
    series = (("gap_k2", "#9cc3ee", "oracle over 2 plans"), ("gap_k3", BLUE, "oracle over 3 plans"),
              ("gap_all", "#1d4f8f", "oracle over all plans"))
    for i, (c, col, lab) in enumerate(series):
        ax.bar(x + (i - 1.5) * (w + 0.015), H[c], w, color=col, label=lab, zorder=2)
    ax.bar(x + 1.5 * (w + 0.015), P.values, w, color=ORANGE, label=f"best adaptive policy (at most {P.max():+.3f})", zorder=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([RN[r] for r in READERS], fontsize=7.4)
    ax.set_ylabel("gain over the hull, same cost"); hgrid(ax, "y"); ax.set_ylim(-0.02, 0.52)
    ax.set_title("Held out headroom exists; policies do not reach it", loc="left")
    ax.legend(fontsize=6.6, ncol=2, loc="upper left")
    names = {"planner_tsr": "TSR planner", "planner_sk": "Planner + self\nknowledge"}
    y = np.arange(len(M)); h = 0.34
    bx.barh(y + h / 2, M.per_plan_auroc, h, color=BLUE, label="is the query answerable")
    bx.barh(y - h / 2, M.pair_auroc, h, color=ORANGE, label="which hull plan wins")
    for yi, a, b in zip(y, M.per_plan_auroc, M.pair_auroc):
        bx.text(a + 0.01, yi + h / 2, f"{a:.2f}", va="center", fontsize=6.6, color=INK2)
        bx.text(b + 0.01, yi - h / 2, f"{b:.2f}", va="center", fontsize=6.6, color=INK2)
    bx.axvline(0.5, color=RED, lw=0.8, ls="--"); bx.text(0.505, -0.75, "chance", color=RED, fontsize=6.4)
    bx.set_yticks(y); bx.set_yticklabels([names.get(i, i) for i in M.index], fontsize=7); bx.set_xlim(0.4, 0.75)
    bx.set_xlabel("AUROC (held out)"); hgrid(bx); bx.set_title("What planners learn", loc="left")
    bx.set_ylim(-0.9, len(M) - 0.4)
    bx.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1)
    fig.tight_layout(); _save(fig, "f10_headroom_matched.png"); plt.close(fig)


# ---------------------------------------------------------------- F11 learning curve
def f11():
    LC = pd.read_csv(AN / "stronger_policies" / "learning_curve.csv")
    g = LC.groupby(["reader", "frac"])[["sum_gain", "n"]].sum()      # pooled gain per question = sum / count
    pts = (g["sum_gain"] / g["n"]).unstack()
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    for rd, col in zip(READERS, cols):
        if rd in pts.index:
            ax.plot(pts.columns * 100, pts.loc[rd], marker="o", ms=3.2, lw=1.4, color=col, label=RN[rd])
    ax.axhline(0, color=INK2, lw=0.9); ax.text(21, 0.002, "hull", fontsize=6.8, color=INK2)
    ax.set_xlabel("share of the training folds used (%)"); ax.set_ylabel("gain over the hull")
    ax.set_xticks([20, 40, 60, 80, 100]); hgrid(ax, "y")
    ax.set_title("Planner with self knowledge: more data approaches the hull", loc="left")
    ax.legend(fontsize=6.6, ncol=2, loc="lower right")
    fig.tight_layout(); _save(fig, "f11_learning_curve.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11):
        f()
    print(f"Written {OUTD}")
