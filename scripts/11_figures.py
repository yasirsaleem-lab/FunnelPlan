"""
Paper figures from the outputs of 08 (grading) and 10 (analysis).

  fig1_forest.pdf     gain vs the held out hull, pooled per reader, with dataset points and Holm survivors
  fig2_frontier.pdf   cost vs success per reader x dataset: fixed plans, the held out hull (the baseline of
                      Fig. 1: plans chosen on training folds, success measured on the held out fold,
                      averaged over folds and seeds), the in sample hull for contrast, held out method points
  fig3_overfit.pdf    training fold gain vs held out gain for every method x reader x dataset
  fig4_grading.pdf    success rate under each grading rule (strict, f1_07, final, lenient)

Reads  results_full/analysis/<label>/{methods,fixed_plans}.csv
       data/processed_full/tsr_labels_<reader>_graded.csv
Writes results_full/figures/<name>.pdf and .png (vector PDF for the paper, PNG for a quick look)

CPU only, about 20 s.
"""

import os
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from config import settings

CONFIG = {"label": "final", "seeds": [0, 1, 2, 3, 4], "k_folds": 5}   # seeds and folds as in 10

ANA = settings.RESULTS_ROOT / "analysis" / CONFIG["label"]
OUT = settings.RESULTS_ROOT / "figures"
PROC = settings.PROCESSED_DATA_ROOT

# ---- style: colour blind safe categorical colours, dark text, light grid
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
METHODS = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade"]
M_LABEL = {"planner_tsr": "TSR planner (ours)", "adaptive_rag": "Adaptive RAG", "adaptive_k": "Adaptive k",
           "gate_closed_book": "Closed book gate", "conf_cascade": "Confidence cascade",
           "reader_cascade": "Reader cascade"}
M_COLOR = dict(zip(METHODS, SLOTS))
M_MARK = dict(zip(METHODS, ["o", "s", "^", "D", "v"]))      # markers as well as colours, so no series relies on colour alone
# stronger routers from 13: the Adaptive RAG variants share its colour (same ladder), per plan routers get new slots
ROUTERS = ["ak_paper", "ar_emb", "ar_t5", "plan_emb", "plan_knn"]
M_LABEL.update({"adaptive_k": "Adaptive k, gap only", "ak_paper": "Adaptive k, published",
                "ar_emb": "Adaptive RAG, BGE-M3", "ar_t5": "Adaptive RAG, FLAN-T5",
                "plan_emb": "Per plan, BGE-M3 LR", "plan_knn": "Per plan, BGE-M3 kNN"})
M_COLOR.update({"ak_paper": SLOTS[2], "ar_emb": SLOTS[1], "ar_t5": SLOTS[1], "plan_emb": "#4a3aa7", "plan_knn": "#008300"})
M_MARK.update({"ak_paper": "<", "ar_emb": "P", "ar_t5": "X", "plan_emb": "h", "plan_knn": "d"})
READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
R_LABEL = {"qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B", "qwen3-14b": "Qwen3 14B (int8)"}
DATASETS = ["nq", "hotpotqa", "2wikimultihopqa", "musique"]
D_LABEL = {"nq": "NQ", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki", "musique": "MuSiQue"}
D_MARK = {"nq": "o", "hotpotqa": "s", "2wikimultihopqa": "^", "musique": "D"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "legend.frameon": False,
})
COL1, COL2 = 3.5, 7.16      # IEEE single and double column widths (inches)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def upper_hull(cost, succ):
    """Upper concave hull of (cost, success), cheapest point to the most accurate point."""
    pts = sorted(zip(cost, succ), key=lambda t: (t[0], -t[1]))
    best = max(range(len(pts)), key=lambda i: (pts[i][1], -pts[i][0]))
    H = []
    for p in pts[: best + 1]:
        while len(H) >= 2 and (H[-1][1] - H[-2][1]) * (p[0] - H[-2][0]) <= (p[1] - H[-2][1]) * (H[-1][0] - H[-2][0]):
            H.pop()
        H.append(p)
    return np.array(H)


# ======================================================================
def fig_forest(M):
    """One panel per reader. Rows: methods, then the stronger routers from 13. Bar: pooled gain and 95% CI.
    Small marks: the four datasets; filled black star: a dataset level test that survives Holm over the
    whole family (every test in 10 and 13)."""
    pooled = M[M.dataset == "pooled"]
    lvl = M[M.dataset != "pooled"]
    rows = METHODS + [m for m in ROUTERS if m in set(pooled.method)]
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[len(METHODS):] -= 0.7                                   # gap before the router block
    lo = min(pooled.ci_low.min(), lvl[lvl.method.isin(rows)].gain.min()) - 0.01
    hi = max(pooled.ci_high.max(), lvl[lvl.method.isin(rows)].gain.max()) + 0.01
    fig, axes = plt.subplots(1, len(READERS), figsize=(COL2, 0.22 * len(rows) + 0.9), sharey=True)
    for ax, r in zip(axes, READERS):
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for y, m in zip(ys, rows):
            p = pooled[(pooled.reader == r) & (pooled.method == m)]
            if p.empty:
                continue
            p = p.iloc[0]
            for _, q in lvl[(lvl.reader == r) & (lvl.method == m)].iterrows():
                sig = q.p_holm < 0.05
                ax.plot(q.gain, y, marker="*" if sig else D_MARK[q.dataset], ms=6 if sig else 3.2,
                        mfc=INK if sig else "none", mec=INK if sig else MUTED, mew=0.6, ls="none", zorder=2)
            ax.plot([p.ci_low, p.ci_high], [y, y], color=M_COLOR[m], lw=2, solid_capstyle="round", zorder=3)
            ax.plot(p.gain, y, marker=M_MARK[m], ms=5, mfc=M_COLOR[m], mec="white", mew=0.8, ls="none", zorder=4)
        ax.set_title(R_LABEL[r], color=INK)
        ax.set_xlim(lo, hi)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([M_LABEL[m] for m in rows])
    axes[1].set_xlabel("Held out success minus held out fixed plan hull at equal mean cost")
    h = [plt.Line2D([], [], color=INK2, lw=2, marker="o", ms=4, mfc=INK2, mec="white", label="pooled, 95% CI"),
         plt.Line2D([], [], color=MUTED, ls="none", marker="o", ms=3.2, mfc="none", label="one dataset"),
         plt.Line2D([], [], color=INK, ls="none", marker="*", ms=6, label="survives Holm")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.55, -0.08), ncol=3, handletextpad=0.3)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig1_forest")


def load_Y(reader, ds):
    """queries x plans success matrix, built exactly as Block in 10 (same query order, same plans)."""
    key = (reader,)
    if key not in _LAB:
        df = pd.read_csv(PROC / f"tsr_labels_{reader}_graded.csv", dtype={"query_id": str}, keep_default_na=False)
        col = f"label_{CONFIG['label']}" if f"label_{CONFIG['label']}" in df.columns else "label_f1_07"
        df = df[df["repeat"].astype(int) == 0].copy()
        df["y"] = df[col].astype(int)
        df["plan"] = df["retriever"] + "@" + df["depth"].astype(int).astype(str)
        _LAB[key] = df
    d = _LAB[key]
    Y = d[d.dataset == ds].pivot_table(index="query_id", columns="plan", values="y")
    return Y[Y.columns[Y.notna().all()]]


_LAB = {}


def heldout_hull(Y, cost, grid):
    """The baseline the gains are measured against, drawn as a curve: for every seed and fold, the hull's
    plans are chosen on the training folds and their success is measured on the held out fold; the
    resulting curves are averaged. Same folds as 10 (seeded shuffle of the sorted query ids)."""
    k, curves = CONFIG["k_folds"], []
    c = cost[Y.columns].to_numpy()
    for sd in CONFIG["seeds"]:
        q = np.array(Y.index).copy()
        np.random.default_rng(sd).shuffle(q)
        folds = [q[i::k] for i in range(k)]
        for i in range(k):
            sel = np.concatenate([folds[j] for j in range(k) if j != i])
            m_tr, m_te = Y.loc[sel].mean().to_numpy(), Y.loc[folds[i]].mean().to_numpy()
            H = upper_hull(c, m_tr)
            idx = [int(np.where((c == x) & (m_tr == y))[0][0]) for x, y in H]
            xs, ys = c[idx], m_te[idx]
            curves.append(np.interp(grid, xs, ys, left=ys[0], right=ys[-1]))
    return np.mean(curves, axis=0)


# ======================================================================
def fig_frontier(M, FP):
    lvl = M[(M.dataset != "pooled") & (M.method.isin(METHODS))]
    fig, axes = plt.subplots(len(READERS), len(DATASETS), figsize=(COL2, 5.0), sharex="col", sharey="col")
    for i, r in enumerate(READERS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            fp = FP[(FP.reader == r) & (FP.dataset == ds)]
            ax.scatter(fp.cost_s, fp.success, s=7, color=MUTED, lw=0, zorder=2, label="fixed plan")
            H = upper_hull(fp.cost_s.to_numpy(), fp.success.to_numpy())
            ax.plot(H[:, 0], H[:, 1], color=MUTED, lw=0.8, ls=(0, (2, 1.5)), zorder=3,
                    label="in sample hull (optimistic)")
            Y = load_Y(r, ds)
            cost = fp.set_index("plan")["cost_s"]
            grid = np.linspace(cost.min(), cost.max(), 200)
            ax.plot(grid, heldout_hull(Y, cost, grid), color=INK, lw=1.1, zorder=3, label="held out hull (baseline)")
            cb = fp[fp.plan == "closed_book@0"]
            if len(cb):
                ax.plot(cb.cost_s, cb.success, marker="x", ms=4, color=INK, mew=0.8, ls="none", zorder=3,
                        label="closed book")
            for m in METHODS:
                q = lvl[(lvl.reader == r) & (lvl.dataset == ds) & (lvl.method == m)]
                if len(q):
                    ax.plot(q.cost_s, q.success, marker=M_MARK[m], ms=4.5, mfc=M_COLOR[m], mec="white",
                            mew=0.6, ls="none", zorder=4, label=M_LABEL[m])
            ax.grid(color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(D_LABEL[ds], color=INK)
            if j == 0:
                ax.set_ylabel(f"{R_LABEL[r]}\nsuccess", color=INK)
            if i == len(READERS) - 1:
                ax.set_xlabel("mean latency per query (s)")
    hd, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), handletextpad=0.3)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, "fig2_frontier")


# ======================================================================
def fig_overfit(M):
    lvl = M[(M.dataset != "pooled") & (M.method.isin(METHODS + ROUTERS))]
    fig, ax = plt.subplots(figsize=(COL1, 3.1))
    lim = [min(lvl.gain_selection_half.min(), lvl.gain.min()) - 0.01,
           max(lvl.gain_selection_half.max(), lvl.gain.max()) + 0.01]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.axvline(0, color=GRID, lw=0.6, zorder=0)
    for m in METHODS + [x for x in ROUTERS if x in set(lvl.method)]:
        q = lvl[lvl.method == m]
        ax.plot(q.gain_selection_half, q.gain, marker=M_MARK[m], ms=4.5, mfc=M_COLOR[m], mec="white", mew=0.6,
                ls="none", label=M_LABEL[m], zorder=3)
    ax.text(0.075, 0.075, "no overfitting (y = x)", ha="left", va="top", fontsize=6.5, color=INK2, rotation=0)
    ax.set_xlim(lim)
    ax.set_ylim(lvl.gain.min() - 0.015, max(lvl.gain.max(), 0.0) + 0.03)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.set_xlabel("Gain on training folds (in sample)")
    ax.set_ylabel("Gain on held out fold")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.3, fontsize=6.5)
    save(fig, "fig3_overfit")


# ======================================================================
def fig_grading():
    rules = ["label_strict", "label_f1_07", "label_final", "label_lenient"]
    r_lab = {"label_strict": "strict (EM or contains)", "label_f1_07": "strict or F1 at least 0.7",
             "label_final": "final (judge on ambiguous)", "label_lenient": "strict or F1 at least 0.4"}
    rows = []
    for r in READERS:
        f = PROC / f"tsr_labels_{r}_graded.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f, usecols=["dataset", "repeat"] + rules)
        g = g[g["repeat"].astype(int) == 0]
        t = g.groupby("dataset")[rules].mean()
        t["reader"] = r
        rows.append(t.reset_index())
    T = pd.concat(rows)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(COL2, 1.9), sharey=False)
    w = 0.19
    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        sub = T[T.dataset == ds].set_index("reader").reindex(READERS)
        x = np.arange(len(READERS))
        for k, ru in enumerate(rules):
            ax.bar(x + (k - 1.5) * w, sub[ru], width=w - 0.02, color=SLOTS[k], label=r_lab[ru], zorder=2,
                   edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["7B", "Llama 8B", "14B"])
        ax.set_title(D_LABEL[ds], color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        if j == 0:
            ax.set_ylabel("success rate (all plans)")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), handletextpad=0.3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig4_grading")


def with_routers(M):
    """Adds 13's routers: dataset rows from routers.csv, pooled rows from routers_per_query.csv (bootstrap as
    in 10), and one Holm family over every dataset level test of 10 and 13."""
    if not (ANA / "routers.csv").exists():
        return M
    R = pd.read_csv(ANA / "routers.csv")
    lvl = pd.concat([M[M.dataset != "pooled"], R], ignore_index=True)
    p = lvl["p"].to_numpy()
    order, run, adj = np.argsort(p), 0.0, np.empty(len(p))
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    lvl["p_holm"] = adj
    PQ = pd.read_csv(ANA / "routers_per_query.csv", dtype={"query_id": str})
    rows = []
    for (rd, m), g in PQ.groupby(["reader", "method"]):
        d = g["gain"].to_numpy()
        rng = np.random.default_rng(0)
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        rows.append({"reader": rd, "dataset": "pooled", "method": m, "gain": d.mean(),
                     "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5)})
    return pd.concat([lvl, M[M.dataset == "pooled"], pd.DataFrame(rows)], ignore_index=True)


def main():
    M = with_routers(pd.read_csv(ANA / "methods.csv"))
    FP = pd.read_csv(ANA / "fixed_plans.csv")
    fig_forest(M)
    fig_frontier(M, FP)
    fig_overfit(M)
    fig_grading()


if __name__ == "__main__":
    main()
