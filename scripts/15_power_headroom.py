"""
Power and headroom: how small a gain could we detect, and how much of the oracle headroom is realistic?

Part A  Minimum detectable effect (MDE) at 80% power, two sided, for every test in 10 and 13:
        MDE = (z_{1-a/2} + z_{0.8}) * SE, SE = sd(per query gain) / sqrt(n).
        Reported at a = 0.05 (nominal) and at the strictest Holm step a = 0.05 / m (m = family size).

Part B  Headroom, all HELD OUT with 10's folds and seeds (choices on the training folds, scored on the test fold):
  best_fixed   the plan with the highest training success
  gate_oracle  per query: closed book if it is right, else the best fixed plan (a perfect retrieve or not gate)
  oracle_K     the K plans whose per query union of successes is largest on the training folds (greedy);
               on the test fold each query takes the cheapest of the K that answers. K = 2, 3, 5, all.
               If most of the all plan oracle is reached with K = 2 or 3, the headroom is structured
               (a few complementary plans); if it needs all plans, much of it is luck across contexts.
  transfer     route reader B's test queries to the cheapest plan that answered for reader A (A's oracle
               choice, a query property if routing were learnable from the question); a ceiling for any
               router that predicts query properties shared across readers.

Part C  Stability: for each retriever, how often the label flips between neighbouring depths (1-3, 3-5, 5-10,
        10-20), and the share of queries whose success is non monotone in depth (right, wrong, right).
        Greedy decoding is deterministic, so flips measure sensitivity to small context changes.

Output: results_full/tables/power_headroom.txt (+ .csv files), tab_oracles.tex
"""

import os
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config import settings

LABEL, SEEDS, K_FOLDS = "final", [0, 1, 2, 3, 4], 5
ANA = settings.RESULTS_ROOT / "analysis" / LABEL
OUT = settings.RESULTS_ROOT / "tables"
PROC = settings.PROCESSED_DATA_ROOT
READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
DATASETS = ["nq", "hotpotqa", "2wikimultihopqa", "musique"]
D_TEX = {"nq": "NQ", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki", "musique": "MuSiQue"}
R_TEX = {"qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B", "qwen3-14b": "Qwen3 14B"}
KS = [2, 3, 5]
L = []


def say(s=""):
    L.append(s)
    print(s, flush=True)


# ======================================================================
def part_a():
    say("== A. Minimum detectable effect, 80% power, two sided")
    pq = [pd.read_csv(ANA / "per_query_gains.csv", dtype={"query_id": str})]
    if (ANA / "routers_per_query.csv").exists():
        pq.append(pd.read_csv(ANA / "routers_per_query.csv", dtype={"query_id": str}))
    PQ = pd.concat(pq, ignore_index=True)
    z80 = stats.norm.ppf(0.8)
    rows = []
    for level, keys in (("dataset", ["reader", "dataset", "method"]), ("pooled", ["reader", "method"])):
        for k, g in PQ.groupby(keys):
            d = g["gain"].to_numpy()
            rows.append({"level": level, **dict(zip(keys, k)), "n": len(d), "sd": d.std(ddof=1),
                         "se": d.std(ddof=1) / np.sqrt(len(d)), "gain": d.mean()})
    R = pd.DataFrame(rows)
    for level, g in R.groupby("level"):
        m = len(g)
        for a, lab in ((0.05, "a=0.05"), (0.05 / m, f"Holm strictest a=0.05/{m}")):
            mde = (stats.norm.ppf(1 - a / 2) + z80) * g["se"]
            R.loc[g.index, f"mde_{'nominal' if a == 0.05 else 'holm'}"] = mde
            say(f"  {level:8s} tests={m:3d}  {lab:28s} MDE median {mde.median():.3f}  range {mde.min():.3f}..{mde.max():.3f}")
    say("  Reading: a true gain smaller than the MDE would usually be missed. Pooled tests (n = 2000) detect")
    say("  about 0.025 at the nominal level; single datasets (n = 500) about 0.05.")
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "power_mde.csv", index=False)
    return R


# ======================================================================
def load(reader):
    df = pd.read_csv(PROC / f"tsr_labels_{reader}_graded.csv", dtype={"query_id": str}, keep_default_na=False,
                     usecols=["dataset", "query_id", "retriever", "depth", "repeat", f"label_{LABEL}"])
    df = df[df["repeat"].astype(int) == 0].copy()
    df["plan"] = df["retriever"] + "@" + df["depth"].astype(int).astype(str)
    df["y"] = df[f"label_{LABEL}"].astype(int)
    return df


def folds(q, sd):
    q = np.array(q).copy()
    np.random.default_rng(sd).shuffle(q)
    return [q[i::K_FOLDS] for i in range(K_FOLDS)]


def greedy_union(Ytr, k):
    """K plans maximising the share of training queries answered by at least one of them."""
    chosen, cover = [], np.zeros(len(Ytr), bool)
    for _ in range(k):
        gain = [(np.logical_or(cover, Ytr[:, j]).mean(), j) for j in range(Ytr.shape[1]) if j not in chosen]
        _, j = max(gain)
        chosen.append(j)
        cover |= Ytr[:, j].astype(bool)
    return chosen


def oracle_on(Yte, cost, idx):
    """success and mean cost when each query takes the cheapest correct plan among idx (else the cheapest)."""
    sub, c = Yte[:, idx], cost[idx]
    ok = sub.max(1) == 1
    cc = np.where(sub == 1, c[None, :], np.inf).min(1)
    return ok.mean(), np.where(ok, cc, c.min()).mean()


def part_b():
    say("\n== B. Headroom, held out (choices on training folds, scored on the held out fold; 5 folds x 5 seeds)")
    FP = pd.read_csv(ANA / "fixed_plans.csv")
    Ys, rows = {}, []
    for rd in READERS:
        df = load(rd)
        for ds in DATASETS:
            Y = df[df.dataset == ds].pivot_table(index="query_id", columns="plan", values="y")
            Y = Y[Y.columns[Y.notna().all()]]
            Ys[(rd, ds)] = Y
            cost = FP[(FP.reader == rd) & (FP.dataset == ds)].set_index("plan")["cost_s"].reindex(Y.columns).to_numpy()
            Yn = Y.to_numpy()
            cb = list(Y.columns).index("closed_book@0") if "closed_book@0" in Y.columns else None
            acc = {k: [] for k in ["best", "gate", "all"] + [f"k{K}" for K in KS]}
            cst = {k: [] for k in acc}
            for sd in SEEDS:
                for i, te in enumerate(f := folds(Y.index, sd)):
                    tr = np.concatenate([f[j] for j in range(K_FOLDS) if j != i])
                    ri, ti = Y.index.get_indexer(tr), Y.index.get_indexer(te)
                    Ytr, Yte = Yn[ri], Yn[ti]
                    b = int(Ytr.mean(0).argmax())
                    acc["best"].append(Yte[:, b].mean()); cst["best"].append(cost[b])
                    if cb is not None:
                        g_ok = np.maximum(Yte[:, cb], Yte[:, b])
                        acc["gate"].append(g_ok.mean())
                        cst["gate"].append(np.where(Yte[:, cb] == 1, cost[cb], cost[cb] + cost[b]).mean())
                    s, c = oracle_on(Yte, cost, list(range(Yn.shape[1])))
                    acc["all"].append(s); cst["all"].append(c)
                    for K in KS:
                        s, c = oracle_on(Yte, cost, greedy_union(Ytr, K))
                        acc[f"k{K}"].append(s); cst[f"k{K}"].append(c)
            row = {"reader": rd, "dataset": ds, "n_plans": Yn.shape[1]}
            for k in acc:
                row[k] = np.mean(acc[k]) if acc[k] else np.nan
                row[f"{k}_cost"] = np.mean(cst[k]) if cst[k] else np.nan
            rows.append(row)
    H = pd.DataFrame(rows)
    H["headroom_all"] = H["all"] - H["best"]
    for K in KS:
        H[f"share_k{K}"] = (H[f"k{K}"] - H["best"]) / H["headroom_all"]
    say(H[["reader", "dataset", "best", "gate", "k2", "k3", "k5", "all"]].round(3).to_string(index=False))
    say(f"  mean over 12 blocks: best {H.best.mean():.3f} | gate oracle {H.gate.mean():+.3f} vs best "
        f"{(H.gate - H.best).mean():+.3f} | oracle over 2 plans {(H.k2 - H.best).mean():+.3f}, 3 plans "
        f"{(H.k3 - H.best).mean():+.3f}, 5 plans {(H.k5 - H.best).mean():+.3f}, all plans {(H['all'] - H.best).mean():+.3f}")
    say(f"  share of the all plan headroom reached with 2 / 3 / 5 plans: {H.share_k2.mean():.2f} / "
        f"{H.share_k3.mean():.2f} / {H.share_k5.mean():.2f}")
    H.to_csv(OUT / "oracles_heldout.csv", index=False)

    # ---- transfer: reader A's oracle choice used for reader B
    say("\n  Transfer ceiling: route B's held out queries to the cheapest plan that answered for reader A")
    trows = []
    for ds in DATASETS:
        for A in READERS:
            for Bn in READERS:
                if A == Bn:
                    continue
                YA, YB = Ys[(A, ds)], Ys[(Bn, ds)]
                common = [p for p in YB.columns if p in YA.columns]
                q = YA.index.intersection(YB.index)
                ya, yb = YA.loc[q, common].to_numpy(), YB.loc[q, common].to_numpy()
                cA = FP[(FP.reader == A) & (FP.dataset == ds)].set_index("plan")["cost_s"].reindex(common).to_numpy()
                bestB = YB.loc[q, common].mean().max()
                pick = np.where(ya == 1, cA[None, :], np.inf).argmin(1)          # A's cheapest correct plan
                solvedA = ya.max(1) == 1
                fallback = int(YB.loc[q, common].mean().to_numpy().argmax())       # unsolved by A: B's best plan
                pick = np.where(solvedA, pick, fallback)
                s = yb[np.arange(len(q)), pick].mean()
                trows.append({"dataset": ds, "from": A, "to": Bn, "transfer": s, "best_fixed_B": bestB,
                              "oracle_B": yb.max(1).mean()})
    T = pd.DataFrame(trows)
    T["gain_vs_best"] = T.transfer - T.best_fixed_B
    say(T.round(3).to_string(index=False))
    say(f"  mean transfer gain over B's best fixed plan: {T.gain_vs_best.mean():+.3f} "
        f"(B's own oracle: {(T.oracle_B - T.best_fixed_B).mean():+.3f}). Note: in sample for the plan choice of B.")
    T.to_csv(OUT / "oracle_transfer.csv", index=False)
    return H, T, Ys


# ======================================================================
def part_c(Ys):
    say("\n== C. Stability of labels across neighbouring depths (same retriever, same query)")
    rows = []
    for (rd, ds), Y in Ys.items():
        rets = sorted({p.split("@")[0] for p in Y.columns} - {"closed_book"})
        for r in rets:
            ds_ = sorted(int(p.split("@")[1]) for p in Y.columns if p.startswith(r + "@"))
            M = Y[[f"{r}@{d}" for d in ds_]].to_numpy()
            flips = [(M[:, i] != M[:, i + 1]).mean() for i in range(len(ds_) - 1)]
            nonmono = np.zeros(len(M), bool)
            for i in range(len(ds_)):
                for j in range(i + 1, len(ds_)):
                    for k in range(j + 1, len(ds_)):
                        nonmono |= (M[:, i] == 1) & (M[:, j] == 0) & (M[:, k] == 1)
            ever = M.max(1) == 1
            rows.append({"reader": rd, "dataset": ds, "retriever": r, "flip_mean": np.mean(flips),
                         "nonmono": nonmono.mean(), "nonmono_given_ever": nonmono[ever].mean() if ever.any() else 0})
    S = pd.DataFrame(rows)
    say(S.groupby("reader")[["flip_mean", "nonmono", "nonmono_given_ever"]].mean().round(3).to_string())
    say(S.groupby("dataset")[["flip_mean", "nonmono", "nonmono_given_ever"]].mean().round(3).to_string())
    say(f"  overall: label flips between neighbouring depths {S.flip_mean.mean():.3f}; queries right, then wrong, "
        f"then right again as depth grows: {S.nonmono.mean():.3f} of all, {S.nonmono_given_ever.mean():.3f} of those "
        f"ever answered")
    S.to_csv(OUT / "label_stability.csv", index=False)
    return S


def tex(H):
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Held out headroom (success). Choices are made on the training folds and scored on the held "
             "out fold (5 folds, 5 seeds). Gate: perfect choice between closed book and the best plan. Oracle $K$: "
             "per query, the cheapest correct plan among the $K$ plans that jointly answer most training queries.}",
             "\\label{tab:oracles}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{llrrrrrr}", "\\toprule",
             "Reader & Data & Best & Gate & Or.~2 & Or.~3 & Or.~5 & Or.~all \\\\", "\\midrule"]
    for i, r in H.iterrows():
        lines.append(f"{R_TEX[r.reader] if r.dataset == DATASETS[0] else ''} & {D_TEX[r.dataset]} & {r.best:.3f} & "
                     f"{r.gate:.3f} & {r.k2:.3f} & {r.k3:.3f} & {r.k5:.3f} & {r['all']:.3f} \\\\")
        if r.dataset == DATASETS[-1] and r.reader != READERS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "tab_oracles.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    part_a()
    H, T, Ys = part_b()
    part_c(Ys)
    tex(H)
    (OUT / "power_headroom.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT / 'power_headroom.txt'}")


if __name__ == "__main__":
    main()
