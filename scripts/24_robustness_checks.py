"""
Robustness checks on the stored answers. CPU only; no new model calls.

  inference  Pooled tests with a bootstrap stratified by dataset, next to the plain bootstrap the paper
             uses, for the primary family and the secondary routers, in both plan sets. Holm adjusted minimum
             detectable effect per pooled test (from tables/power_mde.csv).
  refit      Refit bootstrap for the tests closest to a decision: per plan logistic regression on
             embeddings (plan_emb) for Qwen3 32B in both plan sets, the confidence cascade for Qwen2.5 7B and the
             reader cascade 7B to Qwen3 14B with the two hop plans. Each replicate resamples questions within each
             dataset, then reruns the WHOLE procedure: fold split (copies of a question stay in one fold), policy
             fitting, hull vertices and weights, and scoring. One seed per replicate (seed = replicate number), so the
             interval also carries the variation over fold splits.
  mechanism  What the planners learn, for the TSR planner, the planner with self knowledge AND plan_emb,
             over all 5 seeds: AUROC per plan, correlation of predictions across plans, AUROC for which of the two
             hull plans answers when exactly one does, with a bootstrap interval and the number of such questions.
  headroom   Oracle headroom at the SAME budget as each policy: the best mix of the hull plans and the
             oracle (plans chosen on the training folds) at the policy's own mean cost, minus the hull there. The
             fraction of that gap each policy closes.
  ablation   The weak protocol turned into ours one factor at a time on the same answers: comparator
             (Always and Never RAG, then the best single plan at the same cost, then the hull), seeds, grading, the
             decision rule, and pooling. Runs 11 once for the missing step (strict labels, seed 0, single plan
             comparator) if its output is missing.
  curve      Learning curve of plan_emb: 20% to 100% of the training folds.
  overhead   plan_emb with its own cost charged: --overhead_ms X (from 25_latency_e2e.py) is added to the
             cost of every plan whose retrieval does not embed the query with BGE-M3 anyway; the router tunes lambda
             with these costs, and the hull is read at the policy's charged cost. Also checks that, with no own cost,
             it reproduces script 12.
  pooled     (exploratory) plan_emb trained on the training folds of all four datasets together (about
             1,600 questions instead of 400, dataset indicators added), with and without its own cost. Not part of
             the default run: --parts pooled --overhead_ms X.
  boundary   How often a policy's mean cost falls below the cheapest plan or on the flat part of the hull.

Usage:  python scripts/24_robustness_checks.py                         every part except overhead (about 3 to 5 hours)
        python scripts/24_robustness_checks.py --parts inference,headroom,boundary
        python scripts/24_robustness_checks.py --parts overhead --overhead_ms 25
        python scripts/24_robustness_checks.py --quick                 small run to test the code
Output: results_full/analysis/robustness/*.csv and summary_<parts>.txt
"""

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

from config import settings

ARGS = sys.argv[1:]


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


QUICK = "--quick" in ARGS
ALL_PARTS = ["inference", "refit", "mechanism", "headroom", "ablation", "curve", "boundary"]
PARTS = arg("--parts", ",".join(ALL_PARTS)).split(",")
CONFIG = {
    "n_boot": 2000,
    "refit_reps": 10 if QUICK else 100,
    "fracs": [0.2, 0.4, 0.6, 0.8, 1.0],
    "margin": 0.03,
}
AN = settings.RESULTS_ROOT / "analysis"
OUT = AN / "robustness"
OUT.mkdir(parents=True, exist_ok=True)


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


R = _load("routers12", "12_routers.py")
A = R.A
S20 = _load("stronger20", "20_stronger_policies.py")
if QUICK:
    A.CONFIG["seeds"] = [0, 1]

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def readers():
    rs = A.readers()
    return rs[:1] if QUICK else rs


PRIMARY = ["planner_tsr", "adaptive_rag", "adaptive_k", "gate_closed_book", "conf_cascade"]


def plan_models(Xtr, Ytr):
    """The models of R.m_plan_emb's train_predict (PCA 64, one logistic regression per plan)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    pca = PCA(n_components=min(R.CONFIG["pca_dim"], len(Xtr) - 1), random_state=0).fit(Xtr)
    Ztr = pca.transform(Xtr)
    ms = []
    for j in range(Ytr.shape[1]):
        yj = Ytr[:, j]
        ms.append(float(yj[0]) if yj.min() == yj.max() else LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, yj))
    return pca, ms


def plan_predict(pm, X):
    pca, ms = pm
    Z = pca.transform(X)
    return np.column_stack([np.full(len(X), m) if isinstance(m, float) else m.predict_proba(Z)[:, 1] for m in ms])


def load_E(ds, B):
    E = R.load_emb(ds).reindex([str(q) for q in B.q])
    E.index = B.q
    return E


def boot_strat(parts, n, seed=0):
    """Percentile interval of the pooled mean, resampling within each dataset."""
    rng = np.random.default_rng(seed)
    tot = sum(len(p) for p in parts)
    b = np.empty(n)
    for i in range(n):
        b[i] = sum(p[rng.integers(0, len(p), len(p))].sum() for p in parts) / tot
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(np.percentile(b, 95))


# ====================================================================== inference
def part_inference():
    say("\n== inference: plain versus dataset stratified bootstrap of pooled gains (95% intervals; one sided 95% upper bound)")
    rows = []
    for tag, d in (("single step", "final"), ("with two hop", "final_r3")):
        for fam, f in (("primary", "per_query_gains.csv"), ("secondary routers", "routers_per_query.csv")):
            P = pd.read_csv(AN / d / f, dtype={"query_id": str})
            for (rd, m), g in P.groupby(["reader", "method"]):
                if QUICK and rd not in readers():
                    continue
                parts = [x.gain.to_numpy(float) for _, x in g.groupby("dataset")]
                allg = np.concatenate(parts)
                lo, hi = A.boot_ci(allg, CONFIG["n_boot"], A.CONFIG["seed"])
                slo, shi, sup = boot_strat(parts, CONFIG["n_boot"])
                rows.append({"plan_set": tag, "family": fam, "reader": rd, "method": m, "gain": allg.mean(),
                             "plain_low": lo, "plain_high": hi, "strat_low": slo, "strat_high": shi,
                             "strat_upper_one_sided": sup, "n": len(allg)})
    T = pd.DataFrame(rows)
    mde = pd.read_csv(settings.RESULTS_ROOT / "tables" / "power_mde.csv")
    mde = mde[mde.level == "pooled"][["reader", "method", "mde_nominal", "mde_holm"]]
    T = T.merge(mde, on=["reader", "method"], how="left")
    T.to_csv(OUT / "inference_stratified.csv", index=False)
    T["width_ratio"] = (T.strat_high - T.strat_low) / (T.plain_high - T.plain_low)
    for (tag, fam), g in T.groupby(["plan_set", "family"]):
        flips = g[(np.sign(g.plain_low) != np.sign(g.strat_low)) | ((g.plain_high < CONFIG["margin"]) != (g.strat_high < CONFIG["margin"]))]
        say(f"  {tag:12s} {fam:17s} tests {len(g):2d}  stratified/plain width: median {g.width_ratio.median():.3f} "
            f"range {g.width_ratio.min():.3f}..{g.width_ratio.max():.3f}  interval above 0: plain {int((g.plain_low > 0).sum())}, "
            f"stratified {int((g.strat_low > 0).sum())}  upper bound below {CONFIG['margin']}: plain "
            f"{int((g.plain_high < CONFIG['margin']).sum())}, stratified {int((g.strat_high < CONFIG['margin']).sum())}  "
            f"tests that change: {len(flips)}")
    p = T[T.family == "primary"]
    say(f"  Holm adjusted minimum detectable effect, primary pooled tests: median {p.mde_holm.median():.3f}, "
        f"range {p.mde_holm.min():.3f}..{p.mde_holm.max():.3f} (nominal median {p.mde_nominal.median():.3f})")
    top = T.sort_values("gain", ascending=False).head(6)
    say(top[["plan_set", "reader", "method", "gain", "plain_low", "plain_high", "strat_low", "strat_high", "mde_holm"]]
        .round(4).to_string(index=False))


# ====================================================================== refit bootstrap
def resample_block(B, d_ds, take, tag):
    """A copy of block B on resampled questions: copy j of question q becomes 'q#j'. Plan costs are recomputed as
    the means over the resampled questions, as the paper computes them over the original ones."""
    import copy
    nb = copy.copy(B)
    new = np.array([f"{q}#{j}" for j, q in enumerate(take)])
    nb.Y = B.Y.loc[take].set_axis(new)
    nb.conf = B.conf.loc[take].set_axis(new)
    rt = d_ds.pivot_table(index="query_id", columns="plan", values="retrieval_time_s").reindex(index=B.q, columns=B.plans)
    at = d_ds.pivot_table(index="query_id", columns="plan", values="agent_time_s").reindex(index=B.q, columns=B.plans)
    nb.ret_cost = rt.loc[take].fillna(0.0).mean()
    nb.agent_cost = at.loc[take].mean()
    nb.cost = nb.ret_cost + nb.agent_cost
    nb.q = new
    nb.Yn = nb.Y.to_numpy(dtype=float)
    nb.Cn = nb.conf.reindex(index=nb.Y.index, columns=nb.Y.columns).to_numpy(dtype=float)
    nb.qrow = {q: i for i, q in enumerate(nb.Y.index)}
    nb._rows = {}
    return nb


def heldout_grouped(B, method, seed):
    """A.heldout with folds over ORIGINAL questions, so the copies of a question never sit in train and test."""
    fit, apply_ = method
    k = A.CONFIG["k_folds"]
    orig = np.array([q.split("#")[0] for q in B.q])
    uniq = np.unique(orig)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {u: i % k for i, u in enumerate(uniq)}
    fid = np.array([fold_of[o] for o in orig])
    diffs = []
    for i in range(k):
        test, sel = B.q[fid == i], B.q[fid != i]
        cfg = fit(sel)
        s, c = apply_(cfg, test)
        h = A.hull_at(B.Y, B.cost, sel, c, eval_idx=test)
        diffs.append(s - h)
    return np.concatenate(diffs)


def m_plan_emb_res(B, E):
    """R.m_plan_emb on a resampled block (embeddings looked up by the original question id)."""
    E2 = E.loc[[q.split("#")[0] for q in B.q]].set_axis(B.q)
    return R.m_plan_emb(B, E2)


def part_refit():
    targets = [("plan_emb", "qwen3-32b", False), ("plan_emb", "qwen3-32b", True),
               ("conf_cascade", "qwen2.5-7b", True), ("reader_cascade", ("qwen2.5-7b", "qwen3-14b"), True)]
    if QUICK:
        targets = targets[:1]
    say(f"\n== refit bootstrap ({CONFIG['refit_reps']} replicates; questions resampled within each dataset; "
        f"folds, policy, hull and weights refitted every time)")
    rows = []
    for method, rd, r3 in targets:
        A.CONFIG["include_r3"] = r3
        t0 = time.time()
        rds = rd if isinstance(rd, tuple) else (rd,)
        data = {r: A.load_reader(r)[0] for r in rds}
        dss = sorted(set.intersection(*[set(d.dataset.unique()) for d in data.values()]))
        base = {(r, ds): A.Block(data[r], ds) for r in rds for ds in dss}
        embs = {ds: load_E(ds, base[(rds[0], ds)]) for ds in dss} if method == "plan_emb" else {}
        reps = []
        for b in range(CONFIG["refit_reps"]):
            rng = np.random.default_rng(10_000 + b)
            A._CUR_SEED[0] = b
            gains = []
            for ds in dss:
                q0 = np.intersect1d(*[base[(r, ds)].q for r in rds]) if len(rds) > 1 else base[(rds[0], ds)].q
                take = rng.choice(q0, len(q0), replace=True)
                blocks = {(r, ds): resample_block(base[(r, ds)], data[r][data[r].dataset == ds], take, r) for r in rds}
                Bm = blocks[(rds[0], ds)]
                if method == "plan_emb":
                    mk, Bt = m_plan_emb_res(Bm, embs[ds]), Bm
                elif method == "conf_cascade":
                    mk, Bt = A.m_conf_cascade(Bm, None), Bm
                else:
                    Bt, mk = A.reader_cascade(blocks, rds[0], rds[1], ds)
                gains.append(heldout_grouped(Bt, mk, b))
            reps.append(float(np.concatenate(gains).mean()))
            if b % 10 == 9:
                print(f"    {method} {rd} r3={r3}: {b + 1} replicates ({time.time() - t0:.0f}s)", flush=True)
        reps = np.array(reps)
        name = f"{rds[0]}->{rds[1]}" if len(rds) > 1 else rds[0]
        rows.append({"method": method, "reader": name, "plan_set": "with two hop" if r3 else "single step",
                     "reps": len(reps), "mean": reps.mean(), "sd": reps.std(ddof=1),
                     "low": np.percentile(reps, 2.5), "high": np.percentile(reps, 97.5),
                     "upper_one_sided": np.percentile(reps, 95), "share_above_0": (reps > 0).mean(),
                     "share_above_margin": (reps >= CONFIG["margin"]).mean()})
        pd.DataFrame({"rep": range(len(reps)), "gain": reps}).to_csv(
            OUT / f"refit_reps_{method}_{name.replace('->', '_to_')}_{'r3' if r3 else 'single'}.csv", index=False)
        say(f"  {method:15s} {name:22s} {rows[-1]['plan_set']:12s} refit mean {reps.mean():+.4f}  95% "
            f"[{rows[-1]['low']:+.4f}, {rows[-1]['high']:+.4f}]  one sided upper {rows[-1]['upper_one_sided']:+.4f}  "
            f"above 0 in {rows[-1]['share_above_0']:.2f} of replicates  ({time.time() - t0:.0f}s)")
    A.CONFIG["include_r3"] = False
    pd.DataFrame(rows).to_csv(OUT / "refit_bootstrap.csv", index=False)


# ====================================================================== mechanism
def hull_pair(B, tr, c):
    H = A._hull_vertices(B.Y, B.cost, tr)
    xs = [h[0] for h in H]
    if len(H) == 1:
        return H[0][2], H[0][2]
    j = int(np.clip(np.searchsorted(xs, c), 1, len(H) - 1))
    return H[j - 1][2], H[j][2]


def part_mechanism():
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    say("\n== mechanism: what the policies learn (held out, single step plans, all seeds)")
    A.CONFIG["include_r3"] = False
    rows, pair_all = [], []
    for rd in readers():
        df, _ = A.load_reader(rd)
        for ds in sorted(df.dataset.unique()):
            B = A.Block(df, ds)
            F = A.decision_features(B)
            X = S20.cb_features(df, ds, B.q)
            F2 = F.join(X.add_prefix("cb_"))
            E = load_E(ds, B)
            plans = B.plans
            cost = B.cost[plans].to_numpy()
            for name in ("planner_tsr", "planner_sk", "plan_emb"):
                for sd in A.CONFIG["seeds"]:
                    A._CUR_SEED[0] = sd
                    if name == "planner_tsr":
                        fit, apply_ = A.m_planner_tsr(B, F); FF = F
                    elif name == "planner_sk":
                        fit, apply_ = S20.m_planner_sk(B, F2); FF = F2
                    else:
                        fit, apply_ = R.m_plan_emb(B, E)
                    q = B.q.copy()
                    np.random.default_rng(sd).shuffle(q)
                    folds = [q[i::A.CONFIG["k_folds"]] for i in range(A.CONFIG["k_folds"])]
                    Pall, Yall, picks, pairs = [], [], [], []
                    for i, te in enumerate(folds):
                        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
                        cfg = fit(tr)
                        if name == "plan_emb":
                            trq, lam = cfg
                            p = plan_predict(plan_models(E.loc[trq].to_numpy(), B.Y.loc[trq, plans].to_numpy()),
                                             E.loc[te].to_numpy())
                        else:
                            bst, prior, lam = cfg
                            base = np.tile(np.log(prior / (1 - prior)).to_numpy(), len(te))
                            Xq = FF.loc[te].to_numpy(np.float32)
                            des = np.concatenate([np.repeat(Xq, len(plans), 0), np.tile(np.eye(len(plans)), (len(te), 1))], 1)
                            p = bst.predict(xgb.DMatrix(des, base_margin=base)).reshape(len(te), len(plans))
                        pick = (p - lam * cost[None, :]).argmax(1)
                        _, c = apply_(cfg, te)
                        a_, b_ = hull_pair(B, tr, c)
                        Yte = B.Y.loc[te, plans].to_numpy()
                        ia, ib = plans.index(a_), plans.index(b_)
                        for t in np.where(Yte[:, ia] != Yte[:, ib])[0]:
                            pairs.append((p[t, ia] - p[t, ib], Yte[t, ia], te[t]))
                        Pall.append(p); Yall.append(Yte); picks.append(pick)
                    P_, Y_, pk = np.vstack(Pall), np.vstack(Yall), np.concatenate(picks)
                    aucs = [roc_auc_score(Y_[:, k], P_[:, k]) for k in range(len(plans)) if 0 < Y_[:, k].mean() < 1]
                    cor = np.corrcoef(P_.T)
                    off = cor[~np.eye(len(plans), dtype=bool)]
                    pr = np.array([(x, y) for x, y, _ in pairs], float)
                    pa = roc_auc_score(pr[:, 1], pr[:, 0]) if len(pr) > 10 and 0 < pr[:, 1].mean() < 1 else np.nan
                    rows.append({"reader": rd, "dataset": ds, "policy": name, "seed": sd, "per_plan_auroc": np.mean(aucs),
                                 "cross_plan_corr": np.nanmean(off), "pair_auroc": pa, "n_discordant": len(pr),
                                 "modal_plan_share": np.bincount(pk).max() / len(pk)})
                    pair_all += [(name, rd, ds, sd, x, y, qq) for x, y, qq in pairs]
            print(f"    mechanism {rd} {ds} done", flush=True)
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "mechanism_all_seeds.csv", index=False)
    PP = pd.DataFrame(pair_all, columns=["policy", "reader", "dataset", "seed", "score_diff", "first_wins", "query_id"])
    PP.to_csv(OUT / "mechanism_pairs.csv", index=False)
    rng = np.random.default_rng(0)
    for name, g in M.groupby("policy"):
        pp = PP[PP.policy == name]
        # pair AUROC pooled over blocks, per seed; interval: bootstrap over questions within seed 0
        per_seed = [roc_auc_score(x.first_wins, x.score_diff) for _, x in pp.groupby("seed")]
        p0 = pp[pp.seed == A.CONFIG["seeds"][0]]
        bs = []
        for _ in range(1000):
            s = p0.iloc[rng.integers(0, len(p0), len(p0))]
            if 0 < s.first_wins.mean() < 1:
                bs.append(roc_auc_score(s.first_wins, s.score_diff))
        say(f"  {name:12s} per plan AUROC {g.per_plan_auroc.mean():.3f} (seeds {g.groupby('seed').per_plan_auroc.mean().min():.3f}.."
            f"{g.groupby('seed').per_plan_auroc.mean().max():.3f})  cross plan corr {g.cross_plan_corr.mean():.3f}  "
            f"pair AUROC pooled {np.mean(per_seed):.3f} (seeds {min(per_seed):.3f}..{max(per_seed):.3f}; seed 0 95% "
            f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}])  discordant questions per seed "
            f"{len(p0)}  modal plan share {g.modal_plan_share.mean():.2f}")


# ====================================================================== headroom at the policy's budget
def greedy_union(Ytr, k):
    chosen, cover = [], np.zeros(len(Ytr), bool)
    for _ in range(k):
        gain = [((Ytr[:, j].astype(bool) | cover).sum(), j) for j in range(Ytr.shape[1]) if j not in chosen]
        _, j = max(gain)
        chosen.append(j)
        cover |= Ytr[:, j].astype(bool)
    return chosen


def oracle_on(Yte, cost, idx):
    sub, c = Yte[:, idx], cost[idx]
    ok = sub.max(1) == 1
    cc = np.where(sub == 1, c[None, :], np.inf).min(1)
    return ok.mean(), float(np.where(ok, cc, c.min()).mean())


def envelope_at(pts, c):
    """Best mix of the points (cost, success) with mean cost at most c (upper concave envelope, flat at the top)."""
    pts = sorted(pts, key=lambda t: (t[0], -t[1]))
    best = max(range(len(pts)), key=lambda i: (pts[i][1], -pts[i][0]))
    H = []
    for p in pts[: best + 1]:
        while len(H) >= 2 and (H[-1][1] - H[-2][1]) * (p[0] - H[-2][0]) <= (p[1] - H[-2][1]) * (H[-1][0] - H[-2][0]):
            H.pop()
        H.append(p)
    xs, ys = [h[0] for h in H], [h[1] for h in H]
    if c <= xs[0]:
        return ys[0]
    return float(np.interp(c, xs, ys)) if c < xs[-1] else ys[-1]


def pair_oracle_points(Y, cost, rows):
    """Oracle over every single plan and every PAIR of plans on the given rows: per question run the cheaper plan of
    the pair that answers, else the cheaper plan. Returns {(i, j): (mean cost, success)}."""
    Ys, out = Y[rows], {}
    n = Ys.shape[1]
    for i in range(n):
        out[(i, i)] = (float(cost[i]), float(Ys[:, i].mean()))
        for j in range(i + 1, n):
            a, b = (i, j) if cost[i] <= cost[j] else (j, i)
            use_a = Ys[:, a] == 1
            ok = use_a | (Ys[:, b] == 1)
            c = np.where(use_a, cost[a], np.where(ok, cost[b], cost[a]))
            out[(i, j)] = (float(c.mean()), float(ok.mean()))
    return out


def env_vertices(pts):
    """Keys of the points on the upper concave envelope (up to the most successful point), cheapest first."""
    items = sorted(pts.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
    best = max(range(len(items)), key=lambda i: (items[i][1][1], -items[i][1][0]))
    H = []
    for kv in items[: best + 1]:
        while len(H) >= 2 and (H[-1][1][1] - H[-2][1][1]) * (kv[1][0] - H[-2][1][0]) <= \
                (kv[1][1] - H[-2][1][1]) * (H[-1][1][0] - H[-2][1][0]):
            H.pop()
        H.append(kv)
    return [k for k, _ in H]


def interp_flat(xs, ys, c):
    if c <= xs[0]:
        return ys[0]
    return float(np.interp(c, xs, ys)) if c < xs[-1] else ys[-1]


def part_headroom():
    say("\n== headroom at each policy's own mean cost: oracle frontier indexed by budget")
    say("   (oracle over plan pairs: the pairs on the oracle frontier are chosen on the training folds and measured on")
    say("    the held out fold, exactly as the hull's plans are; the frontier is read at the policy's mean cost)")
    A.CONFIG["include_r3"] = False
    M = pd.read_csv(AN / "final" / "methods.csv")
    Rt = pd.read_csv(AN / "final" / "routers.csv")
    pol = pd.concat([M[(M.dataset != "pooled") & M.method.isin(PRIMARY)][["reader", "dataset", "method", "gain", "cost_s"]],
                     Rt[Rt.method == "plan_emb"][["reader", "dataset", "method", "gain", "cost_s"]]])
    rows = []
    for rd in readers():
        df, _ = A.load_reader(rd)
        for ds in sorted(df.dataset.unique()):
            B = A.Block(df, ds)
            Yn, cost = B.Yn, B.cost[B.Y.columns].to_numpy()
            pr = pol[(pol.reader == rd) & (pol.dataset == ds)]
            gaps = {m: [] for m in pr.method}
            for sd in A.CONFIG["seeds"]:
                q = B.q.copy()
                np.random.default_rng(sd).shuffle(q)
                folds = [q[i::A.CONFIG["k_folds"]] for i in range(A.CONFIG["k_folds"])]
                for i, te in enumerate(folds):
                    tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
                    ri, ti = B.rows(tr), B.rows(te)
                    keys = env_vertices(pair_oracle_points(Yn, cost, ri))
                    te_pts = pair_oracle_points(Yn, cost, ti)
                    pts = sorted(te_pts[k] for k in keys)
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    for _, r in pr.iterrows():
                        h = A.hull_at(B.Y, B.cost, tr, r.cost_s, eval_idx=te)
                        gaps[r.method].append(max(h, interp_flat(xs, ys, r.cost_s)) - h)
            for _, r in pr.iterrows():
                rows.append({"reader": rd, "dataset": ds, "method": r.method, "gain": r.gain, "cost_s": r.cost_s,
                             "gap_pairs": np.mean(gaps[r.method])})
        print(f"    headroom {rd} done", flush=True)
    H = pd.DataFrame(rows)
    H.to_csv(OUT / "headroom_same_budget.csv", index=False)
    g = H.groupby(["reader", "method"])[["gain", "gap_pairs"]].mean().reset_index()
    g["closed"] = g.gain / g.gap_pairs
    g.to_csv(OUT / "headroom_same_budget_pooled.csv", index=False)
    say("  gap of the pair oracle frontier over the hull at the policy's cost, mean over datasets:")
    say(g.pivot(index="method", columns="reader", values="gap_pairs").round(3).to_string())
    say("  fraction of that gap closed (pooled gain / gap; negative = loses):")
    say(g.pivot(index="method", columns="reader", values="closed").round(2).to_string())
    say(f"  largest fraction closed by a primary policy: {g[g.method.isin(PRIMARY)].closed.max():.2f}; by plan_emb: "
        f"{g[g.method == 'plan_emb'].closed.max():.2f}")


# ====================================================================== protocol ablation
def run11(label, seeds, suffix, hull):
    out = AN / (label + suffix)
    if (out / "methods.csv").exists():
        return out
    A11 = _load("a11_ablation", "11_analysis_full.py")
    A11.CONFIG.update(label=label, seeds=seeds, out_suffix=suffix, reader_pairs=[], hull_mode=hull)
    argv = sys.argv
    sys.argv = [sys.argv[0]]
    A11.main()
    sys.argv = argv
    return out


def part_ablation():
    say("\n== protocol ablation: the same 100 cells (5 primary policies x 5 readers x 4 datasets), one factor at a time")
    W = pd.read_csv(AN / "weak_protocol" / "weak_vs_ours.csv")
    step = run11("strict", [0], "_weakseed0_stephull", "step")

    def cells(d):
        M = pd.read_csv(d / "methods.csv")
        return M[(M.dataset != "pooled") & M.method.isin(PRIMARY)].set_index(["reader", "dataset", "method"])

    S1, S2 = cells(step), cells(AN / "strict_weakseed0")
    S3, S4 = cells(AN / "strict"), cells(AN / "final")
    key = W.set_index(["reader", "dataset", "method"]).index
    S1, S2, S3, S4 = (x.reindex(key) for x in (S1, S2, S3, S4))
    p_holm100 = A.holm(S4.p.to_numpy())
    steps = [
        ("weak: beats Always RAG and Never RAG on accuracy (strict, seed 0, no cost matching)",
         ((W.margin_vs_always > 0) & (W.margin_vs_never > 0)).to_numpy()),
        ("comparator: best single plan at the same cost (strict, seed 0)", (S1.gain > 0).to_numpy()),
        ("comparator: fixed plan hull at the same cost (strict, seed 0)", (S2.gain > 0).to_numpy()),
        ("seeds: 5 fold splits instead of 1 (strict)", (S3.gain > 0).to_numpy()),
        ("grading: label_final instead of strict", (S4.gain > 0).to_numpy()),
        ("decision: 95% interval above 0 and p < 0.05 per cell", ((S4.ci_low > 0) & (S4.p < 0.05)).to_numpy()),
        ("decision: Holm over the 100 cells", ((S4.ci_low > 0) & (p_holm100 < 0.05)).to_numpy()),
    ]
    rows = []
    for name, w in steps:
        rows.append({"step": name, "wins": int(np.nansum(w)), "of": int(np.isfinite(w.astype(float)).sum())})
        say(f"  {rows[-1]['wins']:3d} of {rows[-1]['of']:3d}  {name}")
    P = pd.read_csv(AN / "final" / "methods.csv")
    P = P[(P.dataset == "pooled") & P.method.isin(PRIMARY)]
    raw = int((P.gain > 0).sum())
    sig = int(((P.ci_low > 0) & (P.p_holm < 0.05)).sum()) if "p_holm" in P else np.nan
    say(f"  {raw:3d} of {len(P):3d}  pooling: one test per policy and reader over 4 datasets, gain above 0")
    say(f"  {sig:3d} of {len(P):3d}  pooling + Holm over the primary family (the paper's rule)")
    rows += [{"step": "pooled, gain above 0", "wins": raw, "of": len(P)},
             {"step": "pooled, Holm", "wins": sig, "of": len(P)}]
    pd.DataFrame(rows).to_csv(OUT / "protocol_ablation.csv", index=False)


# ====================================================================== learning curve of plan_emb
def m_plan_emb_frac(B, E, frac):
    fit0, apply_ = R.m_plan_emb(B, E)

    def fit(sel):
        sel = np.asarray(sel)
        if frac < 1.0:
            rng = np.random.default_rng(A._CUR_SEED[0] + 7)
            sel = sel[np.sort(rng.permutation(len(sel))[: max(20, int(round(frac * len(sel))))])]
        return fit0(sel)
    return fit, apply_


def part_curve():
    say("\n== learning curve of plan_emb (single step plans; share of the training folds)")
    A.CONFIG["include_r3"] = False
    rows = []
    for rd in readers():
        df, _ = A.load_reader(rd)
        for ds in sorted(df.dataset.unique()):
            B = A.Block(df, ds)
            E = load_E(ds, B)
            for fr in CONFIG["fracs"]:
                d, *_ = A.repeated(B, lambda: m_plan_emb_frac(B, E, fr))
                rows.append({"reader": rd, "dataset": ds, "frac": fr, "n_train": int(round(fr * 400)),
                             "gain": d.mean(), "sum_gain": d.sum(), "n": len(d)})
            print(f"    curve {rd} {ds} done", flush=True)
    LC = pd.DataFrame(rows)
    LC.to_csv(OUT / "learning_curve_plan_emb.csv", index=False)
    g = LC.groupby(["reader", "frac"])[["sum_gain", "n"]].sum()
    say((g.sum_gain / g.n).unstack().round(4).to_string())


# ====================================================================== plan_emb with its overhead charged
# ====================================================================== per plan router: own cost charged, pooled training
BGE_RETRIEVERS = {"bge_m3", "bge_bm25_rrf", "qwen3_reranker_0_6b"}   # plans whose retrieval embeds the query with BGE-M3


def eff_cost(B, plans, delta):
    """Plan costs as the router pays them: + delta (embedding and classifiers) unless the plan's own retrieval embeds
    the query with BGE-M3 anyway (BGE-M3, the RRF hybrid, and the reranker, whose candidates come from that hybrid)."""
    c = B.cost[plans].to_numpy().astype(float)
    if delta:
        c = c + delta * np.array([0.0 if p.split("@")[0] in BGE_RETRIEVERS else 1.0 for p in plans])
    return c


def pick_lambda_eff(B, sel, P_oof, plans, ceff):
    """R.pick_lambda with the router's own plan costs (the hull keeps the plain plan costs)."""
    r = B.rows(sel)
    cols = [B.pcol[p] for p in plans]
    best = None
    for lam in A.CONFIG["lambdas"]:
        k = (P_oof - lam * ceff[None, :]).argmax(1)
        s = B.Yn[r, [cols[i] for i in k]]
        g = s.mean() - A.hull_at(B.Y, B.cost, sel, float(ceff[k].mean()))
        if best is None or g > best[0]:
            best = (g, lam)
    return best[1]


def fit_predict(Xtr, Ytr, Xte, oh_tr=None, oh_te=None):
    """R.m_plan_emb's model (PCA 64 + one logistic regression per plan); pooled training adds dataset indicators
    after the PCA."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    pca = PCA(n_components=min(R.CONFIG["pca_dim"], len(Xtr) - 1), random_state=0).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    if oh_tr is not None:
        Ztr, Zte = np.hstack([Ztr, oh_tr]), np.hstack([Zte, oh_te])
    P = np.empty((len(Xte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        yj = Ytr[:, j]
        P[:, j] = yj[0] if yj.min() == yj.max() else LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, yj).predict_proba(Zte)[:, 1]
    return P


def router_gains(blocks, embs, seed, pooled, deltas):
    """Held out gains of the per plan router for one reader and seed. blocks/embs: dataset -> Block / embeddings.
    pooled=False trains per dataset (R.m_plan_emb); pooled=True trains one model on the training folds of all four
    datasets (dataset indicators added). Every dataset keeps its own folds (the split of A.heldout for this seed),
    its own hull and its own lambda. Returns {delta: {dataset: per question gain Series}}."""
    A._CUR_SEED[0] = seed
    dss = sorted(blocks)
    plans = sorted(set.intersection(*[set(blocks[d].plans) for d in dss])) if pooled else None
    folds = {}
    for d in dss:
        q = blocks[d].q.copy()
        np.random.default_rng(seed).shuffle(q)
        folds[d] = [q[i::A.CONFIG["k_folds"]] for i in range(A.CONFIG["k_folds"])]
    out = {dl: {d: [] for d in dss} for dl in deltas}
    for i in range(A.CONFIG["k_folds"]):
        sel = {d: np.concatenate([folds[d][j] for j in range(A.CONFIG["k_folds"]) if j != i]) for d in dss}
        test = {d: folds[d][i] for d in dss}
        if pooled:
            tr_all = [(d, q) for d in dss for q in sel[d]]
            Xtr = np.vstack([embs[d].loc[sel[d]].to_numpy() for d in dss])
            Ytr = np.vstack([blocks[d].Y.loc[sel[d], plans].to_numpy() for d in dss])
            ohtr = np.vstack([np.tile(np.eye(len(dss))[k], (len(sel[d]), 1)) for k, d in enumerate(dss)])
            oof = np.empty(Ytr.shape)
            for te in R.inner_splits(len(tr_all), seed):
                trn = np.setdiff1d(np.arange(len(tr_all)), te)
                oof[te] = fit_predict(Xtr[trn], Ytr[trn], Xtr[te], ohtr[trn], ohtr[te])
            start = np.cumsum([0] + [len(sel[d]) for d in dss])
            for k, d in enumerate(dss):
                B = blocks[d]
                Pte = fit_predict(Xtr, Ytr, embs[d].loc[test[d]].to_numpy(), ohtr,
                                  np.tile(np.eye(len(dss))[k], (len(test[d]), 1)))
                for dl in deltas:
                    ce = eff_cost(B, plans, dl)
                    lam = pick_lambda_eff(B, sel[d], oof[start[k]:start[k + 1]], plans, ce)
                    kk = (Pte - lam * ce[None, :]).argmax(1)
                    s = B.Yn[B.rows(test[d]), [B.pcol[plans[j]] for j in kk]]
                    h = A.hull_at(B.Y, B.cost, sel[d], float(ce[kk].mean()), eval_idx=test[d])
                    out[dl][d].append(pd.Series(s - h, index=test[d]))
        else:
            for d in dss:
                B, E = blocks[d], embs[d]
                pl = B.plans
                X, Y = E.loc[sel[d]].to_numpy(), B.Y.loc[sel[d], pl].to_numpy()
                oof = np.empty(Y.shape)
                for te in R.inner_splits(len(sel[d]), seed):
                    trn = np.setdiff1d(np.arange(len(sel[d])), te)
                    oof[te] = fit_predict(X[trn], Y[trn], X[te])
                Pte = fit_predict(X, Y, E.loc[test[d]].to_numpy())
                for dl in deltas:
                    ce = eff_cost(B, pl, dl)
                    lam = pick_lambda_eff(B, sel[d], oof, pl, ce)
                    kk = (Pte - lam * ce[None, :]).argmax(1)
                    s = B.Yn[B.rows(test[d]), [B.pcol[pl[j]] for j in kk]]
                    h = A.hull_at(B.Y, B.cost, sel[d], float(ce[kk].mean()), eval_idx=test[d])
                    out[dl][d].append(pd.Series(s - h, index=test[d]))
    return {dl: {d: pd.concat(v).sort_index() for d, v in m.items()} for dl, m in out.items()}


def run_router_variant(pooled, deltas, tag):
    rows, perq = [], []
    for r3 in (False, True):
        A.CONFIG["include_r3"] = r3
        ps = "with two hop" if r3 else "single step"
        for rd in readers():
            t0 = time.time()
            df, _ = A.load_reader(rd)
            blocks = {ds: A.Block(df, ds) for ds in sorted(df.dataset.unique())}
            embs = {ds: load_E(ds, B) for ds, B in blocks.items()}
            acc = {dl: {d: [] for d in blocks} for dl in deltas}
            for sd in A.CONFIG["seeds"]:
                g = router_gains(blocks, embs, sd, pooled, deltas)
                for dl in deltas:
                    for d in blocks:
                        acc[dl][d].append(g[dl][d])
            for dl in deltas:
                parts = [pd.concat(acc[dl][d], axis=1).mean(axis=1) for d in sorted(blocks)]
                for d, p in zip(sorted(blocks), parts):
                    perq.append(pd.DataFrame({"variant": tag, "plan_set": ps, "reader": rd, "delta_ms": dl * 1000,
                                              "dataset": d, "query_id": p.index.astype(str), "gain": p.values}))
                x = np.concatenate([p.to_numpy() for p in parts])
                lo, hi = A.boot_ci(x, A.CONFIG["n_boot"], A.CONFIG["seed"])
                rows.append({"variant": tag, "plan_set": ps, "reader": rd, "delta_ms": dl * 1000, "gain": x.mean(),
                             "ci_low": lo, "ci_high": hi, "p": A.p_two_sided(x), "n": len(x)})
                say(f"  {tag:9s} {ps:12s} {rd:12s} own cost {dl * 1000:4.1f} ms: gain {x.mean():+.4f} "
                    f"[{lo:+.4f}, {hi:+.4f}]  p {rows[-1]['p']:.4f}")
            print(f"    {tag} {ps} {rd} done ({time.time() - t0:.0f}s)", flush=True)
    A.CONFIG["include_r3"] = False
    return pd.DataFrame(rows), pd.concat(perq, ignore_index=True)


def part_overhead():
    ms = float(arg("--overhead_ms", "nan"))
    if not np.isfinite(ms):
        sys.exit("--parts overhead needs --overhead_ms X (the measured embedding + classifier time)")
    say(f"\n== per plan router (per dataset training) with its own cost: +{ms:.1f} ms per question, except when the chosen "
        f"plan embeds the query with BGE-M3 anyway ({sorted(BGE_RETRIEVERS)}); lambda tuned with these costs")
    T, PQ = run_router_variant(False, [0.0, ms / 1000.0], "dataset")
    T.to_csv(OUT / "plan_emb_with_overhead.csv", index=False)
    PQ.to_csv(OUT / "plan_emb_with_overhead_per_query.csv", index=False)
    # check: with no own cost this must reproduce script 12's per plan router
    for r3, d in ((False, "final"), (True, "final_r3")):
        ref = pd.read_csv(AN / d / "routers_pooled.csv")
        ref = ref[ref.method == "plan_emb"].set_index("reader").gain
        mine = T[(T.delta_ms == 0) & (T.plan_set == ("with two hop" if r3 else "single step"))].set_index("reader").gain
        say(f"  check against script 12 ({d}): largest difference {np.abs(mine - ref.reindex(mine.index)).max():.5f}")


def part_pooled():
    ms = float(arg("--overhead_ms", "24.6"))
    say(f"\n== per plan router trained on all four datasets together (dataset indicators added; about 1,600 training "
        f"questions instead of 400); each dataset keeps its own folds, hull and lambda; own cost 0 and {ms:.1f} ms")
    T, PQ = run_router_variant(True, [0.0, ms / 1000.0], "pooled")
    T["p_holm_within_variant"] = np.nan
    for (ps, dl), g in T.groupby(["plan_set", "delta_ms"]):
        T.loc[g.index, "p_holm_within_variant"] = A.holm(g.p.to_numpy())
    T.to_csv(OUT / "plan_emb_pooled_training.csv", index=False)
    PQ.to_csv(OUT / "plan_emb_pooled_training_per_query.csv", index=False)
    say("  Holm within each plan set and cost (5 readers):")
    say(T[["plan_set", "delta_ms", "reader", "gain", "p", "p_holm_within_variant"]].round(4).to_string(index=False))


# ====================================================================== cost boundaries
def part_boundary():
    say("\n== where policy costs fall on the hull (per reader and dataset cell, mean cost over folds and seeds)")
    rows = []
    for tag, d in (("single step", "final"), ("with two hop", "final_r3")):
        F = pd.read_csv(AN / d / "fixed_plans.csv")
        M = pd.read_csv(AN / d / "methods.csv")
        Rt = pd.read_csv(AN / d / "routers.csv")
        P = pd.concat([M[M.dataset != "pooled"], Rt])
        P = P[~P.reader.astype(str).str.contains("->")]
        for (rd, ds), g in P.groupby(["reader", "dataset"]):
            f = F[(F.reader == rd) & (F.dataset == ds)]
            if f.empty:
                continue
            cmin = f.cost_s.min()
            cbest = f.sort_values(["success", "cost_s"], ascending=[False, True]).cost_s.iloc[0]
            for _, r in g.iterrows():
                rows.append({"plan_set": tag, "reader": rd, "dataset": ds, "method": r.method, "cost_s": r.cost_s,
                             "below_cheapest": r.cost_s < cmin, "flat_part": r.cost_s >= cbest})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "cost_boundaries.csv", index=False)
    for tag, g in T.groupby("plan_set"):
        say(f"  {tag:12s} cells {len(g)}: below the cheapest plan {int(g.below_cheapest.sum())}, on the flat part "
            f"(at or above the cost of the most successful plan, in sample) {int(g.flat_part.sum())}")
    say("  Note: the hull's vertices are chosen on the training folds, so the flat part starts at a slightly different "
        "cost in each fold; in sample costs are a close guide.")


def main():
    t0 = time.time()
    say(f"Robustness checks; parts {PARTS}; label {A.CONFIG['label']}; seeds {A.CONFIG['seeds']}"
        + ("; QUICK TEST RUN, numbers not for the paper" if QUICK else ""))
    for p in PARTS:
        {"inference": part_inference, "refit": part_refit, "mechanism": part_mechanism, "headroom": part_headroom,
         "ablation": part_ablation, "curve": part_curve, "overhead": part_overhead, "boundary": part_boundary,
         "pooled": part_pooled}[p]()
        say(f"  ({p} done, {time.time() - t0:.0f}s)")
        (OUT / f"summary_{'_'.join(PARTS)}{'_quick' if QUICK else ''}.txt").write_text("\n".join(LINES), encoding="utf-8")
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
