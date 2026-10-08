#!/usr/bin/env python
"""
Full mode analysis: the fixed plan hull, our planner, and published adaptive
retrieval baselines, all under ONE held out protocol. CPU only, minutes.

Replaces 05, 06 and 07 for the paper runs (those stay for the sub200k pilot).

Per (reader, dataset), on the same 500 queries:
  * fixed plans: success and median measured cost (retrieval + reader) of
    every (retriever, depth) plan, closed book included;
  * the HULL: upper concave hull of the fixed plans (randomised mixing), the
    baseline every method must beat;
  * methods (each picks its settings on half A of the queries, is scored on
    half B, then the halves swap; results pooled, paired bootstrap CI):

    planner_tsr       ours: plan prior + residual XGBoost on decision time
                      features (query text features + a BM25 probe whose cost
                      is charged), pick argmax  p_hat - lambda * cost
    adaptive_rag      Jeong et al., NAACL 2024: a query classifier routes to
                      no retrieval / a cheap plan / a strong plan. Silver
                      labels = cheapest rung that answers correctly, as in the
                      paper. Here TF IDF + logistic regression on the query
                      text replaces their T5 classifier (stated in the paper).
    adaptive_k        Adaptive k (EMNLP 2025 Findings): per query depth from
                      the largest gap in the sorted similarity scores of one
                      retriever's list, mapped to the nearest depth we labelled.
    gate_closed_book  retrieve only when the closed book answer is uncertain
                      (mean token log probability below tau): the self
                      knowledge idea of SKR (Findings of EMNLP 2023) and the
                      adaptive retrieval motivation of Mallen et al. (ACL
                      2023), with model confidence in place of entity
                      popularity, which our datasets do not provide.
    conf_cascade      run a cheap plan; escalate to a strong plan when the
                      answer's mean log probability is below tau (FLARE /
                      AutoMix style self confidence, same reader).
    reader_cascade    escalate the least confident answers of reader A to
                      reader B (only if B's labels exist; the Route B gate).

Gain = method success minus hull success at the method's mean cost, both on
the held out fold. The hull's plans are chosen on the training folds, exactly
like the method's settings, so baseline and method get the same information.

Not reproduced: METIS (SOSP 2025) uses an LLM profiler plus a serving system
(batching, GPU memory scheduling); its configuration space differs from ours.
It is discussed, not rerun.

Output: results_full/analysis/<label>/  fixed_plans.csv, methods.csv, summary.txt
"""

import json
import os
import pickle
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings
from tsr.features import query_lexical, score_stats

CONFIG = {
    "label": "final",             # label_<label> from the graded files; falls back to f1_07
    "readers": "all",             # list or "all" (every tsr_labels_<reader>_graded.csv)
    "datasets": "all",
    "n_boot": 2000,
    "k_folds": 5,
    "seed": 0,                    # bootstrap seed
    "seeds": [0, 1, 2, 3, 4],     # repeats: each changes the fold split AND the XGBoost seed
    "cost_stat": "mean",          # per plan latency statistic: "mean" (expected cost) or "median"
    # robustness: replace a retriever's measured retrieval latency (seconds per query), e.g. {"splade_v3": 0.05}
    # for an inverted index SPLADE; results then go to analysis/<label><out_suffix>/
    "retrieval_cost_override": {},
    "out_suffix": "",
    "lambdas": list(np.round(np.concatenate([[0.0], np.geomspace(0.002, 1.0, 20)]), 5)),
    "taus": list(np.round(np.linspace(-3.0, 0.0, 31), 2)),   # log prob thresholds
    "adaptive_k_retrievers": ["bge_m3", "stella_1_5b"],
    "reader_pairs": [["qwen2.5-7b", "qwen3-14b"], ["qwen2.5-7b", "llama3.1-8b"]],
    "rates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
}

PROC = settings.PROCESSED_DATA_ROOT
CACHE = settings.CACHE_ROOT / "retrievals_v3"
DEPTHS = sorted({0, 1, 3, 5, 10, 20})


# ======================================================================
# data
# ======================================================================
def readers():
    found = sorted(f.stem.replace("tsr_labels_", "").replace("_graded", "")
                   for f in PROC.glob("tsr_labels_*_graded.csv"))
    return found if CONFIG["readers"] == "all" else [r for r in CONFIG["readers"] if r in found]


def load_reader(reader):
    df = pd.read_csv(PROC / f"tsr_labels_{reader}_graded.csv", dtype={"query_id": str}, keep_default_na=False)
    col = f"label_{CONFIG['label']}"
    if col not in df.columns:
        col = "label_f1_07"
    df = df[df["repeat"].astype(int) == 0].copy()
    df["y"] = df[col].astype(int)
    df["depth"] = df["depth"].astype(int)
    df["plan"] = df["retriever"] + "@" + df["depth"].astype(str)
    for c in ("retrieval_time_s", "agent_time_s", "mean_logprob"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["retrieval_time_s"] = df["retrieval_time_s"].fillna(0.0)
    return df, col


class Block:
    """All matrices for one (reader, dataset): queries x plans."""

    def __init__(self, df, ds):
        d = df[df["dataset"] == ds]
        self.ds = ds
        self.Y = d.pivot_table(index="query_id", columns="plan", values="y")
        keep = self.Y.columns[self.Y.notna().all()]
        self.Y = self.Y[keep]
        self.conf = d.pivot_table(index="query_id", columns="plan", values="mean_logprob")[keep]
        # Plan cost = MEAN latency per query (expected cost; what a latency budget constrains and what the
        # cost model predicts). The median understates plans with long generations (14B: mean/median up to 1.8).
        agg = CONFIG["cost_stat"]
        self.ret_cost = d.groupby("plan")["retrieval_time_s"].agg(agg)[keep]
        self.agent_cost = d.groupby("plan")["agent_time_s"].agg(agg)[keep]
        for r_, v_ in CONFIG["retrieval_cost_override"].items():
            self.ret_cost[[p for p in keep if p.split("@")[0] == r_]] = float(v_)
        self.cost = self.ret_cost + self.agent_cost
        self.q = np.array(self.Y.index)
        self.plans = list(keep)
        self.retr_of = {p: p.split("@")[0] for p in self.plans}
        self.depth_of = {p: int(p.split("@")[1]) for p in self.plans}
        cache = {}
        for r in set(self.retr_of.values()) - {"closed_book"}:
            f = CACHE / f"{ds}__{r}.pkl"
            if f.exists():
                cache[r] = pickle.load(open(f, "rb"))
        self.cache = cache
        any_c = next(iter(cache.values()))
        self.text = {q: any_c["queries"].get(q, "") for q in self.q}
        # numpy views for the tuning loops (pandas .loc there was the bottleneck)
        self.Yn = self.Y.to_numpy(dtype=float)
        self.Cn = self.conf.reindex(index=self.Y.index, columns=self.Y.columns).to_numpy(dtype=float)
        self.pcol = {p: i for i, p in enumerate(self.Y.columns)}
        self.qrow = {q: i for i, q in enumerate(self.Y.index)}
        self._rows = {}

    def rows(self, idx):
        """Row positions of query ids idx (memoised per index set)."""
        key = hash(tuple(idx))
        r = self._rows.get(key)
        if r is None:
            r = self._rows[key] = np.fromiter((self.qrow[q] for q in idx), dtype=np.int64, count=len(idx))
        return r

    def plan_cost(self, p, already=()):
        """Cost of plan p when retrievers in `already` have run (their retrieval is free)."""
        r = self.retr_of[p]
        return self.agent_cost[p] + (0.0 if r in already else self.ret_cost[p])


_CUR_SEED = [0]      # set per repeat; read by the planner's XGBoost
_HULL_MEMO = {}
_MEAN_MEMO = {}


def _means(Y, idx):
    # The memo stores Y itself and checks identity: Python reuses id() after an
    # object is freed, so an id-only key can return another frame's means.
    key = (id(Y), hash(tuple(idx)))
    hit = _MEAN_MEMO.get(key)
    if hit is None or hit[0] is not Y:
        hit = _MEAN_MEMO[key] = (Y, Y.loc[idx].mean())
    return hit[1]


def _hull_vertices(Y, cost, idx):
    """Plans on the upper concave hull of (cost, success) measured on queries idx (memoised)."""
    key = (id(Y), id(cost), hash(tuple(idx)))
    hit = _HULL_MEMO.get(key)
    if hit is not None and hit[0] is Y and hit[1] is cost:
        return hit[2]
    m = _means(Y, idx)
    pts = sorted(((cost[p], m[p], p) for p in Y.columns), key=lambda t: (t[0], -t[1]))
    best = max(range(len(pts)), key=lambda i: (pts[i][1], -pts[i][0]))
    H = []
    for p in pts[: best + 1]:
        while len(H) >= 2 and (H[-1][1] - H[-2][1]) * (p[0] - H[-2][0]) <= (p[1] - H[-2][1]) * (H[-1][0] - H[-2][0]):
            H.pop()
        H.append(p)
    _HULL_MEMO[key] = (Y, cost, H)
    return H


def hull_at(Y, cost, idx, c, eval_idx=None):
    """Success of the best randomised mix of fixed plans at mean cost c.

    The hull's plans are CHOSEN on queries idx. Their success is MEASURED on
    eval_idx (default idx). For held out scoring pass idx = training folds and
    eval_idx = test fold: the baseline then picks its plans on the same data
    as the method, and both are scored on unseen queries. (Choosing the hull on
    the test fold itself is optimistic: with 100 test queries the best of 31
    plans is inflated by selection, which would unfairly sink every method.)
    """
    H = _hull_vertices(Y, cost, idx)
    ev = idx if eval_idx is None else eval_idx
    xs = [h[0] for h in H]
    m = _means(Y, ev)
    ys = [m[h[2]] for h in H]
    if c <= xs[0]:
        return float(ys[0])
    if c >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(c, xs, ys))


# ======================================================================
# features
# ======================================================================
def decision_features(B):
    rows = []
    bm25 = B.cache.get("bm25")
    for q in B.q:
        f = query_lexical(B.text[q])
        if bm25 is not None:
            f.update(score_stats([x["score"] for x in bm25["results"][q][:100]], "q_bm25_"))
        rows.append(f)
    return pd.DataFrame(rows, index=B.q)


# ======================================================================
# methods: each returns fit(sel) -> config, apply(config, idx) -> (succ array, mean cost)
# ======================================================================
def m_planner_tsr(B, F):
    import xgboost as xgb
    probe_cost = float(B.ret_cost.get("bm25@1", 0.0)) if "bm25" in B.cache else 0.0
    plans = B.plans
    P = np.eye(len(plans))

    def design(idx):
        Xq = F.loc[idx].to_numpy(np.float32)
        X = np.concatenate([np.repeat(Xq, len(plans), 0), np.tile(P, (len(idx), 1))], 1)
        return X

    def fit(sel):
        prior = B.Y.loc[sel].mean().clip(0.02, 0.98)
        y = B.Y.loc[sel].to_numpy().ravel()
        base = np.tile(np.log(prior / (1 - prior)).to_numpy(), len(sel))
        d = xgb.DMatrix(design(sel), label=y, base_margin=base)
        bst = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": 0.05, "min_child_weight": 5,
                         "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 5.0, "verbosity": 0, "seed": _CUR_SEED[0]},
                        d, 200)
        best = None
        for lam in CONFIG["lambdas"]:
            s, c = apply_((bst, prior, lam), sel)
            g = s.mean() - hull_at(B.Y, B.cost, sel, c)
            if best is None or g > best[0]:
                best = (g, lam)
        return (bst, prior, best[1])

    def apply_(cfg, idx):
        bst, prior, lam = cfg
        base = np.tile(np.log(prior / (1 - prior)).to_numpy(), len(idx))
        p = bst.predict(xgb.DMatrix(design(idx), base_margin=base)).reshape(len(idx), len(plans))
        pick = (p - lam * B.cost[plans].to_numpy()[None, :]).argmax(1)
        chosen = [plans[i] for i in pick]
        succ = np.array([B.Y.at[q, pl] for q, pl in zip(idx, chosen)])
        cost = np.mean([B.cost[pl] + (0.0 if B.retr_of[pl] == "bm25" else probe_cost) for pl in chosen])
        return succ, cost
    return fit, apply_


def m_adaptive_rag(B, F):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def ladder(sel):
        cheap_ret = [p for p in B.plans if B.depth_of[p] > 0]
        # rung 1: the most cost efficient retrieval plan on the selection half; rung 2: the best plan
        acc = B.Y.loc[sel].mean()
        r1 = max(cheap_ret, key=lambda p: (acc[p] - acc.get("closed_book@0", 0)) / max(B.cost[p], 1e-6))
        r2 = acc[cheap_ret].idxmax()
        return ["closed_book@0", r1, r2] if "closed_book@0" in B.plans else [r1, r2]

    def fit(sel):
        rungs = ladder(sel)
        y = []
        for q in sel:
            ok = [i for i, p in enumerate(rungs) if B.Y.at[q, p] == 1]
            y.append(ok[0] if ok else len(rungs) - 1)     # nothing works -> strongest rung (paper's rule)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform([B.text[q] for q in sel])
        if len(set(y)) < 2:
            return (rungs, None, vec, y[0])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)
        return (rungs, clf, vec, None)

    def apply_(cfg, idx):
        rungs, clf, vec, const = cfg
        k = np.full(len(idx), const) if clf is None else clf.predict(vec.transform([B.text[q] for q in idx]))
        chosen = [rungs[i] for i in k]
        return np.array([B.Y.at[q, p] for q, p in zip(idx, chosen)]), float(np.mean([B.cost[p] for p in chosen]))
    return fit, apply_


def m_adaptive_k(B, F):
    # depths this reader was labelled at (the 14B stops at 10)
    avail = sorted({int(c.rsplit('@', 1)[1]) for c in B.Y.columns if '@' in c} - {0})

    def k_star(r, q, kmax=20):
        s = np.array([x["score"] for x in B.cache[r]["results"][q][:kmax]], dtype=float)
        if len(s) < 2:
            return 1
        k = int(np.argmax(s[:-1] - s[1:]) + 1)       # cut at the largest gap
        return min(avail, key=lambda d: (abs(d - k), d))

    retrs = [r for r in CONFIG["adaptive_k_retrievers"] if r in B.cache]
    if not retrs:
        return None

    def run(r, idx):
        chosen = [f"{r}@{k_star(r, q)}" for q in idx]
        chosen = [p if p in B.Y.columns else f"{r}@{avail[-1]}" for p in chosen]
        return np.array([B.Y.at[q, p] for q, p in zip(idx, chosen)]), float(np.mean([B.cost[p] for p in chosen]))

    def fit(sel):
        return max(retrs, key=lambda r: (lambda s, c: s.mean() - hull_at(B.Y, B.cost, sel, c))(*run(r, sel)))

    return fit, (lambda r, idx: run(r, idx))


def m_gate_closed_book(B, F):
    if "closed_book@0" not in B.plans:
        return None
    cb = "closed_book@0"
    rplans = [p for p in B.plans if B.depth_of[p] > 0]

    def run(cfg, idx):
        tau, p = cfg
        r = B.rows(idx)
        conf = np.nan_to_num(B.Cn[r, B.pcol[cb]], nan=-99.0)
        go = conf < tau                                  # uncertain -> retrieve
        succ = np.where(go, B.Yn[r, B.pcol[p]], B.Yn[r, B.pcol[cb]])
        cost = B.cost[cb] + go.mean() * B.cost[p]       # closed book always runs first
        return succ, float(cost)

    def fit(sel):
        best = None
        for p in rplans:
            for tau in CONFIG["taus"]:
                s, c = run((tau, p), sel)
                g = s.mean() - hull_at(B.Y, B.cost, sel, c)
                if best is None or g > best[0]:
                    best = (g, (tau, p))
        return best[1]
    return fit, run


def m_conf_cascade(B, F):
    rplans = [p for p in B.plans if B.depth_of[p] > 0]

    def run(cfg, idx):
        a, b, tau = cfg
        r = B.rows(idx)
        conf = np.nan_to_num(B.Cn[r, B.pcol[a]], nan=-99.0)
        esc = conf < tau
        succ = np.where(esc, B.Yn[r, B.pcol[b]], B.Yn[r, B.pcol[a]])
        cost = B.cost[a] + esc.mean() * B.plan_cost(b, already={B.retr_of[a]})
        return succ, float(cost)

    def fit(sel):
        acc = B.Y.loc[sel].mean()
        # cheap candidates: the 6 cheapest retrieval plans; strong: the 4 most accurate
        cheap = sorted(rplans, key=lambda p: B.cost[p])[:6]
        strong = list(acc[rplans].sort_values(ascending=False).index[:4])
        best = None
        for a in cheap:
            for b in strong:
                if a == b:
                    continue
                for tau in CONFIG["taus"]:
                    s, c = run((a, b, tau), sel)
                    g = s.mean() - hull_at(B.Y, B.cost, sel, c)
                    if best is None or g > best[0]:
                        best = (g, (a, b, tau))
        return best[1]
    return fit, run


# ======================================================================
def heldout(B, method, seed):
    """K fold over queries: settings chosen on K-1 folds, scored on the held
    out fold; every query is scored exactly once. K = 5 gives learned baselines
    (Adaptive RAG's classifier, our planner) 400 training queries per dataset
    instead of 250, which is fairer to them than a 50/50 split."""
    fit, apply_ = method
    k = CONFIG["k_folds"]
    rng = np.random.default_rng(seed)
    q = B.q.copy()
    rng.shuffle(q)
    folds = [q[i::k] for i in range(k)]
    diffs, cfgs, rows = [], [], []
    for i in range(k):
        test = folds[i]
        sel = np.concatenate([folds[j] for j in range(k) if j != i])
        cfg = fit(sel)
        s, c = apply_(cfg, test)
        h = hull_at(B.Y, B.cost, sel, c, eval_idx=test)
        s_sel, c_sel = apply_(cfg, sel)
        rows.append({"success": s.mean(), "cost": c, "hull": h, "gain": s.mean() - h,
                     "gain_selection_half": s_sel.mean() - hull_at(B.Y, B.cost, sel, c_sel)})
        diffs.append(pd.Series(s - h, index=test))
        cfgs.append(cfg if isinstance(cfg, str) else
                    tuple(x for x in cfg if isinstance(x, (str, float, int, np.floating))))
    return pd.concat(diffs), rows, cfgs


def repeated(B, make_method):
    """Run heldout once per seed. Returns the per query gain averaged over seeds
    (aligned by query id), the per seed mean gains, and the fold rows."""
    per, means, rows_all, cfgs_all = [], [], [], []
    for sd in CONFIG["seeds"]:
        _CUR_SEED[0] = sd
        m = make_method()
        if m is None:
            return None
        d, rows, cfgs = heldout(B, m, sd)
        per.append(d.sort_index())
        means.append(float(d.mean()))
        rows_all += rows
        cfgs_all.append(cfgs)
    _CUR_SEED[0] = 0
    D = pd.concat(per, axis=1)
    return D.mean(axis=1).to_numpy(), np.array(means), rows_all, cfgs_all, D.index.to_numpy()


def p_two_sided(d):
    """One sample t test of mean gain = 0 on per query gains (n = 500 per block)."""
    from scipy import stats
    d = np.asarray(d, float)
    if np.allclose(d, d[0]):
        return 1.0
    return float(stats.ttest_1samp(d, 0.0).pvalue)


def holm(pvals):
    """Holm step down adjusted p values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(d, n, seed):
    rng = np.random.default_rng(seed)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def reader_cascade(blocks, A, Bn, ds):
    """Escalate the least confident answers of reader A (any plan) to reader B (any plan)."""
    a, b = blocks[(A, ds)], blocks[(Bn, ds)]
    q = np.intersect1d(a.q, b.q)
    if len(q) < 50:
        return None
    pa = [p for p in a.plans if a.depth_of[p] > 0]
    pb = [p for p in b.plans if b.depth_of[p] > 0]
    Yall = pd.concat([a.Y.loc[q].add_prefix(A + "|"), b.Y.loc[q].add_prefix(Bn + "|")], axis=1)
    Call = pd.concat([a.cost.add_prefix(A + "|"), b.cost.add_prefix(Bn + "|")])

    def run(cfg, idx):
        x, y, rate, thr = cfg
        ra, rb = a.rows(idx), b.rows(idx)
        esc = a.Cn[ra, a.pcol[x]] <= thr                 # NaN confidence -> not escalated (as before)
        succ = np.where(esc, b.Yn[rb, b.pcol[y]], a.Yn[ra, a.pcol[x]])
        same = a.retr_of[x] == b.retr_of[y]
        cost = a.cost[x] + esc.mean() * (b.agent_cost[y] + (0 if same else b.ret_cost[y]))
        return succ, float(cost)

    def fit(sel):
        best = None
        for x in pa:
            for y in pb:
                for rate in CONFIG["rates"]:
                    cx = a.Cn[a.rows(sel), a.pcol[x]]
                    thr = float(np.nanquantile(cx, rate)) if np.isfinite(cx).any() else np.nan
                    s, c = run((x, y, rate, thr), sel)
                    g = s.mean() - hull_at(Yall, Call, sel, c)
                    if best is None or g > best[0]:
                        best = (g, (x, y, rate, thr))
        return best[1]

    class _B:     # hull over BOTH readers' fixed plans: the fair baseline for a two reader system
        pass
    Bj = _B()
    Bj.q, Bj.Y, Bj.cost = q, Yall, Call
    return Bj, (fit, run)


def say(L, s):
    L.append(s)
    print(s, flush=True)


def main():
    import time
    t0 = time.time()
    out = settings.RESULTS_ROOT / "analysis" / (CONFIG["label"] + CONFIG["out_suffix"])
    out.mkdir(parents=True, exist_ok=True)
    blocks, fixed_rows, rows, L = {}, [], [], []
    per_query = {}
    say(L, f"Repeats: seeds {CONFIG['seeds']} (fold split and XGBoost seed). Gains are averaged per query "
           f"over seeds; CIs bootstrap over queries; 'seeds a..b' is the range of per seed mean gains.")
    say(L, "")
    for rd in readers():
        df, col = load_reader(rd)
        say(L, f"== reader {rd}  (label column {col})")
        dss = sorted(df["dataset"].unique()) if CONFIG["datasets"] == "all" else CONFIG["datasets"]
        pooled = {}
        for ds in dss:
            B = Block(df, ds)
            blocks[(rd, ds)] = B
            F = decision_features(B)
            acc = B.Y.mean()
            for p in B.plans:
                fixed_rows.append({"reader": rd, "dataset": ds, "plan": p, "success": acc[p],
                                   "cost_s": B.cost[p], "retrieval_s": B.ret_cost[p], "reader_s": B.agent_cost[p]})
            best = acc.idxmax()
            say(L, f"  {ds}: {len(B.q)} queries, {len(B.plans)} plans; best fixed {best} {acc[best]:.3f} at "
                     f"{B.cost[best]:.2f}s; closed book {acc.get('closed_book@0', np.nan):.3f}")
            makers = {"planner_tsr": lambda: m_planner_tsr(B, F), "adaptive_rag": lambda: m_adaptive_rag(B, F),
                      "adaptive_k": lambda: m_adaptive_k(B, F), "gate_closed_book": lambda: m_gate_closed_book(B, F),
                      "conf_cascade": lambda: m_conf_cascade(B, F)}
            for name, mk in makers.items():
                print(f"    {name} ... ({time.time() - t0:.0f}s elapsed)", flush=True)
                res = repeated(B, mk)
                if res is None:
                    continue
                d, seed_means, halves, cfgs, qids = res
                lo, hi = boot_ci(d, CONFIG["n_boot"], CONFIG["seed"])
                pooled.setdefault(name, []).append(d)
                per_query.setdefault((rd, name), {})[ds] = pd.Series(d, index=qids)
                rows.append({"reader": rd, "dataset": ds, "method": name, "gain": d.mean(), "ci_low": lo,
                             "ci_high": hi, "p": p_two_sided(d),
                             "seed_min": seed_means.min(), "seed_max": seed_means.max(),
                             "success": np.mean([h["success"] for h in halves]),
                             "cost_s": np.mean([h["cost"] for h in halves]),
                             "gain_selection_half": np.mean([h["gain_selection_half"] for h in halves]),
                             "configs": str(cfgs[0])})
                say(L, f"    {name:17s} gain vs hull {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                         f"seeds {seed_means.min():+.3f}..{seed_means.max():+.3f}  "
                         f"(training folds {rows[-1]['gain_selection_half']:+.3f})  cost {rows[-1]['cost_s']:.2f}s")
        for name, ds_list in pooled.items():
            d = np.concatenate(ds_list)
            lo, hi = boot_ci(d, CONFIG["n_boot"], CONFIG["seed"])
            rows.append({"reader": rd, "dataset": "pooled", "method": name, "gain": d.mean(), "ci_low": lo, "ci_high": hi})
            say(L, f"  POOLED {name:17s} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(d)}")
        say(L, "")

    # cross reader cascades
    for A, Bn in CONFIG["reader_pairs"]:
        dss = sorted({ds for (r, ds) in blocks if r == A} & {ds for (r, ds) in blocks if r == Bn})
        if not dss:
            continue
        say(L, f"== reader cascade {A} -> {Bn} (hull over both readers' fixed plans)")
        ds_d = []
        for ds in dss:
            print(f"    {ds} ... ({time.time() - t0:.0f}s elapsed)", flush=True)
            if reader_cascade(blocks, A, Bn, ds) is None:
                continue
            Bj = reader_cascade(blocks, A, Bn, ds)[0]
            d, seed_means, halves, cfgs, qids = repeated(Bj, lambda: reader_cascade(blocks, A, Bn, ds)[1])
            per_query.setdefault((f"{A}->{Bn}", "reader_cascade"), {})[ds] = pd.Series(d, index=qids)
            lo, hi = boot_ci(d, CONFIG["n_boot"], CONFIG["seed"])
            ds_d.append(d)
            rows.append({"reader": f"{A}->{Bn}", "dataset": ds, "method": "reader_cascade", "gain": d.mean(),
                         "ci_low": lo, "ci_high": hi, "p": p_two_sided(d),
                         "seed_min": seed_means.min(), "seed_max": seed_means.max(),
                         "success": np.mean([h["success"] for h in halves]),
                         "cost_s": np.mean([h["cost"] for h in halves]),
                         "gain_selection_half": np.mean([h["gain_selection_half"] for h in halves])})
            say(L, f"    {ds:16s} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  seeds {seed_means.min():+.3f}..{seed_means.max():+.3f}  "
                   f"(training folds {rows[-1]['gain_selection_half']:+.3f})")
        if ds_d:
            d = np.concatenate(ds_d)
            lo, hi = boot_ci(d, CONFIG["n_boot"], CONFIG["seed"])
            say(L, f"  POOLED reader_cascade {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            rows.append({"reader": f"{A}->{Bn}", "dataset": "pooled", "method": "reader_cascade",
                         "gain": d.mean(), "ci_low": lo, "ci_high": hi})
        say(L, "")

    # ---- Holm correction over every dataset level test (method x reader x dataset, plus reader cascades)
    M = pd.DataFrame(rows)
    lvl = M[M["dataset"] != "pooled"].index
    M["p_holm"] = np.nan
    M.loc[lvl, "p_holm"] = holm(M.loc[lvl, "p"].to_numpy())
    say(L, f"== Holm correction over {len(lvl)} dataset level tests (t test on per query gains)")
    sig = M.loc[lvl][M.loc[lvl, "p"] < 0.05].sort_values("p")
    if sig.empty:
        say(L, "  no test has unadjusted p < 0.05")
    for _, r in sig.iterrows():
        say(L, f"  {r['reader']:24s} {r['dataset']:16s} {r['method']:17s} gain {r['gain']:+.3f}  "
               f"p {r['p']:.4f}  Holm p {r['p_holm']:.4f}  {'SURVIVES' if r['p_holm'] < 0.05 else 'does not survive'}")
    say(L, "")

    # ---- paired tests between methods: same queries, gain difference, pooled over datasets per reader
    say(L, "== Paired method comparisons (pooled per reader; difference of per query gains; Holm over all pairs)")
    prs = []
    for rd in readers():
        names = [n for (r, n) in per_query if r == rd]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                dss = sorted(set(per_query[(rd, a)]) & set(per_query[(rd, b)]))
                if not dss:
                    continue
                diff = np.concatenate([(per_query[(rd, a)][ds_] - per_query[(rd, b)][ds_]).dropna().to_numpy()
                                       for ds_ in dss])
                lo, hi = boot_ci(diff, CONFIG["n_boot"], CONFIG["seed"])
                prs.append({"reader": rd, "a": a, "b": b, "diff": diff.mean(), "ci_low": lo, "ci_high": hi,
                            "p": p_two_sided(diff), "n": len(diff)})
    P = pd.DataFrame(prs)
    if len(P):
        P["p_holm"] = holm(P["p"].to_numpy())
        for _, r in P.iterrows():
            flag = "*" if r["p_holm"] < 0.05 else " "
            say(L, f"  {flag} {r['reader']:12s} {r['a']:17s} minus {r['b']:17s} {r['diff']:+.3f} "
                   f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  Holm p {r['p_holm']:.4f}")
        P.to_csv(out / "paired.csv", index=False)
    say(L, "  (* = Holm adjusted p < 0.05)")
    say(L, "")

    pd.DataFrame(fixed_rows).to_csv(out / "fixed_plans.csv", index=False)
    M.to_csv(out / "methods.csv", index=False)
    # per query gains (seed averaged), long format: reader, dataset, method, query_id, gain
    pq = [pd.DataFrame({"reader": rd_, "dataset": ds_, "method": nm_, "query_id": sr.index.astype(str),
                        "gain": sr.to_numpy()})
          for (rd_, nm_), by_ds in per_query.items() for ds_, sr in by_ds.items()]
    pd.concat(pq, ignore_index=True).to_csv(out / "per_query_gains.csv", index=False)
    say(L, "Gain = held out success minus the hull of fixed plans at the same mean cost. A method beats "
             "the hull only if the whole CI is above 0. 'training folds' shows how much of the in sample "
             "gain survives on unseen queries.")
    (out / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.0f}s. Written {out / 'summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
