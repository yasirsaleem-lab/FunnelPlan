"""
Mixed workload condition. CPU only; no new model calls.

Motivation: every hull and every policy in the main protocol is fitted per reader and dataset, so the baseline knows
which dataset each question comes from. On mixed traffic a policy must make that choice from the question itself.
Here the 2,000 questions of each reader are pooled. Every question keeps its own corpus and its stored answers.
Folds are split across the pool, and ONE hull and each policy are fitted on the pooled training folds, with no
dataset indicator.

Two baselines, both chosen on the pooled training folds and read on the held out fold at the policy's mean cost:
  global hull  the best random mix of two fixed plans for the whole pool. It knows nothing about the question.
  label hull   the best mix of fixed plans PER DATASET under one pooled budget. It knows only which dataset each
               question comes from. Each dataset's hull is fitted on that dataset's training questions, and the
               budget goes to the dataset hull segments in order of slope. This is the best fixed plan policy that
               uses the dataset label and nothing else about the question.
So: gain over the global hull = workload part (label hull minus global hull at the policy's cost)
                               + per question part (gain over the label hull).

Plan cost C(p) is the mean measured cost of plan p over the pooled questions (the paper's definition, applied to the
pooled workload). Both arms and both hulls use it. Gains per question are success minus the fold's hull success,
averaged over the 5 seeds, exactly as in 11_analysis_full.py.

Policies (each sees the same question features as in the per dataset runs; no dataset indicator is added):
  primary     planner_tsr, adaptive_rag, adaptive_k, gate_closed_book, conf_cascade (5 readers) and the two reader
              cascades: 27 tests per plan set, Holm over 27.
  secondary   ak_paper, ar_emb, plan_emb, plan_knn and the METIS port (and ar_t5 with --t5, GPU): Holm over the family.
  stronger    clf_gate, planner_sk: Holm over 10.
  ar_pub      Adaptive RAG trained on the pool as published (ar_tfidf of 12_routers.py): silver labels from the
              stored answers (simplest rung that answers); the dataset label is used ONLY to label questions that no
              rung answers (NQ to the single retrieval rung, multi hop data to the strongest rung). Holm over 5.
Decision rule as in the paper: a policy wins only with Holm p < 0.05 and CI above 0 in BOTH plan sets.

Curve part (--parts curve), after the main and extra runs: plan_emb and plan_knn, without and with their own
  cost, trained on 20, 40, 60, 80% of the pooled training folds (about 320 to 1,280 questions; 100% = 1,600 comes
  from the main and extra runs). Against the label hull the gain is per question skill only, so the curve shows
  whether more training data buys per question routing. Output: curve.csv, summary_curve.txt.

Extra part (--parts extra), after the main run:
  own_cost      plan_emb and plan_knn with their own cost: --overhead_ms (default 24.6 ms, 25_latency_e2e.py) per
                question, except when the chosen plan embeds the query with BGE-M3 anyway; lambda tuned with these
                costs (as 24_robustness_checks.py --parts overhead). Holm over 10.
  ds_label      a dataset classifier plus the label hull: the question's dataset is PREDICTED from its BGE-M3
                embedding, and the question runs the label hull's mix for that dataset. It knows nothing else about
                the question. Without (ds_label) and with (ds_label_own) the embedding cost. Holm over 5 each.

Usage:  python scripts/26_mixed_workload.py               both plan sets, every family except ar_t5 (about 2 to 3 h)
        python scripts/26_mixed_workload.py --t5          adds ar_t5 (GPU; one FLAN-T5 fine-tune per fold and seed)
        python scripts/26_mixed_workload.py --quick       small run to test the code (2 readers, 1 seed, single step)
        python scripts/26_mixed_workload.py --parts extra own cost and dataset classifier (about 15 to 25 minutes)
        python scripts/26_mixed_workload.py --parts curve learning curve of the embedding routers (about 1 hour)
        --plan-sets single,r3   --readers qwen2.5-7b,qwen3-32b
Output: results_full/analysis/mixed/  tests.csv, per_query.csv, by_dataset.csv, folds.csv, headroom.csv, summary.txt
        (--parts extra: tests_extra.csv, per_query_extra.csv, by_dataset_extra.csv, folds_extra.csv, summary_extra.txt)
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
WITH_T5 = "--t5" in ARGS
PLAN_SETS = arg("--plan-sets", "single" if QUICK else "single,r3").split(",")
PART = arg("--parts", "main")                    # "main" (the run above) or "extra" (own cost and dataset classifier)
OWN_MS = float(arg("--overhead_ms", "24.6"))     # router's own cost per question (25_latency_e2e.py)
MARGIN = 0.03


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


R = _load("routers12", "12_routers.py")
A = R.A
S20 = _load("stronger20", "20_stronger_policies.py")
try:
    M13 = _load("metis13", "13_metis_profiler.py")
    METIS_PROFILES = M13.load_profiles()
except Exception as e:                                    # profiles or llama.cpp helpers missing
    M13, METIS_PROFILES = None, None
    print(f"METIS port skipped: {e}", flush=True)
MODS = [A, S20.A] + ([M13.A] if M13 is not None else [])   # every copy of 11 whose seed the policies read

if QUICK:
    A.CONFIG["seeds"] = [0]
READERS = arg("--readers", "qwen2.5-7b,llama3.1-8b" if QUICK else "")
READERS = [r for r in READERS.split(",") if r] or A.readers()
OUT = settings.RESULTS_ROOT / "analysis" / ("mixed_quick" if QUICK else "mixed")
OUT.mkdir(parents=True, exist_ok=True)

FAMILY = {"planner_tsr": "primary", "adaptive_rag": "primary", "adaptive_k": "primary", "gate_closed_book": "primary",
          "conf_cascade": "primary", "reader_cascade": "primary",
          "ak_paper": "secondary", "ar_emb": "secondary", "plan_emb": "secondary", "plan_knn": "secondary",
          "metis": "secondary", "ar_t5": "secondary",
          "clf_gate": "stronger", "planner_sk": "stronger", "ar_tfidf": "ar_pub",
          "plan_emb_own": "own_cost", "plan_knn_own": "own_cost",
          "ds_label": "ds_label", "ds_label_own": "ds_label_own"}
SUFFIX = {"extra": "_extra", "curve": "_curve"}.get(PART, "")
FRACS = [0.2, 0.4, 0.6, 0.8]                     # curve part; 1.0 is the main (uncharged) and extra (charged) run
CLF_ACC = []

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def ds_of(ids):
    return np.array([str(x).split("|", 1)[0] for x in ids])


# ====================================================================== the pooled block
class Pooled:
    """The four (reader, dataset) blocks of one reader as ONE block. Question ids become '<dataset>|<id>'.
    Same attributes as A.Block, so every policy of 11, 12, 13 and 20 runs on it unchanged."""

    rows = A.Block.rows
    plan_cost = A.Block.plan_cost

    def __init__(self, blocks):
        plans = [p for p in blocks[0].plans if all(p in b.plans for b in blocks[1:])]
        pre = lambda b: [f"{b.ds}|{q}" for q in b.Y.index]
        self.ds = "pooled"
        self.plans = plans
        self.Y = pd.concat([b.Y[plans].set_axis(pre(b), axis=0) for b in blocks])
        self.conf = pd.concat([b.conf[plans].reindex(b.Y.index).set_axis(pre(b), axis=0) for b in blocks])
        n = np.array([len(b.Y) for b in blocks], float)
        self.ret_cost = sum(b.ret_cost[plans] * k for b, k in zip(blocks, n)) / n.sum()
        self.agent_cost = sum(b.agent_cost[plans] * k for b, k in zip(blocks, n)) / n.sum()
        self.cost = self.ret_cost + self.agent_cost
        self.q = np.array(self.Y.index)
        self.retr_of = {p: p.split("@")[0] for p in plans}
        self.depth_of = {p: int(p.split("@")[1]) for p in plans}
        common = set.intersection(*[set(b.cache) for b in blocks])
        self.cache = {r: {"queries": {f"{b.ds}|{q}": b.cache[r]["queries"].get(q, "") for b in blocks for q in b.q},
                          "results": {f"{b.ds}|{q}": b.cache[r]["results"][q] for b in blocks for q in b.q
                                      if q in b.cache[r]["results"]}} for r in common}
        self.text = {f"{b.ds}|{q}": b.text[q] for b in blocks for q in b.q}
        self.Yn = self.Y.to_numpy(dtype=float)
        self.Cn = self.conf.reindex(index=self.Y.index, columns=self.Y.columns).to_numpy(dtype=float)
        self.pcol = {p: i for i, p in enumerate(self.Y.columns)}
        self.qrow = {q: i for i, q in enumerate(self.Y.index)}
        self._rows = {}


def stack(frames, blocks):
    return pd.concat([f.set_axis([f"{b.ds}|{q}" for q in f.index], axis=0) for f, b in zip(frames, blocks)])


# Adaptive RAG's silver labels on the pool: the dataset label is used only for questions that no rung answers
_silver_per_dataset = R.silver


def silver_pooled(B, sel, rungs):
    if getattr(B, "ds", "") != "pooled":
        return _silver_per_dataset(B, sel, rungs)
    fb = np.where(np.isin(ds_of(sel), list(R.MULTI_HOP)), len(rungs) - 1, len(rungs) - 2)
    Ys = B.Y.loc[sel, list(rungs)].to_numpy()
    return np.where(Ys.max(1) == 1, Ys.argmax(1), fb)


R.silver = silver_pooled


# ====================================================================== the label hull
def allocate(H, w, c):
    """Spread the pooled budget c over the dataset hulls H (dataset -> vertices [(cost, success, plan)]) with
    dataset weights w: segments in order of slope. Returns dataset -> (vertex j, fraction t toward vertex j + 1)."""
    dss = sorted(H)
    budget = c - sum(w[d] * H[d][0][0] for d in dss)
    segs = sorted(((H[d][j + 1][1] - H[d][j][1]) / max(H[d][j + 1][0] - H[d][j][0], 1e-12), d, j)
                  for d in dss for j in range(len(H[d]) - 1))[::-1]
    at = {d: (0, 0.0) for d in dss}
    for _, d, j in segs:
        if budget <= 0:
            break
        L = w[d] * (H[d][j + 1][0] - H[d][j][0])
        if L <= budget:
            at[d], budget = (j + 1, 0.0), budget - L
        else:
            at[d], budget = (j, budget / L), 0.0
    return at


def label_hull(Y, cost, sel, c, test):
    """Best mix of fixed plans per dataset under one pooled budget c: each dataset's hull vertices are chosen on its
    training questions in sel; the budget goes to the dataset hull segments in order of slope (training success
    per second), weighted by each dataset's share of the test questions, so the mean cost on `test` is c. Below the
    cheapest point every dataset runs its cheapest vertex; above the top every dataset runs its best plan.
    Returns the mean success on `test` and the per question value."""
    dsel, dtest = ds_of(sel), ds_of(test)
    dss = sorted(set(dtest))
    w = {d: float(np.mean(dtest == d)) for d in dss}
    H = {d: A._hull_vertices(Y, cost, sel[dsel == d]) for d in dss}
    at = allocate(H, w, c)
    vals = np.empty(len(test))
    for d in dss:
        m = dtest == d
        j, t = at[d]
        v = Y.loc[test[m], H[d][j][2]].to_numpy(float)
        if t > 0:
            v = (1 - t) * v + t * Y.loc[test[m], H[d][j + 1][2]].to_numpy(float)
        vals[m] = v
    return float(vals.mean()), vals


# ====================================================================== extra part: own cost, dataset classifier
BGE_RETRIEVERS = {"bge_m3", "bge_bm25_rrf", "qwen3_reranker_0_6b"}   # as 24_robustness_checks.py


def eff_cost(B, plans, delta):
    """Plan costs as an embedding policy pays them: + delta unless the plan's own retrieval embeds the query with
    BGE-M3 anyway (as 24_robustness_checks.py). The hulls keep the plain plan costs."""
    c = B.cost[plans].to_numpy().astype(float)
    return c + delta * np.array([0.0 if p.split("@")[0] in BGE_RETRIEVERS else 1.0 for p in plans])


def _emb_predict(Xtr, Ytr, Xte):
    """R.m_plan_emb's model: PCA 64, one logistic regression per plan."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    pca = PCA(n_components=min(R.CONFIG["pca_dim"], len(Xtr) - 1), random_state=0).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    P = np.empty((len(Xte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        yj = Ytr[:, j]
        P[:, j] = yj[0] if yj.min() == yj.max() else \
            LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, yj).predict_proba(Zte)[:, 1]
    return P


def _knn_predict(Xtr, Ytr, Xte, exclude_self=False):
    """R.m_plan_knn's model: smoothed mean success of the k nearest training questions."""
    k, a = R.CONFIG["knn_k"], R.CONFIG["knn_prior_weight"]
    S = Xte @ Xtr.T
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    return (Ytr[nn].sum(1) + a * Ytr.mean(0)[None, :]) / (k + a)


def m_plan_own(B, E, kind, delta):
    """R.m_plan_emb (kind 'emb') or R.m_plan_knn (kind 'knn') with the router's own cost in its plan costs: lambda is
    tuned and the policy's cost is measured with these costs. With delta = 0 this is R's router exactly."""
    plans = B.plans
    ce = eff_cost(B, plans, delta)
    cols = [B.pcol[p] for p in plans]

    def fit(sel):
        X, Y = E.loc[sel].to_numpy(), B.Y.loc[sel, plans].to_numpy()
        if kind == "emb":
            oof = np.empty(Y.shape)
            for te in R.inner_splits(len(sel), A._CUR_SEED[0]):
                tr = np.setdiff1d(np.arange(len(sel)), te)
                oof[te] = _emb_predict(X[tr], Y[tr], X[te])
        else:
            oof = _knn_predict(X, Y, X, exclude_self=True)
        r = B.rows(sel)
        best = None
        for lam in A.CONFIG["lambdas"]:
            k = (oof - lam * ce[None, :]).argmax(1)
            g = B.Yn[r, [cols[i] for i in k]].mean() - A.hull_at(B.Y, B.cost, sel, float(ce[k].mean()))
            if best is None or g > best[0]:
                best = (g, lam)
        return (np.asarray(sel), best[1])

    def apply_(cfg, idx):
        tr, lam = cfg
        f = _emb_predict if kind == "emb" else _knn_predict
        P = f(E.loc[tr].to_numpy(), B.Y.loc[tr, plans].to_numpy(), E.loc[idx].to_numpy())
        k = (P - lam * ce[None, :]).argmax(1)
        return B.Yn[B.rows(idx), [cols[i] for i in k]], float(ce[k].mean())
    return fit, apply_


def m_ds_label(B, E, delta):
    """A dataset classifier plus the label hull. A classifier (PCA 64 + multinomial logistic regression on the BGE-M3
    question embedding) predicts each question's dataset; the question then runs the label hull's plan mix for the
    PREDICTED dataset. The label hull (per dataset vertices on the training questions, budget spread by slope with the
    training shares of the predicted labels) and its target budget are chosen on the training folds, using cross
    fitted predictions there. Success and cost are the expectations of the mix, as for the hulls. This policy knows
    nothing about a question beyond its predicted dataset, so it measures how much of a router's gain a 4 way dataset
    classifier already gives. delta: own cost, charged as for the routers."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    plans = B.plans
    ce = dict(zip(plans, eff_cost(B, plans, delta)))
    grid = np.linspace(float(B.cost.min()), float(B.cost[B.Y.mean().idxmax()]) * 1.05, 41)

    def train(sel):
        X = E.loc[sel].to_numpy()
        pca = PCA(n_components=min(R.CONFIG["pca_dim"], len(X) - 1), random_state=0).fit(X)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(pca.transform(X), ds_of(sel))
        return pca, clf

    def predict(m, idx):
        pca, clf = m
        return clf.predict(pca.transform(E.loc[idx].to_numpy()))

    def run(H, at, dhat, idx):
        r = B.rows(idx)
        s, c = np.empty(len(idx)), np.empty(len(idx))
        for d in H:
            m = dhat == d
            j, t = at[d]
            pa = H[d][j][2]
            pb = H[d][min(j + 1, len(H[d]) - 1)][2]
            s[m] = (1 - t) * B.Yn[r[m], B.pcol[pa]] + t * B.Yn[r[m], B.pcol[pb]]
            c[m] = (1 - t) * ce[pa] + t * ce[pb]
        return s, float(c.mean())

    def fit(sel):
        sel = np.asarray(sel)
        oof = np.empty(len(sel), dtype=object)
        for te in R.inner_splits(len(sel), A._CUR_SEED[0]):
            tr = np.setdiff1d(np.arange(len(sel)), te)
            oof[te] = predict(train(sel[tr]), sel[te])
        oof = oof.astype(str)
        dsel = ds_of(sel)
        H = {d: A._hull_vertices(B.Y, B.cost, sel[dsel == d]) for d in sorted(set(dsel))}
        w = {d: float(np.mean(oof == d)) for d in H}
        best = None
        for c in grid:
            at = allocate(H, w, c)
            s, cost = run(H, at, oof, sel)
            g = s.mean() - A.hull_at(B.Y, B.cost, sel, cost)
            if best is None or g > best[0]:
                best = (g, at)
        return (train(sel), H, best[1], len(sel))

    def apply_(cfg, idx):
        m, H, at, n_sel = cfg
        dhat = predict(m, idx)
        if len(idx) < n_sel:                                 # held out fold only
            CLF_ACC.append(float(np.mean(dhat == ds_of(idx))))
        return run(H, at, dhat, idx)
    return fit, apply_


# ====================================================================== protocol (as A.heldout / A.repeated)
def heldout(B, method, seed):
    fit, apply_ = method
    k = A.CONFIG["k_folds"]
    rng = np.random.default_rng(seed)
    q = B.q.copy()
    rng.shuffle(q)
    folds = [q[i::k] for i in range(k)]
    G, L, rows = [], [], []
    for i in range(k):
        test = folds[i]
        sel = np.concatenate([folds[j] for j in range(k) if j != i])
        cfg = fit(sel)
        s, c = apply_(cfg, test)
        s = np.asarray(s, float)
        hg = A.hull_at(B.Y, B.cost, sel, c, eval_idx=test)
        hl, _ = label_hull(B.Y, B.cost, sel, c, test)
        s_sel, c_sel = apply_(cfg, sel)
        rows.append({"seed": seed, "fold": i, "success": s.mean(), "cost": c, "hull_global": hg, "hull_label": hl,
                     "gain_global": s.mean() - hg, "gain_label": s.mean() - hl,
                     "gain_training_folds": np.mean(s_sel) - A.hull_at(B.Y, B.cost, sel, c_sel)})
        G.append(pd.Series(s - hg, index=test))
        L.append(pd.Series(s - hl, index=test))
    return pd.concat(G), pd.concat(L), rows


def repeated(B, make):
    G, L, rows = [], [], []
    for sd in A.CONFIG["seeds"]:
        for m_ in MODS:
            m_._CUR_SEED[0] = sd
        m = make()
        if m is None:
            return None
        g, l_, r = heldout(B, m, sd)
        G.append(g.sort_index())
        L.append(l_.sort_index())
        rows += r
    for m_ in MODS:
        m_._CUR_SEED[0] = 0
    return pd.concat(G, axis=1).mean(axis=1), pd.concat(L, axis=1).mean(axis=1), rows


def hull_gap(B, n_grid=11):
    """Descriptive: label hull minus global hull on held out folds, over a grid of budgets from the cheapest plan
    to the most successful plan (pooled costs), mean over folds and seeds."""
    c = np.linspace(B.cost.min(), B.cost[B.Y.mean().idxmax()], n_grid)
    acc = np.zeros(n_grid)
    n = 0
    for sd in A.CONFIG["seeds"]:
        rng = np.random.default_rng(sd)
        q = B.q.copy()
        rng.shuffle(q)
        k = A.CONFIG["k_folds"]
        folds = [q[i::k] for i in range(k)]
        for i in range(k):
            test = folds[i]
            sel = np.concatenate([folds[j] for j in range(k) if j != i])
            acc += [label_hull(B.Y, B.cost, sel, x, test)[0] - A.hull_at(B.Y, B.cost, sel, x, eval_idx=test) for x in c]
            n += 1
    return c, acc / n


# ====================================================================== run one plan set
def run_plan_set(r3):
    A.CONFIG["include_r3"] = r3
    tag = "with two hop" if r3 else "single step"
    say(f"\n######## plan set: {tag}")
    t0 = time.time()
    pooled, perq, folds, gaps = {}, [], [], []

    def record(rd, name, res):
        if res is None:
            return
        g, l_, rows = res
        perq.append(pd.DataFrame({"plan_set": tag, "family": FAMILY.get(name, "curve"), "reader": rd, "method": name,
                                  "dataset": ds_of(g.index), "query_id": [str(x).split("|", 1)[1] for x in g.index],
                                  "gain_global": g.to_numpy(), "gain_label": l_.reindex(g.index).to_numpy()}))
        folds.extend({"plan_set": tag, "reader": rd, "method": name, **r} for r in rows)
        say(f"    {name:17s} global {g.mean():+.4f}  label {l_.mean():+.4f}  cost "
            f"{np.mean([r['cost'] for r in rows]):.3f}s  ({time.time() - t0:.0f}s)")

    for rd in READERS:
        df, col = A.load_reader(rd)
        dss = sorted(df["dataset"].unique())
        blocks = [A.Block(df, ds) for ds in dss]
        P = Pooled(blocks)
        pooled[rd] = P
        say(f"== {rd}: {len(P.q)} questions from {dss}, {len(P.plans)} common plans; best pooled plan "
            f"{P.Y.mean().idxmax()} {P.Y.mean().max():.3f} at {P.cost[P.Y.mean().idxmax()]:.2f}s")
        F = stack([A.decision_features(b) for b in blocks], blocks).fillna(0.0)
        X = stack([S20.cb_features(df, b.ds, b.q) for b in blocks], blocks)
        F2 = F.join(X.add_prefix("cb_"))
        E = pd.concat([R.load_emb(b.ds).reindex([str(q) for q in b.q]).set_axis([f"{b.ds}|{q}" for q in b.q], axis=0)
                       for b in blocks])
        assert E.notna().all().all(), "missing embeddings; run 12_routers.py stage embed"
        makers = {"planner_tsr": lambda: A.m_planner_tsr(P, F), "adaptive_rag": lambda: A.m_adaptive_rag(P, F),
                  "adaptive_k": lambda: A.m_adaptive_k(P, F), "gate_closed_book": lambda: A.m_gate_closed_book(P, F),
                  "conf_cascade": lambda: A.m_conf_cascade(P, F),
                  "ar_tfidf": lambda: R.m_ar_tfidf(P),
                  "ak_paper": lambda: R.m_ak_paper(P), "ar_emb": lambda: R.m_ar_emb(P, E),
                  "plan_emb": lambda: R.m_plan_emb(P, E), "plan_knn": lambda: R.m_plan_knn(P, E),
                  "clf_gate": lambda: S20.m_clf_gate(P, X), "planner_sk": lambda: S20.m_planner_sk(P, F2)}
        if M13 is not None:
            prof = {("pooled", f"{d}|{q}"): v for (d, q), v in METIS_PROFILES.items()}
            makers["metis"] = lambda: M13.m_metis(P, prof, False)
        if WITH_T5:
            t5 = R.T5Router(rd)
            t5.cache_file = R.DIR / f"t5_cache_{rd}{'_r3' if r3 else ''}_mixed.json"
            t5.cache = __import__("json").load(open(t5.cache_file)) if t5.cache_file.exists() else {}
            makers["ar_t5"] = lambda: R.m_ar_t5(P, t5)
        if PART == "extra":
            dl = OWN_MS / 1000.0
            makers = {"plan_emb_own": lambda: m_plan_own(P, E, "emb", dl),
                      "plan_knn_own": lambda: m_plan_own(P, E, "knn", dl),
                      "ds_label": lambda: m_ds_label(P, E, 0.0), "ds_label_own": lambda: m_ds_label(P, E, dl)}
        if PART == "curve":
            dl = OWN_MS / 1000.0
            makers = {}
            for fr in FRACS:
                for kind in ("emb", "knn"):
                    for own in (0.0, dl):
                        makers[f"plan_{kind}{'_own' if own else ''}@{fr}"] = \
                            (lambda kind=kind, own=own, fr=fr: frac_method(m_plan_own(P, E, kind, own), fr))
        for name, mk in makers.items():
            n0 = len(CLF_ACC)
            record(rd, name, repeated(P, mk))
            if len(CLF_ACC) > n0:
                say(f"      dataset classifier accuracy on held out folds: {np.mean(CLF_ACC[n0:]):.4f}")
        if PART in ("extra", "curve"):
            continue
        c, gap = hull_gap(P)
        gaps += [{"plan_set": tag, "reader": rd, "budget_s": x, "label_minus_global": y} for x, y in zip(c, gap)]
        say(f"    label hull minus global hull, held out: max {gap.max():+.4f} at {c[gap.argmax()]:.2f}s; "
            f"at the most successful plan {gap[-1]:+.4f}")

    for a_, b_ in A.CONFIG["reader_pairs"]:
        if a_ in pooled and b_ in pooled and PART not in ("extra", "curve"):
            blocks = {(a_, "pooled"): pooled[a_], (b_, "pooled"): pooled[b_]}
            say(f"== reader cascade {a_} -> {b_}")
            if A.reader_cascade(blocks, a_, b_, "pooled") is None:
                continue
            Bj = A.reader_cascade(blocks, a_, b_, "pooled")[0]
            record(f"{a_}->{b_}", "reader_cascade",
                   repeated(Bj, lambda: A.reader_cascade(blocks, a_, b_, "pooled")[1]))
    return pd.concat(perq, ignore_index=True), pd.DataFrame(folds), pd.DataFrame(gaps)


# ====================================================================== tests
def tests(PQ, FO):
    from scipy import stats
    rows = []
    for (tag, fam, rd, m), g in PQ.groupby(["plan_set", "family", "reader", "method"], sort=False):
        fo = FO[(FO.plan_set == tag) & (FO.reader == rd) & (FO.method == m)]
        for comp in ("global", "label"):
            d = g[f"gain_{comp}"].to_numpy(float)
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n)
            lo, hi = A.boot_ci(d, A.CONFIG["n_boot"], A.CONFIG["seed"])
            rows.append({"plan_set": tag, "family": fam, "comparator": comp, "reader": rd, "method": m,
                         "gain": d.mean(), "ci_low": lo, "ci_high": hi, "p": A.p_two_sided(d),
                         "ub95": d.mean() + stats.t.ppf(0.95, n - 1) * se, "n": n,
                         "success": fo.success.mean(), "cost_s": fo.cost.mean(),
                         "gain_training_folds": fo.gain_training_folds.mean()})
    T = pd.DataFrame(rows)
    T["p_holm"] = np.nan
    for _, idx in T.groupby(["plan_set", "family", "comparator"]).groups.items():
        T.loc[idx, "p_holm"] = A.holm(T.loc[idx, "p"].to_numpy())
    T["positive"] = (T.p_holm < 0.05) & (T.ci_low > 0)
    T["negative"] = (T.p_holm < 0.05) & (T.ci_high < 0)
    return T


def reference_gains():
    """Gains of the same policies under the per dataset protocol (pooled over datasets), for comparison."""
    out = []
    for tag, d in (("single step", "final"), ("with two hop", "final_r3")):
        for f in ("per_query_gains.csv", "routers_per_query.csv"):
            p = settings.RESULTS_ROOT / "analysis" / d / f
            if p.exists():
                X = pd.read_csv(p, dtype={"query_id": str})
                out.append(X.groupby(["reader", "method"]).gain.mean().rename("per_dataset_gain").reset_index()
                           .assign(plan_set=tag))
    return pd.concat(out, ignore_index=True).drop_duplicates(["plan_set", "reader", "method"]) if out else None


def frac_method(method, frac):
    """Learning curve: the policy trains (and tunes) on a seeded subset of the training folds; the hulls keep the
    full training folds, as in 24_robustness_checks.py --parts curve."""
    fit, apply_ = method

    def fit2(sel):
        sel = np.asarray(sel)
        rng = np.random.default_rng(A._CUR_SEED[0] + 7)
        return fit(sel[np.sort(rng.permutation(len(sel))[: max(20, int(round(frac * len(sel))))])])
    return fit2, apply_


def curve_summary(T):
    """Gain against both hulls by training size, with the full size (1.0) taken from the main and extra runs."""
    T = T.copy()
    T["frac"] = T.method.str.split("@").str[1].astype(float)
    T["method"] = T.method.str.split("@").str[0]
    full = []
    for f in ("tests.csv", "tests_extra.csv"):
        if (OUT / f).exists():
            X = pd.read_csv(OUT / f)
            full.append(X[X.method.isin(["plan_emb", "plan_knn", "plan_emb_own", "plan_knn_own"])].assign(frac=1.0))
    C = pd.concat([T] + full, ignore_index=True)[["plan_set", "comparator", "reader", "method", "frac", "gain", "ci_low", "ci_high"]]
    n_pool = 2000 - 2000 // A.CONFIG["k_folds"]
    C["n_train"] = (C.frac * n_pool).round().astype(int)
    C.to_csv(OUT / "curve.csv", index=False)
    say("\n######## learning curve on the mixed workload: mean gain over readers by training size (pooled training folds "
        f"of about {n_pool} questions; the hulls always use the full training folds)")
    for comp in ("label", "global"):
        say(f"\n== against the {comp} hull" + (" (per question skill: the workload part is removed)" if comp == "label" else ""))
        say(C[C.comparator == comp].pivot_table(index=["plan_set", "method"], columns="n_train", values="gain", aggfunc="mean")
            .round(4).to_string())
    say("\n== against the label hull, per reader (plan_emb, no own cost)")
    say(C[(C.comparator == "label") & (C.method == "plan_emb")].pivot_table(index=["plan_set", "reader"], columns="n_train",
                                                                            values="gain").round(4).to_string())


def main():
    t0 = time.time()
    say(f"Mixed workload condition. Readers {READERS}; seeds {A.CONFIG['seeds']}; {A.CONFIG['k_folds']} folds over "
        f"the pooled questions; plan sets {PLAN_SETS}; ar_t5 {'on' if WITH_T5 else 'off'}; METIS "
        f"{'on' if M13 is not None else 'off'}.")
    PQ, FO, GA = [], [], []
    for ps in PLAN_SETS:
        a, b, c = run_plan_set(ps == "r3")
        PQ.append(a)
        FO.append(b)
        GA.append(c)
    PQ, FO = pd.concat(PQ, ignore_index=True), pd.concat(FO, ignore_index=True)
    GA = pd.concat(GA, ignore_index=True) if PART not in ("extra", "curve") else pd.DataFrame()
    T = tests(PQ, FO)
    ref = reference_gains()
    if ref is not None:
        T = T.merge(ref, on=["plan_set", "reader", "method"], how="left")
    PQ.to_csv(OUT / f"per_query{SUFFIX}.csv", index=False)
    FO.to_csv(OUT / f"folds{SUFFIX}.csv", index=False)
    if PART not in ("extra", "curve"):
        GA.to_csv(OUT / "headroom.csv", index=False)
    T.to_csv(OUT / f"tests{SUFFIX}.csv", index=False)
    BD = PQ.groupby(["plan_set", "reader", "method", "dataset"])[["gain_global", "gain_label"]].mean().reset_index()
    BD.to_csv(OUT / f"by_dataset{SUFFIX}.csv", index=False)

    if PART == "curve":
        curve_summary(T)
        say(f"\nDone in {time.time() - t0:.0f}s. Output: {OUT}")
        (OUT / f"summary{SUFFIX}{'_quick' if QUICK else ''}.txt").write_text("\n".join(LINES), encoding="utf-8")
        return
    say("\n######## tests (pooled questions, n per test as shown; Holm within plan set, family and comparator)")
    for (tag, fam, comp), g in T.groupby(["plan_set", "family", "comparator"], sort=False):
        say(f"\n== {tag} | {fam} | against the {comp} hull: {len(g)} tests, positive {int(g.positive.sum())}, "
            f"negative {int(g.negative.sum())}, upper bound below {MARGIN}: {int((g.ub95 < MARGIN).sum())}")
        for r in g.sort_values("gain", ascending=False).itertuples():
            ref_s = f"  per dataset protocol {r.per_dataset_gain:+.3f}" if pd.notna(getattr(r, "per_dataset_gain", np.nan)) else ""
            say(f"  {r.reader:24s} {r.method:17s} {r.gain:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  Holm p "
                f"{r.p_holm:.4f}  ub {r.ub95:.3f}  cost {r.cost_s:.2f}s  {'POSITIVE' if r.positive else ''}"
                f"{'negative' if r.negative else ''}{ref_s}")
    W = T.pivot_table(index=["plan_set", "reader", "method"], columns="comparator", values="gain").reset_index()
    W["workload_part"] = W["global"] - W["label"]
    say("\n######## decomposition: gain over global hull = workload part + gain over label hull (means over readers)")
    say(W.groupby(["plan_set", "method"])[["global", "workload_part", "label"]].mean().round(4).to_string())
    if len(PLAN_SETS) == 2:
        say("\n######## decision rule: positive (Holm p < 0.05, CI above 0) in BOTH plan sets")
        for comp in ("global", "label"):
            k = [set(map(tuple, T[(T.plan_set == ps) & (T.comparator == comp) & T.positive][["family", "reader", "method"]]
                          .values)) for ps in ("single step", "with two hop")]
            say(f"  against the {comp} hull: {sorted(k[0] & k[1]) or 'none'}")
    if PART == "extra":
        main_t = OUT / "tests.csv"
        if main_t.exists():
            M = pd.read_csv(main_t)
            M = M[M.method.isin(["plan_emb", "plan_knn"])].assign(method=lambda x: x.method + "_own")
            J = T.merge(M[["plan_set", "comparator", "reader", "method", "gain"]], on=["plan_set", "comparator", "reader", "method"],
                        suffixes=("", "_no_own_cost"))
            say("\n######## routers: gain without (main run) and with their own cost")
            say(J[["plan_set", "comparator", "reader", "method", "gain_no_own_cost", "gain", "p_holm"]].round(4).to_string(index=False))
        say(f"\nDone in {time.time() - t0:.0f}s. Output: {OUT}")
        (OUT / f"summary{SUFFIX}{'_quick' if QUICK else ''}.txt").write_text("\n".join(LINES), encoding="utf-8")
        return
    say("\n######## label hull minus global hull on held out folds (descriptive), by budget")
    say(GA.pivot_table(index=["plan_set", "reader"], values="label_minus_global", aggfunc=["max", "last"])
        .round(4).to_string())
    say(f"\nDone in {time.time() - t0:.0f}s. Output: {OUT}")
    (OUT / f"summary{SUFFIX}{'_quick' if QUICK else ''}.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
