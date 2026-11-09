"""
Live end to end latency check. GPU; run on the machine that made the labels.

The paper scores policies by replay: a policy's cost is the mean of the stored plan costs it chooses. This script
checks that replay against a live run, for one reader on the datasets given (default MuSiQue and 2Wiki).

Part A  cells     A random sample of (question, plan) cells, every plan about equally often. Each cell runs live
                  3 times in random order after a warm up. Timed: the retrieval call (a depth 100 search, as when
                  the labels were made), prompt building, tokenisation, generation and decoding. Compared with the
                  stored retrieval_time_s + agent_time_s of that cell.
Part B  policies  Closed book gate, confidence cascade, and per plan logistic regression on BGE-M3 question
                  embeddings (plan_emb, the strongest router in the paper). Each uses the settings it chose on its
                  training folds (seed 0, 5 folds, the same split and fitting code as scripts 11 and 12). Every
                  question is answered live from its text to the decoded answer, including the policy's own
                  signal: the closed book call, the confidence check, the query embedding and the classifiers.
                  Compared with (i) the replayed cost the paper uses (mean of plan mean costs) and (ii) the stored
                  per question cost of the same choices.
Part C  swap      (--swap) Seconds to unload the 7B reader and load Qwen3 14B (int8), and back: the extra cost of
                  the reader cascade 7B -> Qwen3 14B on one GPU, where the two readers do not fit together.

The reader stays loaded. Retrievers are loaded next to it when a group of questions needs them and released after;
loading is not timed. Each row records whether a dense index sat on the GPU or fell back to the CPU. plan_emb runs
in two timed passes per question (embedding and decision, then retrieval and answer), because all six retrievers
and the reader do not fit in GPU memory at once; its live cost is the sum of the two passes.

Usage:  python scripts/25_latency_e2e.py                        MuSiQue and 2Wiki, Qwen2.5 7B (about 1.5 hours)
        python scripts/25_latency_e2e.py --datasets hotpotqa    the large corpus (dense indexes may not fit)
        python scripts/25_latency_e2e.py --swap                 also Part C
        python scripts/25_latency_e2e.py --dry                  CPU only: fits the policies, prints the groups
Output: results_full/latency_e2e/<reader>_<dataset>_cells.csv, _policies.csv, summary_<datasets>.txt
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# torch must load BEFORE numpy, pandas and scikit-learn. On Windows, if scikit-learn's OpenMP and MKL DLLs load
# first, torch fails with "WinError 127 ... fbgemm.dll". The other GPU scripts import torch early for the same reason.
try:
    import torch  # noqa: F401
except ImportError:
    if "--dry" not in sys.argv:
        raise

import numpy as np
import pandas as pd

from config import settings

CONFIG = {
    "reader": "qwen2.5-7b",
    "datasets": ["musique", "2wikimultihopqa"],
    "n_questions": 200,        # Part B: questions per dataset (a random subset of the 500, all folds)
    "n_cells": 100,            # Part A: cells per dataset
    "cell_repeats": 3,
    "warmup": 3,               # untimed calls before each timed group
    "seed": 0,                 # fold split and policy settings: seed 0 of scripts 11 and 12
    "sample_seed": 7,          # which questions and cells are sampled
    "swap_to": "qwen3-14b",
}
if "--datasets" in sys.argv:
    CONFIG["datasets"] = sys.argv[sys.argv.index("--datasets") + 1].split(",")
DRY = "--dry" in sys.argv
OUT = settings.RESULTS_ROOT / "latency_e2e"
OUT.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("routers12", Path(__file__).parent / "12_routers.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
A = R.A                                   # the protocol module (11) that 12 uses

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def retr(p):
    return p.split("@")[0]


def depth(p):
    return int(p.split("@")[1])


# ======================================================================
# offline side: blocks, folds, fitted settings, replayed costs
# ======================================================================
def stored_times(df, ds, B):
    d = df[df["dataset"] == ds]
    rt = d.pivot_table(index="query_id", columns="plan", values="retrieval_time_s").reindex(index=B.q, columns=B.plans)
    at = d.pivot_table(index="query_id", columns="plan", values="agent_time_s").reindex(index=B.q, columns=B.plans)
    ans = d.pivot_table(index="query_id", columns="plan", values="answer", aggfunc="first").reindex(index=B.q, columns=B.plans)
    return rt.fillna(0.0), at, ans


def folds_of(B, seed):
    """The exact split of A.heldout for this seed."""
    rng = np.random.default_rng(seed)
    q = B.q.copy()
    rng.shuffle(q)
    k = A.CONFIG["k_folds"]
    return [q[i::k] for i in range(k)]


def plan_models(B, E, tr):
    """Same models as R.m_plan_emb's train_predict, fitted once per fold so serving only predicts."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    Xtr = E.loc[tr].to_numpy()
    Ytr = B.Y.loc[tr, B.plans].to_numpy()
    pca = PCA(n_components=min(R.CONFIG["pca_dim"], len(Xtr) - 1), random_state=0).fit(Xtr)
    Ztr = pca.transform(Xtr)
    models = []
    for j in range(Ytr.shape[1]):
        yj = Ytr[:, j]
        models.append(float(yj[0]) if yj.min() == yj.max()
                      else LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, yj))
    return pca, models


def plan_predict(pm, x):
    pca, models = pm
    z = pca.transform(x.reshape(1, -1))
    return np.array([m if isinstance(m, float) else m.predict_proba(z)[0, 1] for m in models])


def fit_all(B, E):
    """policy -> list over folds of (test question ids, setting). Seed 0, as in 11 and 12."""
    A._CUR_SEED[0] = CONFIG["seed"]
    folds = folds_of(B, CONFIG["seed"])
    makers = {"gate_closed_book": lambda: A.m_gate_closed_book(B, None),
              "conf_cascade": lambda: A.m_conf_cascade(B, None),
              "plan_emb": lambda: R.m_plan_emb(B, E)}
    out = {}
    for name, mk in makers.items():
        m = mk()
        if m is None:
            continue
        fit, apply_ = m
        rows = []
        for i, test in enumerate(folds):
            sel = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
            cfg = fit(sel)
            if name == "plan_emb":
                tr, lam = cfg
                cfg = {"lam": lam, "models": plan_models(B, E, tr)}
            else:
                cfg = tuple(float(x) if isinstance(x, np.floating) else x for x in cfg)
            rows.append((set(test), cfg))
        out[name] = rows
    return out


def cfg_for(rows, q):
    for test, cfg in rows:
        if q in test:
            return cfg
    raise KeyError(q)


# ======================================================================
# live side
# ======================================================================
class Live:
    def __init__(self, ds):
        from data.data_loader import get_processed
        from data.corpus_utils import build_corpus_list
        from tsr.agent_runner import AgentRunner
        import torch
        self.torch = torch
        corpus, self.queries, _, _ = get_processed(ds)
        self.corpus_list = build_corpus_list(corpus)
        self.agent = AgentRunner(reader=CONFIG["reader"])
        self.agent.set_corpus(corpus)
        del corpus
        self.agent.load()
        self.loaded = {}
        self.nofit = set()

    def sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def free(self):
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()

    def need(self, names):
        """Keep exactly the retrievers in names loaded (closed book needs none)."""
        from retrievers.registry import build_retriever
        names = {n for n in names if n != "closed_book"}
        for n in list(self.loaded):
            if n not in names:
                r = self.loaded.pop(n)
                if hasattr(r, "cleanup"):
                    r.cleanup()
                del r
                self.free()
        for n in sorted(names - set(self.loaded)):
            r = build_retriever(n)
            self.loaded[n] = r
            r.build_index(self.corpus_list)

    def group(self, names, fn):
        """Load the retrievers in names next to the reader and run fn(). If they do not fit in GPU memory together
        with the reader (load or search runs out of memory), record that, release everything and return None."""
        names = {n for n in names if n != "closed_book"}
        try:
            self.need(names)
            return fn()
        except self.torch.OutOfMemoryError as e:
            self.nofit.update(names)
            say(f"    does not fit next to the reader on this GPU: {sorted(names)} ({str(e).splitlines()[0][:90]})")
            for n in list(self.loaded):
                r = self.loaded.pop(n)
                if hasattr(r, "cleanup"):
                    try:
                        r.cleanup()
                    except Exception:
                        pass
                del r
            self.free()
            return None

    def device(self, name):
        if name == "closed_book":
            return ""
        from retrievers.dense.emb_cache import DenseIndex
        seen, stack, devs = set(), [self.loaded[name]], []
        while stack:
            o = stack.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            if isinstance(o, DenseIndex):
                devs.append("gpu" if o.gpu is not None else "cpu")
            elif hasattr(o, "__dict__"):
                stack += [v for v in vars(o).values() if hasattr(v, "__dict__")]
        return "+".join(sorted(set(devs))) or "no dense index"

    def search(self, name, q):
        return self.loaded[name].search(q, 100)

    def answer(self, q, hits, k):
        return self.agent.answer_with_meta(q, hits[:k] if k else [])

    def run_plan(self, p, q, reuse=None):
        """Retrieval (unless closed book or reused) + answer. Returns (answer dict, hits, retrieval seconds)."""
        t = time.perf_counter()
        hits = [] if retr(p) == "closed_book" else (reuse if reuse is not None else self.search(retr(p), q))
        self.sync()
        t_ret = time.perf_counter() - t
        return self.answer(q, hits, depth(p)), hits, t_ret


def conf_of(out):
    v = out["mean_logprob"]
    return -99.0 if v is None or not np.isfinite(v) else float(v)


def part_a(live, B, rt, at, ans, ds, rng):
    plans = list(B.plans)
    per = int(np.ceil(CONFIG["n_cells"] / len(plans)))
    cells = [(q, p) for p in plans for q in rng.choice(B.q, per, replace=False)]
    rng.shuffle(cells)
    cells = cells[:CONFIG["n_cells"]]
    rows = []
    for rname in sorted({retr(p) for _, p in cells}):
        group = [(q, p) for q, p in cells if retr(p) == rname]

        def run():
            dev, got = live.device(rname), []
            for q, p in group[:CONFIG["warmup"]]:
                live.run_plan(p, live.queries[q]["text"])
            for rep in range(CONFIG["cell_repeats"]):
                order = list(group)
                rng.shuffle(order)
                for q, p in order:
                    text = live.queries[q]["text"]
                    live.sync()
                    t0 = time.perf_counter()
                    out, _, t_ret = live.run_plan(p, text)
                    live.sync()
                    got.append({"dataset": ds, "query_id": q, "plan": p, "repeat": rep, "device": dev,
                                "live_total_s": time.perf_counter() - t0, "live_retrieval_s": t_ret,
                                "live_generate_s": out["agent_time_s"],
                                "stored_retrieval_s": float(rt.at[q, p]), "stored_agent_s": float(at.at[q, p]),
                                "same_answer": str(out["answer"]).strip() == str(ans.at[q, p]).strip()})
            return got, dev
        res = live.group([rname], run)
        if res is None:
            say(f"    cells {rname:20s} {len(group):3d} cells SKIPPED (does not fit next to the reader)")
            continue
        rows += res[0]
        say(f"    cells {rname:20s} {len(group):3d} cells x {CONFIG['cell_repeats']}  index {res[1]}")
    return pd.DataFrame(rows)


def part_b(live, B, rt, at, ans, fits, qs, ds):
    rows = []
    # ---- closed book gate and confidence cascade: one continuous timed run per question
    for name in ("gate_closed_book", "conf_cascade"):
        if name not in fits:
            continue
        groups = {}
        for q in qs:
            groups.setdefault(cfg_for(fits[name], q), []).append(q)
        for cfg, group in groups.items():
            if name == "gate_closed_book":
                tau, p = cfg
                first, second = "closed_book@0", p
            else:
                first, second, tau = cfg
            needs = {retr(first), retr(second)}

            def run():
                dev = "+".join(live.device(n) for n in sorted(needs) if live.device(n))
                got = []
                for i, q in enumerate(group[:CONFIG["warmup"]] + group):
                    text = live.queries[q]["text"]
                    live.sync()
                    t0 = time.perf_counter()
                    out1, hits, _ = live.run_plan(first, text)
                    go = conf_of(out1) < tau
                    if go:
                        same = name == "conf_cascade" and retr(first) == retr(second)
                        live.run_plan(second, text, reuse=hits if same else None)
                    live.sync()
                    dt = time.perf_counter() - t0
                    if i < min(CONFIG["warmup"], len(group)):
                        continue
                    conf_st = B.Cn[B.qrow[q], B.pcol[first]]
                    go_st = bool((-99.0 if not np.isfinite(conf_st) else conf_st) < tau)
                    reuse = name == "conf_cascade" and retr(first) == retr(second)
                    c_second = float(B.plan_cost(second, already={retr(first)})) if name == "conf_cascade" \
                        else float(B.cost[second])
                    st_second = float(at.at[q, second]) + (0.0 if reuse else float(rt.at[q, second]))
                    got.append({"dataset": ds, "policy": name, "query_id": q, "setting": str(cfg), "device": dev,
                                "live_s": dt, "live_escalated": go, "stored_escalated": go_st,
                                "replay_s": float(B.cost[first]) + go_st * c_second,
                                "stored_s": float(rt.at[q, first] + at.at[q, first]) + go_st * st_second,
                                "overhead_s": np.nan})
                return got, dev
            res = live.group(needs, run)
            if res is None:
                say(f"    {name:17s} setting {cfg}  {len(group)} questions SKIPPED (does not fit next to the reader)")
                continue
            rows += res[0]
            say(f"    {name:17s} setting {cfg}  {len(group)} questions  index {res[1]}")
    # ---- plan_emb: pass 1 embedding + decision, pass 2 retrieval + answer
    if "plan_emb" in fits:
        from sentence_transformers import SentenceTransformer
        live.need([])
        st = SentenceTransformer(R.CONFIG["embed_model"], device="cuda")
        c = B.cost[B.plans].to_numpy()
        stored_E = R.load_emb(ds)
        choice, t1, cos = {}, {}, {}
        for i, q in enumerate(qs[:CONFIG["warmup"]] + qs):
            cfg = cfg_for(fits["plan_emb"], q)
            text = live.queries[q]["text"]
            live.sync()
            t0 = time.perf_counter()
            x = st.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
            P = plan_predict(cfg["models"], x.astype(np.float32))
            k = int((P - cfg["lam"] * c).argmax())
            live.sync()
            if i >= min(CONFIG["warmup"], len(qs)):
                choice[q], t1[q] = B.plans[k], time.perf_counter() - t0
                cos[q] = float(x @ stored_E.loc[str(q)].to_numpy())
        del st
        live.free()
        stored_choice = {}
        for q in qs:
            cfg = cfg_for(fits["plan_emb"], q)
            P = plan_predict(cfg["models"], stored_E.loc[str(q)].to_numpy())
            stored_choice[q] = B.plans[int((P - cfg["lam"] * c).argmax())]
        for rname in sorted({retr(p) for p in choice.values()}):
            group = [q for q in qs if retr(choice[q]) == rname]

            def run():
                dev, got = live.device(rname), []
                for i, q in enumerate(group[:CONFIG["warmup"]] + group):
                    text = live.queries[q]["text"]
                    live.sync()
                    t0 = time.perf_counter()
                    live.run_plan(choice[q], text)
                    live.sync()
                    dt = time.perf_counter() - t0
                    if i < min(CONFIG["warmup"], len(group)):
                        continue
                    p = stored_choice[q]
                    got.append({"dataset": ds, "policy": "plan_emb", "query_id": q, "setting": choice[q],
                                "device": dev, "live_s": t1[q] + dt, "live_escalated": choice[q] == p,
                                "stored_escalated": True, "replay_s": float(B.cost[p]),
                                "stored_s": float(rt.at[q, p] + at.at[q, p]), "overhead_s": t1[q],
                                "embedding_cosine": cos[q]})
                return got, dev
            res = live.group([rname], run)
            if res is None:
                say(f"    plan_emb          plans on {rname:20s} {len(group)} questions SKIPPED (does not fit next to the reader)")
                continue
            rows += res[0]
            say(f"    plan_emb          plans on {rname:20s} {len(group)} questions  index {res[1]}")
    if live.nofit:
        say(f"    retrievers that do not fit next to the reader: {sorted(live.nofit)}")
    return pd.DataFrame(rows)


def part_c():
    from tsr.agent_runner import AgentRunner
    import torch
    res = {}
    for src, dst in ((CONFIG["reader"], CONFIG["swap_to"]), (CONFIG["swap_to"], CONFIG["reader"])):
        a = AgentRunner(reader=src)
        a.load()
        a.answer_with_meta("Who wrote Hamlet?", [])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a.unload()
        del a
        torch.cuda.empty_cache()
        b = AgentRunner(reader=dst)
        b.load()
        b.answer_with_meta("Who wrote Hamlet?", [])
        torch.cuda.synchronize()
        res[f"{src} -> {dst}"] = time.perf_counter() - t0
        b.unload()
        del b
        torch.cuda.empty_cache()
    return res


# ======================================================================
def summarise_a(C):
    g = C.groupby(["query_id", "plan"]).agg(live=("live_total_s", "median"), gen=("live_generate_s", "median"),
                                             ret=("live_retrieval_s", "median"), st_ret=("stored_retrieval_s", "first"),
                                             st_agent=("stored_agent_s", "first"), same=("same_answer", "mean"),
                                             dev=("device", "first")).reset_index()
    g["stored"] = g.st_ret + g.st_agent
    g["retr"] = g.plan.map(retr)
    say(f"  Part A: {len(g)} cells, median of {CONFIG['cell_repeats']} live runs each")
    say(f"    mean live {g.live.mean():.3f}s  mean stored {g.stored.mean():.3f}s  ratio of means "
        f"{g.live.mean() / g.stored.mean():.3f}  median cell ratio {(g.live / g.stored).median():.3f}  "
        f"Spearman {g[['live', 'stored']].corr('spearman').iloc[0, 1]:.3f}")
    say(f"    generation only: live {g.gen.mean():.3f}s vs stored {g.st_agent.mean():.3f}s; outside generate() "
        f"(prompt, tokenise, decode): {(g.live - g.ret - g.gen).mean() * 1000:.1f} ms per call")
    say(f"    answers identical to the stored answers: {g.same.mean():.3f}")
    t = g.groupby("retr").agg(n=("live", "size"), live=("live", "mean"), stored=("stored", "mean"),
                              ret_live=("ret", "mean"), ret_stored=("st_ret", "mean"), dev=("dev", "first"))
    t["ratio"] = t.live / t.stored
    say(t.round(3).to_string())


def summarise_b(P):
    say("  Part B: live versus replayed cost per policy (seconds per question)")
    for name, g in P.groupby("policy"):
        say(f"    {name:17s} n={len(g):3d}  live {g.live_s.mean():.3f}  replay (paper) {g.replay_s.mean():.3f}  "
            f"stored same choices {g.stored_s.mean():.3f}  live/replay {g.live_s.mean() / g.replay_s.mean():.3f}  "
            f"decisions equal to stored {(g.live_escalated == g.stored_escalated).mean():.3f}")
        if name == "plan_emb":
            say(f"    {'':17s} policy overhead (embedding + classifiers) {g.overhead_s.mean() * 1000:.1f} ms "
                f"= {g.overhead_s.mean() / g.replay_s.mean():.3f} of the replayed cost; embedding cosine to stored "
                f"min {g.embedding_cosine.min():.4f}")


def main():
    say(f"Live end to end latency check. Reader {CONFIG['reader']}, datasets {CONFIG['datasets']}, "
        f"{CONFIG['n_questions']} questions and {CONFIG['n_cells']} cells per dataset, seed {CONFIG['seed']} settings.")
    df, col = A.load_reader(CONFIG["reader"])
    for ds in CONFIG["datasets"]:
        say(f"\n== {ds}")
        B = A.Block(df, ds)
        E = R.load_emb(ds).reindex([str(q) for q in B.q])
        E.index = B.q
        rt, at, ans = stored_times(df, ds, B)
        fits = fit_all(B, E)
        rng = np.random.default_rng(CONFIG["sample_seed"])
        qs = sorted(rng.choice(B.q, min(CONFIG["n_questions"], len(B.q)), replace=False).tolist())
        for name, rows in fits.items():
            sets = [str(cfg) if name != "plan_emb" else f"lambda {cfg['lam']}" for _, cfg in rows]
            say(f"  {name:17s} settings per fold: {sets}")
        if DRY:
            continue
        live = Live(ds)
        C = part_a(live, B, rt, at, ans, ds, rng)
        C.to_csv(OUT / f"{CONFIG['reader']}_{ds}_cells.csv", index=False)
        P = part_b(live, B, rt, at, ans, fits, qs, ds)
        P.to_csv(OUT / f"{CONFIG['reader']}_{ds}_policies.csv", index=False)
        live.need([])
        live.agent.unload()
        del live
        summarise_a(C)
        summarise_b(P)
    if "--swap" in sys.argv and not DRY:
        say("\n== Part C: reader swap on one GPU (unload, load, first answer)")
        for k, v in part_c().items():
            say(f"    {k}: {v:.1f}s")
    if not DRY:
        (OUT / f"summary_{'_'.join(CONFIG['datasets'])}.txt").write_text("\n".join(LINES), encoding="utf-8")
        say(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
