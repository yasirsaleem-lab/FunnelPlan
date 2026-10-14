"""
Fit the plan cost model.

Retrieval: t = c + a * k per retriever, non negative least squares on
latency_profiles.json. The old fit used a 3 parameter power law on curves
that are almost flat in k. That fit is unidentified: it returned values such
as b = 2e-8 or c = 6e-18 that mean nothing. A two parameter line is enough.

Agent: t = a0 + a1 * prompt_tokens, ordinary least squares on the measured
agent_time_s and prompt_tokens columns of tsr_labels_v3.csv. tokens_per_passage
is fitted from prompt_tokens against depth.

Usage:  python -m cost_model.fit_cost_model
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Corpus mode must be set BEFORE config is imported (same switch as 03 and 04).
#   "full":    one cost model per dataset (corpus sizes differ 50x), agent cost
#              per (dataset, reader); writes data/processed_full/cost_model_params.json
#   "sub200k": the pilot fit (pooled over NQ and HotpotQA)
CORPUS_MODE = "full"
import os
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE

from config import settings
from cost_model import CostModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROFILE_FILE = settings.PROCESSED_DATA_ROOT / "latency_profiles.json"
LABELS_FILE = settings.PROCESSED_DATA_ROOT / settings.LABELS_FILE


def fit_line_nonneg(x, y):
    """Least squares y = c + a x with a, c >= 0."""
    from scipy.optimize import nnls
    A = np.column_stack([np.ones_like(x), x])
    (c, a), _ = nnls(A, y)
    return float(c), float(a)


def fit_retrieval(profiles):
    """Fit only on depths inside the plan grid (<= max PLAN_DEPTHS).

    Depth 200 is outside the grid and breaks the line for the rerankers: their
    candidate pool is fixed at 100 up to depth 100 and doubles at 200, so cost
    is flat then jumps. Including 200 made the fit wrong at depths 10 to 100.
    """
    max_d = max(settings.PLAN_DEPTHS)
    out = {}
    for name, pts in profiles.items():
        pts = [p for p in pts if p["depth"] <= max_d]
        x = np.array([p["depth"] for p in pts], dtype=float)
        y = np.array([p["mean_time"] for p in pts], dtype=float)
        c, a = fit_line_nonneg(x, y)
        resid = y - (c + a * x)
        out[name] = {"c": c, "a": a, "rmse": float(np.sqrt(np.mean(resid ** 2)))}
        logger.info(f"{name:22s} c={c*1000:8.2f} ms  a={a*1e6:8.3f} us/doc  rmse={out[name]['rmse']*1000:.2f} ms")
    return out


def fit_agent(labels_path):
    import pandas as pd
    return fit_agent_df(pd.read_csv(labels_path))


def fit_agent_df(df):
    """Reader cost = a0 + a1 * prompt tokens (prefill) + a2 * generated tokens
    (decode), non negative least squares. With bf16 readers decode dominates
    the variance (about 40 ms per generated token against 0.3 ms per prompt
    token), so a prompt only line explains just 55 to 67% of it; adding
    generated tokens raises R2 to 0.81 to 0.99 (full mode, Qwen2.5 7B).
    A planner does not know the answer length in advance, so predictions use
    the mean generated length gen_mean of that (dataset, reader)."""
    need = {"prompt_tokens", "agent_time_s", "depth"}
    if not need.issubset(df.columns):
        logger.warning("Labels have no timing columns; keeping default agent estimates.")
        return None
    cols = list(need | ({"gen_tokens"} & set(df.columns)))
    df = df.dropna(subset=cols)
    x, y = df["prompt_tokens"].to_numpy(float), df["agent_time_s"].to_numpy(float)
    t0, tpp = fit_line_nonneg(df["depth"].to_numpy(float), x)
    if "gen_tokens" in df.columns:
        from scipy.optimize import nnls
        g = df["gen_tokens"].to_numpy(float)
        (a0, a1, a2), _ = nnls(np.column_stack([np.ones_like(x), x, g]), y)
        pred = a0 + a1 * x + a2 * g
        gen_mean = float(g.mean())
    else:
        a0, a1 = fit_line_nonneg(x, y)
        a2, gen_mean = 0.0, 0.0
        pred = a0 + a1 * x
    r2 = 1 - np.var(y - pred) / np.var(y)
    logger.info(f"agent: a0={a0:.3f}s a1={a1*1000:.4f} ms/prompt token a2={a2*1000:.1f} ms/gen token "
                f"(gen mean {gen_mean:.1f}) R2={r2:.3f}; prompt = {t0:.0f} + {tpp:.1f} * depth tokens")
    # Passage length differs by retriever (BM25 favours short passages: on
    # 2Wiki depth 20 gives 2,278 prompt tokens with BM25 against ~3,000 with
    # dense retrievers), so prompt length is fitted per retriever too.
    tpp_by_r = {}
    if "retriever" in df.columns:
        for r, sub in df[df["depth"] > 0].groupby("retriever"):
            rt0, rtpp = fit_line_nonneg(sub["depth"].to_numpy(float), sub["prompt_tokens"].to_numpy(float))
            tpp_by_r[r] = {"t0": rt0, "tokens_per_passage": rtpp}
    return {"a0": float(a0), "a1": float(a1), "a2": float(a2), "gen_mean": gen_mean,
            "t0": t0, "tokens_per_passage": tpp, "prompt_by_retriever": tpp_by_r, "r2": float(r2),
            "estimated": False, "n_rows": int(len(df))}


def pooled_profiles():
    """Average mean_time per (retriever, depth) over all per dataset profile files
    (latency_profiles.json = NQ, latency_profiles_<dataset>.json = others)."""
    files = sorted(settings.PROCESSED_DATA_ROOT.glob("latency_profiles*.json"))
    acc = {}
    for f in files:
        for name, pts in json.loads(f.read_text()).items():
            for p in pts:
                acc.setdefault(name, {}).setdefault(p["depth"], []).append(p["mean_time"])
    pooled = {n: [{"depth": d, "mean_time": float(np.mean(v))} for d, v in sorted(ds.items())]
              for n, ds in acc.items()}
    for n, ds in acc.items():
        counts = {len(v) for v in ds.values()}
        logger.info(f"{n:22s} profiled on {max(counts)} dataset file(s)")
    return pooled, [f.name for f in files]


def main_full():
    """One retrieval fit per dataset (from latency_profiles_<ds>.json) and one
    agent fit per (dataset, reader) (from tsr_labels_<reader>.csv). Pooling
    across datasets is wrong in full mode: SPLADE costs 49 ms on MuSiQue and
    1.3 s on HotpotQA."""
    import pandas as pd
    by_ds = {}
    readers = {}
    for f in sorted(f for f in settings.PROCESSED_DATA_ROOT.glob("tsr_labels_*.csv") if not f.stem.endswith("_graded")):
        reader = f.stem.replace("tsr_labels_", "")
        readers[reader] = pd.read_csv(f, usecols=lambda c: c in
                                      {"dataset", "retriever", "depth", "prompt_tokens", "gen_tokens", "agent_time_s"})
    for ds in settings.DATASETS:
        entry = {"retrieval": {}, "agent": {}}
        pf = settings.PROCESSED_DATA_ROOT / f"latency_profiles_{ds}.json"
        if pf.exists():
            logger.info(f"--- retrieval cost, {ds} ({pf.name})")
            entry["retrieval"] = fit_retrieval(json.loads(pf.read_text()))
        else:
            logger.warning(f"{pf.name} missing; run scripts/03_profile_latency.py")
        for reader, df in readers.items():
            sub = df[df["dataset"] == ds]
            if len(sub):
                logger.info(f"--- agent cost, {ds} / {reader} ({len(sub)} rows)")
                a = fit_agent_df(sub)
                if a:
                    entry["agent"][reader] = a
        by_ds[ds] = entry
    cm = CostModel()
    cm.by_dataset = by_ds
    cm.source = {"mode": "full", "profiles": [f"latency_profiles_{d}.json" for d in by_ds],
                 "labels": [f"tsr_labels_{r}.csv" for r in readers]}
    cm.save()
    logger.info(f"Saved {cm.params_file}")
    for ds in by_ds:
        for reader in by_ds[ds]["agent"]:
            m = CostModel(dataset=ds, reader=reader)
            logger.info(f"{ds:16s} {reader:12s} " + "  ".join(
                f"{r}@10={m.plan_cost(r, 10):.2f}s" for r in sorted(m.retrieval)))


def main():
    # "python -m cost_model.fit_cost_model" imports the cost_model package (and
    # config) before this file runs, so the switch above comes too late. If
    # config loaded the other mode, rerun in a fresh process with it set.
    if settings.CORPUS_MODE != CORPUS_MODE:
        import subprocess
        logger.info(f"config loaded {settings.CORPUS_MODE}; restarting in {CORPUS_MODE} mode")
        env = dict(os.environ, FUNNELPLAN_CORPUS_MODE=CORPUS_MODE)
        sys.exit(subprocess.run([sys.executable, "-m", "cost_model.fit_cost_model"], env=env,
                                cwd=str(Path(__file__).parent.parent)).returncode)
    if settings.CORPUS_MODE == "full":
        return main_full()
    cm = CostModel()
    if PROFILE_FILE.exists():
        profiles, names = pooled_profiles()
        cm.retrieval = fit_retrieval(profiles)
        cm.source["retrieval"] = names
    else:
        logger.error(f"{PROFILE_FILE} not found. Run scripts/03_profile_latency.py.")
    if LABELS_FILE.exists():
        agent = fit_agent(LABELS_FILE)
        if agent:
            cm.agent = agent
            cm.source["agent"] = str(LABELS_FILE.name)
    cm.save()
    logger.info(f"Saved {cm.params_file}")
    for r in sorted(cm.retrieval):
        logger.info("  " + "  ".join(f"{r}@{d}={cm.plan_cost(r, d):.2f}s" for d in settings.PLAN_DEPTHS))


if __name__ == "__main__":
    main()
