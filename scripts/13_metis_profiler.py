"""
R5: a METIS style per query profiler policy (Ray et al., SOSP 2025), ported to our plan space.

METIS: a larger LLM profiles each query (joint reasoning yes/no, complexity high/low, pieces of information
1..10, confidence); rules map the profile to a pruned configuration space (num_chunks in [n, 3n]; synthesis
map_rerank / stuff / map_reduce); a scheduler picks, among pruned configurations that fit the free resources, the
one with the highest resource need; profiles with confidence < 0.9 fall back to a recent default.

Port (our plan space is retriever x depth, plus the fixed two hop plans with --r3):
  profiler  Qwen3 32B (the largest local model; METIS uses GPT-4o or Llama 3.1 70B), thinking off, greedy, the
            question only (no passages). One profile per query, shared by all readers.
  pruning   depth in [n, 3n] intersected with the labelled depths (nearest depth if empty);
            joint reasoning = no             -> single step plans  (METIS: map_rerank)
            joint reasoning = yes, low       -> single step plans  (METIS: stuff)
            joint reasoning = yes, high      -> single step plans, plus the two hop plans with --r3 (METIS: map_reduce)
            confidence < 0.9                 -> the pruned space of the median profile of the training folds
  choice    the most expensive pruned plan with mean cost <= budget b (METIS: highest memory need that fits);
            if none fits, the cheapest pruned plan. The retriever set and b are chosen on the training folds
            (b over the cost quantiles), like every other method in 11_analysis_full.py.
  cost      primary: profiler cost 0 (the most favourable case for METIS; the profiler runs in another runtime,
            so its latency cannot be added to the reader's). Secondary: plus the measured profiler latency.

Stages
  profile   needs llama-server with Qwen3 32B (as for 17). 2,000 calls. -> data/processed_full/metis/profiles.jsonl
  eval      CPU. Held out evaluation with 10's machinery (5 folds x 5 seeds, label_final, hull at equal cost).
            --r3 adds the two hop plans. -> results_full/analysis/metis[_r3]/metis.csv, summary.txt
Usage
  python scripts/13_metis_profiler.py profile
  python scripts/13_metis_profiler.py eval [--r3]
"""

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings

_s = importlib.util.spec_from_file_location("a10", Path(__file__).parent / "11_analysis_full.py")
A = importlib.util.module_from_spec(_s)
_s.loader.exec_module(A)
_s17 = importlib.util.spec_from_file_location("g17", Path(__file__).parent / "04_llamacpp_labels.py")
G = importlib.util.module_from_spec(_s17)
_s17.loader.exec_module(G)

PROFILER = "qwen3-32b"
OUT = settings.PROCESSED_DATA_ROOT / "metis"
PROFILES = OUT / "profiles.jsonl"
CONF_MIN = 0.9
PROFILE_SYSTEM = (
    "You profile questions for a retrieval augmented generation system. The database holds Wikipedia passages of "
    "about 100 words each. Analyse the language and internal structure of the question. Reply with JSON only, "
    "with these keys: \"joint_reasoning\": \"yes\" if answering needs information combined from several passages, "
    "else \"no\"; \"complexity\": \"high\" or \"low\"; \"pieces_of_information\": an integer from 1 to 10, the "
    "number of separate pieces of information that must be retrieved; \"confidence\": a number from 0 to 1, how "
    "sure you are of this profile.")


# ====================================================================== profile
def parse_profile(txt):
    m = re.search(r"\{.*\}", txt or "", flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        jr = str(d.get("joint_reasoning", "")).strip().lower().startswith("y")
        cx = "high" if str(d.get("complexity", "")).strip().lower().startswith("h") else "low"
        n = int(round(float(d.get("pieces_of_information", 1))))
        c = float(d.get("confidence", 0.0))
        return {"joint_reasoning": jr, "complexity": cx, "pieces": int(min(10, max(1, n))), "confidence": c}
    except Exception:
        return None


def stage_profile(url):
    G.check_server(url, PROFILER)
    order, qids, _ = G.design()
    caches = {ds: G.load_caches(ds, ["bm25"])["bm25"]["queries"] for ds in order}
    OUT.mkdir(parents=True, exist_ok=True)
    done = {(x["dataset"], x["query_id"]) for x in (json.loads(l) for l in open(PROFILES, encoding="utf-8")
                                                     if l.strip().endswith("}"))} if PROFILES.exists() else set()
    todo = [(ds, q) for ds in order for q in qids[ds] if (ds, q) not in done]
    print(f"profiles: {len(done)} done, {len(todo)} to do", flush=True)
    t0, n, bad = time.time(), 0, 0
    with open(PROFILES, "a", encoding="utf-8") as f:
        for ds, q in todo:
            msgs = [{"role": "system", "content": PROFILE_SYSTEM},
                    {"role": "user", "content": f"Question: {caches[ds][q]}\nJSON:"}]
            try:
                o = G.call(url, msgs, thinking_off=True)
            except Exception as e:
                print(f"  failed {ds}/{q}: {str(e)[:120]}", flush=True)
                continue
            p = parse_profile(o["answer"])
            bad += p is None
            f.write(json.dumps({"dataset": ds, "query_id": q, "raw": o["answer"], "profile": p,
                                "profiler_time_s": o["agent_time_s"]}) + "\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {len(done) + n}/{len(done) + len(todo)}  {(time.time() - t0) / n:.2f} s/query  unparsable {bad}",
                      flush=True)
    print("done", flush=True)


def load_profiles():
    rows = [json.loads(l) for l in open(PROFILES, encoding="utf-8") if l.strip().endswith("}")]
    out = {}
    for r in rows:
        p = r["profile"] or {"joint_reasoning": False, "complexity": "low", "pieces": 1, "confidence": 0.0}
        out[(r["dataset"], r["query_id"])] = p | {"t": r["profiler_time_s"]}
    return out


# ====================================================================== policy
def m_metis(B, P, charge_profiler):
    depths = sorted({B.depth_of[p] for p in B.plans if B.depth_of[p] > 0 and not A.is_multi(p)})
    single = [p for p in B.plans if B.depth_of[p] > 0 and not A.is_multi(p)]
    multi = [p for p in B.plans if A.is_multi(p)]
    prof = {q: P.get((B.ds, q)) for q in B.q}

    def allowed_depths(n):
        ok = [d for d in depths if n <= d <= 3 * n]
        return ok or [min(depths, key=lambda d: (abs(d - n), d))]

    def pruned(p, retr):
        ds_ = set(allowed_depths(p["pieces"]))
        cand = [x for x in single if B.retr_of[x] in retr and B.depth_of[x] in ds_]
        if p["joint_reasoning"] and p["complexity"] == "high":
            cand += multi
        return cand

    def median_profile(sel):
        ps = [prof[q] for q in sel if prof[q] is not None]
        return {"joint_reasoning": np.mean([x["joint_reasoning"] for x in ps]) >= 0.5,
                "complexity": "high" if np.mean([x["complexity"] == "high" for x in ps]) >= 0.5 else "low",
                "pieces": int(np.median([x["pieces"] for x in ps])), "confidence": 1.0}

    def choose(cand, b):
        fit = [x for x in cand if B.cost[x] <= b]
        return max(fit, key=lambda x: B.cost[x]) if fit else min(cand, key=lambda x: B.cost[x])

    def run(cfg, idx):
        retr, b, med = cfg
        chosen, extra = [], []
        for q in idx:
            p = prof[q]
            if p is None or p["confidence"] < CONF_MIN:
                p = med
            chosen.append(choose(pruned(p, retr), b))
            extra.append(prof[q]["t"] if (charge_profiler and prof[q] is not None) else 0.0)
        succ = np.array([B.Y.at[q, x] for q, x in zip(idx, chosen)])
        return succ, float(np.mean([B.cost[x] for x in chosen]) + np.mean(extra))

    def fit(sel):
        med = median_profile(sel)
        acc = B.Y.loc[sel].mean()
        retrs = sorted({B.retr_of[x] for x in single})
        # retriever sets tried: each single retriever, and all retrievers (METIS has one fixed retriever)
        options = [{r} for r in retrs] + [set(retrs)]
        budgets = sorted(set(np.quantile(B.cost.to_numpy(), np.linspace(0, 1, 21)).round(4)))
        best = None
        for retr in options:
            for b in budgets:
                s, c = run((retr, b, med), sel)
                g = s.mean() - A.hull_at(B.Y, B.cost, sel, c)
                if best is None or g > best[0]:
                    best = (g, (retr, b, med))
        return best[1]
    return fit, run


def stage_eval(with_r3):
    A.CONFIG["include_r3"] = with_r3
    P = load_profiles()
    out = settings.RESULTS_ROOT / "analysis" / ("metis_r3" if with_r3 else "metis")
    out.mkdir(parents=True, exist_ok=True)
    rows, L = [], []
    say = lambda s: (print(s, flush=True), L.append(s))
    pr = pd.DataFrame([v for v in P.values()])
    say(f"profiles: {len(P)}; joint reasoning {pr.joint_reasoning.mean():.2f}, high complexity "
        f"{(pr.complexity == 'high').mean():.2f}, median pieces {pr.pieces.median():.0f}, confidence >= {CONF_MIN}: "
        f"{(pr.confidence >= CONF_MIN).mean():.2f}, mean profiler time {pr.t.mean():.2f}s (Qwen3 32B, llama.cpp)")
    for rd in A.readers():
        df, col = A.load_reader(rd)
        pooled = {}
        for ds in sorted(df.dataset.unique()):
            B = A.Block(df, ds)
            for charge in (False, True):
                name = "metis" + ("_with_profiler_cost" if charge else "")
                res = A.repeated(B, lambda: m_metis(B, P, charge))
                d, sm, halves, cfgs, qids = res
                lo, hi = A.boot_ci(d, A.CONFIG["n_boot"], A.CONFIG["seed"])
                pooled.setdefault(name, []).append(d)
                rows.append({"reader": rd, "dataset": ds, "method": name, "gain": d.mean(), "ci_low": lo, "ci_high": hi,
                             "p": A.p_two_sided(d), "seed_min": sm.min(), "seed_max": sm.max(),
                             "success": np.mean([h["success"] for h in halves]),
                             "cost_s": np.mean([h["cost"] for h in halves])})
                say(f"  {rd:12s} {ds:16s} {name:26s} gain {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  cost {rows[-1]['cost_s']:.2f}s")
        for name, dl in pooled.items():
            d = np.concatenate(dl)
            lo, hi = A.boot_ci(d, A.CONFIG["n_boot"], A.CONFIG["seed"])
            rows.append({"reader": rd, "dataset": "pooled", "method": name, "gain": d.mean(), "ci_low": lo,
                         "ci_high": hi, "p": A.p_two_sided(d)})
            say(f"  POOLED {rd:12s} {name:26s} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    R = pd.DataFrame(rows)
    R.to_csv(out / "metis.csv", index=False)
    say("Pooled p values enter the secondary Holm family (with the routers of 12_routers.py) in 22_tables_final.py.")
    (out / "summary.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else ""
    if st == "profile":
        stage_profile(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--url=")), "http://127.0.0.1:8080"))
    elif st == "eval":
        stage_eval("--r3" in sys.argv)
    else:
        sys.exit("Usage: python scripts/13_metis_profiler.py profile | eval [--r3]")
