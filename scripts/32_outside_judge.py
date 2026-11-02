"""
Regrade of the ambiguous band with a judge from outside the reader pool. GPU (llama.cpp) for stage judge; CPU for report and replay.

Concern: the judge, Qwen3 14B, is also one of the five readers, and Qwen3 32B is from the same family. LLM judges can
favour their own outputs (Panickssery et al., NeurIPS 2024). Every answer of the ambiguous band (grading/judge_queue.csv,
written by 07_grading_full.py: the main labels, the fixed two hop plans, IRCoT and the quantisation control) is judged
again by Mistral Small 3.1 24B Instruct, a model from no reader family, with the same instruction, the same yes/no
reading and the same threshold (P(yes) >= 0.5).

Start the server first (GGUF of Mistral-Small-3.1-24B-Instruct-2503 at about 4 bits, e.g. Q4_K_M, about 14 GB):
   llama-server.exe -m %USERPROFILE%\\models\\gguf\\Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf -ngl 99 -c 4096 -np 1 -fa on --jinja --port 8080

Stages:
  judge    GPU: P(yes) for every queue item from the first token's top 20 log probabilities (yes/Yes against no/No,
           as 07). Resumable. grading/judge_scores_mistral.csv. About 12,000 items, roughly 30 to 60 minutes.
  report   CPU: agreement of the two judges (decisions, by reader, and for the Qwen3 readers against the others);
           both judges against the human majority of audits 1 and 2 on the audited answers in the band (sensitivity,
           specificity, J); label means per reader under each judge.
  replay   CPU, about 1 hour: the primary, secondary and stronger families in both plan sets with label_final rebuilt
           from the Mistral judge (unchanged outside the band); corrected test; compared with the paper's gains.

Usage:  python scripts/32_outside_judge.py judge [--url http://127.0.0.1:8080]
        python scripts/32_outside_judge.py report
        python scripts/32_outside_judge.py replay
Output: data/processed_full/grading/judge_scores_mistral.csv; results_full/analysis/outside_judge/
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
STAGE = ARGS[0] if ARGS else ""
URL = ARGS[ARGS.index("--url") + 1] if "--url" in ARGS else "http://127.0.0.1:8080"
GDIR = settings.PROCESSED_DATA_ROOT / "grading"
SCORES = GDIR / "judge_scores_mistral.csv"
OUT = settings.RESULTS_ROOT / "analysis" / "outside_judge"
THRESHOLD = 0.5


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


# ====================================================================== judge
def stage_judge():
    import requests
    G = _load("grading07", "07_grading_full.py")
    try:
        props = requests.get(f"{URL}/props", timeout=10).json()
    except Exception as e:
        sys.exit(f"No llama-server at {URL} ({e}). Start it with the command in this script's header.")
    path = str(props.get("model_path", ""))
    if "mistral" not in path.lower():
        sys.exit(f"The server runs {path}; start it with the Mistral Small 3.1 GGUF.")
    print(f"server: {path}")
    q = pd.read_csv(G.QUEUE, dtype={"query_id": str}, keep_default_na=False)
    done = set(pd.read_csv(SCORES, keep_default_na=False).pair) if SCORES.exists() else set()
    todo = q[~q.pair.isin(done)]
    print(f"queue {len(q)}, done {len(done)}, to judge {len(todo)}")
    s = requests.Session()
    rows, t0 = [], time.time()
    for k, r in enumerate(todo.itertuples()):
        user = (f"Question: {r.question}\nGold answers: {r.gold}\nModel answer: {r.answer}\n"
                f"Is the model answer correct?")
        body = {"messages": [{"role": "system", "content": G.JUDGE_SYSTEM}, {"role": "user", "content": user}],
                "max_tokens": 1, "temperature": 0.0, "seed": 0, "logprobs": True, "top_logprobs": 20,
                "cache_prompt": False, "stream": False}
        j = s.post(f"{URL}/v1/chat/completions", json=body, timeout=600).json()
        ch = j["choices"][0]
        top = ((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []
        py = sum(np.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
        pn = sum(np.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
        rows.append({"pair": r.pair, "p_yes": py / (py + pn) if py + pn > 0 else np.nan, "yes_no_mass": py + pn,
                     "first_token": (ch["message"].get("content") or "").strip()})
        if len(rows) == 100 or k == len(todo) - 1:
            pd.DataFrame(rows).to_csv(SCORES, mode="a", header=not SCORES.exists(), index=False)
            rows = []
            print(f"  {k + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    S = pd.read_csv(SCORES)
    print(f"Wrote {SCORES}: {len(S)} scores; yes/no mass median {S.yes_no_mass.median():.3f}; "
          f"{int(S.p_yes.isna().sum())} without a yes or no token in the top 20")


# ====================================================================== labels under the Mistral judge
def mistral_map():
    S = pd.read_csv(SCORES, keep_default_na=False).drop_duplicates("pair", keep="last")
    S["p_yes"] = pd.to_numeric(S.p_yes, errors="coerce")
    return S.set_index("pair").p_yes


def relabel(df, pmap):
    """label_final with the Mistral judge: its decision in the ambiguous band where it has a score; elsewhere the label
    stays as it was (the strict rule outside the band)."""
    from tsr.grader import normalize
    pair = df.dataset + "::" + df.query_id.astype(str) + "::" + df.answer.astype(str).map(normalize)
    p = pair.map(pmap)
    amb = pd.to_numeric(df.ambiguous, errors="coerce").fillna(0).astype(int) == 1
    return np.where(amb & p.notna(), (p >= THRESHOLD).astype(int), df.y.astype(int)).astype(int), amb & p.notna()


def stage_report():
    OUT.mkdir(parents=True, exist_ok=True)
    G = _load("grading07", "07_grading_full.py")
    q = pd.read_csv(G.QUEUE, dtype={"query_id": str}, keep_default_na=False)
    Jq = pd.read_csv(G.JUDGE, keep_default_na=False).drop_duplicates("pair", keep="last").set_index("pair").p_yes
    Jm = mistral_map()
    q["qwen"] = pd.to_numeric(q.pair.map(Jq), errors="coerce")
    q["mistral"] = q.pair.map(Jm)
    b = q.dropna(subset=["qwen", "mistral"]).copy()
    b["dq"], b["dm"] = (b.qwen >= THRESHOLD).astype(int), (b.mistral >= THRESHOLD).astype(int)
    L = [f"Ambiguous band: {len(q)} unique answers; both judges scored {len(b)}.",
         f"Decisions agree on {(b.dq == b.dm).mean():.3f}; Cohen's kappa {kappa(b.dq, b.dm):.3f}; "
         f"rated correct: Qwen3 14B {b.dq.mean():.3f}, Mistral Small 3.1 {b.dm.mean():.3f}."]
    rows = []
    for rd in sorted({x.strip() for r in b.reader for x in str(r).split(",")}):
        g = b[b.reader.astype(str).map(lambda r: rd in [x.strip() for x in r.split(",")])]
        rows.append({"reader": rd, "n": len(g), "agree": (g.dq == g.dm).mean(), "qwen_yes": g.dq.mean(),
                     "mistral_yes": g.dm.mean(), "qwen_minus_mistral": g.dq.mean() - g.dm.mean()})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "judges_by_reader.csv", index=False)
    L += ["\nBy reader of the answer (a Qwen3 judge favouring Qwen3 readers would show a larger qwen_minus_mistral "
          "for the Qwen3 readers):", R.round(3).to_string(index=False)]
    # against the human majority on the audited answers in the band
    H = pd.read_csv(GDIR / "audit_adjudicated.csv", dtype={"query_id": str}, keep_default_na=False)
    h = H.merge(q[["pair", "qwen", "mistral"]], on="pair", how="inner").dropna(subset=["qwen", "mistral"])
    out = []
    for nm in ("qwen", "mistral"):
        y, f = h.human.astype(int).to_numpy(), (h[nm] >= THRESHOLD).astype(int).to_numpy()
        se, sp = ((y == 1) & (f == 1)).sum() / max(1, (y == 1).sum()), ((y == 0) & (f == 0)).sum() / max(1, (y == 0).sum())
        out.append({"judge": nm, "n": len(h), "agreement": (y == f).mean(), "sensitivity": se, "specificity": sp,
                    "J": se + sp - 1})
    HJ = pd.DataFrame(out)
    HJ.to_csv(OUT / "judges_vs_humans.csv", index=False)
    L += [f"\nBoth judges against the human majority of audits 1 and 2, on the {len(h)} audited answers in the band:",
          HJ.round(3).to_string(index=False)]
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ====================================================================== replay
def stage_replay():
    OUT.mkdir(parents=True, exist_ok=True)
    K = _load("inference27", "27_inference_refit.py")
    A = K.A
    pmap = mistral_map()
    orig_load = A.load_reader
    changed = []

    def load_mistral(reader):
        df, col = orig_load(reader)
        y, used = relabel(df, pmap)
        changed.append({"reader": reader, "r3": A.CONFIG["include_r3"], "rows": len(df), "in_band_scored": int(used.sum()),
                        "label_changes": int((df.y.to_numpy() != y).sum()), "mean_before": df.y.mean(), "mean_after": y.mean()})
        df["y"] = y
        return df, col

    A.load_reader = load_mistral
    for m_ in K.MODS:
        m_.load_reader = load_mistral
    try:
        K.OUT = OUT                     # 27's part nb writes its files here
        K.LINES.clear()
        K.part_nb()
    finally:
        A.load_reader = orig_load
    C = pd.DataFrame(changed).drop_duplicates(["reader", "r3"])
    C.to_csv(OUT / "label_changes.csv", index=False)
    T = pd.read_csv(OUT / "tests_nb.csv")
    ref = pd.read_csv(settings.RESULTS_ROOT / "analysis" / "inference" / "tests_nb.csv")
    T = T.merge(ref[["plan_set", "reader", "method", "gain"]].rename(columns={"gain": "gain_qwen_judge"}),
                on=["plan_set", "reader", "method"], how="left")
    T.to_csv(OUT / "tests_mistral_vs_qwen.csv", index=False)
    L = ["Label changes when the Mistral judge replaces the Qwen3 judge:", C.round(4).to_string(index=False), ""]
    for (tag, fam), g in T.groupby(["plan_set", "family"], sort=False):
        pos = g[(g.p_nb_holm < 0.05) & (g.nb_low > 0)]
        d = g.gain - g.gain_qwen_judge
        L.append(f"{tag:12s} {fam:9s} {len(g):2d} tests: gain change median {d.median():+.4f} (range {d.min():+.4f} to "
                 f"{d.max():+.4f}); positive {len(pos)}"
                 + (f" ({', '.join(f'{r.reader} {r.method} {r.gain:+.3f}' for r in pos.itertuples())})" if len(pos) else "")
                 + f"; upper bound below 0.03: {int((g.ub95_nb < 0.03).sum())}")
    (OUT / "replay_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("Note: the reproduction check in this replay compares against the Qwen3 judge labels, so differences there "
          "are expected; the uncharged embedding routers (plan_emb, plan_knn) appear without their own cost.")


if __name__ == "__main__":
    {"judge": stage_judge, "report": stage_report, "replay": stage_replay}.get(STAGE, lambda: sys.exit(__doc__))()
