"""
Timing and quality probe for larger readers served by llama.cpp (GGUF), before committing GPU days.

Why llama.cpp: a 27B to 32B model only fits a 20 GB card at about 4 bits; GGUF IQ4_XS through llama-server is
the most reliable way to do that on Windows. The prompt is built exactly as in tsr/agent_runner.py (same system
text, same passage format, the model's own chat template, Qwen3 thinking off, greedy decoding, at most 100 new
tokens), so answers are comparable with the existing readers.

Stages
  prepare  CPU only. Samples 200 (query, plan) items from HotpotQA (40 per depth 0, 1, 3, 5, 10; random retriever),
           fetches the raw passage texts, and stores the chat messages:
           data/processed_full/probe/items_<dataset>.json
  run      needs a running llama-server (see the command printed by `python scripts/16_reader_probe.py help`).
           Sends every item once (plus 3 warm up calls), records wall time, server timings, tokens, the answer and
           its log probabilities, grades it (strict and F1 >= 0.7), and compares with Qwen3 14B int8 on the same items.
           Writes results_full/probe/<name>.csv and <name>_summary.txt, including a projection of the full grid time
           (4 datasets x 500 queries x 25 plans, depth <= 10).

Usage
  python scripts/16_reader_probe.py prepare
  python scripts/16_reader_probe.py run qwen3-32b --thinking-off
  python scripts/16_reader_probe.py run gemma3-27b
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings

# identical to tsr/agent_runner.py (copied, not imported, so this script needs no torch)
SYSTEM_RAG = ("You answer questions using only the given passages. "
              "Reply with the short answer only (a few words). "
              "If the passages do not contain the answer, give your best guess.")
SYSTEM_CLOSED = ("Answer the question from your own knowledge. "
                 "Reply with the short answer only (a few words).")
MAX_NEW_TOKENS = 100
DEPTHS = [0, 1, 3, 5, 10]
PER_DEPTH = 40
REF_READER = "qwen3-14b"
PROC = settings.PROCESSED_DATA_ROOT
CACHE = settings.CACHE_ROOT / "retrievals_v3"
PDIR = PROC / "probe"
OUT = settings.RESULTS_ROOT / "probe"
ID_RE = re.compile(r'^\{"_id":\s*"((?:[^"\\]|\\.)*)"')

HELP = r"""
1) Download llama.cpp for Windows with CUDA from https://github.com/ggml-org/llama.cpp/releases (latest):
   the zip whose name ends in  bin-win-cuda-12.4-x64.zip  and  cudart-llama-bin-win-cuda-12.4-x64.zip
   (or the newest CUDA version listed). Unzip both into one folder, e.g. C:\Users\USER\llama.cpp
2) Download the models (public repos, through the mirror):
   set HF_ENDPOINT=https://hf-mirror.com
   set HF_HUB_DISABLE_IMPLICIT_TOKEN=1
   hf download bartowski/Qwen_Qwen3-32B-GGUF --include "Qwen_Qwen3-32B-IQ4_XS.gguf" --local-dir C:\Users\USER\models\gguf
   hf download bartowski/google_gemma-3-27b-it-GGUF --include "google_gemma-3-27b-it-IQ4_XS.gguf" --local-dir C:\Users\USER\models\gguf
3) Start the server in its own Anaconda Prompt window (one model at a time), e.g. for Qwen3 32B:
   cd C:\Users\USER\llama.cpp
   llama-server.exe -m C:\Users\USER\models\gguf\Qwen_Qwen3-32B-IQ4_XS.gguf -ngl 99 -c 5632 -np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja --port 8080
   (if "-fa on" is rejected by an older build, use "-fa"). Wait for "server is listening".
   If it runs out of memory: lower -c to 5120, or add  --n-gpu-layers 60  (offloads a few layers to the CPU).
4) In a second window:
   python scripts\16_reader_probe.py run qwen3-32b --thinking-off
"""


# ======================================================================
def stage_prepare(ds="hotpotqa", seed=0):
    import pickle
    ref = pd.read_csv(PROC / f"tsr_labels_{REF_READER}_graded.csv", dtype={"query_id": str}, keep_default_na=False)
    ref = ref[(ref.dataset == ds) & (ref["repeat"].astype(int) == 0) & (ref.depth.astype(int).isin(DEPTHS))]
    ref["depth"] = ref["depth"].astype(int)
    rng = random.Random(seed)
    items = []
    for d in DEPTHS:
        sub = ref[ref.depth == d]
        rows = sub.sample(n=PER_DEPTH, random_state=seed + d)
        for r in rows.itertuples():
            items.append({"dataset": ds, "query_id": r.query_id, "retriever": r.retriever, "depth": d,
                          "ref_answer": r.answer, "ref_strict": int(r.label_strict), "ref_final": int(r.label_final),
                          "ref_prompt_tokens": int(r.prompt_tokens), "ref_agent_time_s": float(r.agent_time_s)})
    caches = {}
    need = set()
    for it in items:
        if it["depth"] == 0:
            continue
        r = it["retriever"]
        if r not in caches:
            caches[r] = pickle.load(open(CACHE / f"{ds}__{r}.pkl", "rb"))
        it["doc_ids"] = [x["id"] for x in caches[r]["results"][it["query_id"]][:it["depth"]]]
        need.update(it["doc_ids"])
    qtext = next(iter(caches.values()))["queries"]
    texts = {}
    with open(PROC / ds / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            m = ID_RE.match(line)
            if m and m.group(1) in need:
                dd = json.loads(line)
                title, text = (dd.get("title") or "").strip(), (dd.get("text") or "").strip()
                texts[dd["_id"]] = f"{title}. {text}" if title and text else (title or text)
                if len(texts) == len(need):
                    break
    for it in items:
        q = qtext[it["query_id"]]
        if it["depth"] == 0:
            msgs = [{"role": "system", "content": SYSTEM_CLOSED}, {"role": "user", "content": f"Question: {q}\nAnswer:"}]
        else:
            lines = [f"Passage {i + 1}: {texts[d]}" for i, d in enumerate(it["doc_ids"]) if texts.get(d)]
            user = "Passages:\n" + "\n".join(lines) + f"\n\nQuestion: {q}\nAnswer:"
            msgs = [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}]
        it["messages"] = msgs
    rng.shuffle(items)                      # mixed order, so warm up and drift do not bias one depth
    PDIR.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(PDIR / f"items_{ds}.json", "w", encoding="utf-8"))
    print(f"wrote {len(items)} items, {len(texts)} passages -> {PDIR / f'items_{ds}.json'}")


# ======================================================================
def vram():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, total = (float(x) for x in out.split(","))
        return used / 1024, total / 1024
    except Exception:
        return float("nan"), float("nan")


def call(url, msgs, thinking_off, timeout=600):
    import requests
    body = {"messages": msgs, "max_tokens": MAX_NEW_TOKENS, "temperature": 0.0, "top_p": 1.0, "seed": 0,
            "logprobs": True, "top_logprobs": 1, "cache_prompt": False, "stream": False}
    if thinking_off:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    content = (ch["message"].get("content") or "").strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    lps = [t.get("logprob") for t in ((ch.get("logprobs") or {}).get("content") or []) if t.get("logprob") is not None]
    tm = j.get("timings", {})
    return {"answer": content, "wall_s": wall, "prompt_tokens": j.get("usage", {}).get("prompt_tokens"),
            "gen_tokens": j.get("usage", {}).get("completion_tokens"), "prompt_ms": tm.get("prompt_ms"),
            "predicted_ms": tm.get("predicted_ms"), "prompt_per_second": tm.get("prompt_per_second"),
            "predicted_per_second": tm.get("predicted_per_second"),
            "mean_logprob": float(np.mean(lps)) if lps else float("nan"), "finish": ch.get("finish_reason")}


def stage_run(name, url, thinking_off, ds="hotpotqa", limit=None):
    import requests
    from tsr.grader import apply_rule, grade_detail
    items = json.load(open(PDIR / f"items_{ds}.json", encoding="utf-8"))[:limit]
    gold = json.load(open(PROC / ds / "answers.json", encoding="utf-8"))
    try:
        props = requests.get(f"{url}/props", timeout=10).json()
    except Exception as e:
        sys.exit(f"No llama-server at {url} ({e}). Run: python scripts/16_reader_probe.py help")
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
    model_path = props.get("model_path", "?")
    print(f"server: {model_path}  n_ctx {n_ctx}  thinking_off={thinking_off}", flush=True)
    for it in items[:3]:                                          # warm up, not recorded
        call(url, it["messages"], thinking_off)
    v0 = vram()
    rows = []
    t_start = time.time()
    for i, it in enumerate(items):
        try:
            o = call(url, it["messages"], thinking_off)
            err = ""
        except Exception as e:                                    # e.g. prompt longer than the context
            o, err = {"answer": "", "wall_s": float("nan")}, str(e)[:200]
        g = grade_detail(o["answer"], gold.get(it["query_id"], []))
        rows.append({k: it[k] for k in ("dataset", "query_id", "retriever", "depth", "ref_strict", "ref_final",
                                        "ref_prompt_tokens", "ref_agent_time_s", "ref_answer")}
                    | o | {"strict": apply_rule(g, "strict"), "f1_07": apply_rule(g, "f1_07"), "error": err})
        if (i + 1) % 20 == 0:
            done = pd.DataFrame(rows)
            print(f"  {i + 1}/{len(items)}  mean wall {done.wall_s.mean():.2f}s  strict {done.strict.mean():.3f}"
                  f"  ({time.time() - t_start:.0f}s)", flush=True)
    v1 = vram()
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / f"{name}.csv", index=False)

    ok = D[D.error == ""]
    L = [f"Probe {name}: {len(D)} items on {ds} ({len(D) - len(ok)} errors), model {model_path}, n_ctx {n_ctx}",
         f"VRAM used {v1[0]:.1f} of {v1[1]:.1f} GB (after warm up {v0[0]:.1f})", ""]
    by = ok.groupby("depth").agg(n=("wall_s", "size"), wall_s=("wall_s", "mean"), prompt_tokens=("prompt_tokens", "mean"),
                                 gen_tokens=("gen_tokens", "mean"), prefill_tps=("prompt_per_second", "median"),
                                 decode_tps=("predicted_per_second", "median"), strict=("strict", "mean"),
                                 ref_strict=("ref_strict", "mean"), ref_time=("ref_agent_time_s", "mean"))
    L.append("by depth (ref = Qwen3 14B int8, transformers, same items):")
    L.append(by.round(3).to_string())
    agree = (ok.strict == ok.ref_strict).mean()
    L.append(f"\nstrict accuracy {ok.strict.mean():.3f} vs Qwen3 14B {ok.ref_strict.mean():.3f} on the same items; "
             f"label agreement {agree:.3f}; truncated answers (hit {MAX_NEW_TOKENS} tokens): "
             f"{int((ok.finish == 'length').sum())}")
    # time model: wall = a + b * prompt_tokens + c * gen_tokens, then project the full grid
    X = np.c_[np.ones(len(ok)), ok.prompt_tokens.astype(float), ok.gen_tokens.astype(float)]
    coef, *_ = np.linalg.lstsq(X, ok.wall_s.to_numpy(), rcond=None)
    ratio = float((ok.prompt_tokens / ok.ref_prompt_tokens).median())
    ref = pd.read_csv(PROC / f"tsr_labels_{REF_READER}.csv", usecols=["dataset", "depth", "repeat", "prompt_tokens"])
    ref = ref[(ref["repeat"].astype(int) == 0) & (ref.depth.astype(int) <= 10)]
    grid_s = float((coef[0] + coef[1] * ref.prompt_tokens * ratio + coef[2] * ok.gen_tokens.mean()).sum())
    L.append(f"\ntime model: wall = {coef[0]:.3f} + {coef[1] * 1000:.3f} ms/prompt token + {coef[2] * 1000:.1f} ms/gen token; "
             f"prompt token ratio vs Qwen tokenizer {ratio:.2f}")
    L.append(f"projected full grid ({len(ref):,} calls: 4 datasets x 500 queries x 25 plans, depth <= 10): "
             f"{grid_s / 3600:.1f} GPU hours ({grid_s / 86400:.2f} days), one call at a time")
    L.append(f"longest probe prompt {int(ok.prompt_tokens.max())} tokens; the grid's longest at depth 10 is about "
             f"{int(ref.prompt_tokens.max() * ratio)} (needs n_ctx >= that + {MAX_NEW_TOKENS})")
    (OUT / f"{name}_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "run", "help"])
    ap.add_argument("name", nargs="?", default="probe")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--thinking-off", action="store_true", help="Qwen3: disable thinking (as for Qwen3 14B)")
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "help":
        print(HELP)
    elif a.stage == "prepare":
        stage_prepare(a.dataset)
    else:
        stage_run(a.name, a.url, a.thinking_off, a.dataset, a.limit)
