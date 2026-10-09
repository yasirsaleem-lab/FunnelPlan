#!/usr/bin/env python
"""
Fixed two hop plans (pre-registration run R3, defined in the amendment of 2026-10-09).

Plan bm25_2hop@k, k in {3, 5}. The number of retrieval calls (2) and reader calls (2) is fixed before execution.
  hop 1   BM25 top k for the question (the cached BM25 list of 04, the same as plan bm25@k).
  query   The reader writes ONE search query for the missing information, from the question and the hop 1
          passages (QUERY_SYSTEM below; greedy; at most 32 new tokens; Qwen3 thinking off).
  hop 2   BM25 top k for that query alone, skipping passages already found in hop 1.
  answer  The reader answers from the 2k passages (hop 1 first, then hop 2) with the unchanged answer prompt
          of tsr/agent_runner.py (greedy, at most 100 new tokens).
  cost    retrieval_time_s = hop 1 latency (cache) + hop 2 latency (timed here, one query at a time);
          agent_time_s = query call + answer call (generate() time for transformers readers, server compute
          time for llama.cpp readers, as in 04 and 17).

Stages (each one is resumable; rerun the same command after a stop):
  query  READER   GPU. Writes data/processed_full/r3_2hop/<reader>/query.jsonl
  hop2            CPU. BM25 for every query of every reader that has a query file; loads each index once.
                  Writes data/processed_full/r3_2hop/hop2.jsonl
  answer READER   GPU. Writes data/processed_full/r3_2hop/<reader>/answer.jsonl
  merge  READER   CPU. Writes data/processed_full/r3_2hop/labels_<reader>.csv (columns of 04, plus the parts)
  status          Progress of every reader and stage.

Readers: qwen2.5-7b, llama3.1-8b, qwen3-14b (transformers, as in 04) and qwen3-32b, gemma3-27b (llama.cpp:
start llama-server with the matching model first, as for 17).

Usage
  python scripts/19_two_hop_plans.py query qwen3-32b
  python scripts/19_two_hop_plans.py hop2
  python scripts/19_two_hop_plans.py answer qwen3-32b
  python scripts/19_two_hop_plans.py merge qwen3-32b
"""

import argparse
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

_spec = importlib.util.spec_from_file_location("grid17", Path(__file__).parent / "17_llamacpp_labels.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)              # design(), load_caches(), passage_texts(), messages(), check_server()

KS = [3, 5]
QUERY_MAX_TOKENS = 32
ANSWER_MAX_TOKENS = 100
QUERY_SYSTEM = ("You help answer a question by searching a collection of Wikipedia passages. "
                "Given the question and the passages found so far, write one short search query for the "
                "information that is still missing. Reply with the search query only.")
TRANSFORMERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
LLAMACPP = {"qwen3-32b": True, "gemma3-27b": False}          # name -> thinking_off
ALL_READERS = TRANSFORMERS + list(LLAMACPP)
PLAN = "bm25_2hop"
OUT = settings.PROCESSED_DATA_ROOT / "r3_2hop"
HOP2 = OUT / "hop2.jsonl"


# ====================================================================== helpers
def read_jsonl(p):
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.endswith("}"):
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def key(d):
    return (d["reader"], d["dataset"], d["query_id"], int(d["k"]))


def query_messages(q, doc_ids, texts):
    lines = [f"Passage {i + 1}: {t}" for i, d in enumerate(doc_ids) if (t := texts.get(d))]
    user = "Passages:\n" + "\n".join(lines) + f"\n\nQuestion: {q}\nSearch query:"
    return [{"role": "system", "content": QUERY_SYSTEM}, {"role": "user", "content": user}]


def clean_query(s, question):
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S).strip()
    s = s.splitlines()[0].strip() if s.strip() else ""
    s = re.sub(r"^(search query|query)\s*:\s*", "", s, flags=re.I).strip().strip('"\'`*').strip()
    return (s, False) if s else (question, True)        # empty output -> fall back to the question


def setup(datasets=None):
    order, qids, _ = G.design(datasets)
    data = {}
    for ds in order:
        c = G.load_caches(ds, ["bm25"])["bm25"]
        data[ds] = {"qids": qids[ds], "qtext": c["queries"],
                    "hop1": {q: [x["id"] for x in c["results"][q]] for q in qids[ds]},
                    "lat1": {q: float(c["latency_s"].get(q, float("nan"))) for q in qids[ds]}}
    return order, data


# ====================================================================== readers
class Reader:
    def __init__(self, name, url):
        self.name, self.url = name, url
        if name in LLAMACPP:
            G.check_server(url, name)
            self.thinking_off = LLAMACPP[name]
            self.agent = None
        else:
            from tsr.agent_runner import AgentRunner
            self.agent = AgentRunner(reader=name)
            self.agent.load()

    def generate(self, msgs, max_new):
        if self.agent is None:
            return self._server(msgs, max_new)
        return self._transformers(msgs, max_new)

    def _server(self, msgs, max_new):
        body = {"messages": msgs, "max_tokens": max_new, "temperature": 0.0, "top_p": 1.0, "seed": 0,
                "logprobs": True, "top_logprobs": 1, "cache_prompt": False, "stream": False}
        if self.thinking_off:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        t0 = time.perf_counter()
        r = G.session().post(f"{self.url}/v1/chat/completions", json=body, timeout=600)   # keep-alive
        wall = time.perf_counter() - t0
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        ch = j["choices"][0]
        txt = re.sub(r"<think>.*?</think>", "", (ch["message"].get("content") or ""), flags=re.S).strip()
        lps = [t["logprob"] for t in ((ch.get("logprobs") or {}).get("content") or []) if t.get("logprob") is not None]
        tm = j.get("timings", {})
        pm, gm = tm.get("prompt_ms"), tm.get("predicted_ms")
        return {"text": txt, "prompt_tokens": j.get("usage", {}).get("prompt_tokens"),
                "gen_tokens": j.get("usage", {}).get("completion_tokens"),
                "time_s": (pm + gm) / 1000 if pm is not None and gm is not None else wall,
                "mean_logprob": float(np.mean(lps)) if lps else float("nan"),
                "min_logprob": float(np.min(lps)) if lps else float("nan"),
                "first_token_prob": float(np.exp(lps[0])) if lps else float("nan"),
                "finish": ch.get("finish_reason")}

    def _transformers(self, msgs, max_new):
        import torch
        a = self.agent
        prompt = a.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 **a.spec.get("chat_kwargs", {}))
        inputs = a.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(a.model.device)
        n = int(inputs.input_ids.shape[1])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            out = a.model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                                   top_k=None, pad_token_id=a.tokenizer.pad_token_id, output_scores=True,
                                   return_dict_in_generate=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = time.perf_counter() - t0
        gen = out.sequences[0][n:]
        lp = a.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0][: len(gen)].float()
        keep = [i for i, t in enumerate(gen.tolist()) if t not in a.tokenizer.all_special_ids]
        lpk = lp[keep] if keep else lp
        return {"text": a.tokenizer.decode(gen, skip_special_tokens=True).strip(), "prompt_tokens": n,
                "gen_tokens": int(gen.shape[0]), "time_s": dt,
                "mean_logprob": float(lpk.mean()) if lpk.numel() else float("nan"),
                "min_logprob": float(lpk.min()) if lpk.numel() else float("nan"),
                "first_token_prob": float(lpk[0].exp()) if lpk.numel() else float("nan"),
                "finish": "length" if int(gen.shape[0]) >= max_new else "stop"}


def answer_messages(q, doc_ids, texts):
    return G.messages(q, doc_ids, texts)          # identical to the answer prompt of agent_runner / 17


# ====================================================================== stages
def stage_query(reader, url, datasets):
    order, data = setup(datasets)
    d = OUT / reader
    d.mkdir(parents=True, exist_ok=True)
    out_p = d / "query.jsonl"
    done = {key(x) for x in read_jsonl(out_p)}
    todo = [(ds, q, k) for ds in order for q in data[ds]["qids"] for k in KS if (reader, ds, q, k) not in done]
    print(f"{reader}: {len(done)} queries done, {len(todo)} to do", flush=True)
    if not todo:
        return
    R = Reader(reader, url)
    texts = {}
    t_start, n = time.time(), 0
    with open(out_p, "a", encoding="utf-8") as f:
        for ds in order:
            sub = [(q, k) for (dd, q, k) in todo if dd == ds]
            if not sub:
                continue
            texts = G.passage_texts(ds, {x for q in data[ds]["qids"] for x in data[ds]["hop1"][q][:max(KS)]})
            for q, k in sub:
                msgs = query_messages(data[ds]["qtext"][q], data[ds]["hop1"][q][:k], texts)
                try:
                    o = R.generate(msgs, QUERY_MAX_TOKENS)
                except Exception as e:
                    print(f"  failed {ds}/{q}/k={k}: {str(e)[:120]}", flush=True)
                    continue
                sq, fallback = clean_query(o["text"], data[ds]["qtext"][q])
                f.write(json.dumps({"reader": reader, "dataset": ds, "query_id": q, "k": k, "raw": o["text"],
                                    "subquery": sq, "fallback": fallback, "q_time_s": o["time_s"],
                                    "q_prompt_tokens": o["prompt_tokens"], "q_gen_tokens": o["gen_tokens"]}) + "\n")
                f.flush()
                n += 1
                if n % 200 == 0:
                    el = time.time() - t_start
                    print(f"  {len(done) + n}/{len(done) + len(todo)}  {el / n:.2f} s/call  "
                          f"ETA {(len(todo) - n) * el / n / 60:.0f} min", flush=True)


def stage_hop2(datasets):
    from data.corpus_utils import build_corpus_list
    from data.data_loader import get_processed
    from retrievers.sparse.bm25 import BM25Retriever
    order, data = setup(datasets)
    have = {key(x) for x in read_jsonl(HOP2)}
    qs = [x for r in ALL_READERS for x in read_jsonl(OUT / r / "query.jsonl") if key(x) not in have]
    print(f"hop 2: {len(have)} done, {len(qs)} to do", flush=True)
    if not qs:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(HOP2, "a", encoding="utf-8") as f:
        for ds in order:
            sub = [x for x in qs if x["dataset"] == ds]
            if not sub:
                continue
            corpus = get_processed(ds)[0]
            ret = BM25Retriever()
            ret.build_index(build_corpus_list(corpus))     # loads the cached index of 04
            del corpus
            ret.search("warm up", 10)
            for i, x in enumerate(sub):
                h1 = set(data[ds]["hop1"][x["query_id"]][:x["k"]])
                t0 = time.perf_counter()
                res = ret.search(x["subquery"], x["k"] + len(h1))
                lat = time.perf_counter() - t0
                ids = [r["id"] for r in res if r["id"] not in h1][: x["k"]]
                f.write(json.dumps({"reader": x["reader"], "dataset": ds, "query_id": x["query_id"], "k": x["k"],
                                    "hop2_ids": ids, "hop2_latency_s": lat}) + "\n")
                if (i + 1) % 1000 == 0:
                    f.flush()
                    print(f"  {ds}: {i + 1}/{len(sub)}", flush=True)
            ret.cleanup()
            del ret
    print("hop 2 done", flush=True)


def stage_answer(reader, url, datasets):
    from tsr.grader import apply_rule, grade_detail
    order, data = setup(datasets)
    qmap = {key(x): x for x in read_jsonl(OUT / reader / "query.jsonl")}
    hmap = {key(x): x for x in read_jsonl(HOP2) if x["reader"] == reader}
    out_p = OUT / reader / "answer.jsonl"
    done = {key(x) for x in read_jsonl(out_p)}
    todo = [kk for kk in qmap if kk in hmap and kk not in done and kk[1] in order]
    missing = sum(1 for kk in qmap if kk not in hmap)
    print(f"{reader}: {len(done)} answers done, {len(todo)} to do, {missing} still waiting for hop 2", flush=True)
    if not todo:
        return
    R = Reader(reader, url)
    t_start, n = time.time(), 0
    with open(out_p, "a", encoding="utf-8") as f:
        for ds in order:
            sub = [kk for kk in todo if kk[1] == ds]
            if not sub:
                continue
            gold = json.load(open(settings.PROCESSED_DATA_ROOT / ds / "answers.json", encoding="utf-8"))
            need = {x for kk in sub for x in data[ds]["hop1"][kk[2]][:kk[3]] + hmap[kk]["hop2_ids"]}
            texts = G.passage_texts(ds, need)
            for kk in sub:
                _, _, q, k = kk
                ids = data[ds]["hop1"][q][:k] + hmap[kk]["hop2_ids"]
                try:
                    o = R.generate(answer_messages(data[ds]["qtext"][q], ids, texts), ANSWER_MAX_TOKENS)
                except Exception as e:
                    print(f"  failed {ds}/{q}/k={k}: {str(e)[:120]}", flush=True)
                    continue
                g = grade_detail(o["text"], gold.get(q, []))
                qq = qmap[kk]
                f.write(json.dumps({
                    "reader": reader, "dataset": ds, "query_id": q, "k": k, "answer": o["text"],
                    "label": apply_rule(g, "lenient"), "a_time_s": o["time_s"], "a_prompt_tokens": o["prompt_tokens"],
                    "a_gen_tokens": o["gen_tokens"], "mean_logprob": o["mean_logprob"], "min_logprob": o["min_logprob"],
                    "first_token_prob": o["first_token_prob"], "finish": o["finish"],
                    "n_passages": len(ids), "subquery": qq["subquery"]}) + "\n")
                f.flush()
                n += 1
                if n % 200 == 0:
                    el = time.time() - t_start
                    print(f"  {len(done) + n}/{len(done) + len(todo)}  {el / n:.2f} s/answer  "
                          f"ETA {(len(todo) - n) * el / n / 60:.0f} min", flush=True)


def stage_merge(reader, datasets):
    order, data = setup(datasets)
    qmap = {key(x): x for x in read_jsonl(OUT / reader / "query.jsonl")}
    hmap = {key(x): x for x in read_jsonl(HOP2) if x["reader"] == reader}
    amap = {key(x): x for x in read_jsonl(OUT / reader / "answer.jsonl")}
    expected = [(reader, ds, q, k) for ds in order for q in data[ds]["qids"] for k in KS]
    miss = [kk for kk in expected if kk not in amap]
    if miss:
        print(f"Not merged: {len(miss)} of {len(expected)} answers missing (e.g. {miss[:2]}).")
        return
    rows = []
    for kk in expected:
        _, ds, q, k = kk
        a, qq, h = amap[kk], qmap[kk], hmap[kk]
        rows.append({"dataset": ds, "query_id": q, "retriever": PLAN, "depth": k, "repeat": 0, "label": a["label"],
                     "answer": a["answer"], "prompt_tokens": a["a_prompt_tokens"],
                     "gen_tokens": qq["q_gen_tokens"] + a["a_gen_tokens"],
                     "agent_time_s": qq["q_time_s"] + a["a_time_s"],
                     "retrieval_time_s": data[ds]["lat1"][q] + h["hop2_latency_s"],
                     "mean_logprob": a["mean_logprob"], "min_logprob": a["min_logprob"],
                     "first_token_prob": a["first_token_prob"], "finish": a["finish"],
                     "subquery": qq["subquery"], "subquery_fallback": qq["fallback"],
                     "q_time_s": qq["q_time_s"], "a_time_s": a["a_time_s"], "hop1_latency_s": data[ds]["lat1"][q],
                     "hop2_latency_s": h["hop2_latency_s"], "n_passages": a["n_passages"], "reader": reader})
    df = pd.DataFrame(rows)
    p = OUT / f"labels_{reader}.csv"
    df.to_csv(p, index=False)
    print(f"Wrote {p.name}: {len(df)} rows; fallback queries {int(df.subquery_fallback.sum())}")
    print("lenient accuracy by dataset and k:")
    print(df.pivot_table(index="dataset", columns="depth", values="label").round(3).to_string())


def stage_status(datasets):
    order, data = setup(datasets)
    n = sum(len(data[ds]["qids"]) for ds in order) * len(KS)
    hop = read_jsonl(HOP2)
    for r in ALL_READERS:
        q, a = len(read_jsonl(OUT / r / "query.jsonl")), len(read_jsonl(OUT / r / "answer.jsonl"))
        h = sum(1 for x in hop if x["reader"] == r)
        merged = (OUT / f"labels_{r}.csv").exists()
        print(f"{r:12s} query {q:5d}/{n}  hop2 {h:5d}/{n}  answer {a:5d}/{n}  merged {merged}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["query", "hop2", "answer", "merge", "status"])
    ap.add_argument("reader", nargs="?", choices=ALL_READERS)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args()
    if a.stage in ("query", "answer", "merge") and not a.reader:
        sys.exit(f"Stage {a.stage} needs a reader: {ALL_READERS}")
    {"query": lambda: stage_query(a.reader, a.url, a.datasets),
     "hop2": lambda: stage_hop2(a.datasets),
     "answer": lambda: stage_answer(a.reader, a.url, a.datasets),
     "merge": lambda: stage_merge(a.reader, a.datasets),
     "status": lambda: stage_status(a.datasets)}[a.stage]()
