"""
Plan grid labels for readers served by llama.cpp (run R1, and R2 for the quantization control).

Same design as 02_generate_tsr_labels.py for Qwen3 14B: the same 500 queries per dataset, the same 25 plans
(closed book plus 6 retrievers x depths 1, 3, 5, 10), the same cached retrieval lists, the same prompt text and
system texts as tsr/agent_runner.py, greedy decoding, at most 100 new tokens, one call at a time, no prompt cache
(so a deeper plan never reuses the KV cache of a shallower one).
The query set and the plan list are read from tsr_labels_qwen3-14b.csv, so they are identical by construction.

Output has the columns of 04, so 07_grading_full.py grades it with no change:
  data/processed_full/labels_<reader>_parts/<dataset>__<retriever>.jsonl   progress, one line per answer (resumable)
  data/processed_full/labels_<reader>_parts/<dataset>__<retriever>.csv     written when the block is complete
  data/processed_full/tsr_labels_<reader>.csv                              written only when ALL blocks are complete
  For a control reader (qwen3-14b-gguf) the merged file is control_labels_<reader>.csv instead, so that the
  analysis scripts (which take every tsr_labels_*_graded.csv) never pool it with the paper readers.

agent_time_s = server prompt_ms + predicted_ms (compute only, like generate() time in 04). wall_s is kept too.
Calls that fail (for example a prompt longer than the context) are NOT written as answers; they go to
errors.jsonl and are retried on the next run. The merge refuses to run while any answer is missing.

Usage (llama-server must be running with the matching model; see 03_reader_probe.py help)
  python scripts/04_llamacpp_labels.py run qwen3-32b
  python scripts/04_llamacpp_labels.py run gemma3-27b
  python scripts/04_llamacpp_labels.py run qwen3-14b-gguf --datasets hotpotqa
  python scripts/04_llamacpp_labels.py status qwen3-32b
  python scripts/04_llamacpp_labels.py report qwen3-32b          (accuracy by dataset and depth vs Qwen3 14B)
Stop at any time with Ctrl+C. Rerun the same command to continue.
"""

import argparse
import json
import os
import pickle
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

# identical to tsr/agent_runner.py (copied, not imported, so this script needs no torch)
SYSTEM_RAG = ("You answer questions using only the given passages. "
              "Reply with the short answer only (a few words). "
              "If the passages do not contain the answer, give your best guess.")
SYSTEM_CLOSED = ("Answer the question from your own knowledge. "
                 "Reply with the short answer only (a few words).")
MAX_NEW_TOKENS = 100
MAX_PASSAGES_TO_PROMPT = 100

# name -> substring that must appear in the server's model path, thinking switch, control flag
READERS = {
    "qwen3-32b":      {"model": "qwen3-32b",       "thinking_off": True,  "control": False},
    "gemma3-27b":     {"model": "gemma-3-27b",     "thinking_off": False, "control": False},
    "qwen3-14b-gguf": {"model": "qwen3-14b",       "thinking_off": True,  "control": True},
}
REF_READER = "qwen3-14b"
CLOSED = "closed_book"
PROC = settings.PROCESSED_DATA_ROOT
CACHE = settings.CACHE_ROOT / "retrievals_v3"
TEXTS = PROC / "llamacpp_passages"
ID_RE = re.compile(r'^\{"_id":\s*"((?:[^"\\]|\\.)*)"')


# ====================================================================== design (from the reference reader)
def design(datasets=None):
    """(dataset -> query ids in the reference order, sorted plan list) from tsr_labels_qwen3-14b.csv."""
    ref = pd.read_csv(PROC / f"tsr_labels_{REF_READER}.csv", usecols=["dataset", "query_id", "retriever", "depth", "repeat"],
                      dtype={"query_id": str}, keep_default_na=False)
    ref = ref[ref["repeat"].astype(int) == 0]
    ref["depth"] = ref["depth"].astype(int)
    order = list(dict.fromkeys(ref["dataset"]))
    if datasets:
        bad = [d for d in datasets if d not in order]
        if bad:
            sys.exit(f"Unknown dataset(s) {bad}; choose from {order}")
        order = [d for d in order if d in datasets]
    qids = {d: list(dict.fromkeys(ref.loc[ref.dataset == d, "query_id"])) for d in order}
    plans = {}
    for d in order:
        s = ref[ref.dataset == d]
        plans[d] = {r: sorted(s.loc[s.retriever == r, "depth"].unique().tolist())
                    for r in dict.fromkeys(s["retriever"])}
    return order, qids, plans


def part_dir(reader):
    return PROC / f"labels_{reader}_parts"


def merged_path(reader):
    return PROC / (f"control_labels_{reader}.csv" if READERS[reader]["control"] else f"tsr_labels_{reader}.csv")


# ====================================================================== passages and prompts
def load_caches(ds, retrievers):
    return {r: pickle.load(open(CACHE / f"{ds}__{r}.pkl", "rb")) for r in retrievers if r != CLOSED}


def passage_texts(ds, need):
    """doc id -> 'title. text' (as data/corpus_utils.passage_text), cached per dataset after the first scan."""
    TEXTS.mkdir(parents=True, exist_ok=True)
    f = TEXTS / f"{ds}.json"
    texts = json.load(open(f, encoding="utf-8")) if f.exists() else {}
    missing = set(need) - set(texts)
    if not missing:
        return texts
    print(f"  scanning {ds}/corpus.jsonl for {len(missing):,} passages (once per dataset) ...", flush=True)
    t0 = time.time()
    with open(PROC / ds / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            m = ID_RE.match(line)
            if m:
                if m.group(1) not in missing:
                    continue
                d = json.loads(line)
            else:                                    # other key order: parse fully (slow, but correct)
                d = json.loads(line)
                if str(d.get("_id")) not in missing:
                    continue
            title, text = (d.get("title") or "").strip(), (d.get("text") or "").strip()
            texts[str(d["_id"])] = f"{title}. {text}" if title and text else (title or text)
            missing.discard(str(d["_id"]))
            if not missing:
                break
    if missing:
        sys.exit(f"{len(missing)} passage ids not found in {ds}/corpus.jsonl, e.g. {sorted(missing)[:3]}")
    json.dump(texts, open(f, "w", encoding="utf-8"))
    print(f"  done in {time.time() - t0:.0f}s -> {f}", flush=True)
    return texts


def messages(q, doc_ids, texts):
    if not doc_ids:
        return [{"role": "system", "content": SYSTEM_CLOSED}, {"role": "user", "content": f"Question: {q}\nAnswer:"}]
    lines = [f"Passage {i + 1}: {t}" for i, d in enumerate(doc_ids[:MAX_PASSAGES_TO_PROMPT]) if (t := texts.get(d))]
    user = "Passages:\n" + "\n".join(lines) + f"\n\nQuestion: {q}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}]


# ====================================================================== server
_SESSION = None


def session():
    """One keep-alive HTTP connection for all calls. A new TCP connection per call (50,000 per grid)
    destabilised the Windows network stack. Timing uses the server's own
    prompt_ms + predicted_ms, so connection reuse does not change agent_time_s."""
    global _SESSION
    if _SESSION is None:
        import requests
        _SESSION = requests.Session()
    return _SESSION


def call(url, msgs, thinking_off, timeout=600):
    body = {"messages": msgs, "max_tokens": MAX_NEW_TOKENS, "temperature": 0.0, "top_p": 1.0, "seed": 0,
            "logprobs": True, "top_logprobs": 1, "cache_prompt": False, "stream": False}
    if thinking_off:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.perf_counter()
    r = session().post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
    wall = time.perf_counter() - t0
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    ch = j["choices"][0]
    content = (ch["message"].get("content") or "").strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    lps = [t["logprob"] for t in ((ch.get("logprobs") or {}).get("content") or []) if t.get("logprob") is not None]
    tm = j.get("timings", {})
    pm, gm = tm.get("prompt_ms"), tm.get("predicted_ms")
    return {"answer": content,
            "prompt_tokens": j.get("usage", {}).get("prompt_tokens"),
            "gen_tokens": j.get("usage", {}).get("completion_tokens"),
            "agent_time_s": (pm + gm) / 1000 if pm is not None and gm is not None else wall,
            "wall_s": wall,
            "mean_logprob": float(np.mean(lps)) if lps else float("nan"),
            "min_logprob": float(np.min(lps)) if lps else float("nan"),
            "first_token_prob": float(np.exp(lps[0])) if lps else float("nan"),
            "finish": ch.get("finish_reason")}


def check_server(url, reader):
    import requests
    try:
        props = requests.get(f"{url}/props", timeout=10).json()
    except Exception as e:
        sys.exit(f"No llama-server at {url} ({e}). Start it first (python scripts/03_reader_probe.py help).")
    path = str(props.get("model_path", ""))
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
    want = READERS[reader]["model"]
    if want not in path.lower().replace("_", "-"):
        sys.exit(f"The server runs {path}, but reader '{reader}' needs a model whose file name contains '{want}'. "
                 "Restart the server with the right model.")
    return path, n_ctx


# ====================================================================== run
def done_keys(jsonl):
    if not jsonl.exists():
        return set()
    keys = set()
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                keys.add((d["query_id"], int(d["depth"])))
            except Exception:          # a half written last line after a hard stop
                pass
    return keys


def finish_block(jsonl, n_expected):
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip().endswith("}")]
    df = pd.DataFrame(rows).drop_duplicates(["query_id", "depth"], keep="last")
    if len(df) < n_expected:
        return False
    cols = ["dataset", "query_id", "retriever", "depth", "repeat", "label", "answer", "prompt_tokens", "gen_tokens",
            "agent_time_s", "retrieval_time_s", "mean_logprob", "min_logprob", "first_token_prob", "wall_s", "finish"]
    tmp = jsonl.with_suffix(".csv.tmp")
    df[cols].to_csv(tmp, index=False)
    tmp.replace(jsonl.with_suffix(".csv"))
    return True


def stage_run(reader, url, datasets):
    from tsr.grader import apply_rule, grade_detail
    spec = READERS[reader]
    path, n_ctx = check_server(url, reader)
    order, qids, plans = design(datasets)
    pdir = part_dir(reader)
    pdir.mkdir(parents=True, exist_ok=True)
    errlog = pdir / "errors.jsonl"
    total = sum(len(qids[d]) * len(ds_) for d in order for ds_ in plans[d].values())
    n_done = sum(len(done_keys(pdir / f"{d}__{r}.jsonl")) for d in order for r in plans[d])
    print(f"server {path}  n_ctx {n_ctx}  reader {reader}  thinking_off={spec['thinking_off']}\n"
          f"{n_done:,} of {total:,} answers already done", flush=True)
    warm = False
    t_start, n_new, n_err = time.time(), 0, 0
    for ds in order:
        gold = json.load(open(PROC / ds / "answers.json", encoding="utf-8"))
        caches = load_caches(ds, plans[ds])
        qtext = next(iter(caches.values()))["queries"] if caches else {}
        need = {x["id"] for r, c in caches.items() for q in qids[ds] for x in c["results"][q][:max(plans[ds][r])]}
        texts = passage_texts(ds, need)
        for ret, depths in plans[ds].items():
            jsonl = pdir / f"{ds}__{ret}.jsonl"
            n_expected = len(qids[ds]) * len(depths)
            if jsonl.with_suffix(".csv").exists():
                continue
            done = done_keys(jsonl)
            if len(done) >= n_expected:
                finish_block(jsonl, n_expected)
                continue
            print(f"== {ds} / {ret} / depths {depths}: {n_expected - len(done)} to do", flush=True)
            with open(jsonl, "a", encoding="utf-8") as out:
                for q in qids[ds]:
                    full = [] if ret == CLOSED else [x["id"] for x in caches[ret]["results"][q]]
                    for d in depths:
                        if (q, d) in done:
                            continue
                        msgs = messages(qtext[q], full[:d], texts)
                        if not warm:                       # one warm up call, not recorded
                            try:
                                call(url, msgs, spec["thinking_off"])
                            except Exception:
                                pass
                            warm = True
                        try:
                            o = call(url, msgs, spec["thinking_off"])
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            n_err += 1
                            with open(errlog, "a", encoding="utf-8") as ef:
                                ef.write(json.dumps({"dataset": ds, "query_id": q, "retriever": ret, "depth": d,
                                                     "error": str(e)[:300], "time": time.strftime("%F %T")}) + "\n")
                            continue
                        g = grade_detail(o["answer"], gold.get(q, []))
                        row = {"dataset": ds, "query_id": q, "retriever": ret, "depth": d, "repeat": 0,
                               "label": apply_rule(g, "lenient"),
                               "retrieval_time_s": 0.0 if ret == CLOSED else
                               float(caches[ret]["latency_s"].get(q, float("nan")))} | o
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        n_new += 1
                        if n_new % 200 == 0:
                            el = time.time() - t_start
                            left = total - n_done - n_new
                            print(f"  {n_done + n_new:,}/{total:,}  {el / n_new:.2f} s/answer  "
                                  f"ETA {left * el / n_new / 3600:.1f} h  errors {n_err}", flush=True)
            if finish_block(jsonl, n_expected):
                b = pd.read_csv(jsonl.with_suffix(".csv"))
                print(f"   block done: lenient by depth {b.groupby('depth')['label'].mean().round(3).to_dict()}", flush=True)
            else:
                print(f"   block incomplete ({len(done_keys(jsonl))}/{n_expected}); see {errlog.name}", flush=True)
    merge(reader, datasets)


def merge(reader, datasets=None):
    order, qids, plans = design(datasets)
    pdir = part_dir(reader)
    parts = [pdir / f"{d}__{r}.csv" for d in order for r in plans[d]]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        print(f"Not merged: {len(missing)} of {len(parts)} blocks incomplete (e.g. {missing[:3]}). "
              f"Rerun 'run' to retry failed calls; errors are in {pdir.name}/errors.jsonl")
        return
    df = pd.concat([pd.read_csv(p, dtype={"query_id": str}, keep_default_na=False) for p in parts], ignore_index=True)
    df["reader"] = reader
    out = merged_path(reader)
    tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"Merged {len(parts)} blocks, {len(df):,} rows -> {out.name}")
    if not READERS[reader]["control"]:
        print("Next: stop llama-server (the judge needs the GPU), then 07_grading_full.py stages regrade, judge, finalize.")


def stage_status(reader, datasets):
    order, qids, plans = design(datasets)
    pdir = part_dir(reader)
    tot = got = 0
    for d in order:
        cells = []
        for r, depths in plans[d].items():
            n = len(qids[d]) * len(depths)
            k = n if (pdir / f"{d}__{r}.csv").exists() else len(done_keys(pdir / f"{d}__{r}.jsonl"))
            tot, got = tot + n, got + k
            cells.append(f"{r}:{k}/{n}")
        print(f"{d:16s} " + "  ".join(cells))
    errs = sum(1 for _ in open(pdir / "errors.jsonl", encoding="utf-8")) if (pdir / "errors.jsonl").exists() else 0
    print(f"total {got:,} of {tot:,} ({got / tot:.1%}); failed calls logged: {errs}")


# ====================================================================== report
def stage_report(reader, datasets):
    from tsr.grader import apply_rule, grade_detail
    order, qids, plans = design(datasets)
    pdir = part_dir(reader)
    rows = []
    for d in order:
        for r in plans[d]:
            p = pdir / f"{d}__{r}.csv"
            src = p if p.exists() else pdir / f"{d}__{r}.jsonl"
            if not src.exists():
                continue
            rows.append(pd.read_csv(p, dtype={"query_id": str}, keep_default_na=False) if p.exists()
                        else pd.DataFrame([json.loads(l) for l in open(src, encoding="utf-8") if l.strip().endswith("}")]))
    if not rows:
        sys.exit("Nothing labelled yet.")
    D = pd.concat(rows, ignore_index=True)
    D["query_id"] = D["query_id"].astype(str)
    ref = pd.read_csv(PROC / f"tsr_labels_{REF_READER}_graded.csv", dtype={"query_id": str}, keep_default_na=False)
    ref = ref[ref["repeat"].astype(int) == 0]
    gold = {d: json.load(open(PROC / d / "answers.json", encoding="utf-8")) for d in D.dataset.unique()}
    det = [grade_detail(a, gold[d].get(q, [])) for d, q, a in zip(D.dataset, D.query_id, D.answer)]
    D["strict"] = [apply_rule(x, "strict") for x in det]
    D["f1_07"] = [apply_rule(x, "f1_07") for x in det]
    k = ["dataset", "query_id", "retriever", "depth"]
    ref["depth"] = ref["depth"].astype(int)
    D["depth"] = D["depth"].astype(int)
    M = D.merge(ref[k + ["label_strict", "label_f1_07", "label_final", "agent_time_s"]].rename(
        columns={"agent_time_s": "ref_time"}), on=k, how="left")
    L = [f"Report {reader}: {len(D):,} answers ({M.label_strict.notna().sum():,} matched to {REF_READER})", ""]
    t = M.groupby(["dataset", "depth"]).agg(n=("strict", "size"), strict=("strict", "mean"),
                                            ref_strict=("label_strict", "mean"), time_s=("agent_time_s", "mean"),
                                            ref_time_s=("ref_time", "mean"), prompt_tokens=("prompt_tokens", "max"))
    L.append("strict accuracy and mean time by dataset and depth (ref = Qwen3 14B int8; times are not comparable across runtimes):")
    L.append(t.round(3).to_string())
    m = M.dropna(subset=["label_strict"])
    L.append(f"\nlabel agreement with {REF_READER}: strict {(m.strict == m.label_strict).mean():.3f}, "
             f"f1_07 {(m.f1_07 == m.label_f1_07).mean():.3f}")
    L.append(f"truncated answers (finish = length): {int((D.get('finish', pd.Series(dtype=str)) == 'length').sum())}")
    if READERS[reader]["control"]:
        L.append("Gate G2: label agreement >= 0.85 with Qwen3 14B int8. The decisive number uses "
                 "label_final after grading; strict and f1_07 agreement above are the deterministic proxies.")
    out = settings.RESULTS_ROOT / "probe" / f"grid_{reader}_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "status", "merge", "report"])
    ap.add_argument("reader", choices=list(READERS))
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--datasets", nargs="*", default=None, help="default: all 4 (the control uses hotpotqa)")
    a = ap.parse_args()
    if a.datasets is None and READERS[a.reader]["control"]:
        a.datasets = ["hotpotqa"]                  # R2 runs on HotpotQA only
    if a.stage == "run":
        stage_run(a.reader, a.url, a.datasets)
    elif a.stage == "status":
        stage_status(a.reader, a.datasets)
    elif a.stage == "merge":
        merge(a.reader, a.datasets)
    else:
        stage_report(a.reader, a.datasets)
