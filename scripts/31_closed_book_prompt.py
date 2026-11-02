"""
Closed book prompt check. GPU for stages generate and judge.

Concern: our closed book accuracy for Llama 3.1 8B is 0.06 to 0.10 below the Never RAG values of Moskvoretskii et al.
(ACL 2025, [10]), while Always RAG at depth 10 is within 0.03. The paper blamed corpus and prompt differences, but a
corpus cannot affect a closed book answer. A weaker closed book reader shrinks what the closed book gate and the no
retrieval rung of Adaptive RAG can save. This script matches the closed book prompt of [10] for Llama 3.1 8B,
reports the share of closed book answers that abstain, and reruns the gate tests.

[10] follows the Adaptive RAG protocol (Jeong et al., NAACL 2024) for its no retrieval answers: a few shot chain of
thought prompt (Adaptive_Rag/prompts/<dataset>/no_context_cot_qa_codex.txt, the 20 demonstrations selected in
base_configs/nor_qa_llama_8b_it_<dataset>.jsonnet), raw completion without a chat template, greedy decoding, at most
100 new tokens, stop at the first newline, answer = the text after "answer is:" (the whole text if absent), last full
stop removed. Their files are in data/external/adarague/ (MIT and Apache 2.0 licences; see README.txt there).

Runs (Llama 3.1 8B, bf16, as in our labels):
  their_prompt_their_q   their prompt on their 500 test questions per dataset: checks our implementation against
                         their stored answers (no_retrieve) and their reported Never RAG accuracy.
  our_prompt_their_q     our closed book prompt on their questions: separates the prompt from the question sample.
  their_prompt_our_q     their prompt on our 500 questions per dataset: the closed book answers for the replay.
Their stored answers (their_answers_their_q) are graded the same way.

Stages:
  generate  GPU, about 1 to 1.5 hours (6,000 calls; the few shot prompts are about 1,300 tokens). Resumable.
  judge     GPU, minutes: rules for every answer, then the Qwen3 14B judge (int8, as 07) on the ambiguous band;
            label_final as in 07 (judge in the band, strict rule elsewhere).
  report    CPU: accuracy under answer contained accuracy (the metric of [10]) and label_final, abstention share,
            per dataset and run; their Table 1 Never RAG values next to them.
  replay    CPU: the primary family and the classifier gate for Llama 3.1 8B rerun with the closed book plan's answers
            from their_prompt_our_q (label_final, confidence features), in both plan sets, with (a) the newly measured
            closed book time and (b) the old closed book time; corrected test; compared with the paper's gains.

Usage:  python scripts/31_closed_book_prompt.py generate
        python scripts/31_closed_book_prompt.py judge
        python scripts/31_closed_book_prompt.py report
        python scripts/31_closed_book_prompt.py replay
Output: data/processed_full/m4_closed_book/*.csv; results_full/analysis/closed_book_prompt/
"""

import ast
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import settings

STAGE = sys.argv[1] if len(sys.argv) > 1 else ""
# The Llama weights live in <home>/models/Meta-Llama-3.1-8B-Instruct on the labelling machine; tsr/agent_runner.py reads
# FUNNELPLAN_LLAMA_DIR at import time, so point it there unless it is already set.
_LLAMA = Path.home() / "models" / "Meta-Llama-3.1-8B-Instruct"
if "FUNNELPLAN_LLAMA_DIR" not in os.environ and _LLAMA.is_dir():
    os.environ["FUNNELPLAN_LLAMA_DIR"] = str(_LLAMA)
READER = "llama3.1-8b"
EXT = ROOT / "data" / "external" / "adarague"
DIR = settings.PROCESSED_DATA_ROOT / "m4_closed_book"
OUT = settings.RESULTS_ROOT / "analysis" / "closed_book_prompt"
DATASETS = {"nq": "natural_questions", "hotpotqa": "hotpotqa", "2wikimultihopqa": "2wikimultihopqa", "musique": "musique"}
RUNS = ["their_prompt_their_q", "our_prompt_their_q", "their_prompt_our_q"]
TABLE1_NEVER_RAG = {"nq": 0.446, "hotpotqa": 0.286, "2wikimultihopqa": 0.318, "musique": 0.106}   # [10] Table 1
ABSTAIN = re.compile(r"\b(unknown|not (?:be )?(?:able|possible|sure|known|mentioned|provided|available|specified|"
                     r"stated|given|clear|enough|found|certain)|(?:do|does|did) not (?:mention|say|state|know|contain|"
                     r"have|provide)|(?:don't|doesn't|didn't) (?:mention|say|state|know|contain|have|provide)|cannot|"
                     r"can ?not|can't|no information|unclear|insufficient|no answer|n/a|i'm not sure|i am not sure)\b", re.I)


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


def parse_ref(x):
    try:
        v = ast.literal_eval(str(x))
    except Exception:
        v = re.findall(r"'([^']*)'", str(x)) or [str(x)]
    return [str(a) for a in (v if isinstance(v, (list, tuple)) else [v])]


def their_questions(ds):
    d = pd.read_csv(EXT / f"test_{DATASETS[ds]}.csv")
    return pd.DataFrame({"query_id": d.qid.astype(str), "question": d.question_text.astype(str),
                         "gold": [parse_ref(x) for x in d.reference], "their_answer": d.no_retrieve.astype(str)})


def our_questions(ds):
    lab = pd.read_csv(settings.PROCESSED_DATA_ROOT / f"tsr_labels_{READER}_graded.csv", dtype={"query_id": str},
                      keep_default_na=False, usecols=["dataset", "query_id"])
    ids = sorted(set(lab[lab.dataset == ds].query_id), key=int)
    gold = json.load(open(settings.PROCESSED_DATA_ROOT / ds / "answers.json", encoding="utf-8"))
    text = {}
    with open(settings.PROCESSED_DATA_ROOT / ds / "queries.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if str(i) in ids:
                text[str(i)] = json.loads(line).get("text", "")
    return pd.DataFrame({"query_id": ids, "question": [text[i] for i in ids], "gold": [gold.get(i, []) for i in ids]})


# ====================================================================== their prompt (Adaptive RAG protocol)
def read_demos(ds, tok, limit=8000, gen_len=300):
    """Adaptive_Rag/commaqa/inference/prompt_reader.read_prompt with their arguments: demonstrations filtered to the
    configured ids, in that order, dropped from the end until the estimated length fits 8,000 tokens."""
    valid = json.load(open(EXT / "valid_qids_nor_qa_llama_8b_it.json"))[ds]
    lines = [l + "\n" for l in open(EXT / f"no_context_cot_qa_codex_{ds}.txt", encoding="utf-8").read().strip().split("\n")]
    ex, exs = {"lines": []}, []
    for i, line in enumerate(lines):
        if i == len(lines) - 1:
            exs.append(ex)
        if line.strip().startswith("# METADATA: "):
            exs.append(ex)
            ex = dict(json.loads(line.strip().replace("# METADATA: ", "", 1)), lines=[])
        else:
            ex["lines"].append(line)
    exs = [e for e in exs if e["lines"] and e.get("qid") in valid]
    exs = sorted(exs, key=lambda e: valid.index(e["qid"]))
    texts = ["".join(e["lines"]).strip() for e in exs]
    lens = [len(tok.tokenize(t)) for t in texts]
    while lens and sum(lens) + max(lens) + gen_len > limit:
        texts.pop()
        lens.pop()
    return "\n\n\n".join(t.strip() for t in texts).strip(), len(texts)


def extract(text):
    m = re.match(r".* answer is:? (.*)\.?", text.strip())
    a = m.group(1).strip() if m else text.strip()
    return a[:-1] if a.endswith(".") else a


class TheirPrompt:
    def __init__(self, runner):
        self.r = runner
        self.r.load()
        self.demos = {}
        tok = self.r.tokenizer
        # their server stops when the decoded text ends with a newline: stop on any token that contains one
        self._nl = {i for i in range(len(tok)) if "\n" in tok.decode([i])}

    def answer(self, ds, question):
        import torch
        tok, model = self.r.tokenizer, self.r.model
        if ds not in self.demos:
            self.demos[ds] = read_demos(ds, tok)
            print(f"    {ds}: {self.demos[ds][1]} demonstrations kept", flush=True)
        prompt = (self.demos[ds][0] + "\n" + "\n\nQ: " + question + "\nA:").rstrip()
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)       # their server: tokenizer.encode
        nl = self._nl
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=100, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True,
                                 eos_token_id=sorted(nl | {tok.eos_token_id}))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out.sequences[0][ids.shape[1]:]
        lp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0][: len(gen)].float()
        raw = tok.decode(gen, skip_special_tokens=True).split("\n")[0]
        keep = [i for i, t in enumerate(gen.tolist()) if t not in tok.all_special_ids and "\n" not in tok.decode([t])]
        lpk = lp[keep] if keep else lp
        return {"answer": extract(raw), "raw": raw, "prompt_tokens": int(ids.shape[1]), "gen_tokens": int(len(gen)),
                "agent_time_s": dt, "mean_logprob": float(lpk.mean()) if lpk.numel() else float("nan"),
                "min_logprob": float(lpk.min()) if lpk.numel() else float("nan"),
                "first_token_prob": float(lpk[0].exp()) if lpk.numel() else float("nan")}


def stage_generate():
    from tsr.agent_runner import AgentRunner
    DIR.mkdir(parents=True, exist_ok=True)
    runner = AgentRunner(READER)
    runner.load()
    theirs = TheirPrompt(runner)
    t0 = time.time()
    for run in RUNS:
        for ds in DATASETS:
            f = DIR / f"{run}_{ds}.csv"
            Q = their_questions(ds) if run.endswith("their_q") else our_questions(ds)
            done = set(pd.read_csv(f, dtype={"query_id": str}).query_id) if f.exists() else set()
            todo = Q[~Q.query_id.isin(done)]
            print(f"  {run} {ds}: {len(todo)} to do ({time.time() - t0:.0f}s)", flush=True)
            rows = []
            for k, r in enumerate(todo.itertuples()):
                o = theirs.answer(ds, r.question) if run.startswith("their_prompt") else \
                    dict(runner.answer_with_meta(r.question, []), raw="")
                rows.append({"query_id": r.query_id, **o})
                if len(rows) == 50 or k == len(todo) - 1:
                    pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
                    rows = []
    print(f"Done in {time.time() - t0:.0f}s")


# ====================================================================== judge
def all_answers():
    from tsr.grader import grade_detail
    frames = []
    for ds in DATASETS:
        TQ, OQ = their_questions(ds), our_questions(ds)
        frames.append(TQ.assign(run="their_answers_their_q", answer=TQ.their_answer, dataset=ds))
        for run in RUNS:
            f = DIR / f"{run}_{ds}.csv"
            if not f.exists():
                sys.exit(f"{f.name} missing; run stage generate first")
            g = pd.read_csv(f, dtype={"query_id": str}, keep_default_na=False)
            Q = TQ if run.endswith("their_q") else OQ
            frames.append(g.merge(Q[["query_id", "question", "gold"]], on="query_id").assign(run=run, dataset=ds))
    A = pd.concat(frames, ignore_index=True)
    det = pd.DataFrame([grade_detail(a, g) for a, g in zip(A.answer.astype(str), A.gold)])
    A = pd.concat([A.reset_index(drop=True), det], axis=1)
    A["strict"] = (A.em.astype(int) | A.contains.astype(int)).astype(int)
    A["ambiguous"] = (((A.strict == 0) & (A.f1 >= 0.2)) | ((A.em == 0) & (A.contains == 1) & (A.n_answer_tokens >= 8))).astype(int)
    A["abstain"] = [int(bool(ABSTAIN.search(f"{a} {r}"))) for a, r in zip(A.answer.astype(str), A.get("raw", "").astype(str))]
    return A


def stage_judge():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    G = _load("grading07", "07_grading_full.py")
    A = all_answers()
    q = A[A.ambiguous == 1].drop_duplicates(["dataset", "question", "answer"])
    f = DIR / "judge_scores.csv"
    done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["dataset", "question", "answer", "p_yes"])
    key = set(zip(done.dataset, done.question, done.answer.astype(str)))
    todo = q[[(d, qq, a) not in key for d, qq, a in zip(q.dataset, q.question, q.answer.astype(str))]]
    print(f"judge: {len(q)} ambiguous answers, {len(todo)} to judge")
    if len(todo):
        name = G.CONFIG["judge_model"]
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                                     device_map={"": 0}, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).eval()
        yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
        no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})
        rows = todo.to_dict("records")
        for i in range(0, len(rows), 16):
            b = rows[i: i + 16]
            prompts = [G.judge_prompt(tok, r["question"], " | ".join(r["gold"]), r["answer"]) for r in b]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits[:, -1, :].float(), -1)
            py, pn = p[:, yes].sum(1), p[:, no].sum(1)
            pd.DataFrame({"dataset": [r["dataset"] for r in b], "question": [r["question"] for r in b],
                          "answer": [r["answer"] for r in b], "p_yes": (py / (py + pn + 1e-9)).cpu().numpy()}) \
                .to_csv(f, mode="a", header=not f.exists(), index=False)
    J = pd.read_csv(f).drop_duplicates(["dataset", "question", "answer"])
    A = A.merge(J, on=["dataset", "question", "answer"], how="left")
    A["label_final"] = np.where((A.ambiguous == 1) & A.p_yes.notna(), (A.p_yes >= G.CONFIG["judge_threshold"]).astype(int),
                                A.strict).astype(int)
    A.drop(columns=["gold"]).to_csv(DIR / "graded.csv", index=False)
    print(f"Wrote {DIR / 'graded.csv'}")


# ====================================================================== report
def stage_report():
    OUT.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(DIR / "graded.csv", dtype={"query_id": str}, keep_default_na=False)
    for c in ("contains", "label_final", "abstain", "agent_time_s", "gen_tokens"):
        A[c] = pd.to_numeric(A[c], errors="coerce")
    T = A.groupby(["run", "dataset"]).agg(contains_acc=("contains", "mean"), label_final=("label_final", "mean"),
                                          abstain=("abstain", "mean"), time_s=("agent_time_s", "mean"),
                                          gen_tokens=("gen_tokens", "mean"), n=("contains", "size")).reset_index()
    old = pd.read_csv(settings.PROCESSED_DATA_ROOT / f"tsr_labels_{READER}_graded.csv", dtype={"query_id": str},
                      keep_default_na=False)
    old = old[(old.retriever == "closed_book") & (old["repeat"].astype(int) == 0)].copy()
    old["abstain"] = [int(bool(ABSTAIN.search(str(a)))) for a in old.answer]
    for c in ("contains", "label_final", "agent_time_s", "gen_tokens"):
        old[c] = pd.to_numeric(old[c], errors="coerce")
    o = old.groupby("dataset").agg(contains_acc=("contains", "mean"), label_final=("label_final", "mean"),
                                   abstain=("abstain", "mean"), time_s=("agent_time_s", "mean"),
                                   gen_tokens=("gen_tokens", "mean"), n=("contains", "size")).reset_index()
    T = pd.concat([T, o.assign(run="our_prompt_our_q (paper)")], ignore_index=True)
    T["their_table1_never_rag"] = T.dataset.map(TABLE1_NEVER_RAG)
    T.to_csv(OUT / "accuracy.csv", index=False)
    L = ["Closed book prompt check, Llama 3.1 8B. contains_acc = answer contained accuracy (the metric of [10]);",
         "their Table 1 Never RAG values for reference. Abstain = refusal or 'unknown' style answer (regular expression).", ""]
    L.append(T.pivot_table(index="run", columns="dataset", values="contains_acc").round(3).to_string())
    L.append("\nlabel_final accuracy:")
    L.append(T.pivot_table(index="run", columns="dataset", values="label_final").round(3).to_string())
    L.append("\nabstention share:")
    L.append(T.pivot_table(index="run", columns="dataset", values="abstain").round(3).to_string())
    L.append("\nmean time per answer (s) and generated tokens:")
    L.append(T.groupby("run")[["time_s", "gen_tokens"]].mean().round(3).to_string())
    # agreement of our implementation of their prompt with their stored answers
    a = A[A.run == "their_prompt_their_q"].set_index(["dataset", "query_id"])
    b = A[A.run == "their_answers_their_q"].set_index(["dataset", "query_id"])
    G = _load("grading07", "07_grading_full.py")
    same = (a.answer.astype(str).map(G.normalize) == b.answer.reindex(a.index).astype(str).map(G.normalize))
    L.append(f"\nour run of their prompt reproduces their stored answer exactly (normalised) for {same.mean():.3f} of "
             f"their questions; correctness agrees for {(a.contains == b.contains.reindex(a.index)).mean():.3f}")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# ====================================================================== replay
def stage_replay():
    OUT.mkdir(parents=True, exist_ok=True)
    K = _load("inference27", "27_inference_refit.py")
    A_, R_, S20 = K.A, K.R, K.S20
    G = pd.read_csv(DIR / "graded.csv", dtype={"query_id": str}, keep_default_na=False)
    G = G[G.run == "their_prompt_our_q"].set_index(["dataset", "query_id"])
    for c in ("label_final", "mean_logprob", "min_logprob", "first_token_prob", "gen_tokens", "n_answer_tokens",
              "agent_time_s", "strict", "em", "contains", "f1"):
        G[c] = pd.to_numeric(G[c], errors="coerce")
    rows = []
    for variant in ("new_time", "old_time"):
        for r3 in (False, True):
            A_.CONFIG["include_r3"] = r3
            tag = "with two hop" if r3 else "single step"
            df, _ = A_.load_reader(READER)
            m = df.retriever == "closed_book"
            key = list(zip(df.loc[m, "dataset"], df.loc[m, "query_id"]))
            new = G.reindex(key)
            assert new.label_final.notna().all(), "missing closed book answers for some questions"
            df.loc[m, "y"] = new.label_final.astype(int).to_numpy()
            df.loc[m, "answer"] = new.answer.to_numpy()
            for c in ("mean_logprob", "min_logprob", "first_token_prob", "gen_tokens", "n_answer_tokens"):
                df.loc[m, c] = new[c].to_numpy()
            if variant == "new_time":
                df.loc[m, "agent_time_s"] = new.agent_time_s.to_numpy()
            for ds in sorted(df.dataset.unique()):
                B = A_.Block(df, ds)
                F = A_.decision_features(B)
                X = S20.cb_features(df, ds, B.q)
                F2 = F.join(X.add_prefix("cb_"))
                mk = {"planner_tsr": lambda: A_.m_planner_tsr(B, F), "adaptive_rag": lambda: A_.m_adaptive_rag(B, F),
                      "adaptive_k": lambda: A_.m_adaptive_k(B, F), "gate_closed_book": lambda: A_.m_gate_closed_book(B, F),
                      "conf_cascade": lambda: A_.m_conf_cascade(B, F), "clf_gate": lambda: S20.m_clf_gate(B, X),
                      "planner_sk": lambda: S20.m_planner_sk(B, F2)}
                for name, f in mk.items():
                    D, fr = K.repeated_folds(B, f)
                    rows += [{"variant": variant, "plan_set": tag, "dataset": ds, "method": name, **x} for x in fr]
                print(f"  replay {variant} {tag} {ds} done; closed book accuracy {B.Y['closed_book@0'].mean():.3f}, "
                      f"cost {B.cost['closed_book@0']:.3f}s", flush=True)
    A_.CONFIG["include_r3"] = False
    FO = pd.DataFrame(rows)
    FO.to_csv(OUT / "replay_folds.csv", index=False)
    out = []
    for (v, tag, m), g in FO.groupby(["variant", "plan_set", "method"]):
        cells = [K.nb_cell(x.gain, x.n_test.mean(), x.n_train.mean()) for _, x in g.groupby("dataset")]
        out.append({"variant": v, "plan_set": tag, "method": m, **K.nb_row(*K.nb_pooled(cells))})
    T = pd.DataFrame(out)
    ref = pd.read_csv(K.OUT / "tests_nb.csv") if (K.OUT / "tests_nb.csv").exists() else None
    if ref is not None:
        T = T.merge(ref[ref.reader == READER][["plan_set", "method", "gain", "nb_low", "nb_high"]]
                    .rename(columns={"gain": "gain_paper", "nb_low": "paper_low", "nb_high": "paper_high"}),
                    on=["plan_set", "method"], how="left")
    T.to_csv(OUT / "replay_tests.csv", index=False)
    print("\nLlama 3.1 8B with the closed book answers of the [10] prompt (corrected test; paper gains for comparison):")
    print(T.round(4).to_string(index=False))
    (OUT / "replay_summary.txt").write_text(T.round(4).to_string(index=False), encoding="utf-8")


if __name__ == "__main__":
    {"generate": stage_generate, "judge": stage_judge, "report": stage_report, "replay": stage_replay}.get(
        STAGE, lambda: sys.exit(__doc__))()
