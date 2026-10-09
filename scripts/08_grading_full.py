#!/usr/bin/env python
"""
Grading for the full mode labels: rules, a local LLM judge, and a two rater
human audit that validates both.

Why: every TSR number rests on grading. The lenient rule (F1 >= 0.4) used at
labelling time gives credit for partial matches; the strict rule misses
paraphrases. The paper uses one final label, and reports how well it agrees
with two humans.

Stages (set CONFIG["stage"], run in this order):

  "regrade"    CPU, seconds. For every tsr_labels_<reader>.csv in
               data/processed_full: evidence (exact, contained, F1) and every
               rule. Writes tsr_labels_<reader>_graded.csv and
               grading/judge_queue.csv (unique answers the rules disagree on).

  "audit"      CPU. Writes a blind, stratified sheet of AUDIT_N unique answers
               for two raters: grading/audit_rater1.csv, audit_rater2.csv
               (same items, different order) and grading/audit_key.csv.
               Never resampled once written.

  "judge"      GPU (do not run during labelling). A local LLM judges every
               answer in judge_queue.csv against the gold answers: P(yes).
               Writes grading/judge_scores.csv. Resumable.

  "finalize"   CPU. Reads the two rater sheets (if filled) and the judge.
               Reports Cohen's kappa (rater vs rater, rater vs judge, rater vs
               each rule), picks nothing by hand: label_final = judge decision
               on ambiguous answers, strict rule elsewhere. Writes label_final
               into every graded file and grading/grading_report.txt.

Ambiguous answer: no exact or contained match but token F1 >= 0.2, or a long
answer (>= 8 words) that contains the gold (possible hedging, e.g. "X or Y").
Outside that band the pilot audit found the strict rule right (long contained
answers 30/30, F1 < 0.2 wrong); the audit here re checks both regions with
their own strata, so the claim is measured again, not assumed.
"""

import json
import os
import sys
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings
from tsr.grader import RULES, apply_rule, grade_detail, normalize

CONFIG = {
    "stage": "finalize",            # "regrade" | "audit" | "judge" | "finalize"
    "judge_model": "Qwen/Qwen3-14B",
    "judge_quant": "int8",         # 14B in bf16 does not fit in 20 GB
    "judge_batch": 16,
    "judge_threshold": 0.5,        # P(yes) >= threshold -> correct
    "judge_limit": None,             # smoke test: judge only this many, then check; None = all
    "audit_n": 200,
    "seed": 0,
}

PROC = settings.PROCESSED_DATA_ROOT
GDIR = PROC / "grading"
QUEUE = GDIR / "judge_queue.csv"
JUDGE = GDIR / "judge_scores.csv"
KEY = GDIR / "audit_key.csv"
RATERS = [GDIR / "audit_rater1.csv", GDIR / "audit_rater2.csv"]
REPORT = GDIR / "grading_report.txt"

# Audit strata over unique (dataset, query, normalised answer) pairs.
# Two strata check the regions the final label does NOT send to the judge.
STRATA = [
    ("f1_02_04",      40, lambda d: (d.strict == 0) & (d.f1 >= 0.2) & (d.f1 < 0.4)),
    ("f1_04_07",      50, lambda d: (d.strict == 0) & (d.f1 >= 0.4) & (d.f1 < 0.7)),
    ("f1_07_up",      40, lambda d: (d.strict == 0) & (d.f1 >= 0.7)),
    ("contains_long", 30, lambda d: (d.em == 0) & (d.contains == 1) & (d.n_answer_tokens >= 8)),
    ("strict_short",  20, lambda d: (d.strict == 1) & (d.n_answer_tokens < 8)),
    ("f1_below_02",   20, lambda d: (d.strict == 0) & (d.f1 < 0.2)),
]


def ambiguous(d):
    return ((d.strict == 0) & (d.f1 >= 0.2)) | ((d.em == 0) & (d.contains == 1) & (d.n_answer_tokens >= 8))


def label_files():
    return sorted(f for f in PROC.glob("tsr_labels_*.csv") if not f.stem.endswith("_graded"))


def gold_maps(datasets):
    return {ds: json.load(open(PROC / ds / "answers.json", encoding="utf-8")) for ds in datasets}


def question_texts(pairs):
    out = {}
    for ds in pairs["dataset"].unique():
        want = set(pairs.loc[pairs.dataset == ds, "query_id"])
        with open(PROC / ds / "queries.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if str(i) in want:
                    out[(ds, str(i))] = json.loads(line).get("text", "")
    return out


# ---------------------------------------------------------------- regrade
def regrade_file(f, gold):
    df = pd.read_csv(f, dtype={"query_id": str}, keep_default_na=False)
    df = df.rename(columns={"label": "label_labelling"})
    det = pd.DataFrame([grade_detail(a, gold[ds].get(q, []))
                        for ds, q, a in zip(df["dataset"], df["query_id"], df["answer"])])
    df = pd.concat([df.reset_index(drop=True), det], axis=1)
    df["strict"] = ((df["em"] == 1) | (df["contains"] == 1)).astype(int)
    for r in RULES:
        df[f"label_{r}"] = [apply_rule(x, r) for x in det.to_dict("records")]
    df["pair"] = df["dataset"] + "::" + df["query_id"] + "::" + df["answer"].map(normalize)
    df["ambiguous"] = ambiguous(df).astype(int)
    return df


def stage_regrade():
    GDIR.mkdir(parents=True, exist_ok=True)
    files = label_files()
    if not files:
        sys.exit(f"No tsr_labels_*.csv in {PROC}")
    L, queue = [], []
    for f in files:
        reader = f.stem.replace("tsr_labels_", "")
        head = pd.read_csv(f, usecols=["dataset"])
        df = regrade_file(f, gold_maps(head["dataset"].unique()))
        df.drop(columns=["pair"]).to_csv(f.with_name(f.stem + "_graded.csv"), index=False)
        L.append(f"== {reader}: {len(df)} rows, {df['pair'].nunique()} unique answers, "
                 f"{df['ambiguous'].mean():.1%} of rows ambiguous")
        t = df.groupby("dataset")[[f"label_{r}" for r in RULES]].mean().round(3)
        L.append(t.to_string())
        q = df[df["ambiguous"] == 1].drop_duplicates("pair")[["pair", "dataset", "query_id", "answer"]]
        queue.append(q.assign(reader=reader))
    qa = pd.concat(queue)
    qa = qa.groupby(["pair", "dataset", "query_id", "answer"], as_index=False)["reader"].agg(",".join)
    gold = gold_maps(qa["dataset"].unique())
    qa["gold"] = [" | ".join(gold[d].get(q, [])) for d, q in zip(qa["dataset"], qa["query_id"])]
    qt = question_texts(qa)
    qa["question"] = [qt.get((d, q), "") for d, q in zip(qa["dataset"], qa["query_id"])]
    qa.to_csv(QUEUE, index=False, encoding="utf-8")
    L.append(f"\nJudge queue: {len(qa)} unique ambiguous answers -> {QUEUE}")
    print("\n".join(L))


# ---------------------------------------------------------------- audit
def stage_audit():
    if KEY.exists():
        sys.exit(f"{KEY} exists; the audit sheet is never resampled. Delete the grading/audit_* files to redo it.")
    rng = np.random.default_rng(CONFIG["seed"])
    frames = []
    for f in label_files():
        g = f.with_name(f.stem + "_graded.csv")
        if not g.exists():
            sys.exit("Run stage 'regrade' first.")
        d = pd.read_csv(g, dtype={"query_id": str}, keep_default_na=False)
        d["pair"] = d["dataset"] + "::" + d["query_id"] + "::" + d["answer"].map(normalize)
        frames.append(d.assign(reader=f.stem.replace("tsr_labels_", "")))
    pairs = pd.concat(frames).drop_duplicates("pair")
    total = sum(n for _, n, _ in STRATA)
    scale = CONFIG["audit_n"] / total
    chosen, used = [], set()
    for name, n, cond in STRATA:
        pool = pairs[cond(pairs) & ~pairs["pair"].isin(used)]
        # balance datasets inside each stratum
        k = int(round(n * scale))
        n_ds = max(1, pool["dataset"].nunique())
        parts = [x.iloc[rng.permutation(len(x))[: max(1, k // n_ds)]] for _, x in pool.groupby("dataset")]
        per = pd.concat(parts) if parts else pool.iloc[:0]
        if len(per) < k:    # top up from the rest of the stratum when a dataset is short
            rest = pool[~pool["pair"].isin(per["pair"])]
            per = pd.concat([per, rest.iloc[rng.permutation(len(rest))[: k - len(per)]]])
        take = per.iloc[rng.permutation(len(per))[:k]].assign(stratum=name)
        used.update(take["pair"])
        chosen.append(take)
    a = pd.concat(chosen).reset_index(drop=True)
    a["audit_id"] = np.arange(1, len(a) + 1)
    gold = gold_maps(a["dataset"].unique())
    a["gold_answers"] = [" | ".join(gold[d].get(q, [])) for d, q in zip(a["dataset"], a["query_id"])]
    qt = question_texts(a)
    a["question"] = [qt.get((d, q), "") for d, q in zip(a["dataset"], a["query_id"])]
    sheet = a[["audit_id", "dataset", "question", "gold_answers", "answer"]].rename(columns={"answer": "model_answer"})
    sheet["label_1_correct_0_wrong"] = ""
    sheet["notes"] = ""
    for i, path in enumerate(RATERS):
        order = np.random.default_rng(100 + i).permutation(len(sheet))
        sheet.iloc[order].to_csv(path, index=False, encoding="utf-8-sig")
    a[["audit_id", "stratum", "pair", "reader", "dataset", "query_id"]].to_csv(KEY, index=False)
    print(f"Wrote {len(sheet)} items: {RATERS[0].name}, {RATERS[1].name}; key {KEY.name}")
    print(a["stratum"].value_counts().to_string())


# ---------------------------------------------------------------- judge
JUDGE_SYSTEM = ("You grade answers to questions. A model answer is correct if it gives the same "
                "fact as one of the gold answers, even with different wording, extra detail or a "
                "missing article. It is wrong if it names a different entity, gives a different "
                "number or date, hedges between options, or is too vague to identify the gold. "
                "Reply with one word: yes or no.")


def judge_prompt(tok, question, gold, answer):
    user = (f"Question: {question}\nGold answers: {gold}\nModel answer: {answer}\n"
            f"Is the model answer correct?")
    msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
    kw = {"enable_thinking": False} if "qwen3" in CONFIG["judge_model"].lower() else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def stage_judge():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from tqdm import tqdm
    import logging, warnings
    # bitsandbytes int8 warns "inputs will be cast ... to float16" on every matmul; harmless, floods the console
    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
    q = pd.read_csv(QUEUE, dtype={"query_id": str}, keep_default_na=False)
    done = pd.read_csv(JUDGE, dtype={"query_id": str}, keep_default_na=False) if JUDGE.exists() else None
    todo = q[~q["pair"].isin(set(done["pair"]))] if done is not None else q
    if CONFIG["judge_limit"]:
        todo = todo.iloc[: CONFIG["judge_limit"]]
    print(f"Judge: {len(q)} in queue, {len(q) - len(todo)} done or skipped, {len(todo)} to judge now")
    if todo.empty:
        return
    tok = AutoTokenizer.from_pretrained(CONFIG["judge_model"], trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    qcfg = BitsAndBytesConfig(load_in_8bit=True) if CONFIG["judge_quant"] == "int8" else None
    model = AutoModelForCausalLM.from_pretrained(CONFIG["judge_model"], quantization_config=qcfg,
                                                 device_map={"": 0}, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()
    yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})
    bs = CONFIG["judge_batch"]
    rows = todo.to_dict("records")
    for i in tqdm(range(0, len(rows), bs), desc="judge"):
        batch = rows[i: i + bs]
        prompts = [judge_prompt(tok, r["question"], r["gold"], r["answer"]) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        p = torch.softmax(logits, dim=-1)
        py, pn = p[:, yes_ids].sum(1), p[:, no_ids].sum(1)
        score = (py / (py + pn + 1e-9)).cpu().numpy()
        out = pd.DataFrame({"pair": [r["pair"] for r in batch], "p_yes": score,
                            "yes_no_mass": (py + pn).cpu().numpy()})
        out.to_csv(JUDGE, mode="a", header=not JUDGE.exists(), index=False)
    print(f"Wrote {JUDGE}")
    if CONFIG["judge_limit"]:
        j = pd.read_csv(JUDGE).merge(q, on="pair")
        print("Smoke test sample (check by eye that p_yes makes sense, then set judge_limit None):")
        for r in j.tail(12).itertuples():
            print(f"  p_yes={r.p_yes:.2f} | gold: {r.gold[:40]} | answer: {r.answer[:50]}")


# ---------------------------------------------------------------- finalize
def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def read_rater(path, require_complete=False):
    """Reads <name>.xlsx when it exists (the Excel sheet with a 1/0 dropdown), else <name>.csv.
    require_complete: use the xlsx only once every item is labelled (rater 2: the human sheet replaces the
    LLM labels in audit_rater2.csv only when the second human has finished)."""
    xl = path.with_suffix(".xlsx")
    s = None
    if xl.exists():
        s = pd.read_excel(xl, sheet_name="audit")
        n = pd.to_numeric(s["label_1_correct_0_wrong"], errors="coerce").notna().sum()
        if require_complete and n < len(s):
            print(f"{xl.name}: {n} of {len(s)} labelled, not complete; using {path.name} for now")
            s = None
    if s is not None:
        pass
    elif path.exists():
        s = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    else:
        return None
    s["y"] = pd.to_numeric(s["label_1_correct_0_wrong"], errors="coerce")
    return s.set_index("audit_id")["y"]


def stage_finalize():
    L = []
    j = pd.read_csv(JUDGE) if JUDGE.exists() else None
    if j is None:
        L.append("No judge scores yet: label_final = strict rule everywhere (provisional).")
        jmap = {}
    else:
        jmap = dict(zip(j["pair"], (j["p_yes"] >= CONFIG["judge_threshold"]).astype(int)))
        L.append(f"Judge scores: {len(j)} answers; low yes/no mass (<0.5) on {int((j['yes_no_mass'] < 0.5).sum())}")

    # ---- human audit
    if KEY.exists():
        key = pd.read_csv(KEY, dtype={"query_id": str}, keep_default_na=False).set_index("audit_id")
        r1, r2 = read_rater(RATERS[0]), read_rater(RATERS[1], require_complete=True)
        x2 = RATERS[1].with_suffix(".xlsx")
        human2 = x2.exists() and pd.to_numeric(pd.read_excel(x2, sheet_name="audit")["label_1_correct_0_wrong"],
                                                errors="coerce").notna().all()
        L.append("rater 1: human. rater 2: " + ("HUMAN (audit_rater2.xlsx)" if human2 else
                 "LLM labels (Claude, audit_rater2.csv), NOT human; the human sheet audit_rater2.xlsx is not complete"))
        for name, r in (("rater 1", r1), ("rater 2", r2)):
            n = 0 if r is None else int(r.notna().sum())
            L.append(f"{name}: {n} of {len(key)} labelled")
        if r1 is not None and r2 is not None and r1.notna().all() and r2.notna().all():
            both = pd.DataFrame({"r1": r1, "r2": r2}).join(key)
            L.append(f"Cohen's kappa rater 1 vs rater 2: {kappa(both.r1, both.r2):.3f} "
                     f"(raw agreement {(both.r1 == both.r2).mean():.3f})")
            agreed = both[both.r1 == both.r2].copy()
            agreed["human"] = agreed["r1"].astype(int)
            L.append(f"Items both raters agree on: {len(agreed)}; disagreements listed in grading/audit_disagreements.csv")
            both[both.r1 != both.r2].to_csv(GDIR / "audit_disagreements.csv")
            graded = {}
            for f in label_files():
                g = pd.read_csv(f.with_name(f.stem + "_graded.csv"), dtype={"query_id": str}, keep_default_na=False)
                g["pair"] = g["dataset"] + "::" + g["query_id"] + "::" + g["answer"].map(normalize)
                graded.update(g.drop_duplicates("pair").set_index("pair")[["strict"] + [f"label_{r}" for r in RULES]
                                                                           + ["ambiguous"]].to_dict("index"))
            ev = agreed[agreed["pair"].isin(graded)].copy()
            for c in ["strict"] + [f"label_{r}" for r in RULES] + ["ambiguous"]:
                ev[c] = [graded[p][c] for p in ev["pair"]]
            ev["judge"] = [jmap.get(p, np.nan) for p in ev["pair"]]
            ev["final"] = np.where(ev["ambiguous"] == 1, ev["judge"], ev["strict"])
            L.append("\nAgreement with the human label (items both raters agree on):")
            for c in [f"label_{r}" for r in RULES] + ["judge", "final"]:
                e = ev.dropna(subset=[c])
                if len(e):
                    acc = (e[c].astype(int) == e["human"]).mean()
                    lo, hi = np.percentile([(e[c].astype(int) == e["human"]).sample(frac=1, replace=True,
                                            random_state=s).mean() for s in range(1000)], [2.5, 97.5])
                    L.append(f"  {c:16s} n={len(e):3d} accuracy {acc:.3f} [{lo:.3f}, {hi:.3f}] "
                             f"kappa {kappa(e[c].astype(int), e['human']):.3f}")
            L.append("\nBy stratum (final label vs human):")
            for s, e in ev.dropna(subset=["final"]).groupby("stratum"):
                L.append(f"  {s:14s} n={len(e):3d} accuracy {(e['final'].astype(int) == e['human']).mean():.3f}")
            L.append("\nBy reader (final label vs human; checks judge self preference on qwen3-14b):")
            for rd, e in ev.dropna(subset=["final"]).assign(reader=ev["reader"].str.split(",")).explode("reader").groupby("reader"):
                L.append(f"  {rd:14s} n={len(e):3d} accuracy {(e['final'].astype(int) == e['human']).mean():.3f}")

            # Human only reference: rater 1 on all items (rater 2 is an LLM, so the agreed subset is filtered by it)
            h = both[both["pair"].isin(graded)].copy()
            h["human"] = h["r1"].astype(int)
            for c in ["strict", "ambiguous"]:
                h[c] = [graded[p][c] for p in h["pair"]]
            h["judge"] = [jmap.get(p, np.nan) for p in h["pair"]]
            h["final"] = np.where(h["ambiguous"] == 1, h["judge"], h["strict"])
            e = h.dropna(subset=["final"])
            y, f_ = e["final"].astype(int).values, e["human"].values
            ok = (y == f_)
            lo, hi = np.percentile([ok[np.random.default_rng(s_).integers(0, len(ok), len(ok))].mean()
                                    for s_ in range(2000)], [2.5, 97.5])
            L.append(f"\nHuman only reference (rater 1, all items): final n={len(e)} accuracy {ok.mean():.3f} "
                     f"[{lo:.3f}, {hi:.3f}] kappa {kappa(pd.Series(y), pd.Series(f_)):.3f}")
            L.append(f"  false positives (final 1, human 0): {int(((y == 1) & (f_ == 0)).sum())}; "
                     f"false negatives (final 0, human 1): {int(((y == 0) & (f_ == 1)).sum())}; "
                     f"success rate final {y.mean():.3f} vs human {f_.mean():.3f}")
            L.append("  By reader (FP = judge too lenient; self preference shows as extra FP on qwen3-14b):")
            ex = e.assign(reader=e["reader"].str.split(",")).explode("reader")
            for rd, g_ in ex.groupby("reader"):
                yy, hh = g_["final"].astype(int).values, g_["human"].values
                L.append(f"    {rd:14s} n={len(g_):3d} acc {(yy == hh).mean():.3f} FP {int(((yy == 1) & (hh == 0)).sum())} "
                         f"FN {int(((yy == 0) & (hh == 1)).sum())}")
            if j is not None:
                pmap = dict(zip(j["pair"], j["p_yes"]))
                a = h[h["ambiguous"] == 1].copy()
                a["p"] = [pmap.get(p_, np.nan) for p_ in a["pair"]]
                a = a.dropna(subset=["p"])
                L.append(f"  Judge threshold sweep on ambiguous items (n={len(a)}), human only:")
                for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    yy = (a["p"] >= t).astype(int).values
                    L.append(f"    t={t:.1f} acc {(yy == a['human'].values).mean():.3f} "
                             f"FP {int(((yy == 1) & (a['human'].values == 0)).sum())} "
                             f"FN {int(((yy == 0) & (a['human'].values == 1)).sum())}")
    else:
        L.append("No audit sheet yet (run stage 'audit').")

    # ---- write label_final
    L.append("")
    for f in label_files():
        gpath = f.with_name(f.stem + "_graded.csv")
        g = pd.read_csv(gpath, dtype={"query_id": str}, keep_default_na=False)
        pair = g["dataset"] + "::" + g["query_id"] + "::" + g["answer"].map(normalize)
        jv = pair.map(jmap)
        g["label_final"] = np.where((g["ambiguous"] == 1) & jv.notna(), jv, g["strict"]).astype(int)
        g["label_final_source"] = np.where((g["ambiguous"] == 1) & jv.notna(), "judge", "strict")
        g.to_csv(gpath, index=False)
        miss = int(((g["ambiguous"] == 1) & jv.isna()).sum())
        t = g.groupby("dataset")[["label_strict", "label_f1_07", "label_lenient", "label_final"]].mean().round(3)
        L.append(f"== {f.stem.replace('tsr_labels_', '')}: label_final written ({miss} ambiguous rows without a judge score -> strict)")
        L.append(t.to_string())
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    # The stage can also be given on the command line (python scripts/08_grading_full.py judge), so stages can be
    # chained in one unattended command; without an argument CONFIG["stage"] is used as before.
    stages = {"regrade": stage_regrade, "audit": stage_audit, "judge": stage_judge, "finalize": stage_finalize}
    stage = sys.argv[1] if len(sys.argv) > 1 else CONFIG["stage"]
    if stage not in stages:
        sys.exit(f"Unknown stage {stage}; choose from {list(stages)}")
    stages[stage]()
