"""
Adaptive RAG as published (Jeong et al., NAACL 2024) on the four dataset mix. Plan: docs/ANALYSIS_PLAN_2026-11-05.md, E1.

Ladder as published: A = no retrieval (closed book), B = single step BM25 with 5 passages (the published single step
setting), C = IRCoT (06_ircot.py; NQ with the HotpotQA demonstrations, as Adaptive RAG does).
Classifier as published: t5-large on the question text, AdamW, learning rate 3e-5, batch 32, max length 384; the
number of epochs is chosen once per reader from {15, 20, 25, 30, 35} on an inner split (80/20) of the seed 0, fold 0
training data, by accuracy against the silver labels.
Silver labels as published: the simplest rung whose answer is correct; questions no rung answers get the dataset type
label (NQ: B; HotpotQA, 2Wiki, MuSiQue: C).
Training and scoring: the 2,000 questions of a reader pooled over the four datasets (as published, one classifier for
the mix), 5 folds x 5 seeds across the pool (as 26_mixed_workload.py); costs as there (a plan's cost is its mean over
the pooled questions; IRCoT's likewise), plus the classifier's measured inference time per question.
Comparators on the same held out questions: Always RAG (BM25, 5 passages) and Never RAG (the published comparison,
accuracy only); the global hull of the fixed plans; the label hull; the global hull with IRCoT as one more plan.
Plan sets: single step, and with the fixed two hop plans (the comparators change; the ladder does not).
Tests: per question test, corrected test, corrected test with errors x the largest refit ratio; Holm over the three
readers within each comparator and plan set.

Usage:  python scripts/36_adaptive_rag_published.py            (GPU for the fine tunes; resumable through the cache)
        python scripts/36_adaptive_rag_published.py --fake     (CPU test of the pipeline: the classifier is replaced
                                                                 by the silver label of a random training question)
Output: results_full/analysis/adaptive_rag_published/  tests.csv, folds.csv, per_query.csv, usage.csv, summary.txt,
        t5_<reader>.json (epochs, predictions per seed and fold, inference time)
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from config import settings

FAKE = "--fake" in sys.argv
OUT = settings.RESULTS_ROOT / "analysis" / ("adaptive_rag_published_fake" if FAKE else "adaptive_rag_published")
OUT.mkdir(parents=True, exist_ok=True)
READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "nq"]
SINGLE_HOP = {"nq"}
RUNGS = ["closed_book@0", "bm25@5", "ircot@0"]
T5_MODEL = str(Path.home() / "models" / "t5-large") if (Path.home() / "models" / "t5-large").exists() else "t5-large"
T5 = {"lr": 3e-5, "batch": 32, "max_len": 384, "epoch_grid": [15, 20, 25, 30, 35]}
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


M26 = _load("mixed26", "26_mixed_workload.py")
A, R = M26.A, M26.R
INFL = float((pd.read_csv(settings.RESULTS_ROOT / "analysis" / "inference" / "calibration.csv")
              .eval("sd_refit / se_nb")).max())


# ====================================================================== data
def ircot_pooled(reader, dss):
    P = settings.PROCESSED_DATA_ROOT / "r4_ircot"
    fr = []
    for f in (P / f"labels_{reader}_graded.csv", P / f"labels_{reader}_nq_graded.csv"):
        if f.exists():
            fr.append(pd.read_csv(f, dtype={"query_id": str}, keep_default_na=False))
    d = pd.concat(fr)
    d = d[d.dataset.isin(dss)]
    col = f"label_{A.CONFIG['label']}" if f"label_{A.CONFIG['label']}" in d else "label_f1_07"
    d["pid"] = d.dataset + "|" + d.query_id
    d["y"] = d[col].astype(int)
    d["cost"] = pd.to_numeric(d.agent_time_s) + pd.to_numeric(d.retrieval_time_s)
    return d.set_index("pid")


def build(reader, r3):
    A.CONFIG["include_r3"] = r3
    df, _ = A.load_reader(reader)
    ir_dss = set()
    for f in ("labels_{}_graded.csv", "labels_{}_nq_graded.csv"):
        p = settings.PROCESSED_DATA_ROOT / "r4_ircot" / f.format(reader)
        if p.exists():
            ir_dss |= set(pd.read_csv(p, usecols=["dataset"]).dataset)
    dss = [d for d in DATASETS if d in ir_dss]
    blocks = [A.Block(df, ds) for ds in dss]
    P = M26.Pooled(blocks)
    IR = ircot_pooled(reader, dss)
    missing = [q for q in P.q if q not in IR.index]
    assert not missing, f"{len(missing)} questions have no IRCoT answer"
    Y_all = P.Y.copy()
    Y_all["ircot@0"] = IR.y.reindex(P.q).to_numpy()
    cost_all = P.cost.copy()
    cost_all["ircot@0"] = float(IR.cost.reindex(P.q).mean())
    for r in RUNGS:
        assert r in Y_all.columns, f"rung {r} missing"
    return P, Y_all, cost_all, dss


def silver(Y_all, sel):
    Ys = Y_all.loc[sel, RUNGS].to_numpy()
    fb = np.where(np.isin(M26.ds_of(sel), list(SINGLE_HOP)), 1, 2)
    return np.where(Ys.max(1) == 1, Ys.argmax(1), fb)


# ====================================================================== classifier
class Classifier:
    def __init__(self, reader, texts):
        self.reader, self.texts = reader, texts
        self.file = OUT / f"t5_{reader}.json"
        self.state = json.load(open(self.file)) if self.file.exists() else {"preds": {}}
        self.tok = None

    def save(self):
        tmp = self.file.with_suffix(".tmp")
        json.dump(self.state, open(tmp, "w"))
        tmp.replace(self.file)

    def _setup(self):
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(T5_MODEL)
            self.seqs = [self.tok(l, add_special_tokens=False).input_ids + [self.tok.eos_token_id] for l in "ABC"]

    def _train(self, ids, y, epochs, seed, eval_ids=None, eval_y=None):
        """Fine tune; if eval ids are given, return the accuracy after each epoch in the grid instead of a model."""
        import torch
        from transformers import AutoModelForSeq2SeqLM
        self._setup()
        torch.manual_seed(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=T5["lr"])
        enc = self.tok([self.texts[q] for q in ids], max_length=T5["max_len"], truncation=True, padding=True,
                       return_tensors="pt")
        L = max(len(s) for s in self.seqs)
        lab = torch.tensor([self.seqs[int(c)] + [-100] * (L - len(self.seqs[int(c)])) for c in y])
        bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
        acc = {}
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(ids))
            for i in range(0, len(ids), T5["batch"]):
                b = perm[i:i + T5["batch"]]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                    loss = model(input_ids=enc.input_ids[b].to(dev), attention_mask=enc.attention_mask[b].to(dev),
                                 labels=lab[b].to(dev)).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            if eval_ids is not None and ep in T5["epoch_grid"]:
                acc[ep] = float(np.mean(np.array(self._predict(model, eval_ids, dev, bf16)) == np.asarray(eval_y)))
                say(f"      epoch {ep}: inner accuracy {acc[ep]:.3f}")
        return (acc, None) if eval_ids is not None else (None, (model, dev, bf16))

    def _predict(self, model, ids, dev, bf16):
        import torch
        model.eval()
        enc = self.tok([self.texts[q] for q in ids], max_length=T5["max_len"], truncation=True, padding=True,
                       return_tensors="pt")
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                ii, am = enc.input_ids[i:i + 64].to(dev), enc.attention_mask[i:i + 64].to(dev)
                sc = []
                for c in range(3):
                    seq = torch.tensor([self.seqs[c]] * len(ii), device=dev)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                        lg = model(input_ids=ii, attention_mask=am, labels=seq).logits.float()
                    sc.append(torch.log_softmax(lg, -1).gather(2, seq.unsqueeze(2)).squeeze(2).sum(1))
                out += torch.stack(sc, 1).argmax(1).cpu().tolist()
        return out

    def select_epochs(self, sel, y):
        if "epochs" in self.state:
            return self.state["epochs"]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(sel))
        cut = int(0.8 * len(sel))
        tr, va = perm[:cut], perm[cut:]
        say(f"    {self.reader}: choosing epochs on an inner split ({len(tr)} / {len(va)})")
        acc, _ = self._train(sel[tr], y[tr], max(T5["epoch_grid"]), 0, sel[va], y[va])
        best = max(T5["epoch_grid"], key=lambda e: (acc[e], -e))
        self.state["epochs"], self.state["epoch_accuracy"] = best, acc
        self.save()
        return best

    def inference_time(self, ids):
        """Measured classifier time per question: batch 1, greedy decoding of the label (as a deployed router)."""
        if "inference_s" in self.state:
            return self.state["inference_s"]
        import torch
        from transformers import AutoModelForSeq2SeqLM
        self._setup()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(dev).eval()
        ts = []
        with torch.no_grad():
            for j, q in enumerate(ids[:220]):
                x = self.tok(self.texts[q], max_length=T5["max_len"], truncation=True, return_tensors="pt").to(dev)
                torch.cuda.synchronize() if dev == "cuda" else None
                t0 = time.perf_counter()
                model.generate(**x, max_new_tokens=3, do_sample=False)
                torch.cuda.synchronize() if dev == "cuda" else None
                if j >= 20:                                   # 20 warm up calls
                    ts.append(time.perf_counter() - t0)
        del model
        self.state["inference_s"] = float(np.mean(ts))
        self.save()
        return self.state["inference_s"]

    def predict(self, key, sel, y, test, epochs, seed):
        if key in self.state["preds"]:
            return np.array(self.state["preds"][key])
        if FAKE:
            rng = np.random.default_rng(seed)
            p = np.asarray(y)[rng.integers(0, len(y), len(test))]
        else:
            _, (model, dev, bf16) = self._train(sel, y, epochs, seed)
            p = np.array(self._predict(model, test, dev, bf16))
            del model
            import torch
            torch.cuda.empty_cache()
        self.state["preds"][key] = [int(v) for v in p]
        self.save()
        return p


# ====================================================================== protocol
def run_reader(reader, r3, clf, overhead):
    P, Y_all, cost_all, dss = build(reader, r3)
    tag = "with two hop" if r3 else "single step"
    k = A.CONFIG["k_folds"]
    per_q, folds, usage = {}, [], []
    epochs = None
    for sd in A.CONFIG["seeds"]:
        rng = np.random.default_rng(sd)
        q = P.q.copy()
        rng.shuffle(q)
        F = [q[i::k] for i in range(k)]
        for i in range(k):
            test = F[i]
            sel = np.concatenate([F[j] for j in range(k) if j != i])
            y = silver(Y_all, sel)
            if epochs is None:
                epochs = 1 if FAKE else clf.select_epochs(sel, y)
            t0 = time.time()
            pred = clf.predict(f"{sd}|{i}", sel, y, test, epochs, sd)
            if time.time() - t0 > 1:
                say(f"    {reader} seed {sd} fold {i}: fine tune and predict {time.time() - t0:.0f}s")
            chosen = np.array(RUNGS)[pred]
            s = np.array([Y_all.at[qq, p] for qq, p in zip(test, chosen)], float)
            c = float(np.mean([cost_all[p] for p in chosen])) + overhead
            comp = {"hull_fixed": A.hull_at(P.Y, P.cost, sel, c, eval_idx=test),
                    "hull_label": M26.label_hull(P.Y, P.cost, sel, c, test)[0],
                    "hull_with_ircot": A.hull_at(Y_all, cost_all, sel, c, eval_idx=test),
                    "always_rag": float(Y_all.loc[test, "bm25@5"].mean()),
                    "never_rag": float(Y_all.loc[test, "closed_book@0"].mean())}
            for name, v in comp.items():
                per_q.setdefault(name, []).append(pd.Series(s - v, index=test, name=sd))
                folds.append({"plan_set": tag, "reader": reader, "comparator": name, "seed": sd, "fold": i,
                              "gain": float(s.mean() - v), "n_test": len(test), "n_train": len(sel)})
            usage.append({"plan_set": tag, "reader": reader, "seed": sd, "fold": i, "success": s.mean(), "cost": c,
                          "always_cost": float(cost_all["bm25@5"]), "never_cost": float(cost_all["closed_book@0"]),
                          **{f"share_{r}": float(np.mean(chosen == r)) for r in RUNGS},
                          "silver_accuracy": float(np.mean(pred == silver(Y_all, test)))})
    rows, PQ = [], []
    for name, L in per_q.items():
        g = pd.concat(L).groupby(level=0).mean()          # mean over seeds per question
        fo = [f for f in folds if f["comparator"] == name]
        fg = np.array([f["gain"] for f in fo])
        J = len(fg)
        se_nb = float(np.sqrt((1.0 / J + fo[0]["n_test"] / fo[0]["n_train"]) * fg.var(ddof=1)))
        d = g.to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        m = float(fg.mean())
        r = {"plan_set": tag, "reader": reader, "comparator": name, "n": len(d), "gain": m,
             "gain_per_question": float(d.mean()), "se_plain": se, "p_plain": float(A.p_two_sided(d)),
             "plain_low": d.mean() - stats.t.ppf(0.975, len(d) - 1) * se,
             "plain_high": d.mean() + stats.t.ppf(0.975, len(d) - 1) * se,
             "se_nb": se_nb, "df_nb": J - 1, "p_nb": float(2 * stats.t.sf(abs(m / se_nb), J - 1)),
             "nb_low": m - stats.t.ppf(0.975, J - 1) * se_nb, "nb_high": m + stats.t.ppf(0.975, J - 1) * se_nb,
             "p_infl": float(2 * stats.t.sf(abs(m / (se_nb * INFL)), J - 1)),
             "infl_low": m - stats.t.ppf(0.975, J - 1) * se_nb * INFL,
             "infl_high": m + stats.t.ppf(0.975, J - 1) * se_nb * INFL}
        rows.append(r)
        PQ.append(pd.DataFrame({"plan_set": tag, "reader": reader, "comparator": name, "pid": g.index, "gain": d}))
    U = pd.DataFrame(usage)
    say(f"  {reader} {tag}: success {U.success.mean():.3f} at {U.cost.mean():.2f}s (Always RAG "
        f"{U.always_cost.mean():.2f}s); rungs {U[[c for c in U if c.startswith('share_')]].mean().round(3).to_dict()}; "
        f"classifier vs silver on held out folds {U.silver_accuracy.mean():.3f}; datasets {dss}")
    for r in rows:
        say(f"    vs {r['comparator']:15s} {r['gain']:+.4f}  per question [{r['plain_low']:+.4f}, {r['plain_high']:+.4f}]  "
            f"corrected [{r['nb_low']:+.4f}, {r['nb_high']:+.4f}]  x{INFL:.2f} [{r['infl_low']:+.4f}, {r['infl_high']:+.4f}]")
    return rows, PQ, folds, usage


def main():
    t0 = time.time()
    say(f"Adaptive RAG as published; classifier {T5_MODEL}; {'FAKE classifier (pipeline test)' if FAKE else ''}")
    rows, PQ, FO, US = [], [], [], []
    for rd in READERS:
        A.CONFIG["include_r3"] = False
        P, _, _, _ = build(rd, False)
        clf = Classifier(rd, P.text)
        overhead = 0.0 if FAKE else clf.inference_time(list(P.q))
        say(f"== {rd}: classifier inference {overhead * 1000:.1f} ms per question (charged)")
        for r3 in (False, True):
            r, pq, fo, us = run_reader(rd, r3, clf, overhead)
            rows += r
            PQ += pq
            FO += fo
            US += us
    A.CONFIG["include_r3"] = False
    T = pd.DataFrame(rows)
    for col in ("p_plain", "p_nb", "p_infl"):
        T[col + "_holm"] = np.nan
        for _, idx in T.groupby(["plan_set", "comparator"]).groups.items():
            T.loc[idx, col + "_holm"] = A.holm(T.loc[idx, col].to_numpy())
    T["positive_all_three"] = ((T.p_plain_holm < 0.05) & (T.plain_low > 0) & (T.p_nb_holm < 0.05) & (T.nb_low > 0)
                               & (T.p_infl_holm < 0.05) & (T.infl_low > 0))
    T["negative_corrected"] = (T.p_nb_holm < 0.05) & (T.nb_high < 0)
    T.to_csv(OUT / "tests.csv", index=False)
    pd.concat(PQ).to_csv(OUT / "per_query.csv", index=False)
    pd.DataFrame(FO).to_csv(OUT / "folds.csv", index=False)
    pd.DataFrame(US).to_csv(OUT / "usage.csv", index=False)
    say("\nDecision rule (positive in both plan sets under all three tests), per comparator:")
    for comp, g in T.groupby("comparator"):
        w = g[g.positive_all_three].groupby("reader").plan_set.nunique()
        say(f"  {comp:15s} wins in both plan sets: {sorted(w[w == 2].index)}; negative (corrected): "
            f"{int(g.negative_corrected.sum())} of {len(g)}")
    say(f"({time.time() - t0:.0f}s)")
    (OUT / "summary.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
