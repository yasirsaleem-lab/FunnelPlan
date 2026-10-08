#!/usr/bin/env python
"""
Stronger routers. Answers the reviewer question "is the null result an artifact of a weak router?"

Same protocol as 10 (imported from it): same queries, plans, mean plan costs, 5 fold held out evaluation
over the same 5 seeds, the same hull baseline chosen on the training folds, the same tests and Holm family.

Routers (all predict from the QUESTION only, before any retrieval, like the methods in 10):

  ar_tfidf   Adaptive RAG router, TF-IDF + logistic regression, with the paper's silver label rule
             (simplest rung that answers; if none answers, single hop data -> retrieval rung, multi hop data
             -> strongest rung). 10's adaptive_rag sends unsolved queries to the strongest rung instead.
  ar_emb     same ladder and labels, classifier = logistic regression on BGE-M3 query embeddings
  ar_t5      same ladder and labels, classifier = fine-tuned FLAN-T5 Large (text to label A/B/C), the
             classifier family Adaptive RAG uses. One fine-tune per fold and seed (100 per reader, GPU).
  plan_emb   per plan success model: logistic regression on BGE-M3 embeddings (PCA 64) for every plan;
             pick argmax p(success) - lambda * cost; lambda tuned on out of fold predictions within training
  ak_paper   Adaptive-k with the published rule (largest similarity drop within the top 90% of 100 candidates,
             plus a buffer of 5 passages), on BGE-M3 or Stella scores (chosen on the training folds). Uses the
             retrieval scores, so unlike the routers above it needs the retrieval run first.
  plan_knn   k nearest training questions (cosine, BGE-M3); per plan success = smoothed neighbour mean;
             same lambda rule

Stages (set CONFIG["stage"]):
  "embed"  GPU or CPU, about 1 minute: BGE-M3 embeddings of every question -> data/processed_full/routers/
  "eval"   runs CONFIG["routers"]. ar_t5 needs the GPU and the FLAN-T5 Large weights; its predictions are
           cached per fold (data/processed_full/routers/t5_cache_<reader>.json) so a stopped run resumes.
           Without ar_t5 the eval takes about 5 to 10 minutes on CPU. With ar_t5: about 1 minute per
           fine-tune, 300 fine-tunes for 3 readers x 4 datasets x 5 seeds x 5 folds (about 5 hours).

Output: results_full/analysis/<label>/routers.csv, routers_per_query.csv, routers_summary.txt
"""

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

CORPUS_MODE = "full"
os.environ["FUNNELPLAN_CORPUS_MODE"] = CORPUS_MODE
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import settings

CONFIG = {
    "stage": "embed",                    # "embed" | "eval"
    "routers": ["ar_tfidf", "ar_emb", "plan_emb", "plan_knn", "ar_t5"],
    "readers": "all",
    "embed_model": settings.BGE_M3_MODEL,
    "t5_model": "google/flan-t5-large",  # hub id; a local folder (see T5_LOCAL) is used when it exists
    "t5_epochs": 8,
    "t5_lr": 3e-4,                       # Adafactor, constant (standard T5 fine-tuning recipe)
    "t5_batch": 16,
    "t5_max_len": 64,
    "pca_dim": 64,
    "knn_k": 25,
    "knn_prior_weight": 5.0,
    "inner_folds": 5,
    # Adaptive-k as published (Taguchi et al., EMNLP 2025): largest similarity drop searched within the top
    # 90% of the ranked candidates, then B = 5 more passages after the cut
    "ak_candidates": 100,                # every cached candidate (the paper scores the whole context)
    "ak_top_frac": 0.9,
    "ak_buffer": 5,
}

DIR = settings.PROCESSED_DATA_ROOT / "routers"
T5_LOCAL = Path.home() / "models" / "flan-t5-large"
if T5_LOCAL.exists():
    CONFIG["t5_model"] = str(T5_LOCAL)
MULTI_HOP = {"hotpotqa", "2wikimultihopqa", "musique"}

# ---- import the protocol from 10 (file name starts with a digit, so load it by path)
_spec = importlib.util.spec_from_file_location("analysis10", Path(__file__).parent / "10_analysis_full.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def log(msg):
    print(msg, flush=True)


# ======================================================================
# embeddings
# ======================================================================
def all_questions():
    """dataset -> {query_id: text}, from the retrieval caches 10 uses."""
    import pickle
    out = {}
    for f in sorted(A.CACHE.glob("*__bm25.pkl")):
        ds = f.name.split("__")[0]
        out[ds] = pickle.load(open(f, "rb"))["queries"]
    return out


def stage_embed():
    from sentence_transformers import SentenceTransformer
    import torch
    DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(CONFIG["embed_model"], device=dev)
    for ds, qs in all_questions().items():
        ids = sorted(qs)
        X = model.encode([qs[q] for q in ids], batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
        np.savez(DIR / f"emb_{ds}.npz", ids=np.array(ids), X=X.astype(np.float32))
        log(f"{ds}: {len(ids)} questions -> {X.shape}")


def load_emb(ds):
    z = np.load(DIR / f"emb_{ds}.npz", allow_pickle=False)
    return pd.DataFrame(z["X"], index=[str(i) for i in z["ids"]])


# ======================================================================
# Adaptive RAG ladder and the paper's silver labels
# ======================================================================
def ladder(B, sel):
    """Same ladder as 10: closed book, the most cost efficient retrieval plan, the best plan (training folds)."""
    cheap_ret = [p for p in B.plans if B.depth_of[p] > 0]
    acc = B.Y.loc[sel].mean()
    r1 = max(cheap_ret, key=lambda p: (acc[p] - acc.get("closed_book@0", 0)) / max(B.cost[p], 1e-6))
    r2 = acc[cheap_ret].idxmax()
    return ["closed_book@0", r1, r2] if "closed_book@0" in B.plans else [r1, r2]


def silver(B, sel, rungs):
    """Simplest rung that answers correctly. None answers: the dataset's bias (Adaptive RAG): single hop data
    -> the single retrieval rung, multi hop data -> the strongest rung."""
    fallback = len(rungs) - 1 if B.ds in MULTI_HOP else len(rungs) - 2
    Ys = B.Y.loc[sel, rungs].to_numpy()
    y = np.where(Ys.max(1) == 1, Ys.argmax(1), fallback)
    return y


def ladder_apply(B, rungs, k, idx):
    chosen = [rungs[int(i)] for i in k]
    r = B.rows(idx)
    succ = np.array([B.Yn[ri, B.pcol[p]] for ri, p in zip(r, chosen)])
    return succ, float(np.mean([B.cost[p] for p in chosen]))


def m_ar_tfidf(B):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def fit(sel):
        rungs = ladder(B, sel)
        y = silver(B, sel, rungs)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform([B.text[q] for q in sel])
        if len(set(y)) < 2:
            return (tuple(rungs), None, vec, int(y[0]))
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)
        return (tuple(rungs), clf, vec, None)

    def apply_(cfg, idx):
        rungs, clf, vec, const = cfg
        k = np.full(len(idx), const) if clf is None else clf.predict(vec.transform([B.text[q] for q in idx]))
        return ladder_apply(B, rungs, k, idx)
    return fit, apply_


def m_ar_emb(B, E):
    from sklearn.linear_model import LogisticRegression

    def fit(sel):
        rungs = ladder(B, sel)
        y = silver(B, sel, rungs)
        if len(set(y)) < 2:
            return (tuple(rungs), None, int(y[0]))
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(E.loc[sel].to_numpy(), y)
        return (tuple(rungs), clf, None)

    def apply_(cfg, idx):
        rungs, clf, const = cfg
        k = np.full(len(idx), const) if clf is None else clf.predict(E.loc[idx].to_numpy())
        return ladder_apply(B, rungs, k, idx)
    return fit, apply_


# ======================================================================
# FLAN-T5 classifier (Adaptive RAG's classifier family)
# ======================================================================
class T5Router:
    def __init__(self, reader):
        self.cache_file = DIR / f"t5_cache_{reader}.json"
        self.cache = json.load(open(self.cache_file)) if self.cache_file.exists() else {}
        self.tok = None

    def _save(self):
        tmp = self.cache_file.with_suffix(".tmp")
        json.dump(self.cache, open(tmp, "w"))
        tmp.replace(self.cache_file)

    def _load_tok(self):
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(CONFIG["t5_model"])
            # full token sequence of each label plus EOS; prediction scores each whole sequence, so it is
            # correct even if a tokenizer splits "A" into two pieces
            self.label_seqs = [self.tok(l, add_special_tokens=False).input_ids + [self.tok.eos_token_id]
                               for l in "ABC"]
            assert len({tuple(x) for x in self.label_seqs}) == 3, "label sequences collide"

    def predict_all(self, B, sel, rungs, y, seed):
        """Fine-tune on (sel, y); return the predicted rung for every query of the block (cached)."""
        key = hashlib.sha1(("|".join([B.ds, ",".join(rungs), str(seed), *sorted(map(str, sel))])).encode()).hexdigest()
        if key in self.cache:
            return dict(zip(self.cache[key]["ids"], self.cache[key]["pred"]))
        import torch
        from transformers import AutoModelForSeq2SeqLM
        from transformers.optimization import Adafactor
        self._load_tok()
        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["t5_model"]).to(dev)
        opt = Adafactor(model.parameters(), lr=CONFIG["t5_lr"], scale_parameter=False, relative_step=False,
                        warmup_init=False)
        n_cls = len(rungs)
        texts = [B.text[q] for q in sel]
        enc = self.tok(texts, max_length=CONFIG["t5_max_len"], truncation=True, padding=True, return_tensors="pt")
        L = max(len(x) for x in self.label_seqs)
        lab = torch.tensor([self.label_seqs[int(c)] + [-100] * (L - len(self.label_seqs[int(c)])) for c in y])
        n, bs = len(texts), CONFIG["t5_batch"]
        model.train()
        use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
        for ep in range(CONFIG["t5_epochs"]):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(input_ids=enc.input_ids[b].to(dev), attention_mask=enc.attention_mask[b].to(dev),
                                labels=lab[b].to(dev))
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
        model.eval()
        ids = [str(q) for q in B.q]
        enc_all = self.tok([B.text[q] for q in B.q], max_length=CONFIG["t5_max_len"], truncation=True, padding=True,
                           return_tensors="pt")
        preds = []
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                ii, am = enc_all.input_ids[i:i + 64].to(dev), enc_all.attention_mask[i:i + 64].to(dev)
                scores = []
                for c in range(n_cls):                      # log likelihood of each full label sequence
                    seq = torch.tensor([self.label_seqs[c]] * len(ii), device=dev)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        lg = model(input_ids=ii, attention_mask=am, labels=seq).logits.float()
                    lp = torch.log_softmax(lg, -1).gather(2, seq.unsqueeze(2)).squeeze(2).sum(1)
                    scores.append(lp)
                preds += torch.stack(scores, 1).argmax(1).cpu().tolist()
        del model, opt
        if dev == "cuda":
            torch.cuda.empty_cache()
        self.cache[key] = {"ids": ids, "pred": preds}
        self._save()
        return dict(zip(ids, preds))


def m_ar_t5(B, router):
    def fit(sel):
        rungs = ladder(B, sel)
        y = silver(B, sel, rungs)
        if len(set(y)) < 2:
            return (tuple(rungs), None, int(y[0]))
        t0 = time.time()
        pred = router.predict_all(B, sel, rungs, y, A._CUR_SEED[0])
        if time.time() - t0 > 1:
            log(f"      t5 fine-tune {B.ds} seed {A._CUR_SEED[0]}: {time.time() - t0:.0f}s")
        return (tuple(rungs), pred, None)

    def apply_(cfg, idx):
        rungs, pred, const = cfg
        k = np.full(len(idx), const) if pred is None else np.array([pred[str(q)] for q in idx])
        return ladder_apply(B, rungs, k, idx)
    return fit, apply_


# ======================================================================
# per plan routers: predicted success per plan, then argmax p - lambda * cost
# ======================================================================
def pick_lambda(B, sel, P_oof):
    """lambda that maximises the gain over the training fold hull, using OUT OF FOLD predictions."""
    c = B.cost[B.plans].to_numpy()
    r = B.rows(sel)
    best = None
    for lam in A.CONFIG["lambdas"]:
        k = (P_oof - lam * c[None, :]).argmax(1)
        s = B.Yn[r, [B.pcol[B.plans[i]] for i in k]]
        cost = c[k].mean()
        g = s.mean() - A.hull_at(B.Y, B.cost, sel, cost)
        if best is None or g > best[0]:
            best = (g, lam)
    return best[1]


def plan_apply(B, P, lam, idx):
    c = B.cost[B.plans].to_numpy()
    k = (P - lam * c[None, :]).argmax(1)
    r = B.rows(idx)
    return B.Yn[r, [B.pcol[B.plans[i]] for i in k]], float(c[k].mean())


def inner_splits(n, seed):
    rng = np.random.default_rng(seed + 1000)
    perm = rng.permutation(n)
    k = CONFIG["inner_folds"]
    return [perm[i::k] for i in range(k)]


def m_plan_emb(B, E):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    def train_predict(Xtr, Ytr, Xte):
        pca = PCA(n_components=min(CONFIG["pca_dim"], len(Xtr) - 1), random_state=0).fit(Xtr)
        Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
        P = np.empty((len(Xte), Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            yj = Ytr[:, j]
            if yj.min() == yj.max():
                P[:, j] = yj[0]
            else:
                P[:, j] = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, yj).predict_proba(Zte)[:, 1]
        return P

    def fit(sel):
        X = E.loc[sel].to_numpy()
        Y = B.Y.loc[sel, B.plans].to_numpy()
        oof = np.empty(Y.shape)
        for te in inner_splits(len(sel), A._CUR_SEED[0]):
            tr = np.setdiff1d(np.arange(len(sel)), te)
            oof[te] = train_predict(X[tr], Y[tr], X[te])
        lam = pick_lambda(B, sel, oof)
        return (np.array(sel), float(lam))

    def apply_(cfg, idx):
        tr, lam = cfg
        P = train_predict(E.loc[tr].to_numpy(), B.Y.loc[tr, B.plans].to_numpy(), E.loc[idx].to_numpy())
        return plan_apply(B, P, lam, idx)
    return fit, apply_


def m_plan_knn(B, E):
    k, a = CONFIG["knn_k"], CONFIG["knn_prior_weight"]

    def predict(Xtr, Ytr, Xte, exclude_self=False):
        S = Xte @ Xtr.T                                   # cosine (embeddings are normalised)
        if exclude_self:
            np.fill_diagonal(S, -np.inf)
        nn = np.argpartition(-S, k, axis=1)[:, :k]
        prior = Ytr.mean(0)
        return (Ytr[nn].sum(1) + a * prior[None, :]) / (k + a)

    def fit(sel):
        X = E.loc[sel].to_numpy()
        Y = B.Y.loc[sel, B.plans].to_numpy()
        oof = predict(X, Y, X, exclude_self=True)          # leave one out within training
        return (np.array(sel), float(pick_lambda(B, sel, oof)))

    def apply_(cfg, idx):
        tr, lam = cfg
        P = predict(E.loc[tr].to_numpy(), B.Y.loc[tr, B.plans].to_numpy(), E.loc[idx].to_numpy())
        return plan_apply(B, P, lam, idx)
    return fit, apply_


# ======================================================================
# Adaptive-k with the published rule (10's adaptive_k is the simpler "largest gap in the top 20" variant)
# ======================================================================
def m_ak_paper(B):
    avail = sorted({int(c.rsplit("@", 1)[1]) for c in B.Y.columns if "@" in c} - {0})
    N, frac, buf = CONFIG["ak_candidates"], CONFIG["ak_top_frac"], CONFIG["ak_buffer"]

    def k_paper(r, q):
        s = np.array([x["score"] for x in B.cache[r]["results"][q][:N]], dtype=float)
        if len(s) < 2:
            return avail[0]
        m = max(2, int(np.floor(frac * len(s))))          # gaps between neighbours inside the top 90%
        k = int(np.argmax(s[:m - 1] - s[1:m])) + 1 + buf   # cut at the largest drop, plus the buffer
        k = min(k, len(s))
        return min(avail, key=lambda d: (abs(d - k), -d))  # nearest labelled depth (ties go deeper)

    retrs = [r for r in A.CONFIG["adaptive_k_retrievers"] if r in B.cache]
    if not retrs:
        return None
    memo = {}

    def run(r, idx):
        chosen = [f"{r}@{memo.setdefault((r, q), k_paper(r, q))}" for q in idx]
        rr = B.rows(idx)
        return (np.array([B.Yn[i, B.pcol[p]] for i, p in zip(rr, chosen)]),
                float(np.mean([B.cost[p] for p in chosen])))

    def fit(sel):          # the only choice: which embedding retriever, picked on the training folds
        return (max(retrs, key=lambda r: (lambda sc: sc[0].mean() - A.hull_at(B.Y, B.cost, sel, sc[1]))(run(r, sel))),)

    return fit, (lambda cfg, idx: run(cfg[0], idx))


# ======================================================================
def stage_eval():
    t0 = time.time()
    out = settings.RESULTS_ROOT / "analysis" / A.CONFIG["label"]
    readers = A.readers() if CONFIG["readers"] == "all" else CONFIG["readers"]
    rows, perq, L = [], [], []
    L.append(f"Routers, same protocol as 10 (label {A.CONFIG['label']}, seeds {A.CONFIG['seeds']}, "
             f"{A.CONFIG['k_folds']} folds, plan cost = {A.CONFIG.get('cost_stat', 'median')}).")
    for rd in readers:
        df, col = A.load_reader(rd)
        t5 = T5Router(rd) if "ar_t5" in CONFIG["routers"] else None
        L.append(f"== reader {rd}")
        log(L[-1])
        for ds in sorted(df["dataset"].unique()):
            B = A.Block(df, ds)
            E = None
            if set(CONFIG["routers"]) & {"ar_emb", "plan_emb", "plan_knn"}:
                E = load_emb(ds).reindex([str(q) for q in B.q])
                E.index = B.q
                assert E.notna().all().all(), f"missing embeddings for {ds}; run stage 'embed'"
            makers = {"ar_tfidf": lambda: m_ar_tfidf(B), "ar_emb": lambda: m_ar_emb(B, E),
                      "plan_emb": lambda: m_plan_emb(B, E), "plan_knn": lambda: m_plan_knn(B, E),
                      "ar_t5": lambda: m_ar_t5(B, t5), "ak_paper": lambda: m_ak_paper(B)}
            for name in CONFIG["routers"]:
                log(f"    {ds} {name} ... ({time.time() - t0:.0f}s)")
                d, seed_means, halves, cfgs, qids = A.repeated(B, makers[name])
                lo, hi = A.boot_ci(d, A.CONFIG["n_boot"], A.CONFIG["seed"])
                rows.append({"reader": rd, "dataset": ds, "method": name, "gain": d.mean(), "ci_low": lo,
                             "ci_high": hi, "p": A.p_two_sided(d), "seed_min": seed_means.min(),
                             "seed_max": seed_means.max(), "success": np.mean([h["success"] for h in halves]),
                             "cost_s": np.mean([h["cost"] for h in halves]),
                             "gain_selection_half": np.mean([h["gain_selection_half"] for h in halves])})
                perq.append(pd.DataFrame({"reader": rd, "dataset": ds, "method": name,
                                          "query_id": [str(q) for q in qids], "gain": d}))
                r = rows[-1]
                L.append(f"  {ds:16s} {name:9s} gain {r['gain']:+.3f} [{lo:+.3f}, {hi:+.3f}]  seeds "
                         f"{r['seed_min']:+.3f}..{r['seed_max']:+.3f}  (training folds {r['gain_selection_half']:+.3f})"
                         f"  cost {r['cost_s']:.2f}s")
                log(L[-1])
    R = pd.DataFrame(rows)
    PQ = pd.concat(perq, ignore_index=True)
    # merge with routers from earlier runs (a run replaces only the routers it computed)
    if (out / "routers.csv").exists() and (out / "routers_per_query.csv").exists():
        R0 = pd.read_csv(out / "routers.csv")
        PQ0 = pd.read_csv(out / "routers_per_query.csv", dtype={"query_id": str})
        keep = lambda D: D[~(D.method.isin(CONFIG["routers"]) & D.reader.isin(readers))]
        R = pd.concat([keep(R0).drop(columns=["p_holm"], errors="ignore"), R], ignore_index=True)
        PQ = pd.concat([keep(PQ0), PQ], ignore_index=True)
        L.append(f"(merged with earlier runs: routers now {sorted(R.method.unique())})")

    # Holm over ONE family: 10's dataset level tests plus these
    M10 = pd.read_csv(out / "methods.csv")
    fam = pd.concat([M10[M10.dataset != "pooled"][["reader", "dataset", "method", "p"]],
                     R[["reader", "dataset", "method", "p"]]], ignore_index=True)
    fam["p_holm"] = A.holm(fam["p"].to_numpy())
    R["p_holm"] = fam["p_holm"].to_numpy()[-len(R):]
    L.append(f"\n== Holm over {len(fam)} dataset level tests (10's methods plus these routers)")
    for r in R[R.p < 0.05].sort_values("p").itertuples():
        L.append(f"  {r.reader:12s} {r.dataset:16s} {r.method:9s} gain {r.gain:+.3f} p {r.p:.4f} Holm p {r.p_holm:.4f}"
                 f"  {'SURVIVES' if r.p_holm < 0.05 else 'does not survive'}")

    # pooled per reader, and paired against 10's adaptive_rag (TF-IDF, strongest rung rule) and planner_tsr
    L.append("\n== pooled over datasets")
    PQ10 = pd.read_csv(out / "per_query_gains.csv", dtype={"query_id": str})
    pooled, pairs = [], []
    for (rd, m), g in PQ.groupby(["reader", "method"], sort=False):
        d = g["gain"].to_numpy()
        lo, hi = A.boot_ci(d, A.CONFIG["n_boot"], A.CONFIG["seed"])
        pooled.append({"reader": rd, "method": m, "gain": d.mean(), "ci_low": lo, "ci_high": hi,
                       "p": A.p_two_sided(d)})
        for ref in ("adaptive_rag", "planner_tsr"):
            o = PQ10[(PQ10.reader == rd) & (PQ10.method == ref)]
            j = g.merge(o, on=["dataset", "query_id"], suffixes=("", "_ref"))
            if len(j):
                diff = (j["gain"] - j["gain_ref"]).to_numpy()
                lo2, hi2 = A.boot_ci(diff, A.CONFIG["n_boot"], A.CONFIG["seed"])
                pairs.append({"reader": rd, "a": m, "b": ref, "diff": diff.mean(), "ci_low": lo2, "ci_high": hi2,
                              "p": A.p_two_sided(diff)})
    P = pd.DataFrame(pooled)
    P["p_holm"] = A.holm(P["p"].to_numpy())
    for r in P.itertuples():
        L.append(f"  {r.reader:12s} {r.method:9s} {r.gain:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  Holm p {r.p_holm:.4f}")
    Q = pd.DataFrame(pairs)
    if len(Q):
        Q["p_holm"] = A.holm(Q["p"].to_numpy())
        L.append("\n== paired: router minus 10's method (same queries and seeds; Holm over these pairs)")
        for r in Q.itertuples():
            L.append(f"  {'*' if r.p_holm < 0.05 else ' '} {r.reader:12s} {r.a:9s} minus {r.b:13s} {r.diff:+.3f} "
                     f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]  Holm p {r.p_holm:.4f}")
    L.append("\nAll routers see only the question. Gain = held out success minus the held out hull at equal mean "
             "cost. A router beats the hull only if the CI is above 0 and Holm p < 0.05.")
    R.to_csv(out / "routers.csv", index=False)
    P.to_csv(out / "routers_pooled.csv", index=False)
    Q.to_csv(out / "routers_paired.csv", index=False)
    PQ.to_csv(out / "routers_per_query.csv", index=False)
    (out / "routers_summary.txt").write_text("\n".join(L), encoding="utf-8")
    log("\n".join(L[L.index(next(x for x in L if x.startswith("\n== Holm"))):]))
    log(f"\nDone in {time.time() - t0:.0f}s. Written {out / 'routers_summary.txt'}")


if __name__ == "__main__":
    # command line overrides, so the file never needs editing:
    #   python scripts/13_routers.py embed
    #   python scripts/13_routers.py eval ar_tfidf,ar_emb,plan_emb,plan_knn
    #   python scripts/13_routers.py eval ar_t5
    if len(sys.argv) > 1:
        CONFIG["stage"] = sys.argv[1]
    if len(sys.argv) > 2:
        CONFIG["routers"] = sys.argv[2].split(",")
    log(f"stage {CONFIG['stage']}; routers {CONFIG['routers']}; T5 weights {CONFIG['t5_model']}")
    {"embed": stage_embed, "eval": stage_eval}[CONFIG["stage"]]()
