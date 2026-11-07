"""
Human audit 4: more answers on which a policy and its hull disagree. Plan: docs/ANALYSIS_PLAN_2026-11-05.md, A4.

Tests:
  conf_cascade  Qwen2.5 7B, single step plans: 100 further discordant questions (25 per dataset) that audit 3 did
                not sample, from the same seed 0 run (audit3_questions.csv). With audit 3 this gives 200.
  ircot         Qwen3 14B on 2Wiki against the single step hull at IRCoT's own cost (seed 0 folds of
                15_h3_ircot.py): 50 discordant questions. Above the cost of every plan the hull is the best plan on
                the training folds, so the hull's answer is that plan's answer.
Sampling seed 4. Same guidelines, same raters as audit 3 (raterA, raterB), blind to test, arm, reader and labels.
An answer that audit 3 already rated (same dataset, question and normalised answer) is not rated again; its
audit 3 label is used.

Stages (files in data/processed_full/grading/audit4/):
  make        audit4_raterA.xlsx, audit4_raterB.xlsx, audit4_key.csv, audit4_questions.csv (never send the key or
              the questions file to a rater)
  adjudicate  after audit4_raterA_received.xlsx and audit4_raterB_received.xlsx are back
  finalize    after audit4_adjudication_received.xlsx is back (if there are disagreements); writes
              results_full/analysis/audit4/: the shift of label_final on the gain per test, the confidence cascade
              with audits 3 and 4 together (200 questions)
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import settings

STAGE = sys.argv[1] if len(sys.argv) > 1 else ""
SEED = 4
D3 = settings.PROCESSED_DATA_ROOT / "grading" / "audit3"
DIR = settings.PROCESSED_DATA_ROOT / "grading" / "audit4"
OUT = settings.RESULTS_ROOT / "analysis" / "audit4"
CC = "conf_cascade|qwen2.5-7b|single step"
IR = "ircot|qwen3-14b|2wikimultihopqa, single step hull"


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


A29 = _load("audit29", "29_discordant_audit.py")
for k, v in {"DIR": DIR, "OUT": OUT, "KEY": DIR / "audit4_key.csv", "QUESTIONS": DIR / "audit4_questions.csv",
             "SHEETS": {"A": DIR / "audit4_raterA.xlsx", "B": DIR / "audit4_raterB.xlsx"},
             "RECEIVED": {"A": DIR / "audit4_raterA_received.xlsx", "B": DIR / "audit4_raterB_received.xlsx"},
             "ADJ": DIR / "audit4_adjudication.xlsx", "ADJ_RECEIVED": DIR / "audit4_adjudication_received.xlsx"}.items():
    setattr(A29, k, v)
A29.GUIDE = [A29.GUIDE[0].replace("audit 3", "audit 4")] + A29.GUIDE[1:]


def ircot_questions(G):
    """Seed 0 folds of 15_h3_ircot.py for Qwen3 14B on 2Wiki; the hull's plan per question at IRCoT's cost."""
    A = _load("analysis11", "11_analysis_full.py")
    A.CONFIG["include_r3"] = False
    df, col = A.load_reader("qwen3-14b")
    ds = "2wikimultihopqa"
    B = A.Block(df, ds)
    ir = pd.read_csv(settings.PROCESSED_DATA_ROOT / "r4_ircot" / "labels_qwen3-14b_graded.csv",
                     dtype={"query_id": str}, keep_default_na=False)
    ir = ir[ir.dataset == ds].set_index("query_id")
    ir["cost"] = pd.to_numeric(ir.agent_time_s) + pd.to_numeric(ir.retrieval_time_s)
    q_all = np.array([q for q in B.q if q in ir.index])
    rng = np.random.default_rng(0)
    q = q_all.copy()
    rng.shuffle(q)
    k = A.CONFIG["k_folds"]
    folds = [q[i::k] for i in range(k)]
    coin = np.random.default_rng(2026)
    rows = []
    for i in range(k):
        test = folds[i]
        sel = np.concatenate([folds[j] for j in range(k) if j != i])
        c = float(ir.loc[test, "cost"].mean())
        H = A._hull_vertices(B.Y, B.cost, sel)
        xs = [h[0] for h in H]
        if c >= xs[-1]:
            hull = [H[-1][2]] * len(test)
        else:                                          # not expected: IRCoT costs more than every plan
            j = max(int(np.searchsorted(xs, c, side="right")) - 1, 0)
            t = (c - xs[j]) / (xs[j + 1] - xs[j])
            hull = list(np.where(coin.random(len(test)) < t, H[j + 1][2], H[j][2]))
        rows.append(pd.DataFrame({"query_id": test, "fold": i, "hull_plan": hull}))
    Q = pd.concat(rows, ignore_index=True)
    lab = df[df.dataset == ds].set_index(["plan", "query_id"])
    for c_ in ("answer", "label_final", "label_strict"):
        Q[f"hull_{c_}"] = lab[c_].reindex(list(zip(Q.hull_plan, Q.query_id))).to_numpy()
        Q[f"policy_{c_}"] = ir[c_].reindex(Q.query_id).to_numpy()
    Q["policy_plan"] = "ircot"
    Q["dataset"], Q["test"] = ds, IR
    return Q


def stage_make():
    if A29.KEY.exists():
        sys.exit(f"{A29.KEY} exists; the audit is never resampled.")
    DIR.mkdir(parents=True, exist_ok=True)
    G = _load("grading07", "07_grading_full.py")
    rng = np.random.default_rng(SEED)
    Q3 = pd.read_csv(D3 / "audit3_questions.csv", dtype={"query_id": str}, keep_default_na=False)
    cc = Q3[Q3.test == CC].copy()
    cc["sampled_audit3"] = cc.sampled.astype(str) == "True"
    cc["discordant"] = cc.discordant.astype(str) == "True"
    cc["sampled"] = False
    for ds, g in cc[cc.discordant & ~cc.sampled_audit3].groupby("dataset"):
        cc.loc[g.index[rng.permutation(len(g))[:25]], "sampled"] = True
    ir = ircot_questions(G)
    for arm in ("policy", "hull"):
        ir[f"{arm}_norm"] = ir[f"{arm}_answer"].astype(str).map(G.normalize)
    ir["discordant"] = ir.policy_norm != ir.hull_norm
    ir["sampled"] = False
    ir.loc[ir.index[ir.discordant][rng.permutation(int(ir.discordant.sum()))[:50]], "sampled"] = True
    Qs = pd.concat([cc, ir], ignore_index=True)
    Qs.to_csv(A29.QUESTIONS, index=False)
    S = Qs[Qs.sampled]
    items = pd.concat([S.assign(answer=S[f"{arm}_answer"], norm=S[f"{arm}_norm"])[["dataset", "query_id", "answer", "norm"]]
                       for arm in ("policy", "hull")]).drop_duplicates(["dataset", "query_id", "norm"])
    k3 = pd.read_csv(D3 / "audit3_key.csv", dtype={"query_id": str}, keep_default_na=False)
    seen = set(zip(k3.dataset, k3.query_id, k3.norm))
    items["in_audit3"] = [(d, q, n) in seen for d, q, n in zip(items.dataset, items.query_id, items.norm)]
    items = items[~items.in_audit3].reset_index(drop=True)
    items["item_id"] = np.arange(4001, 4001 + len(items))
    gold = G.gold_maps(items.dataset.unique())
    items["gold_answers"] = [" | ".join(gold[d].get(q, [])) for d, q in zip(items.dataset, items.query_id)]
    qt = G.question_texts(items)
    items["question"] = [qt.get((d, q), "") for d, q in zip(items.dataset, items.query_id)]
    items[["item_id", "dataset", "query_id", "norm", "answer"]].to_csv(A29.KEY, index=False)
    sheet = items[["item_id", "dataset", "question", "gold_answers", "answer"]].rename(columns={"answer": "model_answer"})
    for i, (r, path) in enumerate(A29.SHEETS.items()):
        A29.write_sheet(sheet.iloc[np.random.default_rng(500 + i).permutation(len(sheet))], path, f"Rater {r}")
    print(Qs.groupby("test").agg(questions=("query_id", "size"), discordant=("discordant", "sum"),
                                 sampled=("sampled", "sum")).to_string())
    print(f"{len(items)} items to rate (answers already rated in audit 3 are reused). Sheets in {DIR}: "
          f"{A29.SHEETS['A'].name}, {A29.SHEETS['B'].name}. Never send {A29.KEY.name} or {A29.QUESTIONS.name}.")


def _bias(g, share, rng):
    """Shift of an automatic rule on the gain: mean over discordant questions of (rule gain minus human gain),
    times the share of discordant questions; bootstrap interval over questions."""
    r = {"questions": len(g), "discordant_share": share}
    for arm in ("policy", "hull"):
        for rule in ("label_final", "label_strict"):
            x, h = g[f"{arm}_{rule}"].to_numpy(float), g[f"{arm}_human"].to_numpy(float)
            r[f"{arm}_{rule}_false_pos"] = float(np.mean((x == 1) & (h == 0)))
            r[f"{arm}_{rule}_false_neg"] = float(np.mean((x == 0) & (h == 1)))
            r[f"{arm}_{rule}_error"] = float(np.mean(x != h))
        r[f"{arm}_human_correct"] = float(g[f"{arm}_human"].mean())
    for rule in ("label_final", "label_strict"):
        d = ((g[f"policy_{rule}"] - g[f"hull_{rule}"]) - (g["policy_human"] - g["hull_human"])).to_numpy(float)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        r[f"bias_{rule}_on_gain"], r[f"bias_{rule}_on_gain_low"], r[f"bias_{rule}_on_gain_high"] = (
            d.mean() * share, lo * share, hi * share)
    return r


def stage_finalize():
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = A29.read_labels(A29.RECEIVED["A"]), A29.read_labels(A29.RECEIVED["B"])
    b = b.reindex(a.index)
    C = A29.LABEL_COL
    human = a[C].where(a[C] == b[C])
    dis = human.index[human.isna()]
    n_blank = int((a[C].isna() | b[C].isna()).sum())
    if len(dis):
        if not A29.ADJ_RECEIVED.exists():
            sys.exit(f"{len(dis)} disagreements; run 'adjudicate' and return {A29.ADJ_RECEIVED.name}")
        human.loc[dis] = A29.read_labels(A29.ADJ_RECEIVED)[C].reindex(dis)
        if human.isna().any():
            sys.exit(f"{int(human.isna().sum())} adjudication items unlabelled")
    key = pd.read_csv(A29.KEY, dtype={"query_id": str}, keep_default_na=False)
    key["human"] = human.reindex(key.item_id).to_numpy()
    # audit 3 labels for answers already rated there
    S3 = pd.read_csv(settings.RESULTS_ROOT / "analysis" / "audit3" / "audit3_questions_labelled.csv",
                     dtype={"query_id": str}, keep_default_na=False)
    h3 = pd.concat([S3[["dataset", "query_id", f"{arm}_norm", f"{arm}_human"]].set_axis(
        ["dataset", "query_id", "norm", "human"], axis=1) for arm in ("policy", "hull")])
    hmap = pd.concat([h3, key[["dataset", "query_id", "norm", "human"]]]).drop_duplicates(
        ["dataset", "query_id", "norm"]).set_index(["dataset", "query_id", "norm"]).human
    hmap = pd.to_numeric(hmap)
    Qs = pd.read_csv(A29.QUESTIONS, dtype={"query_id": str}, keep_default_na=False)
    for c in ("sampled", "discordant", "sampled_audit3"):
        Qs[c] = Qs[c].astype(str) == "True"
    for arm in ("policy", "hull"):
        for rule in ("label_final", "label_strict"):
            Qs[f"{arm}_{rule}"] = pd.to_numeric(Qs[f"{arm}_{rule}"])
        Qs[f"{arm}_human"] = hmap.reindex(list(zip(Qs.dataset, Qs.query_id, Qs[f"{arm}_norm"].astype(str)))).to_numpy()
    rng = np.random.default_rng(0)
    x, y = A29.both(a, b)
    L = [f"Audit 4: items {len(key)}; labelled by both raters {len(x)}; raters agree on {(x == y).mean():.3f}, "
         f"kappa {A29.kappa(x, y):.3f}; {len(dis)} adjudicated ({len(dis) - n_blank} disagreements, {n_blank} blank)."]
    cc, ir = Qs[Qs.test == CC], Qs[Qs.test == IR]
    sets = {"conf_cascade audit 4 only": (cc[cc.sampled], cc.discordant.mean()),
            "conf_cascade audits 3 and 4 (pooled)": (cc[cc.sampled | cc.sampled_audit3], cc.discordant.mean()),
            "ircot qwen3-14b 2wiki": (ir[ir.sampled], ir.discordant.mean())}
    T = pd.read_csv(settings.RESULTS_ROOT / "analysis" / "inference" / "tests_nb.csv")
    t = T[(T.family == "primary") & (T.plan_set == "single step") & (T.reader == "qwen2.5-7b") & (T.method == "conf_cascade")].iloc[0]
    H = pd.read_csv(settings.RESULTS_ROOT / "analysis" / "h3_ircot" / "h3.csv")
    h = H[(H.hull == "single_step") & (H.reader == "qwen3-14b") & (H.dataset == "2wikimultihopqa")].iloc[0]
    base = {"conf_cascade": (t.gain, t.ub95_plain, t.ub95_nb), "ircot": (h.gain, h.ci_high, h.refit_high)}
    rows = []
    for name, (g, share) in sets.items():
        miss = int(g[["policy_human", "hull_human"]].isna().any(axis=1).sum())
        if miss:
            sys.exit(f"{name}: {miss} sampled questions have no human label")
        r = {"set": name, **_bias(g, share, rng)}
        gain, ub1, ub2 = base[name.split()[0]]
        bshift = r["bias_label_final_on_gain"]
        r.update(gain=gain, gain_adjusted=gain - bshift, ub_plain=ub1, ub_plain_adjusted=ub1 - bshift,
                 ub_second=ub2, ub_second_adjusted=ub2 - bshift,
                 ub_plain_adjusted_worst=ub1 - r["bias_label_final_on_gain_low"])
        rows.append(r)
        L.append(f"\n== {name}: {len(g)} discordant questions; {share:.3f} of held out questions are discordant")
        for arm in ("policy", "hull"):
            L.append(f"  {arm:6s} human correct {r[f'{arm}_human_correct']:.3f}; label_final error "
                     f"{r[f'{arm}_label_final_error']:.3f} (false pos {r[f'{arm}_label_final_false_pos']:.3f}, "
                     f"false neg {r[f'{arm}_label_final_false_neg']:.3f}); strict error {r[f'{arm}_label_strict_error']:.3f}")
        L.append(f"  shift of label_final on the gain {bshift:+.4f} [{r['bias_label_final_on_gain_low']:+.4f}, "
                 f"{r['bias_label_final_on_gain_high']:+.4f}] (positive: label_final overstates the policy); strict "
                 f"{r['bias_label_strict_on_gain']:+.4f}")
        L.append(f"  gain {gain:+.4f} -> {gain - bshift:+.4f} without the shift; upper bound (per question) {ub1:.4f} -> "
                 f"{ub1 - bshift:.4f} (end of the shift interval: {r['ub_plain_adjusted_worst']:.4f}); "
                 f"{'corrected' if name.startswith('conf') else 'refit'} bound {ub2:.4f} -> {ub2 - bshift:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "audit4_results.csv", index=False)
    Qs.to_csv(OUT / "audit4_questions_labelled.csv", index=False)
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if STAGE == "make":
        stage_make()
    elif STAGE == "adjudicate":
        A29.stage_adjudicate()
    elif STAGE == "finalize":
        stage_finalize()
    else:
        sys.exit("usage: python scripts/35_discordant_audit4.py make | adjudicate | finalize")
