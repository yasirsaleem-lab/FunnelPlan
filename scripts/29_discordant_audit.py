"""
Human audit of the answers on which a policy and its hull disagree.

Concern: the grading audits so far sampled answers in general. A difference in grading error between the two arms
acts only where the policy and the hull give different answers. This script audits those answers
for the tests nearest a decision, by at least two raters who are not authors, with written guidelines and
adjudication, and error rates per arm.

Tests audited (the paper's protocol, seed 0, the fold split of 11_analysis_full.py):
  conf_cascade    Qwen2.5 7B, single step plans (largest single step upper bound in the paper, 0.029)
  reader_cascade  Qwen2.5 7B to Qwen3 14B, with two hop plans (largest two hop upper bound, 0.032)
  plan_emb        Qwen3 32B, single step plans (the router closest to significance)
For every held out question the policy's plan and the hull's plan are recovered (the hull runs vertex a or b; the
coin is drawn with the hull's weight and a fixed seed). A question is discordant when the two answers differ after
normalisation. Per test, N discordant questions are sampled, balanced over the four datasets; each gives two items
(the policy's answer and the hull's answer). Identical answers to the same question are rated once.

Stages:
  make      writes the rating sheets (one per rater, own random order, blind to test, arm, reader and every label),
            the guidelines, and the key (never shown to raters).
            data/processed_full/grading/audit3/audit3_raterA.xlsx, audit3_raterB.xlsx, audit3_key.csv
  adjudicate  after both sheets are back (audit3_raterA_received.xlsx, audit3_raterB_received.xlsx in the same
            folder): kappa between the raters, and audit3_adjudication.xlsx with the items they disagree on.
  finalize  after the adjudication sheet is back (audit3_adjudication_received.xlsx): the human label is the
            agreed label, or the adjudicated one. Reports, per test and arm, the error of label_final (and of the
            strict rule) against the human label, and the bias this error puts on the gain:
              bias = [mean label_final(policy) - mean label_final(hull)] - [mean human(policy) - mean human(hull)]
            on the discordant questions, times the share of discordant questions (concordant questions carry the
            same answer in both arms, so their grading error cancels). Bootstrap intervals over questions.
            Output: results_full/analysis/audit3/

Usage:  python scripts/29_discordant_audit.py make [--n 100]
        python scripts/29_discordant_audit.py adjudicate [--blank-to-adjudication]
                (with the flag, an item one rater left blank goes to adjudication like a disagreement; kappa and
                 agreement are then computed on the items both raters labelled, and the summary says how many)
        python scripts/29_discordant_audit.py finalize
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

ARGS = sys.argv[1:]
STAGE = ARGS[0] if ARGS else ""


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


N_PER_TEST = int(arg("--n", "100"))
DIR = Path(os.environ.get("AUDIT3_DIR", settings.PROCESSED_DATA_ROOT / "grading" / "audit3"))   # env: tests only
OUT = settings.RESULTS_ROOT / "analysis" / "audit3"
KEY = DIR / "audit3_key.csv"
QUESTIONS = DIR / "audit3_questions.csv"
SHEETS = {"A": DIR / "audit3_raterA.xlsx", "B": DIR / "audit3_raterB.xlsx"}
RECEIVED = {"A": DIR / "audit3_raterA_received.xlsx", "B": DIR / "audit3_raterB_received.xlsx"}
ADJ = DIR / "audit3_adjudication.xlsx"
ADJ_RECEIVED = DIR / "audit3_adjudication_received.xlsx"
TESTS = [("conf_cascade", "qwen2.5-7b", False), ("reader_cascade", ("qwen2.5-7b", "qwen3-14b"), True),
         ("plan_emb", "qwen3-32b", False)]
LABEL_COL = "label_1_correct_0_wrong"

GUIDE = [
    "GRADING GUIDELINES (audit 3). Read once before you start; keep this sheet open while you rate.",
    "",
    "Task. Each row has a question, one or more gold answers (separated by ' | '), and one model answer.",
    "Write 1 in the label column if the model answer is correct, 0 if it is wrong. Use the dropdown.",
    "",
    "1. Judge meaning, not wording. 1 = the model answer states the same fact as at least one gold answer.",
    "2. Extra words, a full sentence, or extra correct detail are fine, as long as the answer itself is right.",
    "3. Different entity, person, place, number or date = 0. A wrong year or a wrong count is wrong.",
    "4. A hedge between options ('either X or Y'), a list of guesses, or 'X or possibly Y' = 0, even if one is gold.",
    "5. A refusal ('the passages do not say', 'unknown') = 0. A refusal FOLLOWED by a best guess: grade the guess.",
    "6. Too vague to identify the gold (e.g. 'a city in Europe' for 'Paris') = 0.",
    "7. Partial answers: if the question asks for one thing and the answer gives it, 1. If it asks for two things",
    "   (e.g. 'who and when') and only one is right, 0.",
    "8. Names: a shorter or longer form of the same name is 1 ('Lincoln' for 'Abraham Lincoln') unless it",
    "   could equally mean someone else in context. Spelling slips are fine if the entity is clear.",
    "9. Numbers and dates: equivalent forms are 1 ('1st May 1990' = 'May 1, 1990'; 'six' = '6'). Rounding is 1",
    "   only if the gold itself is approximate.",
    "10. Yes/no questions: the answer must clearly say yes or no and match the gold.",
    "11. If the gold answer itself looks wrong, still grade against the gold, and write a note.",
    "12. Do not look anything up and do not use any AI tool. Rate on your own; do not discuss items with the",
    "    other rater until both sheets are returned.",
    "13. If you cannot decide, pick the more likely label and write 'unsure' in the notes column.",
    "",
    "You will not see which system produced an answer, and the rows are in random order. That is deliberate.",
    "",
    "Examples",
    "Q: Who wrote Hamlet? Gold: William Shakespeare. 'Shakespeare' -> 1. 'Shakespeare or Marlowe' -> 0.",
    "Q: When was X born? Gold: 12 March 1950. 'March 1950' -> 0 (the day is missing and the question asks a date).",
    "   'Born in 1950' -> 0. '12/03/1950' -> 1.",
    "Q: How many members? Gold: 7. 'Seven members.' -> 1. 'About 7 or 8' -> 0.",
    "Q: Where did Y die? Gold: Boston. 'The passages do not mention it. Best guess: Boston.' -> 1.",
]


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    m = importlib.util.module_from_spec(s)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        s.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


# ====================================================================== make
def choices_one_block(K, A, B, method, kind, E=None):
    """Seed 0 run of the paper's protocol on block B. Returns a frame per held out question: policy plan, hull plan
    (the coin drawn with the hull weight), and the fold. Checks that the recovered plans reproduce the policy's
    success and the hull's expected success."""
    fit, apply_ = method
    k = A.CONFIG["k_folds"]
    seed = 0
    for m_ in K.MODS:
        m_._CUR_SEED[0] = seed
    rng = np.random.default_rng(seed)
    q = B.q.copy()
    rng.shuffle(q)
    folds = [q[i::k] for i in range(k)]
    coin = np.random.default_rng(2026)
    rows = []
    for i in range(k):
        test = folds[i]
        sel = np.concatenate([folds[j] for j in range(k) if j != i])
        cfg = fit(sel)
        s, c = apply_(cfg, test)
        # policy plan per question
        if kind == "conf_cascade":
            a, b, tau = cfg
            conf = np.nan_to_num(B.Cn[B.rows(test), B.pcol[a]], nan=-99.0)
            pol = np.where(conf < tau, b, a)
        elif kind == "reader_cascade":
            x, y, rate, thr = cfg
            ra = B._a.rows(test)
            esc = B._a.Cn[ra, B._a.pcol[x]] <= thr
            pol = np.where(esc, f"{B._names[1]}|{y}", f"{B._names[0]}|{x}")
        else:
            tr, lam = cfg
            M26 = K._MIXED
            plans = B.plans
            P = M26._emb_predict(E.loc[tr].to_numpy(), B.Y.loc[tr, plans].to_numpy(), E.loc[test].to_numpy())
            pol = np.array([plans[j] for j in (P - lam * B.cost[plans].to_numpy()[None, :]).argmax(1)])
        ys = np.array([B.Y.at[qq, p] for qq, p in zip(test, pol)], float)
        assert np.allclose(ys, s), f"recovered policy plans do not reproduce the policy ({kind}, fold {i})"
        # hull plan per question
        H = A._hull_vertices(B.Y, B.cost, sel)
        xs = [h[0] for h in H]
        if c <= xs[0] or c >= xs[-1]:
            v = H[0] if c <= xs[0] else H[-1]
            hull = np.array([v[2]] * len(test))
            t, pa, pb = 0.0, v[2], v[2]
        else:
            j = int(np.searchsorted(xs, c, side="right")) - 1
            t = (c - xs[j]) / (xs[j + 1] - xs[j])
            pa, pb = H[j][2], H[j + 1][2]
            hull = np.where(coin.random(len(test)) < t, pb, pa)
        hq = (1 - t) * B.Y.loc[test, pa].to_numpy(float) + t * B.Y.loc[test, pb].to_numpy(float)
        assert abs(hq.mean() - A.hull_at(B.Y, B.cost, sel, c, eval_idx=test)) < 1e-9
        rows.append(pd.DataFrame({"query_id": test, "fold": i, "policy_plan": pol, "hull_plan": hull,
                                  "hull_a": pa, "hull_b": pb, "hull_weight_b": t}))
    for m_ in K.MODS:
        m_._CUR_SEED[0] = 0
    return pd.concat(rows, ignore_index=True)


def stage_make():
    if KEY.exists():
        sys.exit(f"{KEY} exists; the audit is never resampled. Delete the audit3 folder to redo it.")
    DIR.mkdir(parents=True, exist_ok=True)
    K = _load("inference27", "27_inference_refit.py")
    K._MIXED = _load("mixed26", "26_mixed_workload.py")
    G = _load("grading07", "07_grading_full.py")
    A, R = K.A, K.R
    allq = []
    for method, rd, r3 in TESTS:
        A.CONFIG["include_r3"] = r3
        rds = rd if isinstance(rd, tuple) else (rd,)
        data = {r: A.load_reader(r)[0] for r in rds}
        test_name = f"{method}|{'->'.join(rds)}|{'with two hop' if r3 else 'single step'}"
        for ds in sorted(set.intersection(*[set(d.dataset.unique()) for d in data.values()])):
            if method == "reader_cascade":
                blocks = {(r, ds): A.Block(data[r], ds) for r in rds}
                Bj, meth = A.reader_cascade(blocks, rds[0], rds[1], ds)
                Bj._a, Bj._names = blocks[(rds[0], ds)], rds
                Bj.plans = list(Bj.Y.columns)
                ch = choices_one_block(K, A, Bj, meth, method)
            else:
                B = A.Block(data[rds[0]], ds)
                E = None
                if method == "plan_emb":
                    E = R.load_emb(ds).reindex([str(q) for q in B.q])
                    E.index = B.q
                    meth = R.m_plan_emb(B, E)
                else:
                    meth = A.m_conf_cascade(B, None)
                ch = choices_one_block(K, A, B, meth, method, E)
            ch["dataset"], ch["test"] = ds, test_name
            # answers and labels of the two plans
            for arm in ("policy", "hull"):
                rr, pp = [], []
                for p in ch[f"{arm}_plan"]:
                    r_, p_ = p.split("|") if "|" in p else (rds[0], p)
                    rr.append(r_)
                    pp.append(p_)
                ch[f"{arm}_reader"], ch[f"{arm}_plan_only"] = rr, pp
                lab = pd.concat([d.assign(reader=r) for r, d in data.items()])
                lab = lab[lab.dataset == ds].set_index(["reader", "plan", "query_id"])
                idx = list(zip(ch[f"{arm}_reader"], ch[f"{arm}_plan_only"], ch["query_id"]))
                for col in ("answer", "label_final", "label_strict"):
                    ch[f"{arm}_{col}"] = lab[col].reindex(idx).to_numpy()
            allq.append(ch)
            print(f"  {test_name} {ds}: {len(ch)} questions", flush=True)
    A.CONFIG["include_r3"] = False
    Qs = pd.concat(allq, ignore_index=True)
    Qs["policy_norm"] = Qs.policy_answer.astype(str).map(G.normalize)
    Qs["hull_norm"] = Qs.hull_answer.astype(str).map(G.normalize)
    Qs["discordant"] = Qs.policy_norm != Qs.hull_norm
    rng = np.random.default_rng(2027)
    Qs["sampled"] = False
    per_ds = N_PER_TEST // 4
    for (t, ds), g in Qs[Qs.discordant].groupby(["test", "dataset"]):
        take = g.index[rng.permutation(len(g))[:per_ds]]
        Qs.loc[take, "sampled"] = True
    Qs.to_csv(QUESTIONS, index=False)
    S = Qs[Qs.sampled]
    items = pd.concat([S.assign(arm=arm, answer=S[f"{arm}_answer"], norm=S[f"{arm}_norm"])[
        ["dataset", "query_id", "answer", "norm"]] for arm in ("policy", "hull")])
    items = items.drop_duplicates(["dataset", "query_id", "norm"]).reset_index(drop=True)
    items["item_id"] = np.arange(3001, 3001 + len(items))
    gold = G.gold_maps(items.dataset.unique())
    items["gold_answers"] = [" | ".join(gold[d].get(q, [])) for d, q in zip(items.dataset, items.query_id)]
    qt = G.question_texts(items)
    items["question"] = [qt.get((d, q), "") for d, q in zip(items.dataset, items.query_id)]
    items[["item_id", "dataset", "query_id", "norm", "answer"]].to_csv(KEY, index=False)
    sheet = items[["item_id", "dataset", "question", "gold_answers", "answer"]].rename(columns={"answer": "model_answer"})
    for i, (r, path) in enumerate(SHEETS.items()):
        write_sheet(sheet.iloc[np.random.default_rng(300 + i).permutation(len(sheet))], path, f"Rater {r}")
    print(f"\nDiscordant questions per test (seed 0, 500 per dataset):")
    print(Qs.groupby("test").discordant.agg(["sum", "mean"]).round(3).to_string())
    print(f"Sampled {int(Qs.sampled.sum())} discordant questions; {len(items)} items to rate "
          f"(identical answers to the same question rated once).")
    print(f"Sheets: {SHEETS['A'].name}, {SHEETS['B'].name} in {DIR}. Key: {KEY.name}; never send the key or "
          f"{QUESTIONS.name} to a rater.")


def write_sheet(df, path, title, extra_cols=()):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "audit"
    cols = list(df.columns) + list(extra_cols) + [LABEL_COL, "notes"]
    ws.append(cols)
    for r in df.itertuples(index=False):
        ws.append(list(r) + [""] * (len(extra_cols) + 2))
    widths = {"item_id": 8, "dataset": 14, "question": 60, "gold_answers": 30, "model_answer": 50, LABEL_COL: 14,
              "notes": 30, "rater_A": 9, "rater_B": 9}
    for j, c in enumerate(cols, start=1):
        ws.column_dimensions[ws.cell(1, j).column_letter].width = widths.get(c, 14)
        ws.cell(1, j).font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    lc = cols.index(LABEL_COL) + 1
    dv = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
    ws.add_data_validation(dv)
    col = ws.cell(1, lc).column_letter
    dv.add(f"{col}2:{col}{len(df) + 1}")
    ws.freeze_panes = "A2"
    g = wb.create_sheet("guide")
    g.column_dimensions["A"].width = 120
    g.append([f"{title}. {len(df)} items."])
    for line in GUIDE:
        g.append([line])
    wb.save(path)


# ====================================================================== adjudicate / finalize
def read_labels(path):
    d = pd.read_excel(path, sheet_name="audit")
    d[LABEL_COL] = pd.to_numeric(d[LABEL_COL], errors="coerce")
    return d.set_index("item_id")


BLANK_OK = "--blank-to-adjudication" in ARGS


def both(a, b):
    """The items both raters labelled (all items unless a blank was sent to adjudication)."""
    m = a[LABEL_COL].notna() & b[LABEL_COL].notna()
    return a.loc[m, LABEL_COL].astype(int), b.loc[m, LABEL_COL].astype(int)


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def stage_adjudicate():
    for r, p in RECEIVED.items():
        if not p.exists():
            sys.exit(f"{p.name} missing: save rater {r}'s returned sheet under that name in {DIR}")
    a, b = read_labels(RECEIVED["A"]), read_labels(RECEIVED["B"])
    b = b.reindex(a.index)
    blank = a.index[a[LABEL_COL].isna() | b[LABEL_COL].isna()]
    if len(blank) and not BLANK_OK:
        sys.exit(f"{len(blank)} items are unlabelled (item_id {list(blank)}); send the sheets back to the raters, or "
                 f"rerun with --blank-to-adjudication to send them to adjudication")
    x, y = both(a, b)
    print(f"items {len(a)}; labelled by both {len(x)}; blank {len(blank)} (item_id {list(blank)}, sent to adjudication)"
          if len(blank) else f"items {len(a)}")
    print(f"agreement {(x == y).mean():.3f}; Cohen's kappa {kappa(x, y):.3f}; rated correct: A {x.mean():.3f}, "
          f"B {y.mean():.3f} (items labelled by both)")
    dis = x.index[x != y].append(blank)
    lab = lambda s_: s_.map(lambda v: "" if pd.isna(v) else int(v)).to_numpy()
    sheet = a.loc[dis, ["dataset", "question", "gold_answers", "model_answer"]].reset_index()
    sheet = sheet.assign(rater_A=lab(a.loc[dis, LABEL_COL]), rater_B=lab(b.loc[dis, LABEL_COL]))
    write_sheet(sheet.iloc[np.random.default_rng(400).permutation(len(sheet))], ADJ, "Adjudication")
    print(f"{len(sheet)} items ({len(dis) - len(blank)} disagreements, {len(blank)} blank) written to {ADJ.name}. Both raters (or a third person who is not an author) "
          f"agree one label per item, then save it as {ADJ_RECEIVED.name}.")


def stage_finalize():
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = read_labels(RECEIVED["A"]), read_labels(RECEIVED["B"])
    b = b.reindex(a.index)
    human = a[LABEL_COL].where(a[LABEL_COL] == b[LABEL_COL])        # blank or disagreement: NaN, adjudicated
    dis = human.index[human.isna()]
    n_blank = int((a[LABEL_COL].isna() | b[LABEL_COL].isna()).sum())
    if len(dis):
        if not ADJ_RECEIVED.exists():
            sys.exit(f"{len(dis)} disagreements; run 'adjudicate' and return {ADJ_RECEIVED.name}")
        adj = read_labels(ADJ_RECEIVED)
        human.loc[dis] = adj[LABEL_COL].reindex(dis)
        if human.isna().any():
            sys.exit(f"{int(human.isna().sum())} adjudication items unlabelled")
    key = pd.read_csv(KEY, dtype={"query_id": str})
    key["human"] = human.reindex(key.item_id).to_numpy()
    hmap = key.set_index(["dataset", "query_id", "norm"]).human
    Qs = pd.read_csv(QUESTIONS, dtype={"query_id": str}, keep_default_na=False)
    for arm in ("policy", "hull"):
        Qs[f"{arm}_label_final"] = pd.to_numeric(Qs[f"{arm}_label_final"])
        Qs[f"{arm}_label_strict"] = pd.to_numeric(Qs[f"{arm}_label_strict"])
    S = Qs[Qs.sampled == True].copy()  # noqa: E712
    for arm in ("policy", "hull"):
        S[f"{arm}_human"] = hmap.reindex(list(zip(S.dataset, S.query_id, S[f"{arm}_norm"]))).to_numpy()
    rng = np.random.default_rng(0)
    rows, L = [], []
    x, y = both(a, b)
    L.append(f"Audit 3: answers on which the policy and its hull disagree. Items {len(key)}; labelled by both raters "
             f"{len(x)}; raters agree on {(x == y).mean():.3f}, kappa {kappa(x, y):.3f}; {len(dis)} adjudicated "
             f"({len(dis) - n_blank} disagreements, {n_blank} left blank by one rater).")
    for t, g in S.groupby("test"):
        share = Qs[Qs.test == t].discordant.mean()
        r = {"test": t, "questions": len(g), "discordant_share": share}
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
            r[f"bias_{rule}_discordant"] = d.mean()
            r[f"bias_{rule}_discordant_low"], r[f"bias_{rule}_discordant_high"] = np.percentile(bs, [2.5, 97.5])
            r[f"bias_{rule}_on_gain"] = d.mean() * share
            r[f"bias_{rule}_on_gain_low"] = np.percentile(bs, 2.5) * share
            r[f"bias_{rule}_on_gain_high"] = np.percentile(bs, 97.5) * share
        rows.append(r)
        L.append(f"\n== {t}: {len(g)} discordant questions sampled; {share:.3f} of all held out questions are discordant")
        for arm in ("policy", "hull"):
            L.append(f"  {arm:6s} human correct {r[f'{arm}_human_correct']:.3f}; label_final error "
                     f"{r[f'{arm}_label_final_error']:.3f} (false positive {r[f'{arm}_label_final_false_pos']:.3f}, "
                     f"false negative {r[f'{arm}_label_final_false_neg']:.3f}); strict error {r[f'{arm}_label_strict_error']:.3f}")
        L.append(f"  bias of label_final on the gain: {r['bias_label_final_on_gain']:+.4f} "
                 f"[{r['bias_label_final_on_gain_low']:+.4f}, {r['bias_label_final_on_gain_high']:+.4f}] "
                 f"(positive: label_final overstates the policy's gain); strict rule {r['bias_label_strict_on_gain']:+.4f}")
    pd.DataFrame(rows).to_csv(OUT / "audit3_results.csv", index=False)
    S.to_csv(OUT / "audit3_questions_labelled.csv", index=False)
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"make": stage_make, "adjudicate": stage_adjudicate, "finalize": stage_finalize}.get(
        STAGE, lambda: sys.exit(__doc__))()
