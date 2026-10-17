"""
Human audit of the answers of the readers added in the extension (Qwen3 32B and Gemma 3 27B; pre-registration R1).

Why a second audit: the first audit (07_grading_full.py, 200 items) sampled only the three original readers.
The new readers answer differently (the 32B writes sentences and abstains more often), so the grading rule must
be checked again on their answers. The first audit is never resampled; this one is separate.

Stages
  sample   CPU. Needs tsr_labels_<reader>.csv of both new readers. Draws 200 unique answers (pairs not in the
           first audit), half per reader, datasets balanced inside each stratum. One extra stratum checks long
           answers with F1 < 0.2 and no gold string, which the strict rule marks wrong and the judge never sees.
           Writes grading/audit2_rater1.xlsx and audit2_rater2.xlsx (same items, different order, 1/0 dropdown)
           and grading/audit2_key.csv. Refuses to overwrite an existing key.
  report   CPU. After both raters filled their sheets (and, for the final label, after 08 regrade, judge and
           finalize). Kappa between raters, accuracy of every rule and of label_final against the human label,
           by reader and by stratum, and the estimated share of mislabelled rows in the long, low F1 group.
           Writes grading/audit2_report.txt.

Usage
  python scripts/08_audit_new_readers.py sample
  python scripts/08_audit_new_readers.py report
"""

import argparse
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

READERS = ["qwen3-32b", "gemma3-27b"]
PROC = settings.PROCESSED_DATA_ROOT
GDIR = PROC / "grading"
OLD_KEY = GDIR / "audit_key.csv"
KEY = GDIR / "audit2_key.csv"
SHEETS = [GDIR / "audit2_rater1.xlsx", GDIR / "audit2_rater2.xlsx"]
REPORT = GDIR / "audit2_report.txt"
SEED = 1
LABEL_COL = "label_1_correct_0_wrong"

# (name, n, condition on unique answers); conditions are disjoint by construction of the order
STRATA = [
    ("long_low_f1",   35, lambda d: (d.strict == 0) & (d.f1 < 0.2) & (d.n_answer_tokens >= 8)),
    ("f1_02_04",      30, lambda d: (d.strict == 0) & (d.f1 >= 0.2) & (d.f1 < 0.4)),
    ("f1_04_07",      35, lambda d: (d.strict == 0) & (d.f1 >= 0.4) & (d.f1 < 0.7)),
    ("f1_07_up",      30, lambda d: (d.strict == 0) & (d.f1 >= 0.7)),
    ("contains_long", 30, lambda d: (d.em == 0) & (d.contains == 1) & (d.n_answer_tokens >= 8)),
    ("strict_short",  20, lambda d: (d.strict == 1) & (d.n_answer_tokens < 8)),
    ("short_low_f1",  20, lambda d: (d.strict == 0) & (d.f1 < 0.2) & (d.n_answer_tokens < 8)),
]


def load(reader, graded=False):
    f = PROC / (f"tsr_labels_{reader}_graded.csv" if graded else f"tsr_labels_{reader}.csv")
    if not f.exists():
        sys.exit(f"Missing {f.name}. Both new readers must be labelled first.")
    return pd.read_csv(f, dtype={"query_id": str}, keep_default_na=False)


def evidence(df):
    gold = {d: json.load(open(PROC / d / "answers.json", encoding="utf-8")) for d in df.dataset.unique()}
    det = pd.DataFrame([grade_detail(a, gold[d].get(q, [])) for d, q, a in zip(df.dataset, df.query_id, df.answer)])
    df = pd.concat([df.reset_index(drop=True), det], axis=1)
    df["strict"] = ((df.em == 1) | (df.contains == 1)).astype(int)
    df["pair"] = df.dataset + "::" + df.query_id + "::" + df.answer.map(normalize)
    return df, gold


def question_texts(a):
    out = {}
    for ds in a.dataset.unique():
        want = set(a.loc[a.dataset == ds, "query_id"])
        with open(PROC / ds / "queries.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if str(i) in want:
                    out[(ds, str(i))] = json.loads(line).get("text", "")
    return out


def write_sheet(df, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "audit"
    cols = ["audit_id", "dataset", "question", "gold_answers", "model_answer", LABEL_COL, "notes"]
    ws.append(cols)
    for r in df[cols].itertuples(index=False):
        ws.append(list(r))
    for c in ws[1]:
        c.font = Font(bold=True)
    widths = {"A": 9, "B": 16, "C": 60, "D": 35, "E": 60, "F": 14, "G": 30}
    for k, w in widths.items():
        ws.column_dimensions[k].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        row[5].fill = PatternFill("solid", fgColor="FFF2CC")
    dv = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
    dv.error, dv.errorTitle = "Enter 1 (correct) or 0 (wrong).", "Invalid label"
    ws.add_data_validation(dv)
    dv.add(f"F2:F{len(df) + 1}")
    ws.freeze_panes = "A2"
    g = wb.create_sheet("guide")
    for line in [
        "Judge meaning, not wording. 1 = the model answer gives the same fact as one of the gold answers.",
        "Extra detail or a full sentence is fine if the fact is right and nothing wrong is added as the answer.",
        "0 = a different entity, number or date; a hedge between options; a refusal ('the passages do not mention');",
        "or too vague to identify the gold. If the gold itself looks wrong, still grade against the gold and add a note.",
        "Do not use any AI tool or search engine. Do not discuss items with the other rater.",
    ]:
        g.append([line])
    g.column_dimensions["A"].width = 120
    wb.save(path)


def stage_sample():
    if KEY.exists():
        sys.exit(f"{KEY.name} exists; the second audit is never resampled.")
    old = set(pd.read_csv(OLD_KEY)["pair"]) if OLD_KEY.exists() else set()
    frames = []
    for r in READERS:
        d, _ = evidence(load(r))
        frames.append(d.assign(reader=r))
    allp = pd.concat(frames)
    # one row per (pair, reader); a pair produced by both readers keeps both names
    readers_of = allp.groupby("pair")["reader"].agg(lambda s: ",".join(sorted(set(s))))
    pairs = allp.drop_duplicates("pair").copy()
    pairs["reader"] = pairs["pair"].map(readers_of)
    pairs = pairs[~pairs["pair"].isin(old)]
    rng = np.random.default_rng(SEED)
    chosen, used = [], set()
    for name, n, cond in STRATA:
        pool = pairs[cond(pairs) & ~pairs["pair"].isin(used)]
        take = []
        for r in READERS:                                       # half per reader, datasets balanced
            pr = pool[pool["reader"].str.contains(r, regex=False) & ~pool["pair"].isin(used)]
            k = n // len(READERS) + (1 if r == READERS[0] and n % len(READERS) else 0)
            per_ds = [x.iloc[rng.permutation(len(x))[: max(1, k // max(1, pr.dataset.nunique()))]]
                      for _, x in pr.groupby("dataset")]
            t = pd.concat(per_ds) if per_ds else pr.iloc[:0]
            if len(t) < k:
                rest = pr[~pr["pair"].isin(t["pair"])]
                t = pd.concat([t, rest.iloc[rng.permutation(len(rest))[: k - len(t)]]])
            t = t.iloc[:k]
            used.update(t["pair"])
            take.append(t)
        chosen.append(pd.concat(take).assign(stratum=name))
    a = pd.concat(chosen).reset_index(drop=True)
    a = a.iloc[rng.permutation(len(a))].reset_index(drop=True)
    a["audit_id"] = np.arange(1001, 1001 + len(a))             # ids never collide with the first audit
    gold = {d: json.load(open(PROC / d / "answers.json", encoding="utf-8")) for d in a.dataset.unique()}
    a["gold_answers"] = [" | ".join(gold[d].get(q, [])) for d, q in zip(a.dataset, a.query_id)]
    qt = question_texts(a)
    a["question"] = [qt.get((d, q), "") for d, q in zip(a.dataset, a.query_id)]
    a["model_answer"] = a["answer"]
    a[LABEL_COL] = ""
    a["notes"] = ""
    GDIR.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(SHEETS):
        write_sheet(a.iloc[np.random.default_rng(200 + i).permutation(len(a))], p)
    a[["audit_id", "stratum", "pair", "reader", "dataset", "query_id"]].to_csv(KEY, index=False)
    print(f"Wrote {len(a)} items -> {SHEETS[0].name}, {SHEETS[1].name}; key {KEY.name}")
    print(pd.crosstab(a["stratum"], a["reader"]).to_string())
    print(a.groupby("dataset").size().to_string())


# ---------------------------------------------------------------- report
def kappa(x, y):
    x, y = np.asarray(x, int), np.asarray(y, int)
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def read_sheet(p):
    if not p.exists():
        return None
    s = pd.read_excel(p, sheet_name="audit")
    return s.set_index("audit_id")[LABEL_COL].pipe(pd.to_numeric, errors="coerce")


def ci(ok, n_boot=2000):
    ok = np.asarray(ok, float)
    b = [ok[np.random.default_rng(s).integers(0, len(ok), len(ok))].mean() for s in range(n_boot)]
    return np.percentile(b, [2.5, 97.5])


def stage_report():
    key = pd.read_csv(KEY, dtype={"query_id": str}, keep_default_na=False).set_index("audit_id")
    r1, r2 = (read_sheet(p) for p in SHEETS)
    L = [f"Second audit (new readers {', '.join(READERS)}): {len(key)} items"]
    for name, r in (("rater 1", r1), ("rater 2", r2)):
        L.append(f"  {name}: {0 if r is None else int(r.notna().sum())} of {len(key)} labelled")
    if r1 is None or r2 is None or r1.isna().any() or r2.isna().any():
        L.append("Both sheets must be complete. Stopping.")
        print("\n".join(L))
        return
    b = pd.DataFrame({"r1": r1, "r2": r2}).join(key)
    L.append(f"Cohen's kappa rater 1 vs rater 2: {kappa(b.r1, b.r2):.3f} (raw agreement {(b.r1 == b.r2).mean():.3f})")
    b[b.r1 != b.r2].to_csv(GDIR / "audit2_disagreements.csv")

    # labels of every rule, and label_final when 08 finalize has run
    lab = {}
    have_final = True
    for r in READERS:
        g = PROC / f"tsr_labels_{r}_graded.csv"
        if g.exists():
            d = pd.read_csv(g, dtype={"query_id": str}, keep_default_na=False)
            d["pair"] = d.dataset + "::" + d.query_id + "::" + d.answer.map(normalize)
            have_final &= "label_final" in d.columns
        else:
            d, _ = evidence(load(r))
            for rule in RULES:
                d[f"label_{rule}"] = [apply_rule(x, rule) for x in d[["em", "contains", "f1", "proposed"]].to_dict("records")]
            have_final = False
        keep = ["strict"] + [f"label_{x}" for x in RULES] + (["label_final"] if "label_final" in d else [])
        lab.update(d.drop_duplicates("pair").set_index("pair")[keep].to_dict("index"))
    cols = ["strict"] + [f"label_{x}" for x in RULES if x != "strict"] + (["label_final"] if have_final else [])
    for c in cols:
        b[c] = [lab.get(p, {}).get(c, np.nan) for p in b["pair"]]
    if not have_final:
        L.append("label_final not available yet (run 08 regrade, judge, finalize); rules only.")

    def block(e, title, human):
        L.append(f"\n{title} (n={len(e)}):")
        for c in cols:
            x = e.dropna(subset=[c])
            ok = (x[c].astype(int) == x[human].astype(int)).values
            lo, hi = ci(ok)
            fp = int(((x[c] == 1) & (x[human] == 0)).sum())
            fn = int(((x[c] == 0) & (x[human] == 1)).sum())
            L.append(f"  {c:14s} accuracy {ok.mean():.3f} [{lo:.3f}, {hi:.3f}]  kappa {kappa(x[c], x[human]):.3f}  FP {fp}  FN {fn}")

    agreed = b[b.r1 == b.r2].assign(human=lambda x: x.r1)
    block(agreed, "Items both raters agree on", "human")
    adj_f = GDIR / "audit_adjudicated.csv"
    if adj_f.exists():
        # main reference: majority of three raters (a third rater labelled every item the two raters disagree on)
        adj = pd.read_csv(adj_f)
        adj = adj[adj["audit"] == "audit2"].set_index("audit_id")["human"]
        maj = adj.reindex(b.index)
        if maj.notna().all():
            block(b.assign(human=maj.astype(int)), "All items against the majority of three raters (main reference)", "human")
        else:
            L.append("\naudit_adjudicated.csv misses audit 2 items; majority block skipped")
    block(b.assign(human=b.r1), "All items against rater 1 alone", "human")
    block(b.assign(human=b.r2), "All items against rater 2 alone", "human")
    main = "label_final" if have_final else "strict"
    L.append(f"\n{main} vs agreed human label, by reader and by stratum:")
    ex = agreed.assign(reader=agreed["reader"].str.split(",")).explode("reader")
    for k, e in ex.groupby("reader"):
        L.append(f"  {k:14s} n={len(e):3d} accuracy {(e[main].astype(int) == e.human).mean():.3f}")
    for k, e in agreed.groupby("stratum"):
        L.append(f"  {k:14s} n={len(e):3d} accuracy {(e[main].astype(int) == e.human).mean():.3f}  "
                 f"human=1 share {e.human.mean():.3f}")

    # how many rows of each reader sit in the long, low F1 group, and how many of them humans call correct
    s = agreed[agreed.stratum == "long_low_f1"]
    L.append("\nLong answers with F1 < 0.2 (graded wrong by strict, never judged):")
    for r in READERS:
        d, _ = evidence(load(r))
        share = ((d.strict == 0) & (d.f1 < 0.2) & (d.n_answer_tokens >= 8)).mean()
        sr = s[s["reader"].str.contains(r, regex=False)]
        p = sr.human.mean() if len(sr) else np.nan
        L.append(f"  {r:12s} {share:.1%} of rows; humans call {p:.3f} of the audited ones correct (n={len(sr)}) "
                 f"-> about {share * p:.1%} of all rows mislabelled as wrong")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "report"])
    a = ap.parse_args()
    {"sample": stage_sample, "report": stage_report}[a.stage]()
