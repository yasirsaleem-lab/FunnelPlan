"""
Regrade the stored agent answers under several rules, and audit the rules by hand.

No GPU, no agent reruns: every answer is already in tsr_labels_v3.csv.

RUN 1 (no audit yet):
    python scripts/04d_regrade_labels.py
  writes
    data/processed/tsr_labels_v3_graded.csv   evidence columns + label_<rule> per row
    data/processed/grading_audit.csv          blind hand check sheet (~150 answers)
    data/processed/grading_audit_key.csv      which stratum each item came from (do not open while auditing)
    results/tsr/grading_report.txt            success rates and depth effect under each rule

HAND CHECK: open grading_audit.csv in Excel. For each row read the question,
the gold answers and the agent answer, and put 1 (correct) or 0 (wrong) in
human_label. Judge meaning, not wording: "Nancy Cartwright" for gold
"Nancy Jean Cartwright" is correct; "Maine" for "Bath, Maine" is wrong if
the question asks for the town. Leave notes if unsure. Save as CSV (UTF-8).

RUN 2 (after the hand check):
    python scripts/04d_regrade_labels.py
  The sheet is NOT resampled. The script measures each rule against your
  labels, estimates each rule's false positive and false negative counts over
  all 9,600 rows, picks the rule with the fewest estimated errors, and adds
  label_audited (your label where you judged that answer, else the chosen rule).
  05_train_predictor.py uses label_adjudicated (or label_audited) when it exists.

ADJUDICATION (optional, recommended): grading_adjudicated.csv holds a judgment
for every remaining ambiguous answer. With it, the script writes
label_adjudicated = judgment for every ambiguous answer, strict rule elsewhere.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from tsr.grader import RULES, apply_rule, grade_detail, normalize

PROC = settings.PROCESSED_DATA_ROOT
IN_CSV = PROC / settings.LABELS_FILE
OUT_CSV = PROC / "tsr_labels_v3_graded.csv"
AUDIT_CSV = PROC / "grading_audit.csv"
AUDIT_KEY = PROC / "grading_audit_key.csv"
ADJ_CSV = PROC / "grading_adjudicated.csv"   # judgments for every ambiguous answer not in the audit
REPORT = settings.RESULTS_ROOT / "tsr" / "grading_report.txt"
SEED = 0

# Strata of UNIQUE (dataset, query, normalised answer) pairs. Exact matches and
# answers with F1 < 0.2 are not audited; they are assumed correct and wrong.
STRATA = {
    "f1_04_07":      (60, lambda d: (d.strict == 0) & (d.f1 >= 0.4) & (d.f1 < 0.7)),
    "f1_07_up":      (40, lambda d: (d.strict == 0) & (d.f1 >= 0.7)),
    "contains_long": (30, lambda d: (d.em == 0) & (d.contains == 1) & (d.n_answer_tokens >= 8)),
    "f1_02_04":      (20, lambda d: (d.strict == 0) & (d.f1 >= 0.2) & (d.f1 < 0.4)),
}


def load_questions(ds, qids):
    out, want = {}, set(qids)
    with open(PROC / ds / "queries.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if str(i) in want:
                out[str(i)] = json.loads(line).get("text", "")
    return out


def regrade():
    df = pd.read_csv(IN_CSV, dtype={"query_id": str}, keep_default_na=False)
    df = df.rename(columns={"label": "label_original"})
    gold = {ds: json.load(open(PROC / ds / "answers.json", encoding="utf-8")) for ds in df["dataset"].unique()}
    det = [grade_detail(a, gold[ds].get(q, [])) for ds, q, a in zip(df["dataset"], df["query_id"], df["answer"])]
    det = pd.DataFrame(det)
    df = pd.concat([df.reset_index(drop=True), det], axis=1)
    df["strict"] = ((df["em"] == 1) | (df["contains"] == 1)).astype(int)
    for r in RULES:
        df[f"label_{r}"] = [apply_rule(row, r) for row in det.to_dict("records")]
    df["pair"] = df["dataset"] + "::" + df["query_id"] + "::" + df["answer"].map(normalize)
    df["gold"] = [" | ".join(gold[ds].get(q, [])) for ds, q in zip(df["dataset"], df["query_id"])]
    return df


def make_audit(df):
    pairs = df.drop_duplicates("pair").copy()
    rng = np.random.default_rng(SEED)
    chosen, used = [], set()
    for name, (n, cond) in STRATA.items():
        pool = pairs[cond(pairs) & ~pairs["pair"].isin(used)]
        take = pool.iloc[rng.permutation(len(pool))[:n]].assign(stratum=name)
        used.update(take["pair"])
        chosen.append(take)
    audit = pd.concat(chosen)
    audit = audit.iloc[rng.permutation(len(audit))].reset_index(drop=True)   # mix strata
    audit["audit_id"] = np.arange(1, len(audit) + 1)
    qtext = {}
    for ds in audit["dataset"].unique():
        qtext.update({(ds, k): v for k, v in load_questions(ds, audit.loc[audit.dataset == ds, "query_id"]).items()})
    audit["question"] = [qtext.get((d, q), "") for d, q in zip(audit["dataset"], audit["query_id"])]
    sheet = audit[["audit_id", "dataset", "query_id", "question", "gold", "answer"]].rename(
        columns={"gold": "gold_answers", "answer": "agent_answer"})
    sheet["human_label"] = ""
    sheet["notes"] = ""
    sheet.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    audit[["audit_id", "stratum", "pair"]].to_csv(AUDIT_KEY, index=False)
    return len(sheet)


def read_audit():
    if not (AUDIT_CSV.exists() and AUDIT_KEY.exists()):
        return None
    sheet = pd.read_csv(AUDIT_CSV, encoding="utf-8-sig", dtype={"query_id": str}, keep_default_na=False)
    key = pd.read_csv(AUDIT_KEY)
    a = sheet.merge(key, on="audit_id")
    a["human_label"] = pd.to_numeric(a["human_label"], errors="coerce")
    return a


def audit_report(df, a, L):
    done = a.dropna(subset=["human_label"])
    L.append(f"Hand check: {len(done)} of {len(a)} items labelled.")
    if len(done) < 0.8 * len(a):
        L.append("  Fewer than 80% labelled; rule choice skipped until the sheet is complete.")
        return None
    pairs = df.drop_duplicates("pair").set_index("pair")
    done = done.join(pairs[[f"label_{r}" for r in RULES]], on="pair")
    rows_per_pair = df.groupby("pair").size()
    est = {r: {"fp": 0.0, "fn": 0.0} for r in RULES}
    L.append(f"{'stratum':15s} {'n':>3s} {'human=1':>8s} " + " ".join(f"{r+' agree':>15s}" for r in RULES))
    for name, (_, cond) in STRATA.items():
        s = done[done["stratum"] == name]
        if s.empty:
            continue
        pop = pairs[cond(pairs)]
        # rows (not pairs) in this stratum, excluding pairs claimed by earlier strata
        pop_rows = rows_per_pair.reindex(pop.index).sum()
        line = f"{name:15s} {len(s):3d} {s.human_label.mean():8.2f} "
        for r in RULES:
            agree = (s[f"label_{r}"] == s["human_label"]).mean()
            fp = ((s[f"label_{r}"] == 1) & (s["human_label"] == 0)).mean()
            fn = ((s[f"label_{r}"] == 0) & (s["human_label"] == 1)).mean()
            est[r]["fp"] += fp * pop_rows
            est[r]["fn"] += fn * pop_rows
            line += f"{agree:15.2f} "
        L.append(line)
    L.append("")
    L.append("Estimated label errors over all rows (audited strata only; exact matches and F1<0.2 assumed right):")
    for r in RULES:
        L.append(f"  {r:9s} false positives ~{est[r]['fp']:6.0f}   false negatives ~{est[r]['fn']:6.0f}   "
                 f"total ~{est[r]['fp'] + est[r]['fn']:6.0f}  ({(est[r]['fp'] + est[r]['fn']) / len(df):.1%} of rows)")
    best = min(RULES, key=lambda r: est[r]["fp"] + est[r]["fn"])
    L.append(f"Chosen rule: {best}")
    human = done.set_index("pair")["human_label"]
    df["label_audited"] = df["pair"].map(human).fillna(df[f"label_{best}"]).astype(int)
    df["audited_rule"] = best
    return best


def adjudicate(df, a, L):
    """label_adjudicated: a direct judgment for EVERY ambiguous answer, the
    strict rule elsewhere.

    Ambiguous = no exact or contained match but token F1 >= 0.2. These are the
    only answers where the rules disagree in a way the audit showed matters.
    Outside that band the audit found the strict rule right: long answers that
    contain the gold were 30/30 correct, and F1 < 0.2 answers are wrong.
    Judgments come from the audit sheet plus grading_adjudicated.csv.
    """
    if a is None or not ADJ_CSV.exists():
        return False
    judged = {}
    done = a.dropna(subset=["human_label"])
    judged.update(dict(zip(done["pair"], done["human_label"].astype(int))))
    adj = pd.read_csv(ADJ_CSV, encoding="utf-8-sig", dtype={"query_id": str}, keep_default_na=False)
    judged.update(dict(zip(adj["pair"], adj["label"].astype(int))))
    amb = (df["strict"] == 0) & (df["f1"] >= 0.2)
    amb_pairs = set(df.loc[amb, "pair"])
    missing = amb_pairs - set(judged)
    L.append("")
    L.append(f"Adjudication: {len(amb_pairs)} ambiguous answers ({amb.sum()} rows); "
             f"{len(amb_pairs) - len(missing)} judged, {len(missing)} not judged.")
    if missing:
        L.append("  Some ambiguous answers have no judgment; label_adjudicated not written.")
        return False
    df["label_adjudicated"] = df["pair"].map(judged).where(amb | df["pair"].isin(judged), df["strict"])
    df["label_adjudicated"] = df["label_adjudicated"].fillna(df["strict"]).astype(int)
    sub = df[amb].drop_duplicates("pair")
    L.append("  Agreement of each rule with the judgments on all ambiguous answers (unique answers):")
    for r in RULES:
        y, j = sub[f"label_{r}"], sub["pair"].map(judged)
        L.append(f"    {r:9s} agree {(y == j).mean():.2f}   false pos {((y == 1) & (j == 0)).sum():3d}   "
                 f"false neg {((y == 0) & (j == 1)).sum():3d}")
    diff = (df["label_adjudicated"] != df["label_lenient"]).sum()
    L.append(f"  label_adjudicated differs from label_lenient on {diff} rows ({diff / len(df):.1%}).")
    return True


def summary(df, L, cols):
    L.append("")
    L.append("Task success by rule (rows = rule, columns = dataset @ depth)")
    t = []
    for c in cols:
        p = df.pivot_table(index="dataset", columns="depth", values=c, aggfunc="mean")
        t.append(pd.Series({f"{ds}@{d}": p.loc[ds, d] for ds in p.index for d in p.columns}, name=c))
    L.append(pd.DataFrame(t).round(3).to_string())
    L.append("")
    L.append("Depth effect: success at depth 10 minus depth 100, per rule (positive = shallow is better)")
    for c in cols:
        p = df.pivot_table(index="dataset", columns="depth", values=c, aggfunc="mean")
        L.append(f"  {c:16s} " + "  ".join(f"{ds}: {p.loc[ds, 10] - p.loc[ds, 100]:+.3f}" for ds in p.index))
    L.append("")
    L.append("Best fixed plan per rule")
    for c in cols:
        for ds, sub in df.groupby("dataset"):
            s = sub.assign(plan=sub["retriever"] + "@" + sub["depth"].astype(str)).groupby("plan")[c].mean()
            L.append(f"  {c:16s} {ds:9s} {s.idxmax():24s} {s.max():.3f}")
    L.append("")
    v = df[df["n_answer_tokens"] > 15]
    L.append(f"Long answers (>15 words): {len(v)} rows; by depth {v.groupby('depth').size().to_dict()}; "
             f"success under strict {v['label_strict'].mean():.2f}, lenient {v['label_lenient'].mean():.2f}")


def main():
    df = regrade()
    L = [f"Regraded {len(df)} rows from {IN_CSV.name}", ""]
    changed = (df["label_original"] != df["label_lenient"]).sum()
    L.append(f"Lenient rule with article removal differs from the original label on {changed} rows.")

    a = read_audit()
    if a is None:
        n = make_audit(df)
        n_pairs = df["pair"].nunique()
        L.append(f"Wrote hand check sheet with {n} items ({AUDIT_CSV.name}); {n_pairs} unique answers in total.")
        L.append("Fill human_label (1/0) and run this script again.")
        best = None
    else:
        L.append(f"Using existing hand check sheet {AUDIT_CSV.name} (not resampled).")
        best = audit_report(df, a, L)

    adj = adjudicate(df, a, L)

    cols = ["label_original"] + [f"label_{r}" for r in RULES] + (["label_audited"] if best else []) \
        + (["label_adjudicated"] if adj else [])
    summary(df, L, cols)

    out = df.drop(columns=["pair"])
    out.to_csv(OUT_CSV, index=False)
    L.append("")
    L.append(f"Wrote {OUT_CSV}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
