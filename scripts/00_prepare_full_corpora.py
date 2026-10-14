"""
Prepare the four full datasets for CORPUS_MODE=full.

Writes data/processed_full/<dataset>/{corpus.jsonl, queries.jsonl, qrels.tsv, answers.json}
in the format data.data_loader.load_processed already reads, so get_processed()
works unchanged in full mode.

  nq, hotpotqa       Full BEIR corpora from data/raw/<ds>/corpus.jsonl (2.68M and
                     5.23M passages). No subsampling. Queries, answers and qrels
                     are copied from data/processed/<ds>/, so query ids match the
                     pilot (the 500 query sample contains the 200 pilot queries).
  musique            Corpus = all unique paragraphs of train + dev (the IRCoT
                     construction, ACL 2023). Queries = dev (answerable split).
                     Put musique_ans_v1.0_train.jsonl and musique_ans_v1.0_dev.jsonl
                     in data/raw/musique/  (Hugging Face: bdsaglam/musique).
  2wikimultihopqa    Same construction. Put train.json and dev.json (original
                     format with context and supporting_facts) in
                     data/raw/2wikimultihopqa/  (Hugging Face: voidful/2WikiMultihopQA).

Run once:  python scripts/00_prepare_full_corpora.py
(It does not need FUNNELPLAN_CORPUS_MODE; it always writes data/processed_full.)
"""

import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SUB = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed_full"


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare_beir(ds):
    out = OUT / ds
    out.mkdir(parents=True, exist_ok=True)
    src = RAW / ds / "corpus.jsonl"
    n = 0
    with open(src, encoding="utf-8") as fin, open(out / "corpus.jsonl", "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"{ds} corpus"):
            d = json.loads(line)
            fout.write(json.dumps({"_id": str(d["_id"]), "title": d.get("title", "") or "",
                                   "text": d.get("text", "") or ""}, ensure_ascii=False) + "\n")
            n += 1
    for name in ("queries.jsonl", "answers.json", "qrels.tsv"):
        shutil.copyfile(SUB / ds / name, out / name)
    print(f"{ds}: {n} passages (full corpus); queries, answers and qrels copied from {SUB / ds}")


class Pool:
    """Deduplicated paragraph pool: (title, text) -> passage id."""

    def __init__(self, prefix):
        self.prefix, self.ids, self.rows = prefix, {}, []

    def add(self, title, text):
        key = (title.strip(), " ".join(text.split()))
        if key not in self.ids:
            pid = f"{self.prefix}{len(self.rows)}"
            self.ids[key] = pid
            self.rows.append({"_id": pid, "title": key[0], "text": key[1]})
        return self.ids[key]


def prepare_musique():
    src = RAW / "musique"
    files = {s: src / f"musique_ans_v1.0_{s}.jsonl" for s in ("train", "dev")}
    for f in files.values():
        if not f.exists():
            sys.exit(f"Missing {f}. Download from Hugging Face: bdsaglam/musique")
    pool = Pool("msq")
    queries, answers, qrels = [], {}, []
    for split, f in files.items():
        with open(f, encoding="utf-8") as fin:
            for line in tqdm(fin, desc=f"musique {split}"):
                ex = json.loads(line)
                pids = [(pool.add(p["title"], p["paragraph_text"]), p.get("is_supporting", False))
                        for p in ex["paragraphs"]]
                if split == "dev" and ex.get("answerable", True):
                    idx = str(len(queries))
                    queries.append({"_id": ex["id"], "text": ex["question"]})
                    answers[idx] = [ex["answer"]] + list(ex.get("answer_aliases", []))
                    qrels += [(ex["id"], pid) for pid, sup in pids if sup]
    _write(OUT / "musique", pool.rows, queries, answers, qrels)


def prepare_2wiki():
    src = RAW / "2wikimultihopqa"
    files = {s: src / f"{s}.json" for s in ("train", "dev")}
    for f in files.values():
        if not f.exists():
            sys.exit(f"Missing {f}. Download from Hugging Face: voidful/2WikiMultihopQA")
    pool = Pool("w2")
    queries, answers, qrels = [], {}, []
    for split, f in files.items():
        data = json.load(open(f, encoding="utf-8"))
        for ex in tqdm(data, desc=f"2wiki {split}"):
            title_to_pid = {}
            for title, sents in ex["context"]:
                title_to_pid[title] = pool.add(title, " ".join(sents))
            if split == "dev":
                idx = str(len(queries))
                queries.append({"_id": ex["_id"], "text": ex["question"]})
                answers[idx] = [ex["answer"]]
                sup_titles = {t for t, _ in ex.get("supporting_facts", [])}
                qrels += [(ex["_id"], title_to_pid[t]) for t in sup_titles if t in title_to_pid]
        del data
    _write(OUT / "2wikimultihopqa", pool.rows, queries, answers, qrels)


def _write(out, corpus, queries, answers, qrels):
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(corpus, out / "corpus.jsonl")
    write_jsonl(queries, out / "queries.jsonl")
    json.dump(answers, open(out / "answers.json", "w", encoding="utf-8"), ensure_ascii=False)
    with open(out / "qrels.tsv", "w", encoding="utf-8") as f:
        for qid, pid in qrels:
            f.write(f"{qid}\t{pid}\t1\n")
    print(f"{out.name}: {len(corpus)} unique passages, {len(queries)} dev queries, {len(qrels)} gold links")


if __name__ == "__main__":
    which = sys.argv[1:] or ["nq", "hotpotqa", "musique", "2wikimultihopqa"]
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in which:
        if (OUT / ds / "corpus.jsonl").exists() and (OUT / ds / "answers.json").exists():
            print(f"[skip] {ds} already prepared")
            continue
        {"nq": lambda: prepare_beir("nq"), "hotpotqa": lambda: prepare_beir("hotpotqa"),
         "musique": prepare_musique, "2wikimultihopqa": prepare_2wiki}[ds]()
