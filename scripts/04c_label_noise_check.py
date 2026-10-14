"""
Measure TSR label noise properly.

Reruns the agent N_REPEATS times on a random subset of (query, plan) pairs,
using the SAME cached passages the labels were built from. Reports:
  * repeat agreement: fraction of pairs where all reruns agree
  * flip rate against the stored label
  * implied ceiling: with label flip probability e, a perfect predictor can
    match a noisy label at most 1 - e of the time.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from data.data_loader import get_processed
from data.corpus_utils import resolve_gold
from tsr.agent_runner import AgentRunner
from tsr.grader import Grader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

N_QUERIES_PER_DATASET = 25
N_REPEATS = 3
SEED = 0
# Set to True to also rerun with the agent reloaded between repeats. Reloading
# exposes nondeterminism from kernel selection that a warm model can hide.
RELOAD_BETWEEN_REPEATS = True

LABELS = settings.PROCESSED_DATA_ROOT / settings.LABELS_FILE
CACHE_DIR = settings.CACHE_ROOT / "retrievals_v3"
OUT = settings.RESULTS_ROOT / "tsr" / "label_noise.json"


def main():
    labels = pd.read_csv(LABELS, dtype={"query_id": str})
    labels = labels[labels["repeat"] == 0]
    rng = np.random.default_rng(SEED)
    agent = AgentRunner()
    records = []

    for ds in labels["dataset"].unique():
        corpus, queries, _, answers = get_processed(ds)
        grader = Grader(ds)
        agent.set_corpus(corpus)
        sub = labels[labels["dataset"] == ds]
        qids = rng.choice(sub["query_id"].unique(), N_QUERIES_PER_DATASET, replace=False)
        sub = sub[sub["query_id"].isin(qids)]
        caches = {}
        for r in sub["retriever"].unique():
            with open(CACHE_DIR / f"{ds}__{r}.pkl", "rb") as f:
                caches[r] = pickle.load(f)

        outs = {i: [] for i in sub.index}
        for rep in range(N_REPEATS):
            agent.load()
            for i, row in tqdm(sub.iterrows(), total=len(sub), desc=f"{ds} repeat {rep}"):
                q = queries[row["query_id"]]["text"]
                res = caches[row["retriever"]]["results"][row["query_id"]][: int(row["depth"])]
                a = agent.answer(q, res)
                gold = resolve_gold(queries, answers, row["query_id"])
                outs[i].append((a, int(grader.is_correct(q, a, gold))))
            if RELOAD_BETWEEN_REPEATS:
                agent.unload()
        agent.unload()

        for i, row in sub.iterrows():
            ans = [a for a, _ in outs[i]]
            labs = [l for _, l in outs[i]]
            records.append({
                "dataset": ds, "query_id": row["query_id"], "retriever": row["retriever"],
                "depth": int(row["depth"]), "stored_label": int(row["label"]),
                "rerun_labels": labs, "answers_identical": len(set(ans)) == 1,
                "labels_agree": len(set(labs)) == 1,
                "flip_vs_stored": float(np.mean([l != row["label"] for l in labs])),
            })

    df = pd.DataFrame(records)
    summary = {
        "n_pairs": int(len(df)),
        "n_repeats": N_REPEATS,
        "answer_string_identical_rate": float(df["answers_identical"].mean()),
        "label_agreement_rate": float(df["labels_agree"].mean()),
        "flip_rate_vs_stored": float(df["flip_vs_stored"].mean()),
        "by_dataset": df.groupby("dataset")[["answers_identical", "labels_agree", "flip_vs_stored"]].mean().to_dict(),
        "note": "If answers are identical across reruns, decoding noise is ~0 and the depth flips "
                "in the labels are real context effects, not noise.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT.with_suffix(".csv"), index=False)
    OUT.write_text(json.dumps(summary, indent=2))
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
