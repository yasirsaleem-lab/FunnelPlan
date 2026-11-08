"""
Where the time of IRCoT and of the fixed step chains F_T goes. CPU only; reads the stored logs.

Per reader and dataset, for IRCoT and for F_1 to F_4: mean retrievals, reader calls, reader calls made after the last
retrieval (IRCoT keeps reasoning when its 15 paragraph cap stops further retrieval), paragraphs in the prompt,
generated tokens, prompt tokens, and reader and retrieval seconds per question.

Usage:  python scripts/42_multistep_calls.py
Output: results_full/analysis/multistep/calls.csv, calls_summary.txt
"""
import os
import sys
from pathlib import Path

os.environ["FUNNELPLAN_CORPUS_MODE"] = "full"
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import settings

READERS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-14b"]
PROC = settings.PROCESSED_DATA_ROOT
OUT = settings.RESULTS_ROOT / "analysis" / "multistep"
NUM = ["lm_calls", "retrieval_calls", "n_paragraphs", "prompt_tokens", "gen_tokens", "agent_time_s", "retrieval_time_s"]


def read(path):
    d = pd.read_csv(path, dtype={"query_id": str}, keep_default_na=False)
    for c in NUM + ["reason_prompt_tokens"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def main():
    rows = []
    for rd in READERS:
        ir = pd.concat([read(p) for p in (PROC / "r4_ircot" / f"labels_{rd}_graded.csv",
                                          PROC / "r4_ircot" / f"labels_{rd}_nq_graded.csv") if p.exists()])
        ir["plan"] = "IRCoT"
        ft = read(PROC / "r5_cotfixed" / f"labels_{rd}_graded.csv")
        ft["plan"] = ft.retriever.str.replace("cotfixed", "F", regex=False)
        for d in (ir, ft):
            # reasoning calls after the last retrieval (the answer call excluded): a chain with r retrievals makes
            # r - 1 reasoning calls between them and one answer call, so lm_calls - retrieval_calls is what is left
            d["calls_after_last_retrieval"] = (d.lm_calls - d.retrieval_calls).clip(lower=0)
            for (ds, plan), g in d.groupby(["dataset", "plan"]):
                rows.append({"reader": rd, "dataset": ds, "plan": plan, "n": len(g),
                             **{c: g[c].mean() for c in NUM + ["calls_after_last_retrieval"]},
                             "share_with_calls_after_last_retrieval": float((g.calls_after_last_retrieval > 0).mean()),
                             "seconds": float((g.agent_time_s + g.retrieval_time_s).mean())})
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "calls.csv", index=False)
    L = ["Calls, tokens and time per question (mean over questions), IRCoT and fixed step chains F_T.",
         "lm_calls counts every reader call, the answer call included; calls_after_last_retrieval = reasoning calls",
         "after the last retrieval (F_T makes none). F_T tokens cover only its new calls. Mean over the three readers:"]
    cols = ["retrieval_calls", "lm_calls", "calls_after_last_retrieval", "share_with_calls_after_last_retrieval",
            "n_paragraphs", "gen_tokens", "seconds"]
    L.append(T.groupby(["dataset", "plan"])[cols].mean().round(2).to_string())
    L.append("\nPer reader, IRCoT against F_4:")
    L.append(T[T.plan.isin(["IRCoT", "F4"])].set_index(["reader", "dataset", "plan"])[cols].round(2).to_string())
    (OUT / "calls_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
