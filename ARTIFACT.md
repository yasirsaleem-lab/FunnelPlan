# FunnelPlan: reproducibility package

This package reproduces every number, table and figure of the FunnelPlan study of adaptive retrieval policies
against the fixed plan hull. It has three levels. Level 1 needs nothing but a text
editor. Level 2 needs a CPU and a few hours. Level 3 regenerates the answers and needs a GPU for several days.

## 1. Check the results without running anything

| What | Where |
|---|---|
| Every preregistered decision, computed from the result files | `results_full/tables_final/verdicts.txt` |
| Every table in readable form | `results_full/tables_final/tables.md` |
| LaTeX tables used in the paper | `results_full/tables_final/tab_*.tex` |
| Figures | `results_full/figures_final/f2` to `f11`, `fig_framework`, `fig_policies` (`.pdf` vector, `.png` at 600 dpi) |
| Per query gains of every test | `results_full/analysis/*/per_query_gains.csv` |
| Preregistration, amendments, commit record | `docs/PREREGISTRATION.md` (section 7) |
| Design decisions and their history | `docs/DECISIONS.md` |
| Human raters and how the audits were run | `docs/RATERS.md` |

## 2. Rerun every analysis on a CPU (level 2)

All answers, labels and costs are in the package, so no model is needed.

```
pip install -e .                      # Python 3.10 or later; a CPU build of torch is enough for this level
python scripts/run_all_analysis.py    # 18 steps; see --list. Resume after an error with --from <step>
```

The run rewrites `results_full/analysis/`, `results_full/tables/`, `results_full/tables_final/` and
`results_full/figures_final/`. Compare the new `verdicts.txt` with the paper. The analyses are deterministic
(fixed fold seeds 0 to 4, fixed XGBoost and bootstrap seeds), so the numbers should match on the library versions
in `docs/environment.txt`; other versions can move the third decimal. The full run took 3.7 hours on the author's machine (Ryzen 7 5700X, 8 cores).

Inputs used at this level:

| Input | Path |
|---|---|
| Graded answers of every (query, plan, reader) | `data/processed_full/tsr_labels_<reader>_graded.csv` |
| Fixed two hop plans (R3) | `data/processed_full/r3_2hop/labels_<reader>_graded.csv` |
| IRCoT runs (R4) | `data/processed_full/r4_ircot/labels_<reader>_graded.csv` |
| Query text and ranked lists with scores | `data/cache_full/retrievals_v3/*.pkl` (in the repository; pandas and Python objects written by `04_generate_tsr_labels.py`) |
| Router embeddings and FLAN T5 predictions | `data/processed_full/routers/emb_*.npz`, `t5_cache_*.json` |
| METIS style profiles (Qwen3 32B) | `data/processed_full/metis/profiles.jsonl` |
| LLM judge scores | `data/processed_full/grading/judge_scores.csv` |
| Human audits | `data/processed_full/grading/` (`received/` holds the rater files exactly as received) |
| Measured latency per answer | columns `agent_time_s`, `retrieval_time_s` of the graded files |

## 3. Regenerate the answers (level 3, GPU)

This repeats the measurement itself. It needs the corpora, the indexes and the model weights, none of which are in
the package because of size (about 45 GB of indexes and embeddings, plus the models).

| Step | Command | Notes |
|---|---|---|
| Corpora | `python scripts/00_prepare_full_corpora.py` | NQ and HotpotQA from BEIR; 2Wiki and MuSiQue built from train and dev paragraphs, as in IRCoT |
| Retrieval caches | `python scripts/04_generate_tsr_labels.py` (writes `retrievals_v3/*.pkl` before answering) | BM25 (bm25s), SPLADE v3, BGE M3, Stella 1.5B v5, BGE M3 plus BM25 with RRF, Qwen3 Reranker 0.6B over the RRF list |
| Answers, transformers readers | `python scripts/04_generate_tsr_labels.py` | Qwen2.5 7B Instruct, Llama 3.1 8B Instruct (bf16), Qwen3 14B (int8, LLM.int8()); greedy decoding |
| Answers, llama.cpp readers | `python scripts/17_llamacpp_labels.py run <reader>` | Gemma 3 27B and Qwen3 32B, GGUF IQ4_XS (bartowski), llama.cpp build b11491; see `16_reader_probe.py help` |
| Two hop plans (R3) | `python scripts/19_two_hop_plans.py query/hop2/answer/merge <reader>` | |
| IRCoT (R4) | `python scripts/20_ircot.py run <reader>`, then `merge` | |
| Grading | `python scripts/08_grading_full.py regrade`, `judge`, `finalize` | strict rule plus a Qwen3 14B judge on ambiguous answers |
| Latency profiles | `python scripts/03_profile_latency.py` | |

Time on one RTX 3080 20 GB: several GPU days per large reader (gate G1 allowed up to 3 GPU days for the Qwen3 32B
grid). Latency is hardware dependent, so a different machine gives different absolute costs; the paper therefore
also reports the results under token, FLOP and call costs (`scripts/27_cost_axes.py`), which do not depend on the
machine.

## 4. Hardware and software used

One desktop: RTX 3080 with 20 GB (modified card), Ryzen 7 5700X, 64 GB RAM, Windows, Anaconda environment
"FunnelPlan". `pyproject.toml` gives the version ranges; `docs/environment.txt` (output of `pip freeze`) gives the
exact versions used. No API model was used.

## 5. Integrity

* `MANIFEST.sha256` inside the package lists the sha256 and size of every file. The same list is kept next to the
  zip file. Check it with `sha256sum -c` after removing the size column, or with a short Python loop.
* Build the package with `python scripts/make_artifact.py` on a clean tree; the file name carries the git commit.
* The preregistration file and every amendment are under version control. Section 7 of `docs/PREREGISTRATION.md`
  records, for each amendment, when it was written and when it was first committed. The repository had no remote
  until 14 October 2026; the first push to GitHub (11:42 UTC+8, every commit up to c38d1aa) is the earliest
  external timestamp, so no commit before it has one.
* The human audit files in `data/processed_full/grading/received/` are the files as the raters sent them.

## 6. What is not included, and why

| Item | Reason | How to rebuild |
|---|---|---|
| Corpora (`corpus.jsonl`) | 3 GB; public datasets | `scripts/00_prepare_full_corpora.py` |
| BM25, SPLADE and dense indexes, passage embeddings | about 41 GB | retrievers in `retrievers/`, run by `scripts/04_generate_tsr_labels.py` |
| Model weights | licences and size | Hugging Face ids in `config.py` and `scripts/16_reader_probe.py` |
| Per call progress logs (`labels_*_parts/*.jsonl`) | superseded by the merged label files | not needed |
