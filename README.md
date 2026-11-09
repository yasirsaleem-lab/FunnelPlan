# FunnelPlan

Code, answers and analyses for the paper **Adaptive Retrieval Versus a Coin Flip: Cost-Matched Baselines Change the
Verdict** (ICDE 2027, Experiment, Analysis, and Benchmark track).
The extended version of the paper, with every table and figure, is
[docs/extended/FunnelPlan_ICDE2027_extended.pdf](docs/extended/FunnelPlan_ICDE2027_extended.pdf).

FunnelPlan is a benchmark for retrieval augmented generation (RAG). A *plan* is a retriever and a depth (how many
passages the reader gets), or no retrieval. We ran every plan on every question for five open readers, graded every
answer and measured its cost. Each adaptive policy is then compared with the **fixed plan hull**: the best random mix
of two fixed plans at the policy's own mean cost, chosen on training folds.

All runs are local. No API model is used.

## Main findings

* The baseline decides the verdict. Under one test, five primary policies beat Always RAG and Never RAG in 23 of 27
  pooled tests, the best single plan at the same cost in 5 to 7, and the hull in none.
* Adaptive RAG as published beats Always RAG by 0.070 to 0.091, but a random mix of its own rungs at its cost does as well.
* No primary policy beats the hull. 53 of 54 upper bounds exclude a gain of 0.03 under the per question test, the
  corrected test and a refit bootstrap.
* IRCoT beats the hull on 2Wiki, but fixed chains that run its reasoning for a set number of steps match it, so its
  stopping rule adds nothing.

## Benchmark

| Part | Setting |
|---|---|
| Datasets | NQ, HotpotQA, 2WikiMultihopQA, MuSiQue; 500 questions each |
| Retrievers | BM25, SPLADE v3, BGE M3, Stella 1.5B, BGE M3 + BM25 (RRF), Qwen3 Reranker 0.6B |
| Depths | 0 (closed book), 1, 3, 5, 10, 20; plus fixed two hop plans |
| Readers | Qwen2.5 7B, Llama 3.1 8B, Qwen3 14B, Gemma 3 27B, Qwen3 32B; greedy decoding |
| Multi step | IRCoT, and fixed step chains F1 to F4 |
| Cost | measured latency; also tokens, FLOPs and calls |
| Grading | strict rule plus a local judge, checked by four human audits |

## Reproduce

All answers, grades and costs are in the repository, so the analyses need only a CPU.

```bash
conda create -n FunnelPlan python=3.10 -y
conda activate FunnelPlan
pip install -e .
python scripts/run_all_analysis.py     # core analyses, tables and figures (about 7 hours)
```

Scripts 26 to 43 add the later analyses (mixed workload, refit bootstrap, audits 3 and 4, Adaptive RAG as published,
fixed step chains, tail latency, split halves, the comparator at a fixed test, IRCoT call counts). Their outputs are in `results_full/analysis/`; each script states how
to run it at the top. The plan for these analyses is in `docs/ANALYSIS_PLAN_2026-11-05.md`.
[ARTIFACT.md](ARTIFACT.md) explains the inputs and how to regenerate the answers on a GPU.

## Layout

```
config.py          paths and models
data/              questions, cached retrievals, graded answers, audits (data/processed_full/)
retrievers/, tsr/  retrievers, readers, prompts and grading rules
cost_model/        plan cost model and latency profiler
scripts/           every step, numbered in run order
results_full/      analysis outputs, tables (tables_final/) and figures (figures_final/)
docs/              analysis plan, rater procedure, exact package versions
```

## Answer files

`data/processed_full/tsr_labels_<reader>_graded.csv` has one row per question and plan: the answer, `label_final`
(1 correct, 0 wrong), the alternative grades, token counts, and the measured reader and retrieval time.

## Hardware

One desktop: RTX 3080 (20 GB), Ryzen 7 5700X, 64 GB RAM, Windows. Latency depends on the machine, so the results are
also reported under machine independent costs.
