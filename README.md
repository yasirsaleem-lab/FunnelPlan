# FunnelPlan

**Do adaptive retrieval policies beat fixed retrieval plans at the same cost?**

FunnelPlan is an experimental benchmark for retrieval augmented generation (RAG). A *plan* is a choice of retriever
and retrieval depth (how many passages the reader LLM receives), or no retrieval at all. We run every plan on every
question for five open readers, grade every answer, and record its measured cost. On this complete plan matrix we test
whether adaptive policies, which pick a plan per question, do better than the **fixed plan hull**: the best randomised
mix of fixed plans at the policy's own mean cost, fitted on training folds only.

All runs are local. No API model is used.

## Main findings

| Result | Value |
|---|---|
| Primary policies against the hull (27 tests, Holm) | 0 positive, in both plan sets (with and without two hop plans) |
| Secondary routers and a METIS style profiler (30 tests) | 0 positive under the decision rule |
| Stronger learned policies (classifier gate, planner with self knowledge) | 0 of 10 positive |
| Same tests with token, price weighted token, FLOP and call costs | 0 positive on every cost axis |
| Gains of 0.03 or more ruled out (one sided 95% bound) | 27 of 27 tests (single step plans), 25 of 27 (with two hop plans) |
| Adaptive RAG router | significantly worse than the hull for 4 of 5 readers (latency), 5 of 5 (tokens, FLOPs) |
| Weak protocol (Always RAG baseline, one seed, strict EM) on the same answers | 64 of 100 apparent wins; 0 under our protocol |
| Held out headroom at matched cost (oracle over 2 plans) | +0.10 over the hull; the best policy closes at most 13% of it |
| What planners learn (held out AUROC) | 0.62 to 0.65 for "is this question answerable", 0.48 to 0.50 for "which hull plan wins" |
| IRCoT (multi step) against the single step hull | significantly better on 2WikiMultihopQA for all 3 readers, under all 3 grading rules |

Every number comes from the files in `results_full/`. `results_full/tables_final/verdicts.txt` lists each decision,
computed from the result files.

## Benchmark

| Component | Setting |
|---|---|
| Datasets | Natural Questions, HotpotQA, 2WikiMultihopQA, MuSiQue; 500 questions each, seeded sample |
| Corpora | BEIR corpora for NQ (2.68M passages) and HotpotQA (5.23M); pooled train and dev paragraphs for 2WikiMultihopQA (413k) and MuSiQue (102k), as in IRCoT |
| Retrievers | BM25, SPLADE v3, BGE M3, Stella 1.5B v5, BGE M3 plus BM25 fused by reciprocal rank fusion (k = 60), Qwen3 Reranker 0.6B over the fused top 100 |
| Depths | 0 (closed book), 1, 3, 5, 10, 20 |
| Fixed two hop plans | BM25, 3 or 5 passages per hop; the reader writes the second query |
| Readers | Qwen2.5 7B Instruct and Llama 3.1 8B Instruct (bf16, 31 plans); Qwen3 14B (int8, 25 plans, depth up to 10); Gemma 3 27B and Qwen3 32B (GGUF IQ4_XS on llama.cpp, 25 plans). Greedy decoding |
| Multi step reference | IRCoT with BM25, for the three smaller readers on the multi hop datasets |
| Cost | mean measured latency per question (retrieval plus reader); also tokens, price weighted tokens, FLOPs and calls |
| Grading | `label_final`: a strict string rule, plus a Qwen3 14B judge on ambiguous answers; checked against two human audits of 200 answers each, adjudicated by a third rater |

**Policies tested.** Primary: TSR planner (predicts the success of every plan and trades it against cost), Adaptive
RAG router (query complexity classifier), Adaptive k (largest gap in similarity scores), closed book confidence gate,
confidence cascade, and two reader cascades (a small reader escalates to a larger one). Secondary: Adaptive k with the
published rule, Adaptive RAG routers on BGE M3 embeddings and on FLAN T5, per plan logistic regression and kNN, and a
port of the METIS profiler. Stronger policies: a gradient boosted classifier gate on closed book confidence signals,
and the TSR planner with those signals added.

**Protocol.** 5 folds x 5 seeds. Every policy and the hull choose their settings on four folds and are scored on the
fifth, so each question is scored once per seed. Gains are averaged per question over seeds and tested with a one
sample t test on 2,000 questions per test, with bootstrap 95% intervals and Holm correction within each family. A
result counts only if it survives Holm with its interval above zero in both plan sets. The hypotheses, families and
decision rule were written down before the runs: see `docs/PREREGISTRATION.md`, including every dated amendment.

## Reproduce the results

All answers, grades and costs are included, so the analyses need only a CPU.

```bash
conda create -n FunnelPlan python=3.10 -y
conda activate FunnelPlan
pip install -e .                       # a CPU build of torch is enough for the analyses
python scripts/run_all_analysis.py     # 18 steps, about 4 hours on an 8 core desktop CPU
```

This rebuilds `results_full/analysis/`, `results_full/tables_final/` (tables and `verdicts.txt`) and
`results_full/figures_final/`. A full rerun on the reference machine reproduced every verdict exactly.
[ARTIFACT.md](ARTIFACT.md) lists every input file, what is not included, and how to regenerate the answers on a GPU.

## Regenerate the answers (GPU)

| Step | Script |
|---|---|
| Build corpora | `scripts/00_prepare_full_corpora.py` |
| Retrieve, then answer with transformers readers | `scripts/04_generate_tsr_labels.py` |
| Answer with llama.cpp readers | `scripts/16_reader_probe.py` (setup), `scripts/17_llamacpp_labels.py` |
| Fixed two hop plans | `scripts/19_two_hop_plans.py` |
| IRCoT | `scripts/20_ircot.py` |
| Grade (strict rule, judge, final label) | `scripts/08_grading_full.py` |
| Human audit sheets and reports | `scripts/08_grading_full.py`, `scripts/18_audit_new_readers.py` |
| Latency profiles | `scripts/03_profile_latency.py` |

Each script documents its usage at the top. The analysis scripts are `10` (primary family), `13` (secondary routers),
`15` (power and headroom), `21` (IRCoT), `22` (weak protocol), `23` (METIS port), `24` (grading sensitivity),
`25` (single plan baseline), `27` (cost axes), `28` (stronger policies), `29` (sanity, grading and mechanism checks),
`12` (tables) and `26` (figures).

## Repository layout

```
FunnelPlan/
├── ARTIFACT.md                 reproduction guide
├── config.py                   paths, models, corpus mode, offline mode
├── docs/
│   ├── PREREGISTRATION.md      hypotheses, families, decision rules, dated amendments
│   ├── DECISIONS.md            design decisions and their history
│   └── RATERS.md               human audit procedure and raters
├── data/
│   ├── data_loader.py, corpus_utils.py
│   └── processed_full/
│       ├── tsr_labels_<reader>_graded.csv     every (question, plan) answer: grade, cost, confidence
│       ├── r3_2hop/, r4_ircot/                two hop plans and IRCoT
│       ├── grading/                           judge scores, audit keys, rater files as received, adjudication
│       ├── routers/, metis/                   router inputs and METIS profiles
│       └── latency_*.json                     latency profiles
├── retrievers/                 BM25, SPLADE, dense, fusion, rerankers
├── tsr/                        readers and prompts, grading rules, features, cross validation
├── scripts/                    numbered pipeline (see above)
└── results_full/
    ├── analysis/               one folder per analysis, with per question gains
    ├── tables_final/           tables.md, verdicts.txt, LaTeX tables
    └── figures_final/          figures (PDF, and PNG at 600 dpi)
```

## Answer file format

`data/processed_full/tsr_labels_<reader>_graded.csv` has one row per (question, plan):

| Column | Meaning |
|---|---|
| `dataset`, `query_id`, `retriever`, `depth` | the question and the plan (`closed_book` with depth 0 is no retrieval) |
| `answer` | raw reader output |
| `label_final` | final grade (1 correct, 0 wrong); `label_final_source` says whether the strict rule or the judge decided |
| `label_strict`, `label_lenient`, `label_f1_07` | alternative grading rules used in the sensitivity analysis |
| `em`, `contains`, `f1` | exact match, answer contained, token F1 against the gold answers |
| `prompt_tokens`, `gen_tokens` | prompt and output length |
| `agent_time_s`, `retrieval_time_s` | measured reader and retrieval time in seconds |
| `mean_logprob`, `min_logprob`, `first_token_prob` | reader confidence signals |

## Hardware

One desktop: NVIDIA RTX 3080 with 20 GB, AMD Ryzen 7 5700X, 64 GB RAM, Windows. Generating all answers takes several
GPU days per large reader. Latency depends on the machine, which is why the results are also reported under
machine independent costs.

## Notes

* **NQ corpus.** The BEIR NQ corpus is a 2.68M passage slice of Wikipedia, not the 21M passage DPR dump used in much
  open domain QA work. Absolute accuracies are not comparable to results on the DPR corpus.
* **Models without peer reviewed papers.** SPLADE v3, Stella 1.5B v5 and Qwen3 Reranker 0.6B are cited by their
  model cards and technical reports.
* **Offline mode.** `config.py` loads models from the local cache only. Set `FUNNELPLAN_HF_OFFLINE=0` once to
  download a model.
* **Windows GPU memory.** When VRAM runs out, the driver silently spills into system RAM and runs about 20 times
  slower. The scripts run one process per block and check free VRAM after loading.
* **Do not profile latency while labelling.** Running scripts 03 and 04 at the same time corrupts both timings.

## Citation

A citation entry will be added when the paper is public.

## License

Code: MIT. The datasets, corpora and models keep their own licences.
