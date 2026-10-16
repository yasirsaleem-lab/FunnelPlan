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
| Corpora | BEIR corpora for NQ (2.68M passages) and HotpotQA (5.23M); pooled train and dev paragraphs for 2WikiMultihopQA (413k) and MuSiQue (102k); smaller than the IRCoT corpora (430k and 139k), so not directly comparable |
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

The analysis scripts are listed under [Repository layout](#repository-layout).

## Repository layout

```
FunnelPlan/
├── README.md, ARTIFACT.md      overview; reproduction guide and file inventory
├── config.py                   paths, models, corpus mode, offline mode
├── logutil.py                  run logs for the long GPU scripts (results_full/logs/)
├── pyproject.toml              package metadata and version ranges
├── docs/
│   ├── PREREGISTRATION.md      hypotheses, families, decision rules, dated amendments
│   ├── DECISIONS.md            design decisions and their history
│   ├── RATERS.md               human audit procedure and raters
│   └── environment.txt         exact package versions (pip freeze)
├── data/
│   ├── data_loader.py, corpus_utils.py
│   ├── ircot_prompts/          IRCoT demonstrations from the authors' release, with their licence
│   ├── cache_full/retrievals_v3/   cached ranked lists per dataset and retriever (inputs to every plan)
│   └── processed_full/
│       ├── <dataset>/                         queries.jsonl, qrels.tsv, answers.json (corpus.jsonl is not tracked)
│       ├── tsr_labels_<reader>[_graded].csv   every (question, plan) answer: grade, cost, confidence
│       ├── labels_<reader>_parts/             per dataset and retriever parts of the answer files
│       ├── control_labels_qwen3-14b-gguf*.csv quantization control (Qwen3 14B in GGUF)
│       ├── r3_2hop/, r4_ircot/                fixed two hop plans and IRCoT runs
│       ├── grading/                           judge scores, audit keys, rater files as received, adjudication
│       ├── routers/, metis/                   router embeddings and caches, METIS profiles
│       ├── ablation_prompt/                   prompt ablation answers (script 09)
│       ├── retrieval_quality/                 retrieval hit rates and reader use of the context (script 14)
│       ├── probe/                             probe items for the llama.cpp readers (script 16)
│       └── latency_*.json, cost_model_params.json, cpu_fallback_hotpotqa.json   latency profiles and cost model inputs
├── retrievers/                 BM25, SPLADE v3, BGE M3, Stella 1.5B, fusion, Qwen3 rerankers
│                               (GTE Qwen2 7B and Qwen3 Reranker 4B are kept from the pilot, not used in the paper runs)
├── tsr/                        readers and prompts, grading rules, features, cross validation, predictor
├── planner/                    budget aware planner and its fallback
├── cost_model/                 plan cost model and latency profiler
├── evaluation/                 retrieval metrics and the paired t test
├── figures_src/                diagram sources for Fig. 1 and Fig. 2 (palette.py is shared with the data figures)
├── tests/                      unit tests (retrievers, reranker padding, planner, cost model)
├── scripts/                    every pipeline step (table below)
└── results_full/
    ├── analysis/               one folder per analysis, with per question gains
    ├── tables_final/           tables.md, verdicts.txt, LaTeX tables (12_tables_final.py)
    ├── figures_final/          figures, PDF and PNG at 600 dpi (26_figures.py, figures_src/)
    ├── tables/                 power, held out oracles, label stability, retrieval quality (scripts 14, 15)
    ├── probe/                  llama.cpp reader probes and the quantization control report (script 16)
    └── run_all_log.txt         log of the last full analysis run
```

### Scripts

Scripts run from the repository root (`python scripts/<name>.py`). Each documents its usage at the top.

| Script | Stage | What it does |
|---|---|---|
| `00_prepare_full_corpora.py` | data | builds the four corpora, queries, qrels and answers |
| `03_profile_latency.py` | GPU | profiles retrieval latency per retriever and depth |
| `04_generate_tsr_labels.py` | GPU | retrieves once per retriever, answers every plan with the transformers readers |
| `08_grading_full.py` | grading | strict rule, local judge, final label, human audit sheets and reports |
| `09_prompt_ablation.py` | GPU | prompt ablation: the effect of the "use only the passages" instruction |
| `14_retrieval_quality.py` | analysis | retrieval hit rates per retriever and depth, and how readers use the context |
| `16_reader_probe.py` | GPU | timing and quality probe for the llama.cpp readers (gate G1) |
| `17_llamacpp_labels.py` | GPU | answers every plan with the llama.cpp readers (R1) and the quantization control (R2) |
| `18_audit_new_readers.py` | grading | second human audit, for Gemma 3 27B and Qwen3 32B |
| `19_two_hop_plans.py` | GPU | fixed two hop plans (R3) |
| `20_ircot.py` | GPU | IRCoT runs (R4) |
| `10_analysis_full.py` | analysis | H1 primary family: the fixed plan hull against every primary policy |
| `13_routers.py` | analysis | secondary routers (published Adaptive k rule, Adaptive RAG on embeddings and FLAN T5, per plan LR and kNN) |
| `15_power_headroom.py` | analysis | minimum detectable effect, held out oracles, label stability |
| `21_h3_ircot.py` | analysis | H3: IRCoT against the hull at its own latency |
| `22_weak_baselines.py` | analysis | R6: the same answers under the weak protocol |
| `23_metis_profiler.py` | GPU, analysis | R5: METIS style profiler (profiles with Qwen3 32B, then evaluation) |
| `24_label_sensitivity.py` | analysis | H1 and H3 under the strict and lenient grading rules |
| `25_step_hull.py` | analysis | baseline without mixing: the best single plan at the policy's cost |
| `27_cost_axes.py` | analysis | H1 under tokens, price weighted tokens, FLOPs and calls |
| `28_stronger_policies.py` | analysis | classifier gate and planner with closed book confidence features, learning curve |
| `29_checks.py` | analysis | sanity check against published numbers, grading error, matched cost headroom, mechanism |
| `12_tables_final.py` | output | final tables, `tables.md` and `verdicts.txt` |
| `26_figures.py` | output | final data figures |
| `run_all_analysis.py` | output | runs every analysis step above in order (CPU only) |
| `make_artifact.py` | output | builds the reproducibility package |
| `01_subsample_corpora.py` | pilot | subsampled corpora for the pilot (not used in the paper runs) |
| `04b_build_features.py`, `04c_label_noise_check.py`, `04d_regrade_labels.py` | pilot | pilot features, label noise and regrading |
| `05_train_predictor.py`, `06_run_planner_eval.py`, `07_cascade_gate.py` | pilot | pilot predictor, planner and cascade evaluation, replaced by `10` |
| `11_figures.py`, `12_tables.py` | superseded | earlier three reader figures and tables, replaced by `26` and `12_tables_final.py` |

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
