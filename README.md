# FunnelPlan

Budget aware retrieval planning for retrieval augmented agents.

For each query, a *plan* is a pair (retriever, depth): which retriever to call
and how many passages to hand to the reader LLM. FunnelPlan measures the task
success rate (TSR) and the real cost of every plan, on every query, for several
readers. On top of these labels it studies whether a planner, a TSR predictor
or a reader cascade can beat the best fixed plans at the same cost.

The repository produces the experiments for an ICDE 2027 Experiment, Analysis
and Benchmark (EAB) submission (deadline 11 November 2026). Every run is local.
No API model is used.

**To reproduce the paper, start with [ARTIFACT.md](ARTIFACT.md).** The final results are in
`results_full/tables_final/` (`verdicts.txt`, `tables.md`) and `results_full/figures_final/`. The status section
below is historical (4 October 2026).

---

## Contents

1. [Status](#1-status)
2. [Hardware and software](#2-hardware-and-software)
3. [Installation](#3-installation)
4. [Models](#4-models)
5. [Data](#5-data)
6. [Corpus modes](#6-corpus-modes)
7. [Retrievers and readers](#7-retrievers-and-readers)
8. [Pipeline](#8-pipeline)
9. [Outputs and file formats](#9-outputs-and-file-formats)
10. [Grading](#10-grading)
11. [Evaluation protocol](#11-evaluation-protocol)
12. [Project layout](#12-project-layout)
13. [Known pitfalls](#13-known-pitfalls)
14. [Tests](#14-tests)
15. [References](#15-references)

---

## 1. Status

Last updated 2026-10-04.

| Item | State |
|---|---|
| Full corpora for 4 datasets | built (`scripts/00_prepare_full_corpora.py`) |
| TSR labels, Qwen2.5 7B, full mode | MuSiQue done; 2Wiki, NQ, HotpotQA running |
| TSR labels, Llama 3.1 8B | model downloaded; runs after the 7B labels |
| TSR labels, Qwen3 14B, reduced grid | runs after Llama |
| Latency profiling, full mode | script ready; runs after labelling |
| `04b`, `04c`, `04d`, `05`, `06`, `07`, `fit_cost_model` | still written for sub200k mode (NQ and HotpotQA, `data/processed`); full mode port pending |
| Pilot and gate (sub200k) | done; results in `results/` |

Findings so far (pilot and gate, sub200k, 200 queries per dataset):

* A learned TSR planner, a confidence cascade (7B to 14B) and an answer
  agreement cascade all fail to beat the hull of fixed plans on held out
  queries. Gate result: pooled gain −0.040, 95% CI [−0.083, +0.005].
* Reader failures dominate retrieval failures.
* The depth penalty seen with a 4 bit reader disappears in bf16.
* Decoding is deterministic (greedy); repeated runs give identical labels.

These are pilot results. The paper uses the full mode runs.

---

## 2. Hardware and software

Reference machine for all reported numbers:

* Windows 11, one NVIDIA GPU with 20 GB VRAM, 64 GB RAM.
* Python 3.10 or newer, Anaconda environment `FunnelPlan`.
* CUDA build of PyTorch.

Disk space (full mode, engineering estimates): raw and processed corpora about
15 GB; dense embeddings about 11 GB per model for HotpotQA and 5.5 GB for NQ;
model weights about 60 GB in total.

---

## 3. Installation

```bash
conda create -n FunnelPlan python=3.10 -y
conda activate FunnelPlan

# PyTorch with CUDA first (pick the wheel that matches your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Project and dependencies (editable)
pip install -e .

# Optional extras
pip install -e ".[download]"   # ModelScope client, for gated models such as Llama
pip install -e ".[dev]"        # pytest, ruff, black
```

Version control (strongly recommended; the EAB track requires released artifacts):

```bash
git init
echo data/>> .gitignore
echo results*/>> .gitignore
echo __pycache__/>> .gitignore
echo _archive_v2/>> .gitignore
echo _backup_20261002/>> .gitignore
git add . && git commit -m "initial"
```

---

## 4. Models

`config.py` turns on Hugging Face offline mode by default, so no run touches
the network. To download a new model, run once with downloads enabled:

```bash
set FUNNELPLAN_HF_OFFLINE=0
set HF_ENDPOINT=https://hf-mirror.com
hf download <model id>
set FUNNELPLAN_HF_OFFLINE=1
```

Llama 3.1 is gated on Hugging Face. Download it from ModelScope instead:

```bash
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct ^
  --local_dir C:\Users\USER\models\Meta-Llama-3.1-8B-Instruct --exclude "original/*"
```

`tsr/agent_runner.py` loads Llama from that folder when it exists (the `local`
field of the `llama3.1-8b` reader). Change the path there if you store it elsewhere.

| Role | Model | Size |
|---|---|---|
| Sparse retriever | `naver/splade-v3` | 110M |
| Dense retriever | `BAAI/bge-m3` | 568M |
| Dense retriever | `NovaSearch/stella_en_1.5B_v5` | 1.5B |
| Dense retriever (sub200k only) | `Alibaba-NLP/gte-Qwen2-7B-instruct` | 7B |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` | 0.6B |
| Reranker (sub200k only) | `Qwen/Qwen3-Reranker-4B` | 4B |
| Reader | `Qwen/Qwen2.5-7B-Instruct` | 7B |
| Reader | `meta-llama/Llama-3.1-8B-Instruct` | 8B |
| Reader | `Qwen/Qwen3-14B` | 14B |

---

## 5. Data

Raw files go in `data/raw/`:

| Dataset | Files | Source |
|---|---|---|
| NQ | `data/raw/nq/corpus.jsonl`, `queries.jsonl`, `qrels.tsv` | BEIR |
| HotpotQA | `data/raw/hotpotqa/corpus.jsonl`, `queries.jsonl`, `qrels/` | BEIR |
| MuSiQue | `data/raw/musique/musique_ans_v1.0_train.jsonl`, `musique_ans_v1.0_dev.jsonl` | Hugging Face `bdsaglam/musique` |
| 2WikiMultihopQA | `data/raw/2wikimultihopqa/train.json`, `dev.json` | Hugging Face `voidful/2WikiMultihopQA` |

Prepared data (written by the scripts):

| Folder | Content | Written by |
|---|---|---|
| `data/processed/<ds>/` | 200k gold preserving subsample of NQ and HotpotQA, with queries, qrels and answers | `01_subsample_corpora.py` |
| `data/processed_full/<ds>/` | full corpora for all 4 datasets | `00_prepare_full_corpora.py` |

Full corpora:

| Dataset | Passages | Queries with gold answer | Corpus construction |
|---|---|---|---|
| NQ | 2,681,468 | 2,255 | BEIR NQ corpus |
| HotpotQA | 5,233,329 | 7,405 | official abstract corpus (BEIR) |
| MuSiQue | 101,958 | 2,417 | unique train and dev paragraphs, pooled (IRCoT) |
| 2WikiMultihopQA | 413,024 | 12,576 | unique train and dev contexts, pooled (IRCoT) |

Each processed folder holds `corpus.jsonl`, `queries.jsonl`, `qrels.tsv` and
`answers.json`. `answers.json` is keyed by the query's line index in
`queries.jsonl` (the same key `data_loader.load_processed` uses).

Limitation to state in the paper: the BEIR NQ corpus is a 2.68M passage slice
of Wikipedia, not the 21M passage DPR dump used by most open domain QA work.

---

## 6. Corpus modes

`CORPUS_MODE` decides which corpora, queries and output folders a run uses.
It is set at the top of `scripts/03_profile_latency.py` and
`scripts/04_generate_tsr_labels.py`, before `config` is imported. Other scripts
read the environment variable `FUNNELPLAN_CORPUS_MODE` (default `sub200k`).

| Setting | `sub200k` (pilot) | `full` (paper) |
|---|---|---|
| Corpus | 200k subsample | full corpus |
| Datasets | NQ, HotpotQA | NQ, HotpotQA, MuSiQue, 2Wiki |
| Queries per dataset | 200 | 500 |
| Retrievers | 8 (adds GTE Qwen2 7B, GTE+BM25 fusion, 4B reranker) | 6 |
| Reranker base | BM25 | BGE+BM25 fusion |
| Processed data | `data/processed` | `data/processed_full` |
| Caches | `data/cache` | `data/cache_full` |
| Results | `results` | `results_full` |

The two modes never share files. GTE Qwen2 7B and the 4B reranker are dropped in
full mode: encoding the full corpora with a 7B encoder would take on the order
of 80 GPU hours (engineering estimate), and the 4B reranker added cost without
gain in the pilot.

---

## 7. Retrievers and readers

Retrievers (`retrievers/registry.py` builds all of them):

| Name | Type | Notes |
|---|---|---|
| `bm25` | sparse | `bm25s`; index saved under `cache/bm25` |
| `splade_v3` | learned sparse | log(1+ReLU) weights; cached in chunks of 100k under `cache/splade` |
| `bge_m3` | dense | fp16 embeddings, resumable memmap cache |
| `stella_1_5b` | dense | `s2p_query` prompt on queries; NaN rows from fp16 overflow are re-encoded in fp32 |
| `bge_bm25_rrf` | hybrid | reciprocal rank fusion (k=60) of BGE M3 and BM25, pool of 100 each |
| `qwen3_reranker_0_6b` | reranker | reranks a fixed pool of the top 100 from `RERANK_BASE` |
| `gte_qwen2_7b`, `gte_bm25_rrf`, `qwen3_reranker_4b` | sub200k only | |

Retrieval always runs once per (dataset, retriever) at depth 100 and is cached.
Every depth is a prefix of that list, so plans at different depths are nested.

Dense search is exact inner product. The fp16 matrix goes on the GPU when it
fits next to the encoder, else search runs in chunks on the CPU. Profiling
records which case applied.

Readers (`tsr/agent_runner.py`, greedy decoding, chat template, short answer prompt):

| Name | Precision | Max new tokens | Use |
|---|---|---|---|
| `qwen2.5-7b` | bf16 | 100 | main reader, full grid |
| `llama3.1-8b` | bf16 | 100 | second model family, full grid |
| `qwen3-14b` | int8 (bitsandbytes), thinking disabled | 100 | larger model, reduced grid |
| `qwen2.5-7b-4bit` | nf4 | 32 | legacy pilot labels; 4 bit ablation only |

Depth 0 is closed book: the reader gets no passages and a closed book prompt.

---

## 8. Pipeline

Run from the project root. Each script has a `CONFIG` block at the top; edit it,
then run. Scripts that loop over datasets or methods start one fresh Python
process per block, so the GPU is clean each time, and they skip finished blocks.

### Step 0. Prepare data (once)

```bash
python scripts/01_subsample_corpora.py      # data/processed (needed by 00: it copies NQ and HotpotQA queries and answers)
python scripts/00_prepare_full_corpora.py   # data/processed_full
```

### Step 1. TSR labels

`scripts/04_generate_tsr_labels.py`:

```python
CORPUS_MODE = "full"
CONFIG = {
    "dataset": "musique",          # "nq", "hotpotqa", "musique", "2wikimultihopqa" or "all"
    "method": "all",               # one retriever, a list, or "all" (includes "closed_book")
    "reader": "qwen2.5-7b",        # "qwen2.5-7b", "llama3.1-8b", "qwen3-14b"
    "depths": [0, 1, 3, 5, 10, 20],
    "num_queries": settings.NUM_QUERIES,
    "seed": 42,
    "n_repeats": 1,
    "skip_done": True,
}
```

Run order for the paper:

1. Qwen2.5 7B, full grid: MuSiQue, 2Wiki, NQ, HotpotQA.
2. Llama 3.1 8B, full grid: same order.
3. Qwen3 14B, reduced grid: `depths [0, 3, 10]`, methods
   `["closed_book", "bm25", "bge_m3", "bge_bm25_rrf", "qwen3_reranker_0_6b"]`.

Keep `method` in registry order: `bge_bm25_rrf` needs the BGE M3 embeddings,
and the reranker needs the fusion results.

After each dataset, check recall@k of every retriever and TSR against closed
book before moving on. A retriever below closed book means a bug (this check
caught the Stella NaN bug).

### Step 2. Latency profiling (after labelling, never in parallel)

`scripts/03_profile_latency.py` with `CONFIG = {"dataset": "all", "method": "all"}`.
Times 50 queries (the first 50 of the labelled sample) three times each at
depths 1, 3, 5, 10, 20 and 100, after warm up. Also saves index load time, peak
VRAM, RAM, cache size on disk and GPU or CPU placement of the dense index.

### Step 3. Cost model

```bash
python -m cost_model.fit_cost_model
```

Retrieval cost is a line in depth per retriever; reader cost is
`a0 + a1 × prompt tokens`, fitted from the label timings
(sub200k fit: a0 = 0.064 s, a1 = 0.339 ms per token, R² = 0.954).

### Step 4. Features, noise and grading

```bash
python scripts/04b_build_features.py     # query, retrieval and reranker features
python scripts/04c_label_noise_check.py  # reruns a subset 3 times to measure label noise
python scripts/04d_regrade_labels.py     # applies every grading rule; writes the audit sheet
```

### Step 5. Prediction, planning and cascades

```bash
python scripts/05_train_predictor.py     # 5 fold grouped CV: plan prior, residual XGBoost, budget sweeps
python scripts/06_run_planner_eval.py    # planner versus the fixed plan hull
python scripts/07_cascade_gate.py        # reader cascade versus the hull, held out, bootstrap CI
```

---

## 9. Outputs and file formats

| File | Content |
|---|---|
| `<processed>/labels_<reader>_parts/<ds>__<retriever>.csv` | one part file per block; written atomically |
| `<processed>/tsr_labels_<reader>.csv` | merged labels (the legacy 4 bit reader writes `tsr_labels_v3.csv`) |
| `<cache>/retrievals_v3/<ds>__<retriever>.pkl` | top 100 ids and scores per query, per query latency, metadata |
| `<cache>/embeddings/<model>__<hash>.f16.npy` | dense embeddings; `.ok` marks a finished NaN scan |
| `<processed>/latency_profiles_<ds>.json` | latency by retriever and depth |
| `<processed>/latency_meta_<ds>.json` | index load time, memory, disk, GPU placement |
| `<processed>/cost_model_params.json` | fitted cost model |
| `results/tsr/` | CV metrics, budget sweeps, cascade and gate reports |
| `results/planner/` | planner frontier and figure |

Label columns:

| Column | Meaning |
|---|---|
| `dataset`, `query_id`, `retriever`, `depth`, `repeat` | the plan and query |
| `label` | success under the lenient rule at labelling time (regrade with 04d) |
| `answer` | raw reader output |
| `prompt_tokens`, `gen_tokens` | prompt and output length |
| `agent_time_s` | reader wall time |
| `retrieval_time_s` | retrieval wall time at depth 100 (0 for closed book) |
| `mean_logprob`, `min_logprob`, `first_token_prob` | reader confidence signals |
| `reader` | reader name |

---

## 10. Grading

`tsr/grader.py`, SQuAD normalisation (lowercase, strip punctuation and the
articles a, an, the). Rules:

| Rule | Correct when |
|---|---|
| `strict` | exact match, or the gold appears as a word sequence in the answer |
| `f1_07` | strict, or token F1 ≥ 0.7 (main rule) |
| `lenient` | strict, or token F1 ≥ 0.4 |
| `proposed` | strict, else F1 ≥ 0.7 unless a multi token gold has a much shorter answer |

`04d` also writes `label_adjudicated`: an LLM judgment for the 244 ambiguous
unique answers in the pilot, the strict rule elsewhere. It is not a human
label and the paper says so. A human spot check of about 30 items is required
before submission.

---

## 11. Evaluation protocol

* **Baseline.** The upper hull of fixed plans in (median cost, success) space.
  It never falls with cost and stays flat beyond the best plan. A method must
  beat this hull, not a single plan.
* **Held out selection.** Choose any setting on one half of the queries,
  evaluate on the other half, then swap. Gains on the selection half are
  reported but never used as results.
* **Uncertainty.** Paired bootstrap over queries, 2,000 resamples, 95% CI.
* **Gate rule (Route B).** A method counts as a contribution only with a held
  out gain of at least 0.03 over the hull and a CI above zero.

---

## 12. Project layout

```
FunnelPlan/
├── config.py                  settings, corpus mode, offline mode
├── pyproject.toml
├── data/
│   ├── data_loader.py         load raw and processed datasets
│   ├── corpus_utils.py        passage text, gold answer lookup
│   ├── raw/                   downloaded datasets
│   ├── processed/             sub200k data and labels
│   ├── processed_full/        full mode data and labels
│   ├── cache/                 sub200k caches
│   └── cache_full/            full mode caches
├── retrievers/
│   ├── registry.py            name -> retriever
│   ├── sparse/                bm25, splade_v3
│   ├── dense/                 bge_m3, stella_1_5b, gte_qwen2_7b, emb_cache
│   ├── hybrid/                RRF fusion
│   └── rerankers/             Qwen3 rerankers
├── tsr/
│   ├── agent_runner.py        readers and prompts
│   ├── grader.py              grading rules
│   ├── features.py            plan features
│   ├── predictor.py           TSR predictor
│   └── cv.py                  CV, hull, budget sweeps, cascades
├── cost_model/                profiler and cost model fit
├── planner/                   planner and fallback logic
├── evaluation/                metrics, significance tests
├── scripts/                   numbered pipeline (00 to 07)
├── tests/
├── results/                   sub200k results
└── results_full/              full mode results
```

---

## 13. Known pitfalls

* **Windows shared GPU memory.** When VRAM runs out, the NVIDIA driver spills to
  system RAM silently and runs about 20 times slower. Scripts run one process
  per block and check free VRAM after loading. If a block is suddenly very
  slow, check Task Manager for "Shared GPU memory" use.
* **fp16 overflow.** Stella v5 (Qwen2 backbone) gives NaN embeddings for some
  passages in fp16. `emb_cache` re-encodes those rows in fp32, and search never
  ranks NaN scores.
* **PyCharm overwrites.** If a file changes on disk while open, use
  File > Reload All from Disk before running, or PyCharm may write its old copy back.
* **Offline mode.** A model that is not in the local cache fails to load. See
  section 4 to download it.
* **Profiling during labelling.** Never run 03 and 04 at the same time; both
  timings become wrong.
* **Corpus mode mismatch.** 04b to 07 still default to sub200k. Do not mix their
  outputs with full mode labels until they are ported.

---

## 14. Tests

```bash
pytest -q
```

Covers the cost model, planner, reranker padding and retriever basics.

---

## 15. References

Datasets and corpora: Natural Questions (Kwiatkowski et al., TACL 2019);
HotpotQA (Yang et al., EMNLP 2018); MuSiQue (Trivedi et al., TACL 2022);
2WikiMultihopQA (Ho et al., COLING 2020); BEIR (Thakur et al., NeurIPS 2021
Datasets and Benchmarks); corpus pooling and 500 query protocol from IRCoT
(Trivedi et al., ACL 2023); DPR Wikipedia (Karpukhin et al., EMNLP 2020).

Methods: SPLADE (Formal et al., SIGIR 2021); reciprocal rank fusion
(Cormack et al., SIGIR 2009); LLM.int8() (Dettmers et al., NeurIPS 2022);
SQuAD evaluation (Rajpurkar et al., EMNLP 2016).

Related work the paper positions against: Adaptive RAG (NAACL 2024); Lost in the
Middle (TACL 2024); The Power of Noise (SIGIR 2024); The Powerless Noise
(SIGIR 2026); Sufficient Context (ICLR 2025); Long Context LLMs Meet RAG
(ICLR 2025); Adaptive k (EMNLP 2025); AutoMix (NeurIPS 2024); RouteLLM
(ICLR 2025); METIS (SOSP 2025); Retriever Portfolios (ICML 2026); FlashRAG
(WWW 2025); RAGO (ISCA 2025); BrowseComp Plus (ACL 2026).

---

## License

MIT
