# Pre-registration of the extension runs

Written 2026-10-08 (UTC+8), before any of the runs below was started. Committed to git with this date.
Nothing in this file may be changed after a run starts without a dated entry in section 7.
Background and the history of earlier decisions: `docs/DECISIONS.md`.

## 1. Runs

| ID | Run | Readers | Data | Plans |
|---|---|---|---|---|
| R1 | Larger readers on the plan grid | Qwen3 32B (GGUF IQ4_XS), Gemma 3 27B (GGUF IQ4_XS), llama.cpp | NQ, HotpotQA, 2Wiki, MuSiQue, the same 500 queries | the 25 plans with depth <= 10 (as for Qwen3 14B) |
| R2 | Quantization and runtime control | Qwen3 14B GGUF IQ4_XS, llama.cpp | HotpotQA | 25 plans |
| R3 | Fixed two hop plans (cost known before execution) | all readers | all 4 datasets | BM25, k = 3 and k = 5 per hop |
| R4 | IRCoT reference point (data dependent cost; not a plan) | Qwen2.5 7B, Llama 3.1 8B, Qwen3 14B | HotpotQA, 2Wiki, MuSiQue | BM25, published IRCoT settings |
| R5 | METIS style per query profiler policy | all readers | all 4 datasets | same plan space as the policy |
| R6 | Weak baseline rescoring (no new GPU runs) | all readers | all 4 datasets | Always RAG (one retriever, k = 5), Never RAG, single seed, strict EM |

Grading of all new answers: the same `label_final` rule (Qwen3 14B judge on ambiguous answers, threshold 0.5,
strict elsewhere). Both human raters label 200 new answers drawn by the same stratified design.

## 2. Hypotheses and tests

**H1 (main).** No adaptive single step policy beats the held out fixed plan hull at equal mean latency.
Test: pooled per reader gain, 5 folds x 5 seeds, t test on per query gains, Holm within the family.
Primary family for the final paper: the 5 policies of `10_analysis_full.py` x 5 readers, plus the reader
cascades (7B to Llama 8B, 7B to Qwen3 14B, and 7B to Qwen3 32B): **28 tests**.
Secondary family: the 5 baselines of `13_routers.py` (Adaptive k published rule, Adaptive RAG with BGE-M3 and FLAN-T5, per plan
LR and kNN) x 5 readers, plus the METIS style profiler x 5 readers: **30 tests**.
Exploratory: every dataset level test, Holm over all of them.

**H2 (scale, descriptive).** Headroom and its structure change with reader size. Reported for each reader with
95% bootstrap CIs, no significance claim: two plan oracle gain, gate oracle gain, all plan oracle gain,
share of best plan failures with the answer string in context, label flip rate between neighbouring depths,
and the 5 x 5 routing transfer matrix.

**H3 (multi step).** IRCoT beats the held out single step hull at IRCoT's own mean latency.
Test: per reader and dataset (3 x 3 = 9 tests), paired per query gain, Holm over the 9.
The fixed two hop plans (R3) enter the plan set; every H1 result is reported with and without them.

Minimum detectable effects (from `15_power_headroom.py`): about 0.027 pooled, 0.055 per dataset, nominal.
A non significant result is reported as "no gain larger than about 0.03 pooled", never as "no effect".

## 3. Gates (go or no go)

| Gate | Date | Condition | If it fails |
|---|---|---|---|
| G1 probe | 2026-10-09 | `16_reader_probe.py`: the projected full grid time for Qwen3 32B <= 3 GPU days, no context errors at depth 10, strict accuracy on the probe within 0.10 of Qwen3 14B | Use Qwen3 30B A3B; if that also fails, a 12 plan subset taken from the union of the existing readers' hull plans, with every small reader statistic recomputed on the same subset |
| G2 quantization | after R2 | label agreement of Qwen3 14B GGUF with Qwen3 14B int8 >= 0.85 on HotpotQA | Report the large readers as "4 bit, llama.cpp" with the measured agreement; do not pool them with the int8 or bf16 readers |
| G3 IRCoT sanity | 2026-10-25 | Llama 3.1 8B IRCoT on HotpotQA within 0.05 of the published value of Moskvoretskii et al. (ACL 2025), on their metric, after checking which metric they report | Drop R4; keep R3; narrow every claim to "single step and fixed two hop plans" |
| G4 freeze | 2026-10-28 | R1 (at least one large reader) and R3 complete | Do not submit to ICDE 2027 (no revision round); target SIGMOD 2027 |

## 4. Decision rules for the framing

| Outcome | Title | Claim |
|---|---|---|
| H1 holds for all readers; IRCoT not significantly above the hull on any dataset | "Fixed Retrieval Plans Are Hard to Beat: Evaluating Adaptive RAG Policies at Equal Latency" | Even an iterative reference point does not exceed the single step frontier at its own latency |
| H1 holds; IRCoT significantly above the hull on exactly one dataset | same | Choosing among single step plans does not pay; iterating pays on that dataset at its measured latency multiple |
| IRCoT significantly above the hull on two or more datasets | "When Does Adaptive Retrieval Pay Off? A Latency Matched Evaluation Against Fixed Plan Frontiers" | Adaptivity across steps beats adaptivity across configurations |
| A policy significantly beats the hull for a larger reader | same neutral title | Adaptive headroom that policies can reach grows with reader size |

## 5. Reported regardless of outcome

Every policy, reader and dataset in the families above; the in sample hull next to the held out hull;
mean and median latency versions; the fast SPLADE robustness run; all seeds; the decision log.

## 6. Fixed implementation choices

Prompts, system texts and chat templates exactly as in `tsr/agent_runner.py` (Qwen3 thinking off); greedy
decoding; at most 100 new tokens; one query at a time (no batching), so latencies are comparable within a reader.
Latencies of llama.cpp readers are compared only within the same reader (each reader has its own hull).

## 7. Amendments

Every change is dated and states whether any result of the affected run had been seen.

- **2026-10-08, gate G1 result (Qwen3 32B, before R1 started).** Probe of 200 HotpotQA items with
  `16_reader_probe.py`: 0 errors; VRAM 18.5 of 20 GB; strict accuracy 0.420 vs 0.410 for Qwen3 14B on the same
  items (label agreement 0.920); projected full grid 12.6 GPU hours (0.52 days). The longest probe prompt was
  1,603 tokens; the longest grid prompt is about 4,948, below n_ctx 5,632. The context condition is therefore
  checked again on the full run: `17_llamacpp_labels.py` logs every failed call and refuses to merge while any
  answer is missing. G1 passes.
- **2026-10-08, gate G1 result (Gemma 3 27B, before R1 started).** Same probe: 0 errors; VRAM 16.1 of 20 GB;
  strict accuracy 0.360 vs 0.410 for Qwen3 14B (label agreement 0.880; discordant 7 vs 17, exact McNemar
  p = 0.064); projected full grid 10.7 GPU hours. Longest grid prompt about 4,775 tokens, n_ctx 6,144. G1 passes.
  Observed, not acted on: Gemma answers "Not specified." on 4 of 160 retrieval items despite the "give your best
  guess" instruction, and writes dates as "November 22, 1761" (strict misses these; they are ambiguous and go to
  the judge). Gemma's chat template has no system role; llama.cpp puts the system text into the user turn.
  No change to prompts or grading.
- **2026-10-08 19:50, primary family corrected (NQ labels of Qwen3 32B seen, in sample descriptives only; no
  H1 test run).** The cascade "7B to Qwen3 32B" is removed from the primary family, which becomes 27 tests.
  Reason: its cost would add a transformers latency (7B, bf16) to a llama.cpp latency (32B, IQ4_XS), which
  section 6 already rules out. It is reported as exploratory only if the 7B is re-timed under llama.cpp.
  Observed on NQ, not acted on: Qwen3 32B answers in sentences more often (21% of retrieval answers above
  20 tokens vs 6% for Qwen3 14B) and abstains more ("the passages do not mention", 5.8% vs 3.6%). Of 25 random
  long answers with F1 < 0.2 and no gold string, 0 were correct, so the strict rule is not under crediting them.
- **2026-10-09 07:20, R3 defined exactly (before any R3 run; R1 labels of Qwen3 32B seen, in sample only).**
  Section 1 left open how the second hop query is formed. Fixed now, in `scripts/19_two_hop_plans.py`:
  plans `bm25_2hop@3` and `bm25_2hop@5`. Hop 1: BM25 top k for the question. Then one reader call writes one
  search query for the missing information (fixed system text in the script, greedy, at most 32 new tokens;
  empty output falls back to the question). Hop 2: BM25 top k for that query alone, skipping hop 1 passages.
  Then the unchanged answer prompt on the 2k passages. "Cost known before execution" means the number of
  retrieval calls (2) and reader calls (2) is fixed in advance; latency still varies with token counts, as it
  does for single step plans. Run on all 5 readers and all 4 datasets (NQ included). Cost = hop 1 latency +
  timed hop 2 latency + both reader calls. The alternative (hop 2 query = question plus hop 1 titles, no reader
  call) was rejected as a weak method that would lose for trivial reasons.
- **2026-10-09 23:50, gate G2 result (R2 complete).** Qwen3 14B GGUF IQ4_XS (llama.cpp) against Qwen3 14B
  int8 (transformers), HotpotQA, 12,500 cells: label agreement strict 0.960, F1 >= 0.7 0.958, label_final
  0.957 (judge scores reused where they exist; 190 of 1,875 ambiguous answers fall back to strict until the
  next judge pass). Identical answer text on 72% of cells. label_final accuracy 0.491 vs 0.500. G2 passes:
  4 bit llama.cpp readers are reported next to the int8 and bf16 readers. Mean reader time 0.32 s (llama.cpp)
  vs 1.58 s (transformers int8) for the same model: the runtime changes latency about 5 times, which confirms
  that latencies are comparable only within a reader (section 6).
- **2026-10-10 12:00, R4 bug fix (IRCoT results seen: Llama 3.1 8B, HotpotQA, 147 questions).** The sentence
  splitter of `20_ircot.py` ended a sentence at every period followed by a space, including initials and
  abbreviations, so "The W. H. Auden poem ..." became the kept sentence and retrieval query "The W." (9 of 388
  queries were cut to nothing BM25 can match; more were cut short). This broke the pre-registered rule (keep the
  first generated sentence). Fixed: no split after an initial, a common abbreviation, or before a lowercase word.
  The 147 answers produced with the bug are kept apart (`hotpotqa_buggy_splitter.jsonl`) and not used; Llama 3.1 8B
  on HotpotQA is rerun from the start. No other IRCoT run had started.
- **2026-10-10 12:30, R5 and R6 fixed exactly, before either was run.**
  R5 (`scripts/23_metis_profiler.py`): the profiler is Qwen3 32B (llama.cpp, thinking off, greedy, question only),
  the largest local model; METIS used GPT-4o or Llama 3.1 70B. It returns joint reasoning (yes/no), complexity
  (high/low), pieces of information n (1 to 10) and a confidence; one profile per query, shared by all readers.
  METIS's rules mapped to our plan space: depth in [n, 3n] (nearest labelled depth if empty); single step plans for
  no joint reasoning (METIS map_rerank) and for joint reasoning with low complexity (stuff); with joint reasoning and
  high complexity the fixed two hop plans are added when they are in the plan set (map_reduce). Confidence below 0.9
  falls back to the median profile of the training folds (METIS: the recent default). Choice: the most expensive
  pruned plan with mean cost at most a budget b, else the cheapest pruned plan (METIS: the configuration with the
  highest resource need that fits). The retriever set (one retriever or all) and b are tuned on the training folds,
  like every other method. METIS's feedback loop (a golden configuration every 30 queries) is not ported, because
  it needs gold answers at serving time. Primary: profiler cost 0, the most favourable case for METIS (the profiler
  runs in another runtime, so its latency cannot be added to the reader's). Secondary: plus the measured profiler
  latency. It enters the secondary Holm family, as pre-registered.
  R6 (`scripts/22_weak_baselines.py`): Always RAG = BM25 at k = 5 (the retriever of Adaptive-RAG), Never RAG =
  closed book, accuracy only with no cost matching, one seed (0), label_strict. Reported for every policy, reader
  and dataset next to our protocol's result. Descriptive only, no test.
- **2026-10-10 11:25, gate G3 replaced (IRCoT results seen: Llama 3.1 8B, HotpotQA, 49 of 500 questions;
  strict 0.571, 3.55 reason calls, 3.37 retrieval calls, 98% of chains stop on "answer is"; no other IRCoT
  result seen).** G3 as written compares our strict accuracy with the In-Accuracy of Moskvoretskii et al.
  (0.438). That paper retrieves from the DPR Wikipedia dump of Karpukhin et al. (2020; 21M passages of 100 words);
  we retrieve from IRCoT's own HotpotQA corpus (5.2M abstracts), on a different 500 question sample. The gate
  therefore mixes a corpus effect and a sample effect with the implementation check it was meant to be. This flaw
  was ours and was found only after the first IRCoT answers were seen; we disclose that order here.
  New G3 (implementation check, all on Llama 3.1 8B, HotpotQA, full 500): (i) mean LM reason calls per question
  within 0.5 of the published 3.5, (ii) mean retrieval calls within 0.5 of the published 3.45, (iii) at least 95%
  of chains end with "answer is" (the published stopping rule). If G3 fails, the consequence is unchanged: drop R4
  and narrow every claim to single step and fixed two hop plans. The paper reports the published 0.438 next to our
  value and names both differences (corpus and sample). Accuracy is not used as a gate.
- **2026-10-10 04:30, R4 (IRCoT) fixed exactly, before any IRCoT run.** `scripts/20_ircot.py`. Published
  settings kept: BM25; the query of each later step is the last generated chain of thought sentence; only the
  first generated sentence is kept; stop on "answer is" or after 8 reason steps; at most 15 paragraphs; a separate
  reader at the end; the authors' own demonstrations (github.com/StonyBrookNLP/ircot, Apache 2.0, Flan-T5 files)
  packed in file order up to about 6,000 tokens (24,000 characters), as for their Flan-T5 runs. Choices our setting
  forces, fixed a priori with no tuning on any test query: K = 4 paragraphs per step (the paper tunes K in
  {2, 4, 6, 8} on 100 dev questions; we have no separate tuning split with retrieval labels); M = 1 distractor in
  the demonstrations (paper tunes M in {1, 2, 3}); one demonstration set (the paper averages three); the final
  reader is our unchanged answer prompt (direct reader, as in the paper's Flan-T5 setting), so that IRCoT differs
  from the single step plans only in retrieval; raw text completion for the reason step, greedy, at most 64 new
  tokens. Cost = all reason calls + the answer call + all BM25 calls, measured as for every plan. Readers: Qwen2.5
  7B, Llama 3.1 8B, Qwen3 14B (H3, 9 tests). Gate G3 compares Llama 3.1 8B on HotpotQA with Moskvoretskii et al.
  (ACL 2025), whose primary metric is In-Accuracy (prediction contains the gold answer; 0.438, EM 0.414, F1 0.516,
  3.5 LM calls and 3.45 retrieval calls per question). Our closest metric is the strict rule (exact match or
  contained). Their 500 question subset and corpus differ from ours (our HotpotQA corpus is the same 5.2M paragraph
  corpus IRCoT used), so G3 tests the implementation, not a replication.
- **2026-10-08, implementation detail (no R1 result seen).** For llama.cpp readers, `agent_time_s` is the
  server's compute time (prompt_ms + predicted_ms), the analogue of the `generate()` time measured in 04.
  Wall time per call is stored as `wall_s`. Prompts were checked identical to the probe items (200 of 200),
  whose token counts matched the transformers prompts of Qwen3 14B (median ratio 1.00).
