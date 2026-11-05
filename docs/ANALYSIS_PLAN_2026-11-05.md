# Analysis plan for the added experiments (frozen 5 November 2026, before any run)

This file fixes, before any of the runs below, what is run, how it is scored and what is reported.
Every outcome is reported, whichever way it goes. No test is dropped after its result is seen.
The decision rule is unchanged: a policy beats a hull only if, in both plan sets, its Holm adjusted p is below
0.05 and its interval is above zero under the per question test, the corrected test, and the corrected test with
standard errors times the largest refit ratio (now 1.64). IRCoT style methods, which fit nothing but the hull,
use a refit bootstrap of each test instead of the two corrected tests.
Anything not finished by 8 November 2026, 12:00 (UTC+8) goes to the extended version or to future work.

## E0. IRCoT on NQ (data only)
Script 06, readers Qwen2.5 7B, Llama 3.1 8B, Qwen3 14B, the same 500 NQ questions. IRCoT has no NQ
demonstrations; Adaptive RAG runs IRCoT on NQ with the HotpotQA demonstrations (its NQ prompt files are byte
identical to the HotpotQA ones), and so do we. Everything else as for the other datasets (K = 4, at most 8 steps,
at most 15 paragraphs). Graded by script 07 like every other answer. Merged into labels_<reader>_nq.csv, so the
existing IRCoT analyses (H2) do not change.

## E1. Adaptive RAG as published, on the four dataset mix
Ladder: A = no retrieval (closed book), B = single step BM25 with 5 passages (the published single step setting),
C = IRCoT. Classifier: t5-large (the published model), learning rate 3e-5, batch 32, max length 384, trained on
the question text only. Silver labels as published: the simplest rung whose answer is correct; questions that no
rung answers get the dataset type label (NQ: B; HotpotQA, 2Wiki, MuSiQue: C). The number of epochs is chosen once
per reader from {15, 20, 25, 30, 35} on an inner split of the seed 0, fold 0 training data, then fixed.
Training: pooled over the four datasets, as published; 5 folds x 5 seeds across the pool, as in the mixed workload.
Costs: each question pays its rung's measured cost plus the measured classifier inference time.
Comparators, on the same held out questions: (i) Always RAG (BM25, 5 passages) and Never RAG, the published
comparison; (ii) the global hull of the fixed plans; (iii) the label hull; (iv) the global hull with IRCoT added as
one more plan. Single step plan set and with the fixed two hop plans. Readers: the three IRCoT readers.

## E2. Fixed step chain of thought controllers against IRCoT
F_T runs IRCoT's reasoning chain for exactly T retrievals (T = 1, 2, 3, 4) with stopping switched off, then
answers with the plan answer prompt. The number of calls is fixed in advance, so F_T is a plan.
Construction: decoding is greedy, so the first T steps of a stored chain with at least T retrievals are reused and
only the answer call is new. A chain that stopped earlier is continued: the stored sentence is the next query, and
new reasoning calls are made where needed (timed live). The 15 paragraph cap is kept (F_4 adds 3 paragraphs).
Costs: answer calls, new reasoning calls and new BM25 calls are timed live. Reused reasoning calls take the
question's own measured IRCoT time per reasoning call. Check: for 30 questions per reader and dataset the whole
chain is rerun with per call timing; we report the median absolute error of the reused cost, and that the rerun
reproduces the stored sentences.
Analyses: (a) quality and latency frontier of single step plans, the two hop plans, F_1 to F_4 and IRCoT, per
reader and dataset; (b) IRCoT against the hull extended with F_1 to F_4 (and with the two hop plans), per question
test and refit bootstrap (100 resamples), Holm over the tests of each hull; (c) IRCoT minus F_4 on the same
questions, success and cost (the value of adaptive stopping). Readers: the three IRCoT readers; datasets: all four.

## E3. FLARE (stretch; run only if E0 to E2 finish by 7 November)
Settings as published for 2WikiMultihopQA: the question is the first query; the answer is generated sentence by
sentence; retrieval is triggered when any token of the next sentence has probability below theta; the query is the
sentence with tokens below beta = 0.4 masked; BM25. theta is chosen on the training folds from
{0.2, 0.4, 0.6, 0.8}. Details not fixed here are written into an addendum to this file before the run.

## A4. Human audit 4 (same raters as audit 3, same guidelines, blind, no AI help)
100 further discordant questions (25 per dataset, not in audit 3) for the confidence cascade of Qwen2.5 7B with
single step plans, the one open test; and 50 discordant questions (IRCoT answer against the best fixed plan's answer)
for IRCoT with Qwen3 14B on 2Wiki, its largest win. Sampling seed 4. Disagreements are settled by both raters
together. Reported as for audit 3: the shift of label_final on the gain, with a bootstrap interval.

## CPU analyses
1. Margin curve: counts of the 54 primary upper bounds below 0.010, 0.015, 0.020, 0.025, 0.030, 0.040 and 0.050
   under the three variance estimates (computed on 5 November from tests_nb.csv; the numbers are final).
2. Refit bootstrap on 12 randomly drawn tests (numpy seed 20261105, drawn from the primary and secondary families
   and both plan sets, excluding the four already run), 100 resamples each. If the largest ratio of the refit
   spread to the corrected error exceeds 1.64, the larger ratio replaces 1.64 everywhere.
3. Tail latency: the hull matched on the policy's P95 (and P99) latency instead of its mean, from the per question
   costs on the held out fold; and success under P95 service levels on a grid. Same tests and rule as the main
   analysis.
4. Grading: population weighted label accuracy by audit stratum; the per arm error rates of audits 3 and 4
   propagated into the gains of the audited tests; audits 1 to 3 by dataset and reader.
5. Mechanism: the share of questions where the two hull plans disagree; held out pair AUROC against training size.
6. Split half replication: the primary family on two disjoint halves of each dataset (question ids split by a
   seeded hash, seed 20261105).

## Addendum (5 November 2026, before any CPU analysis was run)
- CPU 2, the random draw (seed 20261105) is fixed in scripts/37_refit_random.py: 12 tests from the primary and
  secondary families. Policies outside the draw, for technical reasons only: the reader cascades (two reader
  blocks), the FLAN-T5 Adaptive RAG classifier (a GPU fine tune per replicate) and the METIS style router
  (profiles stored per question).
- CPU 3 (scripts/38_tail_latency.py): the target is the policy's realized 95th (and 99th) percentile latency on the
  held out fold; the matched mix is chosen on the training questions among plans on the training
  (percentile, success) frontier, weights in steps of 0.05. The service level grid is dropped: it would need every
  policy retuned per level.
- CPU 4 (scripts/39_grading_mechanism.py): the population accuracy weights the two regions (inside and outside the
  ambiguous band) by their share of all single step answers. The grading sensitivity shifts every primary upper
  bound by the largest audited shift of label_final against the policy (and by the end of its interval).
- CPU 6: the split half replication covers the five single reader primary policies (not the two reader cascades).
- Timed GPU runs (E0, E2, E3) run alone; the CPU analyses run after them, never at the same time.
