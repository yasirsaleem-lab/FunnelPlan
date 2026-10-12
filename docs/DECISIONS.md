# Decision log and test families

This file records what was decided, when, and whether it was decided before or after seeing results.
It exists so that a reader can judge the analysis for forking paths. Dates are local (UTC+8), 2026.

## 1. Hypotheses and test families

**Primary hypothesis H1.** No adaptive policy beats the best randomised mix of fixed retrieval plans (the
fixed plan hull) at equal mean latency, on held out queries.

| Family | Tests | Correction | Role |
|---|---|---|---|
| Primary | pooled gain (4 datasets, 2,000 queries) of the 5 policies in `10_analysis_full.py` (TSR planner, Adaptive RAG router, Adaptive k gap only, closed book gate, confidence cascade) for 3 readers, plus the 2 reader cascades: **17 tests** | Holm within the family | Confirms or rejects H1 |
| Secondary | pooled gain of the 5 baselines added in `13_routers.py` (Adaptive k published rule, Adaptive RAG with BGE-M3 and with FLAN-T5, per plan BGE-M3 LR and kNN) for 3 readers: **15 tests** (ar_tfidf, a near duplicate of the Adaptive RAG router, is reported only in the appendix) | Holm within the family | Robustness: is the result an artifact of weak or unfaithful baselines? |
| Exploratory | every dataset level test of 10 and 13: **140 tests** | Holm over all 140 | Locates effects; never used alone for a claim |

A policy "beats the hull" only if its pooled 95% bootstrap CI is above 0 and its Holm adjusted p < 0.05 within
its family. Minimum detectable effects (80% power): pooled 0.027 nominal, 0.039 at the strictest Holm step;
single dataset 0.055 and 0.087 (`15_power_headroom.py`). A null result is therefore stated as "no gain larger
than about 0.03 pooled".

## 2. Fixed before any full corpus result was seen

- Datasets, query samples (500 per dataset, seeded), corpora, the 6 retrievers, depths {0, 1, 3, 5, 10, 20},
  greedy decoding, the strict instruction prompt.
- The plan matrix design (every query x every plan x every reader).
- Readers: Qwen2.5 7B and Llama 3.1 8B in bf16. Qwen3 14B in int8 was added for scale. Depth 20 was dropped for
  the 14B because it did not fit in 20 GB of VRAM (a hardware limit, decided before its results).
- The fixed plan hull as the baseline; 5 fold held out evaluation.

## 3. Changed after results were seen (with the effect on the conclusions)

| When | Change | Why | Effect on conclusions |
|---|---|---|---|
| Oct 7 | Hull chosen on the training folds instead of the test fold | The test fold hull is optimistic (selection over 31 plans on 100 queries), biasing every method downward | Gate result moved from -0.040 to +0.017 (still not significant). A correction of a bias, applied to all methods |
| Oct 7 | Grading: `label_final` (local judge on ambiguous answers, strict elsewhere), threshold 0.5 | Strict and lenient rules disagree with humans (accuracy 0.67 and 0.67 on items two humans agree on) | Threshold fixed at 0.5 before the human audit; the audit sweep (0.3 to 0.8) changes accuracy by less than 0.013, so it was kept |
| Oct 8 | 5 seeds (fold split and XGBoost seed) and Holm correction | One seed moved the TSR planner by up to 0.02, as large as the effects | Removed the only nominal positive (confidence cascade on 7B MuSiQue, Holm p 1.0); reader cascade fell from +0.021 to +0.006 |
| Oct 8 | Plan cost = mean latency instead of median | The latency budget constrains expected cost; the cost model predicts the mean (median error 3.5% to 4.3%); for the 14B the mean is up to 1.8x the median | No conclusion changed; Holm survivors went from 2 to 4 (all negative). Both versions are reported |
| Oct 8 | Stronger routers (13): BGE-M3, FLAN-T5, per plan LR and kNN | Reviewer risk that the null result comes from a weak router | Secondary family; nothing beats the hull; Adaptive RAG loses with every classifier |
| Oct 8 | Adaptive k with the published rule (buffer of 5, top 90% search) | Literature check: our first version lacked the buffer | Secondary family; better than gap only on Llama (+0.027), still no gain over the hull |
| Oct 8 | Second rater sheet added (corrected Oct 12) | | Both audit 1 sheets of Oct 7 were labelled by the author (the first author), so kappa 0.837 is the author agreeing with himself, NOT agreement between two people. Corrected Oct 12: audit 1 is relabelled on blank sheets by rater 1 (done) and rater 2 (pending); see docs/RATERS.md |

## 4. Known implementation dependent quantities

- **SPLADE latency** is an exhaustive sparse matrix product on the CPU (`retrievers/sparse/splade_v3.py`):
  0.25 to 0.40 microseconds per passage, so 41 ms on MuSiQue (102k passages) and 1.29 s on HotpotQA
  (5.2M). An inverted index would be much faster. Robustness run with SPLADE retrieval set to 50 ms:
  `10_analysis_full.py` with `retrieval_cost_override = {"splade_v3": 0.05}`, results in
  `results_full/analysis/final_splade50ms/`.
- **XGBoost** results differ slightly across machines and builds (TSR planner up to 0.007 after averaging
  5 seeds). The reported numbers come from the author's machine.
- **BM25** (bm25s, default parameters, title and text) scores 0.03 to 0.05 nDCG@10 below the published BEIR
  numbers on NQ and HotpotQA. Not tuned on the evaluation queries.

## 5. Out of scope (stated in the paper)

Multi step retrieval (IRCoT, Self-RAG, FLARE), query rewriting, readers above 14B parameters, API models,
batched serving, cost in money.
