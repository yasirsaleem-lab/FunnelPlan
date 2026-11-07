# FunnelPlan: reproducibility guide

There are three levels. Level 1 needs no software. Level 2 needs a CPU. Level 3 needs a GPU for several days.

## Level 1: read the results

| What | Where |
|---|---|
| Verdict of every test family | `results_full/tables_final/verdicts.txt` |
| Every table | `results_full/tables_final/tables.md` and `tab_*.tex` |
| Figures | `results_full/figures_final/` |
| Per question gains and test outputs | `results_full/analysis/` |
| Human audits and how they were run | `data/processed_full/grading/`, `docs/RATERS.md` |

## Level 2: rerun the analyses (CPU)

```bash
pip install -e .
python scripts/run_all_analysis.py
```

This rebuilds the core analyses, the tables and the figures from the stored answers. The later analyses are scripts
26 to 40; run their CPU stages one by one as each script's header says. Seeds are fixed, so results match on the
versions in `docs/environment.txt`.

Main inputs: the graded answers (`data/processed_full/tsr_labels_<reader>_graded.csv`), the two hop plans
(`r3_2hop/`), IRCoT (`r4_ircot/`), the fixed step chains (`r5_cotfixed/`), the cached retrievals
(`data/cache_full/retrievals_v3/`) and the judge scores and audits (`grading/`).

## Level 3: regenerate the answers (GPU)

| Step | Script |
|---|---|
| Corpora | `00_prepare_full_corpora.py` |
| Retrieval and transformers readers | `02_generate_tsr_labels.py` |
| llama.cpp readers | `03_reader_probe.py`, `04_llamacpp_labels.py` |
| Two hop plans, IRCoT, fixed step chains | `05_two_hop_plans.py`, `06_ircot.py`, `34_fixed_step_controllers.py` |
| Grading | `07_grading_full.py` |
| Adaptive RAG as published (T5 fine tunes) | `36_adaptive_rag_published.py` |

Latency depends on the hardware, so a new machine gives different absolute costs.

## Not included

Corpora (3 GB), indexes and embeddings (about 41 GB) and model weights (licences and size). The scripts above rebuild
them; model ids are in `config.py`.

## Integrity

The rater files are kept as returned, with only the raters' names removed from the file metadata. Rater names are not
in this repository.
