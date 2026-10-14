# Human raters

Two independent raters labelled both audits. Each received a blank sheet (labels and notes empty), worked without
AI, and saw no other rater's labels and no judge output. Sheets were exchanged by WeChat; the dated messages are kept
by the author as confirmation.

| Audit | Rater 1 | Rater 2 |
|---|---|---|
| audit 1 (Qwen2.5 7B, Llama 3.1 8B, Qwen3 14B; 200 items) | rater 1, returned 2026-10-11 18:07 (`grading/audit_rater1.xlsx`) | rater 2, returned 2026-10-12 21:39 (`grading/audit_rater2.xlsx`) |
| audit 2 (Gemma 3 27B, Qwen3 32B; 200 items) | rater 1, returned 2026-10-11 18:07 (`grading/audit2_rater1.xlsx`) | rater 2, returned 2026-10-11 16:05 (`grading/audit2_rater2.xlsx`) |

Adjudication: the 88 items where rater 1 and rater 2 disagree (57 in audit 1, 31 in audit 2) were labelled by a third
independent rater, rater 3, returned 2026-10-13 14:42 (`grading/received/adjudication_rater3_BLANK.xlsx`), on a blank
sheet in random order without rater labels, audit or reader names, and without AI. The human reference is the majority
of the three labels (`grading/audit_adjudicated.csv`).
