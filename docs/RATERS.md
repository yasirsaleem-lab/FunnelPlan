# Human raters

Two independent raters labelled audits 1 and 2. Each received a blank sheet (labels and notes empty), worked without
AI, and saw no other rater's labels and no judge output. Sheets were exchanged by WeChat; the dated messages are kept
by the author as confirmation.

| Audit | Rater 1 | Rater 2 |
|---|---|---|
| audit 1 (Qwen2.5 7B, Llama 3.1 8B, Qwen3 14B; 200 items) | rater 1 (a coauthor), returned 2026-10-11 18:07 (`grading/audit_rater1.xlsx`) | rater 2 (not an author), returned 2026-10-12 21:39 (`grading/audit_rater2.xlsx`) |
| audit 2 (Gemma 3 27B, Qwen3 32B; 200 items) | rater 1 (a coauthor), returned 2026-10-11 18:07 (`grading/audit2_rater1.xlsx`) | rater 2 (not an author), returned 2026-10-11 16:05 (`grading/audit2_rater2.xlsx`) |
| audit 3 (answers where a policy and its hull disagree; 586 answers) | rater A (not an author) | rater B (not an author) |
| audit 4 (same design, same raters; 278 answers) | rater A (not an author) | rater B (not an author) |

Rater names are not given in this repository.

Adjudication: the 88 items where rater 1 and rater 2 disagree (57 in audit 1, 31 in audit 2) were labelled by a third
independent rater (rater 3, not an author), returned 2026-10-13 14:42 (`grading/received/adjudication_rater3_BLANK.xlsx`), on a blank
sheet in random order without rater labels, audit or reader names, and without AI. The human reference is the majority
of the three labels (`grading/audit_adjudicated.csv`). So every reference label is that of a rater who is not an author:
rater 2 on the 312 items the two raters agree on, rater 3 on the 88 they do not.

Audits 3 and 4: raters A and B worked blind to the test, the arm, the reader and every automatic label, with written
guidelines and without AI. They settled their disagreements together (35 items in audit 3, 8 in audit 4). Files are in
`grading/audit3/` and `grading/audit4/`.
