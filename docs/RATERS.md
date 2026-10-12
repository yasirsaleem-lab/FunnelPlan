# Human raters

Who labelled each audit sheet. Every rater confirms by email: no AI was used, no other rater's labels or judge
output were seen, and labels were entered on a blank sheet (labels and notes empty).

| Sheet | File | Rater | Date | How entered | Status |
|---|---|---|---|---|---|
| audit 1, rater 1 | `grading/audit_rater1.xlsx` (copy `audit_rater1_ghalib_20261012.xlsx`) | rater 1 | 2026-10-12 | blank sheet (labels and notes cleared), entered on the first author's computer | done |
| audit 1, rater 2 | `grading/audit_rater2.xlsx` (from `audit_rater2_BLANK_for_usman.xlsx`) | rater 2 | [date] | blank sheet | pending |
| audit 2, rater 1 | `grading/audit2_rater1.xlsx` | rater 1 | by 2026-10-12 | blank sheet | done |
| audit 2, rater 2 | `grading/audit2_rater2.xlsx` | rater 2 | 2026-10-11 | blank sheet | done |

Reference only, not raters:

- the first author (author): both audit 1 sheets of 2026-10-07 (`audit_rater1_yasir_20261007.xlsx`,
  `audit_rater2_yasir_20261007.xlsx`). The same person labelled both, so their kappa (0.837) measures one rater's
  consistency, not agreement between raters.
- Claude (LLM): `grading/audit_rater2.csv` (identical to `audit_llm.csv`). Never used as a rater;
  `08_grading_full.py` skips the audit until the human rater 2 sheet is complete.

Confirmation emails: [rater 1, date] [rater 2, date]
