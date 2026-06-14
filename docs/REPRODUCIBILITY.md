# Reproducibility

This repository supports the manuscript question: **Does cross-wave evaluation overestimate screening performance?** The public code release focuses on participant overlap, ID-isolated evaluation, model-complexity comparison, threshold behaviour, calibration, decision utility, and robustness analyses for possible sarcopenia screening in CHARLS.

## What can be reproduced from this repository alone

- Script syntax and structure can be inspected.
- Public aggregate tables and final publication figures can be reviewed.
- The public-release safety checker can be run.

## What requires authorised CHARLS data

- Cohort construction.
- Possible-sarcopenia label construction.
- Model-input table preparation.
- Model fitting.
- Row-level evaluation prediction generation.
- Strict-label sensitivity reruns from individual records.

## Primary methodological questions

1. Did non-isolated Wave 3 evaluation yield more favourable estimates than ID-isolated evaluation?
2. Did XGBoost A+B provide a meaningful advantage over Logistic A-only after participant isolation?
3. Did accuracy at threshold 0.50 conceal inadequate sensitivity?

## Exclusions

Raw CHARLS data, processed row-level datasets, row-level predictions, participant IDs, and trained model objects are intentionally excluded.

## Environment

The included `requirements.txt` and `environment.yml` reflect the current reproducibility environment used for post hoc checks. The exact original training-environment versions were not fully recoverable.
