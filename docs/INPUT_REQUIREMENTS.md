# Input Requirements

Required input families:

- CHARLS Wave 1 data for development.
- CHARLS Wave 3 data for cross-wave evaluation.
- Stable participant identifier for ID isolation.
- Age and sex variables.
- Handgrip strength variables with sex-specific thresholds.
- Five-chair-stand time.
- Low-missing A-core predictor domains: demographic, residence, marital status, education, hukou, selected chronic disease indicators, smoking, and drinking.
- B anthropometric/body-size variables: height, weight, BMI, and waist circumference.

Outcome components:

- Male grip strength <28 kg.
- Female grip strength <18 kg.
- Five-chair-stand time >=12 seconds.

Primary label:

- Classifiable when at least one outcome component is observed.
- Positive when any observed component is positive.
- Negative when no observed positive component is present.

Strict-label sensitivity:

- Positive when any observed component is positive.
- Confirmed negative only when both grip and chair-stand components are observed and negative.
- One observed negative component plus the other missing is uncertain and excluded from strict-label sensitivity analysis.

Missing values:

- Scripts expect missing or invalid values to be coerced according to documented configuration rules.
- Complete-case model inputs are used for the highlighted main models.

No participant-level example rows are provided in this public release.

## Manuscript Focus

The final manuscript focuses on three methodological questions: whether non-isolated Wave 3 evaluation produced more favourable estimates than ID-isolated evaluation, whether XGBoost A+B materially improved on Logistic A-only after participant isolation, and whether accuracy at threshold 0.50 concealed inadequate sensitivity.
