# Code for participant-isolated evaluation of possible sarcopenia screening models in CHARLS

Version: 1.0.0

This repository contains code and public aggregate examples supporting the manuscript:

**Does cross-wave validation overestimate screening performance? An ID-isolated evaluation of model complexity for possible sarcopenia in CHARLS**

## Authors

- Yu Yue
- Chunhua Zhang

Corresponding author: Chunhua Zhang, zch20080808@126.com

## Study Summary

This repository supports a CHARLS-based study examining whether cross-wave screening-model evaluation can appear more favourable when model-development and later-wave evaluation samples include overlapping participants. The manuscript compares performance before and after removal of Wave 1 participant IDs from Wave 3 and evaluates two prespecified models: a lower-burden Logistic A-only model and an XGBoost A+B model combining A-core predictors with anthropometric/body-size variables.

The analysis emphasizes discrimination, AUPRC, calibration, threshold performance, decision-curve analysis, and robustness checks. It also highlights why acceptable-looking accuracy at the default 0.50 threshold can conceal low sensitivity in screening settings. Strict-label, subgroup, and calibration-uncertainty analyses are included as supplementary robustness evidence.

Participant overlap was examined as a key methodological concern, but performance changes after ID isolation should not be attributed to overlap alone because participant removal also changed case mix and event prevalence. The ID-isolated Wave 3 analysis is a stricter within-CHARLS participant-isolated cross-wave evaluation, not fully independent external validation.

## Study Design

- Wave 1 model development.
- Original non-isolated Wave 3 evaluation retained as contextual evidence.
- ID-isolated Wave 3 evaluation after removing participants appearing in Wave 1.
- Primary available-component possible-sarcopenia label.
- Strict-label sensitivity analysis excluding uncertain partial-negative classifications.
- Screening and risk-identification focus; this is not a confirmed sarcopenia diagnostic model.

## Main Models

- Logistic A-only: low-missing A-core predictors.
- XGBoost A+B: A-core predictors plus anthropometric/body-size variables.

## Repository Structure

- `config/`: non-confidential configuration templates and predictor-domain specifications.
- `scripts/core/`: data import, cohort, label, missingness, analysis-base, and ID-isolation workflow.
- `scripts/evaluation/`: main model development, model evaluation, calibration/table generation, and learning-curve scripts.
- `scripts/figures/`: final/historical figure workflows, including the final grayscale participant-flow workflow.
- `scripts/sensitivity/`: threshold analysis, DCA, feature importance, strict-label sensitivity, subgroup robustness, calibration uncertainty, and final v2 figure/table scripts.
- `scripts/checks/`: checker scripts.
- `docs/`: data access, reproducibility, workflow, output map, input requirements, output dictionary, privacy, and audit notes.
- `data/`: placeholder only; CHARLS data are not included.
- `example_outputs/`: aggregate example tables and publication figures only.
- `tools/`: public-release safety checker.

## Data Access Restrictions

CHARLS participant-level microdata are not included and cannot be redistributed here. Users must obtain authorised access from the official CHARLS data portal and comply with CHARLS data-use terms. Do not commit raw data, processed row-level datasets, model-input files, prediction files, or trained model objects to GitHub.

## Required Software

This release includes `requirements.txt` and `environment.yml` based on the current reproducibility environment. The exact original training environment was not fully recoverable, so these files should be treated as a current reproducibility snapshot rather than a historical guarantee.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda/mamba:

```bash
conda env create -f environment.yml
conda activate charls-sarcopenia-screening
```

## Configuration

Copy `config/path_config.example.json` to a local untracked configuration file and replace placeholder paths with authorised local CHARLS-derived input files. The data folder is ignored by Git except for `data/README.md`.

## Reproduction Workflow

See `docs/WORKFLOW_ORDER.md` for the recommended order. Full reproduction requires authorised CHARLS data and may require adapting import paths to the user's authorised local file layout.

## Expected Outputs

The workflow produces cohort summaries, model-input summaries, main model performance outputs, calibration and threshold outputs, decision-curve outputs, strict-label sensitivity tables, subgroup/calibration uncertainty summaries, and publication figures. Public aggregate examples are provided in `example_outputs/`.

## Included and Excluded Materials

Included:
- Analysis scripts and configuration templates.
- Aggregate example tables.
- Publication figure files.

Excluded:
- Raw CHARLS data.
- Processed participant-level datasets.
- Model-input tables with individual records.
- Row-level prediction files.
- Trained model objects.
- Manuscript drafts.

## Citation

Use `CITATION.cff` for software citation metadata. A repository URL and DOI should be added by the authors after public upload and any archive deposition.

## License

Code is released under the MIT License. CHARLS data remain governed by CHARLS data-use conditions and are not covered by this repository license.

## Contact

Chunhua Zhang, zch20080808@126.com

## Reproducibility Limitations

The included aggregate examples allow users to inspect final public outputs and verify the package structure. Recomputing participant-level analyses requires authorised CHARLS data and will not be possible from this repository alone.
