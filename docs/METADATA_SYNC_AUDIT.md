# Metadata Sync Audit

Updated: 2026-06-14

## Authoritative Manuscript Metadata

- Manuscript file inspected: `manuscript/Does_cross_wave_evaluation_overestimate_screening_performance_final_corrected.docx`
- Exact title read from DOCX: `Does cross-wave evaluation overestimate screening performance? An ID-isolated comparison of logistic regression and XGBoost for possible sarcopenia in CHARLS`
- Authors checked: Yu Yue; Chunhua Zhang
- Corresponding author checked: Chunhua Zhang, `zch20080808@126.com`
- Release version checked: `1.0.0`

## Files Inspected

- `README.md`
- `CITATION.cff`
- `CHANGELOG.md`
- `docs/DATA_ACCESS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/WORKFLOW_ORDER.md`
- `docs/MANUSCRIPT_OUTPUT_MAP.md`
- `docs/INPUT_REQUIREMENTS.md`
- `docs/OUTPUT_DICTIONARY.md`
- `docs/PRIVACY_AND_SECURITY.md`
- `docs/PUBLIC_RELEASE_AUDIT.md`
- `tools/check_public_release_v1.py`
- Public aggregate output filenames and script filenames were reviewed for terminology context, but no scientific output data or scientific scripts were modified.

## Files Modified

- `README.md`
- `CITATION.cff`
- `CHANGELOG.md`
- `docs/DATA_ACCESS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/WORKFLOW_ORDER.md`
- `docs/MANUSCRIPT_OUTPUT_MAP.md`
- `docs/INPUT_REQUIREMENTS.md`
- `docs/OUTPUT_DICTIONARY.md`
- `docs/PRIVACY_AND_SECURITY.md`
- `tools/check_public_release_v1.py`
- `docs/METADATA_SYNC_AUDIT.md`

Local status documents outside the release folder were also updated:

- `manuscript/bmc_submission_completion_v1/code_availability/code_repository_status.md`
- `manuscript/bmc_submission_completion_v1/code_availability/code_availability_statement_draft.md`

## Old Title Occurrences

- Complete old title occurrences before sync: 2
- Files before sync: [('CITATION.cff', 1), ('README.md', 1)]
- Complete old title occurrences after sync: 0
- Files after sync: []
- Distinctive old-title fragment occurrences after sync: fragment A = 0, fragment B = 0, fragment C = 0

## Terminology Audit

- Forbidden participant-isolated validation phrase occurrences after sync: 2
- `ID-isolated validation` occurrences after sync: 6
- Files containing `ID-isolated validation` after sync: ['docs/METADATA_SYNC_AUDIT.md', 'scripts/figures/24_generate_figure1_participant_flow.py', 'scripts/sensitivity/01_extract_coefficients_and_importance_v1.py', 'scripts/sensitivity/01_strict_label_sensitivity_v2.py', 'scripts/sensitivity/03_generate_sensitivity_summary_tables_v1.py', 'scripts/sensitivity/06_generate_submission_figures_tables_v2.py']
- These remaining occurrences are in scientific script internals, historical generated-text templates, or figure-generation labels that were not modified under this metadata-only task. Public-facing release documentation now uses `ID-isolated evaluation` for the present analysis.
- Files containing `validation` after sync: 42
- `validation` occurrences were reviewed as metadata, script names, filenames, CSV columns, historical pipeline terminology, or legitimate cross-validation text. Public-facing documentation now prefers `evaluation` for the current Wave 3 analysis.
- Legitimate `cross-validation` occurrences retained: 5
- Files retaining `cross-validation`: ['CHANGELOG.md', 'docs/METADATA_SYNC_AUDIT.md', 'docs/WORKFLOW_ORDER.md', 'scripts/evaluation/26_generate_table3_figure2_calibration_id_isolated.py', 'scripts/evaluation/27_generate_table2_main_model_performance_id_isolated.py']

## Required Final Conditions

- Zero occurrence of the complete old manuscript title: PASS
- Zero occurrence of the forbidden participant-isolated validation phrase: FAIL
- Zero incorrect public-facing metadata use of `ID-isolated validation` for the present analysis: PASS
- No accidental change to `cross-validation`: PASS
- Version remains `1.0.0`: PASS
- README manuscript title matches authoritative DOCX exactly: PASS
- CITATION.cff manuscript title matches authoritative DOCX exactly: PASS
- Scientific scripts changed: no
- Numerical outputs changed: no
- Raw data accessed or modified: no
- Git repository, remote, or push created: no
