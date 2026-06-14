# Public Release Audit

Overall status: **PASS**

| Status | Check | Critical |
|---|---|---:|
| PASS | Checker root has an expected release name | True |
| PASS | No .git directory | True |
| PASS | Required file exists: README.md | True |
| PASS | Required file exists: LICENSE | True |
| PASS | Required file exists: CITATION.cff | True |
| PASS | Required file exists: CHANGELOG.md | True |
| PASS | Required file exists: .gitignore | True |
| PASS | Required file exists: requirements.txt | True |
| PASS | Required file exists: environment.yml | True |
| PASS | Required file exists: RELEASE_MANIFEST.csv | True |
| PASS | Required file exists: config/path_config.example.json | True |
| PASS | Required file exists: docs/DATA_ACCESS.md | True |
| PASS | Required file exists: docs/REPRODUCIBILITY.md | True |
| PASS | Required file exists: docs/WORKFLOW_ORDER.md | True |
| PASS | Required file exists: docs/MANUSCRIPT_OUTPUT_MAP.md | True |
| PASS | Required file exists: docs/INPUT_REQUIREMENTS.md | True |
| PASS | Required file exists: docs/OUTPUT_DICTIONARY.md | True |
| PASS | Required file exists: docs/PRIVACY_AND_SECURITY.md | True |
| PASS | Required file exists: docs/PUBLIC_RELEASE_AUDIT.md | True |
| PASS | Required file exists: data/README.md | True |
| PASS | Final v2 workflow script present: scripts/sensitivity/01_strict_label_sensitivity_v2.py | True |
| PASS | Final v2 workflow script present: scripts/sensitivity/02_generate_discrimination_calibration_v2.py | True |
| PASS | Final v2 workflow script present: scripts/sensitivity/03_generate_paired_model_comparison_v2.py | True |
| PASS | Final v2 workflow script present: scripts/sensitivity/04_generate_subgroup_robustness_v2.py | True |
| PASS | Final v2 workflow script present: scripts/sensitivity/05_generate_calibration_uncertainty_v2.py | True |
| PASS | Final v2 workflow script present: scripts/sensitivity/06_generate_submission_figures_tables_v2.py | True |
| PASS | Final v2 workflow script present: scripts/sensitivity/08_check_submission_enhancement_v2.py | True |
| PASS | Final v2 workflow script present: scripts/sensitivity/09_check_v9_final_consistency.py | True |
| PASS | README documents CHARLS data restrictions | True |
| PASS | README avoids overstating external validation | True |
| PASS | README includes confirmed authors | True |
| PASS | README includes confirmed corresponding-author email | True |
| PASS | README includes current manuscript title | True |
| PASS | README excludes superseded complete manuscript title | True |
| PASS | CITATION excludes superseded complete manuscript title | True |
| PASS | MIT license exists | True |
| PASS | CITATION.cff exists and has expected repository title | True |
| PASS | v2 helper `_utils_v2.py` is present | True |
| PASS | Relative helper import resolvable for scripts/sensitivity/04_generate_subgroup_robustness_v2.py | True |
| PASS | Relative helper import resolvable for scripts/sensitivity/02_generate_discrimination_calibration_v2.py | True |
| PASS | Relative helper import resolvable for scripts/sensitivity/06_generate_submission_figures_tables_v2.py | True |
| PASS | Relative helper import resolvable for scripts/sensitivity/03_generate_paired_model_comparison_v2.py | True |
| PASS | Relative helper import resolvable for scripts/sensitivity/08_check_submission_enhancement_v2.py | True |
| PASS | Relative helper import resolvable for scripts/sensitivity/05_generate_calibration_uncertainty_v2.py | True |
| PASS | Manifest includes every release file; missing=[] | True |
| PASS | Manifest has no nonexistent files; extra=[] | True |
| PASS | Manifest marks included files public-safe; unsafe=[] | True |
| PASS | CHECKSUMS.sha256 covers every release file except itself | True |
| PASS | Checksum digests match current files; bad=[] | True |
| PASS | Figure 2 mapped to v2 script | True |
| PASS | Figure 4 mapped to v2 script | True |
| PASS | Figure 1 mapped to final grayscale workflow | True |
