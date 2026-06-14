"""Batch 3 output checker.

Safety notes:
- Checks only Batch 3 output/script locations.
- Does not read raw data.
- Does not rerun models.
- Does not modify existing project outputs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output" / "submission_enhancement_v1"
SCRIPT_ROOT = ROOT / "scripts" / "submission_enhancement_v1"
LOG_DIR = OUT_ROOT / "logs"

EXPECTED_OUTPUTS = [
    OUT_ROOT / "README_outputs.md",
    OUT_ROOT / "tables/SuppTable_logistic_A_only_coefficients.csv",
    OUT_ROOT / "tables/SuppTable_feature_importance_xgboost_A_plus_B.csv",
    OUT_ROOT / "tables/SuppTable_feature_importance_group_summary.csv",
    OUT_ROOT / "figures/SuppFig_feature_importance_xgboost_A_plus_B.png",
    OUT_ROOT / "tables/SuppTable_decision_curve_id_isolated_wave3.csv",
    OUT_ROOT / "figures/SuppFig_decision_curve_id_isolated_wave3.png",
    OUT_ROOT / "tables/SuppTable_threshold_sensitivity_id_isolated.csv",
    OUT_ROOT / "tables/SuppTable_complete_case_vs_eligible_samples.csv",
    OUT_ROOT / "tables/SuppTable_original_vs_id_isolated_validation.csv",
    OUT_ROOT / "logs/input_discovery_log.md",
    OUT_ROOT / "logs/logistic_coefficients_methods_note.md",
    OUT_ROOT / "logs/xgboost_feature_importance_methods_note.md",
    OUT_ROOT / "logs/decision_curve_methods_note.md",
    OUT_ROOT / "logs/sensitivity_analysis_methods_note.md",
    OUT_ROOT / "logs/submission_enhancement_v1_run_manifest.md",
]

EXPECTED_SCRIPTS = [
    SCRIPT_ROOT / "01_extract_coefficients_and_importance_v1.py",
    SCRIPT_ROOT / "02_generate_dca_net_benefit_v1.py",
    SCRIPT_ROOT / "03_generate_sensitivity_summary_tables_v1.py",
    SCRIPT_ROOT / "04_check_submission_enhancement_outputs_v1.py",
]


def require_output_path(path: Path) -> None:
    resolved = path.resolve()
    allowed = OUT_ROOT.resolve()
    if not str(resolved).startswith(str(allowed) + "/") and resolved != allowed:
        raise RuntimeError(f"Unsafe checker output path outside Batch 3 folder: {path}")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Submission Enhancement v1 Output Check",
        "",
        f"Check time: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Output folder exists: {'yes' if OUT_ROOT.exists() else 'no'}",
        f"- Script folder exists: {'yes' if SCRIPT_ROOT.exists() else 'no'}",
        "",
        "## Expected outputs",
        "",
        "| File | Status |",
        "|---|---|",
    ]
    missing = []
    for path in EXPECTED_OUTPUTS:
        status = "exists" if path.exists() else "missing"
        if status == "missing":
            missing.append(path)
        lines.append(f"| `{path.relative_to(ROOT)}` | {status} |")

    lines.extend(["", "## Expected scripts", "", "| File | Status |", "|---|---|"])
    for path in EXPECTED_SCRIPTS:
        status = "exists" if path.exists() else "missing"
        if status == "missing":
            missing.append(path)
        lines.append(f"| `{path.relative_to(ROOT)}` | {status} |")

    shap_files = list(OUT_ROOT.rglob("*shap*")) + list(OUT_ROOT.rglob("*SHAP*"))
    lines.extend(
        [
            "",
            "## Safety checks",
            "",
            f"- SHAP output created: {'yes' if shap_files else 'no'}",
            "- Existing original output files modified: not assessed by file mtime; checker only verifies Batch 3 paths.",
            "- Batch 3 files are expected only under `output/submission_enhancement_v1/` and `scripts/submission_enhancement_v1/`.",
        ]
    )
    if shap_files:
        lines.append("- SHAP files found:")
        for path in shap_files:
            lines.append(f"  - `{path.relative_to(ROOT)}`")

    lines.extend(["", "## Missing or skipped outputs", ""])
    if missing:
        for path in missing:
            lines.append(f"- `{path.relative_to(ROOT)}` is missing and should be explained by a methods note if intentionally skipped.")
    else:
        lines.append("- None.")

    out = LOG_DIR / "submission_enhancement_v1_output_check.md"
    require_output_path(out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Batch 3 output check complete: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
