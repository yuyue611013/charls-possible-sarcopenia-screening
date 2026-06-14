"""Batch 3 supplementary sensitivity summary tables from existing outputs.

Safety notes:
- Reads existing summary outputs only.
- Does not rerun models or estimate broader eligible-sample performance.
- Does not modify existing outputs, scripts, configs, models, or raw data.
- Writes only under output/submission_enhancement_v1/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = ROOT / ".python_packages"
if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import pandas as pd


OUT_ROOT = ROOT / "output" / "submission_enhancement_v1"
TABLE_DIR = OUT_ROOT / "tables"
LOG_DIR = OUT_ROOT / "logs"


def require_output_path(path: Path) -> None:
    resolved = path.resolve()
    allowed = OUT_ROOT.resolve()
    if not str(resolved).startswith(str(allowed) + "/") and resolved != allowed:
        raise RuntimeError(f"Unsafe output path outside Batch 3 folder: {path}")


def write_text(path: Path, text: str) -> None:
    require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_jsonish(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def threshold_sensitivity() -> None:
    summary_path = ROOT / "output/tables/23_threshold_summary_main_models_id_isolated.csv"
    rec_path = ROOT / "output/tables/23_threshold_recommendation_main_models_id_isolated.csv"
    frames = []
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary["threshold_type"] = summary["scenario"].fillna("threshold_grid_summary")
        frames.append(summary)
    if rec_path.exists():
        rec = pd.read_csv(rec_path)
        rec["scenario"] = rec.get("recommendation_target", "recommended_screening_threshold")
        rec["threshold_type"] = "recommended_screening_threshold"
        frames.append(rec)
    if not frames:
        write_text(LOG_DIR / "threshold_sensitivity_skipped.md", "# Threshold Sensitivity Skipped\n\nRequired threshold outputs were missing.\n")
        return
    df = pd.concat(frames, ignore_index=True, sort=False)
    rename = {
        "model_name": "model",
        "false_positive_n": "false_positives",
        "false_negative_n": "false_negatives",
        "ppv": "PPV",
        "npv": "NPV",
    }
    df = df.rename(columns=rename)
    keep = [
        "model",
        "scenario",
        "threshold_type",
        "threshold",
        "sensitivity",
        "specificity",
        "accuracy",
        "false_positives",
        "false_negatives",
        "PPV",
        "NPV",
        "n",
        "positive_n",
        "negative_n",
    ]
    keep = [col for col in keep if col in df.columns]
    write_csv(TABLE_DIR / "SuppTable_threshold_sensitivity_id_isolated.csv", df[keep])


def complete_case_summary() -> None:
    rows: list[dict[str, Any]] = []
    log_path = ROOT / "output/logs/21b_prepare_model_input_id_isolated_summary.txt"
    if log_path.exists():
        payload = read_jsonish(log_path)
        for item in payload.get("summaries", []):
            eligible = int(item["eligible_with_label_n"])
            complete = int(item["complete_case_n"])
            rows.append(
                {
                    "wave": "wave1" if item["dataset_name"].startswith("wave1") else "wave3",
                    "model_path": item["dataset_name"].replace("wave1_", "").replace("wave3_", ""),
                    "validation_context": "id_isolated_main_pipeline",
                    "label_eligible_sample_size": eligible,
                    "complete_case_sample_size": complete,
                    "sample_retained": complete,
                    "sample_lost": eligible - complete,
                    "retention_percentage": complete / eligible * 100 if eligible else None,
                    "predictor_count": item.get("predictor_count"),
                    "notes": "main ID-isolated model-input summary; no broader eligible-sample performance estimated",
                }
            )
    matrix_path = ROOT / "output/tables/12b_model_matrix_sample_size_summary.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
        for _, item in matrix.iterrows():
            eligible = int(item["eligible_with_label_n"])
            complete = int(item["complete_case_n"])
            rows.append(
                {
                    "wave": item["wave"],
                    "model_path": item["matrix_name"],
                    "validation_context": "original_non_isolated_formal_matrix_context",
                    "label_eligible_sample_size": eligible,
                    "complete_case_sample_size": complete,
                    "sample_retained": complete,
                    "sample_lost": eligible - complete,
                    "retention_percentage": complete / eligible * 100 if eligible else None,
                    "predictor_count": item.get("predictor_count"),
                    "notes": "formal matrix sample-size context; high-missingness paths show sample compression; no broader eligible-sample performance estimated",
                }
            )
    if not rows:
        write_text(LOG_DIR / "complete_case_summary_skipped.md", "# Complete-case Summary Skipped\n\nRequired sample-size outputs were missing.\n")
        return
    write_csv(TABLE_DIR / "SuppTable_complete_case_vs_eligible_samples.csv", pd.DataFrame(rows))


def original_vs_id_isolated() -> None:
    path = ROOT / "output/tables/21_compare_original_vs_id_isolated_main_models.csv"
    if not path.exists():
        write_text(LOG_DIR / "original_vs_id_isolated_skipped.md", "# Original vs ID-isolated Skipped\n\nRequired comparison output was missing.\n")
        return
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model": row["model_name"],
                "validation_setting": "original_non_isolated_wave3_with_participant_overlap",
                "N": "",
                "AUROC": row["original_wave3_auroc"],
                "AUPRC": row["original_wave3_auprc"],
                "F1": row["original_wave3_f1"],
                "accuracy": row["original_wave3_accuracy"],
                "sensitivity": row["original_wave3_sensitivity"],
                "specificity": row["original_wave3_specificity"],
                "note": "supplementary/contextual validation; contains participant overlap",
            }
        )
        rows.append(
            {
                "model": row["model_name"],
                "validation_setting": "id_isolated_wave3_overlap_removed",
                "N": row["id_isolated_validation_n"],
                "AUROC": row["id_isolated_wave3_auroc"],
                "AUPRC": row["id_isolated_wave3_auprc"],
                "F1": row["id_isolated_wave3_f1"],
                "accuracy": row["id_isolated_wave3_accuracy"],
                "sensitivity": row["id_isolated_wave3_sensitivity"],
                "specificity": row["id_isolated_wave3_specificity"],
                "note": "primary stricter validation; overlapping IDs removed",
            }
        )
    write_csv(TABLE_DIR / "SuppTable_original_vs_id_isolated_validation.csv", pd.DataFrame(rows))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    threshold_sensitivity()
    complete_case_summary()
    original_vs_id_isolated()
    write_text(
        LOG_DIR / "sensitivity_analysis_methods_note.md",
        "\n".join(
            [
                "# Sensitivity Analysis Methods Note",
                "",
                "- Sensitivity summaries are based on existing outputs.",
                "- No models were rerun.",
                "- No broader eligible-sample model performance was estimated.",
                "- Threshold sensitivity uses existing ID-isolated threshold grid/recommendation outputs.",
                "- Complete-case summaries compare label-eligible sample sizes with complete-case model-input sizes only.",
                "- Original non-isolated results are supplementary/contextual, not primary.",
                "- ID-isolated validation remains the primary manuscript evidence.",
            ]
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
