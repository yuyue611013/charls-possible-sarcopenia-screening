from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import pandas as pd

from utils_io import load_json, read_csv, write_dataframe_csv, write_text


def build_model_input(
    df: pd.DataFrame,
    predictors: list[str],
    dataset_name: str,
    id_col: str,
    wave_col: str,
    label_col: str,
    eligibility_col: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required_cols = [id_col, wave_col, label_col, eligibility_col] + predictors
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"{dataset_name} 缺少必要字段: {missing_cols}")

    eligible = df.loc[df[eligibility_col].eq(1) & df[label_col].notna(), required_cols].copy()
    complete = eligible.loc[eligible[predictors].notna().all(axis=1), [id_col, wave_col, label_col] + predictors].copy()
    summary = {
        "dataset_name": dataset_name,
        "eligible_with_label_n": int(len(eligible)),
        "complete_case_n": int(len(complete)),
        "dropped_due_to_predictor_missingness": int(len(eligible) - len(complete)),
        "positive_n": int((complete[label_col] == 1).sum()),
        "negative_n": int((complete[label_col] == 0).sum()),
        "predictor_count": len(predictors),
        "predictors": " | ".join(predictors),
    }
    return complete, summary


def main() -> None:
    spec = load_json("config/model_matrix_spec.json")
    id_col = spec["id_col"]
    wave_col = spec["wave_col"]
    label_col = spec["label_col"]
    eligibility_col = spec["eligibility_col"]
    a_predictors = spec["matrix_sets"]["A_only"]
    ab_predictors = spec["matrix_sets"]["A_only"] + spec["matrix_sets"]["B_only"]

    wave1 = read_csv("output/tables/21_wave1_development_full_analysis_base_id_isolated.csv")
    wave3 = read_csv("output/tables/21_wave3_temporal_validation_full_analysis_base_id_isolated.csv")

    input_specs = [
        ("wave1_A_only_id_isolated", wave1, a_predictors, "output/tables/21b_wave1_model_input_A_only_id_isolated.csv"),
        ("wave3_A_only_id_isolated", wave3, a_predictors, "output/tables/21b_wave3_model_input_A_only_id_isolated.csv"),
        (
            "wave1_A_plus_B_id_isolated",
            wave1,
            ab_predictors,
            "output/tables/21b_wave1_model_input_A_plus_B_id_isolated.csv",
        ),
        (
            "wave3_A_plus_B_id_isolated",
            wave3,
            ab_predictors,
            "output/tables/21b_wave3_model_input_A_plus_B_id_isolated.csv",
        ),
    ]

    summaries = []
    for dataset_name, df, predictors, output_path in input_specs:
        model_df, summary = build_model_input(df, predictors, dataset_name, id_col, wave_col, label_col, eligibility_col)
        write_dataframe_csv(model_df, output_path)
        summary["output_path"] = output_path
        summaries.append(summary)

    wave1_ids = set(read_csv("output/tables/21b_wave1_model_input_A_only_id_isolated.csv")[id_col].astype(str))
    wave3_ids = set(read_csv("output/tables/21b_wave3_model_input_A_only_id_isolated.csv")[id_col].astype(str))
    overlap = wave1_ids & wave3_ids
    if overlap:
        raise RuntimeError(f"21b A-only 输入仍存在 ID 重叠: {len(overlap)}")

    write_text(
        "output/logs/21b_prepare_model_input_id_isolated_summary.txt",
        json.dumps(
            {
                "stage": "21b_prepare_model_input_id_isolated",
                "complete_case_only": True,
                "remaining_overlap_in_A_only_inputs": int(len(overlap)),
                "summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
