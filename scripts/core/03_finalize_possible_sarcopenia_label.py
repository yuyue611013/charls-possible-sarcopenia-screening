from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import pandas as pd

from utils_io import (
    as_numeric,
    column_exists,
    ensure_output_dirs,
    load_json,
    non_missing_mask,
    read_csv,
    write_dataframe_csv,
    write_text,
)


REQUIRED_RULE_VALUES = {
    "sex_code_male": 1,
    "sex_code_female": 0,
    "grip_primary_rule": "use_gripsum_as_primary_dominant_hand_max",
    "use_gait_speed_in_main_label": False,
}


def validate_configuration(columns_config: dict, rules_config: dict, df: pd.DataFrame) -> None:
    required_columns = [
        columns_config["id_col"],
        columns_config["wave_col"],
        columns_config["age_col"],
        columns_config["sex_col"],
        columns_config["grip_main_col"],
        columns_config["grip_left_col"],
        columns_config["grip_right_col"],
        columns_config["chair_time_col"],
        columns_config["chair_complete_col"],
        columns_config["chair_count_col"],
        columns_config["height_col"],
        columns_config["weight_col"],
        columns_config["bmi_col"],
        columns_config["waist_col"],
    ]
    missing_columns = [column for column in required_columns if not column_exists(df, column)]
    if missing_columns:
        raise ValueError(f"关键字段缺失，停止生成标签数据：{missing_columns}")

    for key, expected_value in REQUIRED_RULE_VALUES.items():
        actual_value = rules_config.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"规则冲突：{key} 当前为 {actual_value!r}，预期为 {expected_value!r}。"
            )

    if rules_config.get("grip_unit") != "kg":
        raise ValueError("规则冲突：grip_unit 未确认为 'kg'。")
    if rules_config.get("chair_time_unit") != "seconds":
        raise ValueError("规则冲突：chair_time_unit 未确认为 'seconds'。")


def build_output_columns(columns_config: dict) -> list[str]:
    return [
        columns_config["id_col"],
        columns_config["wave_col"],
        columns_config["age_col"],
        columns_config["sex_col"],
        columns_config["grip_main_col"],
        columns_config["grip_left_col"],
        columns_config["grip_right_col"],
        columns_config["chair_time_col"],
        columns_config["chair_complete_col"],
        columns_config["chair_count_col"],
        columns_config["height_col"],
        columns_config["weight_col"],
        columns_config["bmi_col"],
        columns_config["waist_col"],
        "possible_sarcopenia",
        "low_grip_flag",
        "poor_chair_flag",
        "label_eligible",
        "label_missing_reason",
    ]


def label_wave_subset(
    df: pd.DataFrame,
    wave_value: int,
    columns_config: dict,
    rules_config: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    wave_col = columns_config["wave_col"]
    age_col = columns_config["age_col"]
    sex_col = columns_config["sex_col"]
    grip_main_col = columns_config["grip_main_col"]
    chair_time_col = columns_config["chair_time_col"]

    wave_numeric = as_numeric(df[wave_col])
    age_numeric = as_numeric(df[age_col])
    sex_numeric = as_numeric(df[sex_col])
    grip_numeric = as_numeric(df[grip_main_col])
    chair_numeric = as_numeric(df[chair_time_col])

    subset = df.loc[wave_numeric == wave_value].copy()
    subset = subset.loc[as_numeric(subset[age_col]) >= rules_config["min_age"]].copy()
    subset = subset.loc[non_missing_mask(subset[sex_col])].copy()

    subset["sex_code_numeric"] = as_numeric(subset[sex_col])
    invalid_sex_mask = ~subset["sex_code_numeric"].isin(
        [rules_config["sex_code_male"], rules_config["sex_code_female"]]
    )
    if invalid_sex_mask.any():
        invalid_values = sorted(subset.loc[invalid_sex_mask, "sex_code_numeric"].dropna().unique().tolist())
        raise ValueError(
            f"wave {wave_value} 存在未定义的 sex 编码，停止生成标签数据：{invalid_values}"
        )

    subset["grip_main_numeric"] = as_numeric(subset[grip_main_col])
    subset["chair_time_numeric"] = as_numeric(subset[chair_time_col])

    subset["grip_available"] = subset["grip_main_numeric"].notna()
    subset["chair_available"] = subset["chair_time_numeric"].notna()
    subset["label_eligible"] = (subset["grip_available"] | subset["chair_available"]).astype(int)

    subset["label_missing_reason"] = ""
    subset.loc[subset["label_eligible"] == 0, "label_missing_reason"] = "missing_grip_and_chair"

    male_mask = subset["sex_code_numeric"] == rules_config["sex_code_male"]
    female_mask = subset["sex_code_numeric"] == rules_config["sex_code_female"]

    subset["low_grip_flag"] = pd.NA
    subset.loc[male_mask & subset["grip_available"], "low_grip_flag"] = (
        subset.loc[male_mask & subset["grip_available"], "grip_main_numeric"]
        < rules_config["male_grip_cutoff"]
    ).astype(int)
    subset.loc[female_mask & subset["grip_available"], "low_grip_flag"] = (
        subset.loc[female_mask & subset["grip_available"], "grip_main_numeric"]
        < rules_config["female_grip_cutoff"]
    ).astype(int)

    subset["poor_chair_flag"] = pd.NA
    subset.loc[subset["chair_available"], "poor_chair_flag"] = (
        subset.loc[subset["chair_available"], "chair_time_numeric"]
        >= rules_config["chair_stand_cutoff_sec"]
    ).astype(int)

    subset["possible_sarcopenia"] = pd.NA
    eligible_mask = subset["label_eligible"] == 1
    low_grip_eval = subset.loc[eligible_mask, "low_grip_flag"].fillna(0).astype(int)
    poor_chair_eval = subset.loc[eligible_mask, "poor_chair_flag"].fillna(0).astype(int)
    subset.loc[eligible_mask, "possible_sarcopenia"] = ((low_grip_eval == 1) | (poor_chair_eval == 1)).astype(int)

    summary = {
        "wave": wave_value,
        "final_cohort_n": int(len(subset)),
        "label_eligible_n": int(subset["label_eligible"].sum()),
        "label_ineligible_n": int((subset["label_eligible"] == 0).sum()),
        "possible_sarcopenia_positive_n": int(
            subset.loc[subset["possible_sarcopenia"] == 1, "possible_sarcopenia"].shape[0]
        ),
        "possible_sarcopenia_positive_rate_among_eligible": round(
            float((subset.loc[subset["possible_sarcopenia"] == 1, "possible_sarcopenia"].shape[0]))
            / max(int(subset["label_eligible"].sum()), 1),
            6,
        ),
        "gripsum_missing_n_in_final_cohort": int(subset["grip_main_numeric"].isna().sum()),
        "chr5sec_missing_n_in_final_cohort": int(subset["chair_time_numeric"].isna().sum()),
    }

    output_columns = build_output_columns(columns_config)
    output_df = subset[output_columns].copy()
    return output_df, summary


def main() -> None:
    ensure_output_dirs()

    columns_config = load_json("config/first_part_columns_template.json")
    rules_config = load_json("config/first_part_rules_template.json")
    df = read_csv(columns_config["input_csv"])

    validate_configuration(columns_config, rules_config, df)

    development_waves = rules_config["development_waves"]
    temporal_validation_waves = rules_config["temporal_validation_waves"]
    all_target_waves = development_waves + temporal_validation_waves

    if all_target_waves != [1, 3]:
        raise ValueError(
            f"当前阶段仅支持 wave 1 / wave 3 主分析标签生成，收到波次设置：{all_target_waves}"
        )

    wave_outputs: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    for wave_value in all_target_waves:
        output_df, summary = label_wave_subset(df, wave_value, columns_config, rules_config)
        wave_outputs[wave_value] = output_df
        summary_rows.append(summary)

    wave1_path = "output/tables/03_wave1_development_cohort_labeled.csv"
    wave3_path = "output/tables/03_wave3_temporal_validation_cohort_labeled.csv"
    summary_path = "output/tables/03_label_summary_by_wave.csv"
    log_path = "output/logs/03_finalize_possible_sarcopenia_label_summary.txt"

    write_dataframe_csv(wave_outputs[1], wave1_path)
    write_dataframe_csv(wave_outputs[3], wave3_path)

    summary_df = pd.DataFrame(summary_rows)
    write_dataframe_csv(summary_df, summary_path)

    lines = [
        "第一部分 03_finalize_possible_sarcopenia_label 摘要",
        "",
        f"输入文件: {columns_config['input_csv']}",
        f"输出开发集: {wave1_path}",
        f"输出时间验证集: {wave3_path}",
        f"输出汇总表: {summary_path}",
        "",
        "主标签规则:",
        f"- sex_code_male = {rules_config['sex_code_male']}",
        f"- sex_code_female = {rules_config['sex_code_female']}",
        f"- grip_primary_rule = {rules_config['grip_primary_rule']}",
        f"- male_grip_cutoff = {rules_config['male_grip_cutoff']}",
        f"- female_grip_cutoff = {rules_config['female_grip_cutoff']}",
        f"- chair_stand_cutoff_sec = {rules_config['chair_stand_cutoff_sec']}",
        f"- use_gait_speed_in_main_label = {rules_config['use_gait_speed_in_main_label']}",
        "",
        "各波次汇总:",
    ]

    for row in summary_rows:
        lines.extend(
            [
                f"- wave {row['wave']}:",
                f"  final_cohort_n = {row['final_cohort_n']}",
                f"  label_eligible_n = {row['label_eligible_n']}",
                f"  label_ineligible_n = {row['label_ineligible_n']}",
                f"  possible_sarcopenia_positive_n = {row['possible_sarcopenia_positive_n']}",
                "  possible_sarcopenia_positive_rate_among_eligible = "
                f"{row['possible_sarcopenia_positive_rate_among_eligible']}",
                f"  gripsum_missing_n_in_final_cohort = {row['gripsum_missing_n_in_final_cohort']}",
                f"  chr5sec_missing_n_in_final_cohort = {row['chr5sec_missing_n_in_final_cohort']}",
            ]
        )

    lines.extend(
        [
            "",
            "说明:",
            "- 本脚本只生成正式主分析标签数据集，不进行建模、插补或特征筛选。",
            "- 主标签未纳入 wspeed；步速继续仅保留为敏感性分析候选字段。",
            "- 原始数据未被修改。",
        ]
    )

    write_text(log_path, "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
