from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class ConfirmationState:
    ready_for_final_label: bool
    blockers: list[str]


def resolve_required_columns(columns_config: dict) -> dict[str, str]:
    return {
        "id_col": columns_config["id_col"],
        "wave_col": columns_config["wave_col"],
        "age_col": columns_config["age_col"],
        "sex_col": columns_config["sex_col"],
        "grip_main_col": columns_config["grip_main_col"],
        "chair_time_col": columns_config["chair_time_col"],
        "chair_complete_col": columns_config["chair_complete_col"],
        "chair_count_col": columns_config["chair_count_col"],
    }


def evaluate_confirmation_state(columns_config: dict, rules_config: dict) -> ConfirmationState:
    blockers: list[str] = []
    if rules_config.get("sex_code_male") is None:
        blockers.append("sex_code_male 未确认，不能安全判断男性低握力。")
    if rules_config.get("sex_code_female") is None:
        blockers.append("sex_code_female 未确认，不能安全判断女性低握力。")
    if not rules_config.get("grip_unit"):
        blockers.append("grip_unit 未确认，不能安全使用握力阈值。")
    if not rules_config.get("chair_time_unit"):
        blockers.append("chair_time_unit 未确认，不能安全使用 5 次起立阈值。")
    if not rules_config.get("grip_primary_rule"):
        blockers.append("grip_primary_rule 未确认，不能锁定主握力字段使用规则。")

    if rules_config.get("use_gait_speed_in_main_label") and not rules_config.get("gait_speed_unit"):
        blockers.append("主标签若包含步速，则 gait_speed_unit 必须先确认。")

    for field_name, info in columns_config.get("field_confirmation", {}).items():
        if info.get("status") == "needs_manual_confirmation":
            blockers.append(f"{field_name} 仍需人工确认：{info.get('reason')}")

    deduplicated_blockers = []
    seen = set()
    for blocker in blockers:
        if blocker not in seen:
            seen.add(blocker)
            deduplicated_blockers.append(blocker)
    return ConfirmationState(
        ready_for_final_label=len(deduplicated_blockers) == 0,
        blockers=deduplicated_blockers,
    )


def candidate_possible_sarcopenia(row: pd.Series, columns_config: dict, rules_config: dict) -> int | None:
    """
    这是主标签构建逻辑的占位实现。
    在 sex 编码、单位和握力主字段规则未完成人工确认前，不应被用于正式落盘。
    """
    sex_value = row[columns_config["sex_col"]]
    grip_value = pd.to_numeric(row[columns_config["grip_main_col"]], errors="coerce")
    chair_value = pd.to_numeric(row[columns_config["chair_time_col"]], errors="coerce")

    low_grip = False
    if pd.notna(grip_value):
        if sex_value == rules_config.get("sex_code_male"):
            low_grip = grip_value < rules_config["male_grip_cutoff"]
        elif sex_value == rules_config.get("sex_code_female"):
            low_grip = grip_value < rules_config["female_grip_cutoff"]

    poor_chair = pd.notna(chair_value) and chair_value >= rules_config["chair_stand_cutoff_sec"]
    if pd.isna(grip_value) and pd.isna(chair_value):
        return None
    return int(low_grip or poor_chair)


def main() -> None:
    ensure_output_dirs()

    columns_config = load_json("config/first_part_columns_template.json")
    rules_config = load_json("config/first_part_rules_template.json")
    df = read_csv(columns_config["input_csv"])

    required_columns = resolve_required_columns(columns_config)
    missing_columns = [column for column in required_columns.values() if not column_exists(df, column)]

    wave_col = columns_config["wave_col"]
    age_col = columns_config["age_col"]
    sex_col = columns_config["sex_col"]
    grip_col = columns_config["grip_main_col"]
    chair_col = columns_config["chair_time_col"]

    development_waves = rules_config["development_waves"]
    validation_waves = rules_config["temporal_validation_waves"]
    target_waves = development_waves + validation_waves
    min_age = rules_config["min_age"]

    wave_numeric = as_numeric(df[wave_col])
    age_numeric = as_numeric(df[age_col])

    selected_wave_mask = wave_numeric.isin(target_waves)
    age_mask = age_numeric >= min_age
    sex_available_mask = non_missing_mask(df[sex_col])
    grip_available_mask = non_missing_mask(df[grip_col])
    chair_available_mask = non_missing_mask(df[chair_col])
    label_component_available_mask = grip_available_mask | chair_available_mask

    prep_df = df.loc[selected_wave_mask].copy()
    prep_df["__age_ge_min__"] = age_mask.loc[selected_wave_mask]
    prep_df["__sex_available__"] = sex_available_mask.loc[selected_wave_mask]
    prep_df["__grip_available__"] = grip_available_mask.loc[selected_wave_mask]
    prep_df["__chair_available__"] = chair_available_mask.loc[selected_wave_mask]
    prep_df["__label_component_available__"] = label_component_available_mask.loc[selected_wave_mask]
    prep_df["__provisional_label_eligible__"] = (
        prep_df["__age_ge_min__"]
        & prep_df["__sex_available__"]
        & prep_df["__label_component_available__"]
    )

    cohort_rows: list[dict[str, object]] = []
    for wave_value in target_waves:
        wave_subset = prep_df.loc[as_numeric(prep_df[wave_col]) == wave_value]
        cohort_rows.extend(
            [
                {"wave": wave_value, "stage": "selected_wave_rows", "count": int(len(wave_subset))},
                {"wave": wave_value, "stage": f"age_ge_{min_age}", "count": int(wave_subset["__age_ge_min__"].sum())},
                {"wave": wave_value, "stage": "sex_available", "count": int(wave_subset["__sex_available__"].sum())},
                {
                    "wave": wave_value,
                    "stage": "grip_available",
                    "count": int(wave_subset["__grip_available__"].sum()),
                },
                {
                    "wave": wave_value,
                    "stage": "chair_available",
                    "count": int(wave_subset["__chair_available__"].sum()),
                },
                {
                    "wave": wave_value,
                    "stage": "label_component_available",
                    "count": int(wave_subset["__label_component_available__"].sum()),
                },
                {
                    "wave": wave_value,
                    "stage": "provisional_label_eligible",
                    "count": int(wave_subset["__provisional_label_eligible__"].sum()),
                },
            ]
        )

    cohort_counts_df = pd.DataFrame(cohort_rows)
    write_dataframe_csv(cohort_counts_df, "output/tables/02_cohort_counts.csv")

    confirmation_state = evaluate_confirmation_state(columns_config, rules_config)

    summary_lines = [
        "第一部分 02_define_cohort_label 准备摘要",
        "",
        f"输入文件: {columns_config['input_csv']}",
        f"目标开发波次: {development_waves}",
        f"目标时间验证波次: {validation_waves}",
        f"最小年龄阈值: {min_age}",
        "",
        "必要字段检查:",
    ]
    if missing_columns:
        summary_lines.extend([f"- 缺失字段: {column}" for column in missing_columns])
    else:
        summary_lines.append("- 关键字段均存在。")

    summary_lines.extend(
        [
            "",
            "cohort 准备统计已写入: output/tables/02_cohort_counts.csv",
            "",
            "possible_sarcopenia 主标签逻辑（待确认后才能正式启用）:",
            "- 男性握力 < 28 kg",
            "- 女性握力 < 18 kg",
            "- 或 5 次起立 >= 12 秒",
            f"- use_gait_speed_in_main_label = {rules_config['use_gait_speed_in_main_label']}",
            "",
            "当前可确认边界:",
            "- 已完成 wave 1 / wave 3 选择框架。",
            f"- 已完成 age >= {min_age} 选择框架。",
            "- 已完成标签必要字段可用性预检。",
            "- 已定义主标签函数占位实现，但未用于正式落盘。",
            "",
        ]
    )

    if confirmation_state.ready_for_final_label:
        summary_lines.extend(
            [
                "人工确认状态:",
                "- 所有关键参数均已确认，下一步可进入正式标签生成。",
                "- 但当前脚本仍按本阶段要求只输出 cohort preparation summary，不自动落盘最终标签数据。",
            ]
        )
    else:
        summary_lines.extend(
            [
                "人工确认状态:",
                "- 当前仍存在关键阻断项，脚本不会落盘最终标签数据。",
                "阻断项列表:",
            ]
        )
        summary_lines.extend([f"- {item}" for item in confirmation_state.blockers])

    summary_lines.extend(
        [
            "",
            "说明:",
            "- 本脚本不建模、不做特征选择。",
            "- 在关键参数未确认前，只输出 cohort preparation summary 与预备统计。",
        ]
    )

    write_text("output/logs/02_define_cohort_label_summary.txt", "\n".join(summary_lines))
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
