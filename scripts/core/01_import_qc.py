from __future__ import annotations

import sys
from pathlib import Path
from textwrap import indent

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


QC_PROMPT_RULES = {
    "age_col": {"low": 40, "high": 110},
    "grip_main_col": {"low": 0},
    "grip_left_col": {"low": 0},
    "grip_right_col": {"low": 0},
    "chair_time_col": {"low": 0},
    "gait_speed_col": {"low": 0},
    "height_col": {"low": 1.0, "high": 2.5},
    "weight_col": {"low": 20, "high": 250},
    "bmi_col": {"low": 10, "high": 80},
    "waist_col": {"low": 30, "high": 200},
}


def build_core_column_map(columns_config: dict) -> dict[str, str]:
    column_map = {
        key: value
        for key, value in columns_config.items()
        if key.endswith("_col") and isinstance(value, str)
    }
    for biomarker_name, column_name in columns_config.get("blood_biomarker_cols", {}).items():
        column_map[f"blood::{biomarker_name}"] = column_name
    return column_map


def summarize_wave_counts(df: pd.DataFrame, wave_col: str) -> pd.DataFrame:
    wave_counts = (
        df[wave_col]
        .value_counts(dropna=False)
        .rename_axis("wave_value")
        .reset_index(name="row_count")
        .sort_values(by="wave_value", na_position="last")
    )
    return wave_counts


def summarize_missingness(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for logical_name, actual_name in column_map.items():
        exists = actual_name in df.columns
        missing_count = None
        missing_rate = None
        if exists:
            mask = non_missing_mask(df[actual_name])
            missing_count = int((~mask).sum())
            missing_rate = round(missing_count / len(df), 6)
        rows.append(
            {
                "logical_name": logical_name,
                "column_name": actual_name,
                "exists": exists,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
            }
        )
    return pd.DataFrame(rows)


def summarize_numeric_series(series: pd.Series) -> dict[str, float | int | None]:
    numeric = as_numeric(series).dropna()
    if numeric.empty:
        return {
            "non_missing": 0,
            "min": None,
            "p01": None,
            "median": None,
            "p99": None,
            "max": None,
        }
    return {
        "non_missing": int(numeric.shape[0]),
        "min": float(numeric.min()),
        "p01": float(numeric.quantile(0.01)),
        "median": float(numeric.median()),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
    }


def build_extreme_value_prompts(df: pd.DataFrame, columns_config: dict) -> list[str]:
    prompts: list[str] = []
    for logical_name, thresholds in QC_PROMPT_RULES.items():
        column_name = columns_config.get(logical_name)
        if not column_exists(df, column_name):
            continue
        stats = summarize_numeric_series(df[column_name])
        if stats["non_missing"] == 0:
            prompts.append(f"- {logical_name} | {column_name}: 全部缺失，需人工确认字段是否可用。")
            continue

        reasons: list[str] = []
        low = thresholds.get("low")
        high = thresholds.get("high")
        minimum = stats["min"]
        maximum = stats["max"]
        if low is not None and minimum is not None and minimum < low:
            reasons.append(f"最小值 {minimum} < 参考下限 {low}")
        if high is not None and maximum is not None and maximum > high:
            reasons.append(f"最大值 {maximum} > 参考上限 {high}")

        if reasons:
            prompt = (
                f"- {logical_name} | {column_name}: "
                + "; ".join(reasons)
                + "。仅作 QC 提示，不代表自动排除规则。"
            )
            prompts.append(prompt)
    return prompts


def main() -> None:
    ensure_output_dirs()

    columns_config = load_json("config/first_part_columns_template.json")
    rules_config = load_json("config/first_part_rules_template.json")
    df = read_csv(columns_config["input_csv"])

    core_column_map = build_core_column_map(columns_config)
    missingness_df = summarize_missingness(df, core_column_map)
    write_dataframe_csv(missingness_df, "output/tables/01_variable_missingness.csv")

    wave_col = columns_config["wave_col"]
    wave_counts_df = summarize_wave_counts(df, wave_col)
    write_dataframe_csv(wave_counts_df, "output/tables/01_basic_wave_counts.csv")

    age_col = columns_config["age_col"]
    id_col = columns_config["id_col"]
    wave_numeric = as_numeric(df[wave_col]) if column_exists(df, wave_col) else pd.Series(dtype=float)
    age_numeric = as_numeric(df[age_col]) if column_exists(df, age_col) else pd.Series(dtype=float)

    development_waves = rules_config.get("development_waves", [])
    validation_waves = rules_config.get("temporal_validation_waves", [])
    min_age = rules_config.get("min_age")

    dev_count = int(((wave_numeric.isin(development_waves)) & (age_numeric >= min_age)).sum())
    val_count = int(((wave_numeric.isin(validation_waves)) & (age_numeric >= min_age)).sum())

    duplicate_id_wave_count = None
    if column_exists(df, id_col) and column_exists(df, wave_col):
        duplicate_id_wave_count = int(df.duplicated(subset=[id_col, wave_col]).sum())

    age_stats = summarize_numeric_series(df[age_col]) if column_exists(df, age_col) else {}
    extreme_prompts = build_extreme_value_prompts(df, columns_config)

    summary_lines = [
        "第一部分 01_import_qc 审计摘要",
        "",
        f"输入文件: {columns_config['input_csv']}",
        f"数据维度: {df.shape[0]} 行 x {df.shape[1]} 列",
        "",
        "关键字段存在性:",
    ]
    for logical_name, actual_name in core_column_map.items():
        status = "存在" if actual_name in df.columns else "缺失"
        summary_lines.append(f"- {logical_name}: {actual_name} | {status}")

    summary_lines.extend(
        [
            "",
            "年龄概览:",
            f"- age 非缺失: {age_stats.get('non_missing')}",
            f"- age 最小值: {age_stats.get('min')}",
            f"- age 中位数: {age_stats.get('median')}",
            f"- age P99: {age_stats.get('p99')}",
            f"- age 最大值: {age_stats.get('max')}",
            "",
            "第一部分目标样本预览:",
            f"- development_waves: {development_waves}",
            f"- temporal_validation_waves: {validation_waves}",
            f"- min_age: {min_age}",
            f"- 开发集 age >= {min_age} 样本量: {dev_count}",
            f"- 时间验证集 age >= {min_age} 样本量: {val_count}",
            "",
            f"ID + wave 重复记录数: {duplicate_id_wave_count}",
            "",
            "核心变量缺失率文件: output/tables/01_variable_missingness.csv",
            "波次分布文件: output/tables/01_basic_wave_counts.csv",
            "",
            "极端值 QC 提示:",
        ]
    )

    if extreme_prompts:
        summary_lines.extend(extreme_prompts)
    else:
        summary_lines.append("- 未发现基于当前启发式规则的极端值提示。")

    summary_lines.extend(
        [
            "",
            "字段待确认提醒:",
            indent(
                pd.Series(columns_config.get("field_confirmation", {}), dtype="object")
                .to_json(force_ascii=False, indent=2),
                prefix="  ",
            ),
            "",
            "说明:",
            "- 本脚本只做读取与审计，不修改原始数据。",
            "- 极端值提示仅用于后续人工 QC，不代表自动修正规则。",
        ]
    )

    write_text("output/logs/01_import_qc_summary.txt", "\n".join(summary_lines))
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
