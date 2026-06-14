from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import pandas as pd

from utils_io import ensure_output_dirs, load_json, read_csv, write_dataframe_csv, write_text


CANDIDATE_VARIABLES_BY_GROUP = {
    "demographic_social": [
        "age (年龄)",
        "ragender (性别)",
        "hrural (居住在农村或城市)",
        "marry (婚姻)",
        "raeduc_c (教育)",
        "hukou (户口4分类)",
    ],
    "anthropometric": [
        "mheight (身高/m)",
        "mweight (体重/kg)",
        "bmi (BMI)",
        "mwaist (腰围/cm)",
    ],
    "health_behavior": [
        "smoken (现在是否吸烟)",
        "drinkl (现在是否饮酒)",
        "vgact_c (重度身体活动)",
        "mdact_c (中度身体活动)",
        "ltact_c (轻度身体活动)",
    ],
    "comorbidity": [
        "hibpe (高血压)",
        "diabe (糖尿病)",
        "hearte (心脏病)",
        "stroke (中风)",
        "arthre (关节炎)",
    ],
    "biomarker": [
        "bl_glu (Glucose (mg/dl)血糖)",
        "bl_hbalc (Glycated Hemoglobin糖化血红蛋白 (%))",
        "bl_crp (C-Reactive Protein (CRP) C反应蛋白(mg/l))",
        "bl_crea (Creatinine (mg/dl)肌酐)",
        "bl_bun (Blood Urea Nitrogen (BUN) (mg/dl)尿素氮)",
        "bl_ua (Uric Acid尿酸(mg/dl))",
        "bl_hgb (Hemoglobin血红蛋白 (g/dl))",
        "bl_cysc (Cystatin C胱抑素C(mg/l))",
        "bl_cho (Total Cholesterol (mg/dl)总胆固醇)",
        "bl_tg (Triglycerides甘油三酯 (mg/dl))",
        "bl_hdl (Hdl Cholesterol高密度脂蛋白胆固醇 (mg/dl))",
        "bl_ldl (Ldl Cholesterol低密度脂蛋白胆固醇 (mg/dl))",
    ],
}


def candidate_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for module_group, variables in CANDIDATE_VARIABLES_BY_GROUP.items():
        for variable_name in variables:
            rows.append({"module_group": module_group, "variable_name": variable_name})
    return rows


def check_unique_key(df: pd.DataFrame, key_cols: list[str], dataset_name: str) -> None:
    duplicate_count = int(df.duplicated(subset=key_cols).sum())
    if duplicate_count > 0:
        raise ValueError(f"{dataset_name} 中 {key_cols} 存在重复主键，重复数为 {duplicate_count}。")


def build_join_status(raw_df: pd.DataFrame, exclusion_df: pd.DataFrame) -> pd.DataFrame:
    excluded_variables = set(exclusion_df["variable_name"].tolist())
    rows = []
    for item in candidate_rows():
        variable_name = item["variable_name"]
        found_in_raw = variable_name in raw_df.columns
        notes = ""
        if variable_name in excluded_variables:
            notes = "该字段在 04 排除清单中，当前仅为状态记录，不进入主模型候选池。"
        elif not found_in_raw:
            notes = "raw data 中未找到该字段。"
        else:
            notes = "字段存在于 raw data，可参与 full analysis base 回接。"

        rows.append(
            {
                "variable_name": variable_name,
                "found_in_raw_data": found_in_raw,
                "joined_successfully": False,
                "module_group": item["module_group"],
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def merge_candidate_variables(
    labeled_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    key_cols: list[str],
    candidate_columns: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    before_rows = len(labeled_df)
    join_columns = [column for column in candidate_columns if column not in labeled_df.columns]
    raw_subset = raw_df[key_cols + join_columns].copy()
    merged = labeled_df.merge(raw_subset, on=key_cols, how="left", validate="one_to_one")
    after_rows = len(merged)
    if before_rows != after_rows:
        raise ValueError(
            f"{dataset_name} 回接前后行数不一致：before={before_rows}, after={after_rows}"
        )
    check_unique_key(merged, key_cols, f"{dataset_name} 回接结果")
    return merged


def variable_health_notes(df: pd.DataFrame, variables: list[str]) -> tuple[list[str], list[str]]:
    not_found = []
    all_missing_or_near = []
    for variable_name in variables:
        if variable_name not in df.columns:
            not_found.append(variable_name)
            continue
        missing_pct = float(df[variable_name].isna().mean())
        if missing_pct == 1.0:
            all_missing_or_near.append(f"{variable_name} | all_missing")
        elif missing_pct >= 0.95:
            all_missing_or_near.append(f"{variable_name} | missing_pct={missing_pct:.3f}")
    return not_found, all_missing_or_near


def summarize_group_join_status(join_status_df: pd.DataFrame) -> list[str]:
    lines = []
    grouped = join_status_df.groupby("module_group", dropna=False)
    for module_group, subset in grouped:
        found_n = int(subset["found_in_raw_data"].sum())
        joined_n = int(subset["joined_successfully"].sum())
        total_n = int(len(subset))
        lines.append(f"- {module_group}: found={found_n}/{total_n}, joined={joined_n}/{total_n}")
    return lines


def main() -> None:
    ensure_output_dirs()

    columns_config = load_json("config/first_part_columns_template.json")
    _rules_config = load_json("config/first_part_rules_template.json")
    exclusion_df = pd.read_csv(PROJECT_ROOT / "output/tables/04_main_model_exclusion_list.csv")

    raw_df = read_csv(columns_config["input_csv"])
    wave1_labeled = pd.read_csv(PROJECT_ROOT / "output/tables/03_wave1_development_cohort_labeled.csv")
    wave3_labeled = pd.read_csv(PROJECT_ROOT / "output/tables/03_wave3_temporal_validation_cohort_labeled.csv")

    id_col = columns_config["id_col"]
    wave_col = columns_config["wave_col"]
    key_cols = [id_col, wave_col]

    check_unique_key(raw_df, key_cols, "raw_df")
    check_unique_key(wave1_labeled, key_cols, "wave1_labeled")
    check_unique_key(wave3_labeled, key_cols, "wave3_labeled")

    join_status_df = build_join_status(raw_df, exclusion_df)
    candidate_columns = join_status_df.loc[join_status_df["found_in_raw_data"], "variable_name"].tolist()

    wave1_full = merge_candidate_variables(
        labeled_df=wave1_labeled,
        raw_df=raw_df,
        key_cols=key_cols,
        candidate_columns=candidate_columns,
        dataset_name="wave1_full_analysis_base",
    )
    wave3_full = merge_candidate_variables(
        labeled_df=wave3_labeled,
        raw_df=raw_df,
        key_cols=key_cols,
        candidate_columns=candidate_columns,
        dataset_name="wave3_full_analysis_base",
    )

    join_status_df["joined_successfully"] = join_status_df["found_in_raw_data"]

    wave1_not_found, wave1_all_missing = variable_health_notes(wave1_full, candidate_columns)
    wave3_not_found, wave3_all_missing = variable_health_notes(wave3_full, candidate_columns)

    write_dataframe_csv(
        wave1_full,
        "output/tables/05_wave1_development_full_analysis_base.csv",
    )
    write_dataframe_csv(
        wave3_full,
        "output/tables/05_wave3_temporal_validation_full_analysis_base.csv",
    )
    write_dataframe_csv(
        join_status_df,
        "output/tables/05_variable_join_status.csv",
    )

    summary_lines = [
        "第一部分 05_build_full_analysis_base 摘要",
        "",
        f"raw data 行数: {len(raw_df)}",
        f"wave1 labeled cohort 行数: {len(wave1_labeled)}",
        f"wave3 labeled cohort 行数: {len(wave3_labeled)}",
        f"wave1 full analysis base 行数: {len(wave1_full)}",
        f"wave3 full analysis base 行数: {len(wave3_full)}",
        "",
        "主键检查:",
        "- raw_df: ID + wave 唯一",
        "- wave1 labeled cohort: ID + wave 唯一",
        "- wave3 labeled cohort: ID + wave 唯一",
        "- 回接后 wave1/wave3 行数未改变，且 ID + wave 仍唯一",
        "",
        "各模块回接状态:",
    ]
    summary_lines.extend(summarize_group_join_status(join_status_df))

    not_found_overall = join_status_df.loc[~join_status_df["found_in_raw_data"], "variable_name"].tolist()
    summary_lines.extend(
        [
            "",
            "raw data 中未找到的候选变量:",
        ]
    )
    if not_found_overall:
        summary_lines.extend([f"- {variable_name}" for variable_name in not_found_overall])
    else:
        summary_lines.append("- 无")

    summary_lines.extend(
        [
            "",
            "join 后全部缺失或几乎全缺失（wave1）:",
        ]
    )
    if wave1_all_missing:
        summary_lines.extend([f"- {item}" for item in wave1_all_missing])
    else:
        summary_lines.append("- 无")

    summary_lines.extend(
        [
            "",
            "join 后全部缺失或几乎全缺失（wave3）:",
        ]
    )
    if wave3_all_missing:
        summary_lines.extend([f"- {item}" for item in wave3_all_missing])
    else:
        summary_lines.append("- 无")

    summary_lines.extend(
        [
            "",
            "说明:",
            "- 当前只构建 full analysis base，不建模、不插补、不做特征选择。",
            "- 标签构成变量与衍生标签变量不会因此自动进入主模型候选池。",
            "- wspeed 未纳入本轮回接清单，继续只保留为敏感性分析候选。",
        ]
    )

    write_text(
        "output/logs/05_build_full_analysis_base_summary.txt",
        "\n".join(summary_lines),
    )
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
