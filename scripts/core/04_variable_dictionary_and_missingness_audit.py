from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import pandas as pd

from utils_io import ensure_output_dirs, load_json, write_dataframe_csv, write_text


LABEL_COMPONENT_COLUMNS = {
    "gripsum (优势手的最大测量值)",
    "lgrip (左手的最大握力测试)",
    "rgrip (右手的最大握力测试)",
    "chr5sec (5次站立的时间)",
    "chr5comp (是否完成5次站立)",
    "chr5num (完成站立的次数)",
}

DERIVED_LABEL_COLUMNS = {
    "poor_chair_flag",
    "low_grip_flag",
    "possible_sarcopenia",
    "label_eligible",
    "label_missing_reason",
}

MAIN_MODEL_EXCLUSION_REASONS = {
    "gripsum (优势手的最大测量值)": "label leakage / label component",
    "lgrip (左手的最大握力测试)": "label leakage / label component",
    "rgrip (右手的最大握力测试)": "label leakage / label component",
    "chr5sec (5次站立的时间)": "label leakage / label component",
    "chr5comp (是否完成5次站立)": "label leakage / label component",
    "chr5num (完成站立的次数)": "label leakage / label component",
    "poor_chair_flag": "derived from label",
    "low_grip_flag": "derived from label",
    "possible_sarcopenia": "derived label",
    "label_eligible": "derived from label workflow",
    "label_missing_reason": "derived from label workflow",
}


def load_labeled_cohorts() -> tuple[pd.DataFrame, pd.DataFrame]:
    wave1 = pd.read_csv(PROJECT_ROOT / "output/tables/03_wave1_development_cohort_labeled.csv")
    wave3 = pd.read_csv(PROJECT_ROOT / "output/tables/03_wave3_temporal_validation_cohort_labeled.csv")
    return wave1, wave3


def infer_data_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    return "string"


def unit_for_variable(variable_name: str, rules_config: dict) -> str:
    if variable_name in {"gripsum (优势手的最大测量值)", "lgrip (左手的最大握力测试)", "rgrip (右手的最大握力测试)"}:
        return rules_config.get("grip_unit") or ""
    if variable_name == "chr5sec (5次站立的时间)":
        return rules_config.get("chair_time_unit") or ""
    if variable_name == "wspeed (步行速度测试-均值)":
        return rules_config.get("gait_speed_unit") or ""
    if variable_name == "mheight (身高/m)":
        return "m"
    if variable_name == "mweight (体重/kg)":
        return "kg"
    if variable_name == "mwaist (腰围/cm)":
        return "cm"
    if variable_name == "bmi (BMI)":
        return "kg/m^2"
    return ""


def role_for_variable(variable_name: str, columns_config: dict) -> str:
    if variable_name == columns_config["id_col"]:
        return "id"
    if variable_name == columns_config["wave_col"]:
        return "wave"
    if variable_name == columns_config["age_col"] or variable_name == columns_config["sex_col"]:
        return "demographic"
    if variable_name in {
        columns_config["height_col"],
        columns_config["weight_col"],
        columns_config["bmi_col"],
        columns_config["waist_col"],
    }:
        return "anthropometric"
    if variable_name in LABEL_COMPONENT_COLUMNS:
        return "label_component"
    if variable_name in DERIVED_LABEL_COLUMNS:
        return "derived_label"
    return "qc_only"


def module_group_for_variable(variable_name: str, columns_config: dict) -> str:
    role = role_for_variable(variable_name, columns_config)
    if role in {"id", "wave"}:
        return "dataset_key"
    if role == "demographic":
        return "demographic_social"
    if role == "anthropometric":
        return "anthropometric"
    if role == "label_component":
        return "label_component"
    if role == "derived_label":
        return "derived_label"
    return "qc_or_other"


def source_support_for_variable(variable_name: str, columns_config: dict) -> str:
    field_confirmation = columns_config.get("field_confirmation", {})

    if variable_name in {
        columns_config["id_col"],
        columns_config["wave_col"],
        columns_config["age_col"],
    }:
        return "mapped_support"

    if variable_name == columns_config["sex_col"]:
        return "mapped_support"

    if variable_name in {
        columns_config["grip_main_col"],
        columns_config["grip_left_col"],
        columns_config["grip_right_col"],
        columns_config["chair_time_col"],
    }:
        return "mapped_support"

    if variable_name in columns_config.get("blood_biomarker_cols", {}).values():
        return "mapped_support"

    if variable_name == "wspeed (步行速度测试-均值)":
        return "unresolved"

    status = field_confirmation.get(variable_name, {}).get("status")
    if status == "mapping_supported_confirmed":
        return "mapped_support"
    return "unresolved"


def human_label(variable_name: str) -> str:
    if "(" in variable_name and variable_name.endswith(")"):
        return variable_name.split("(", 1)[1][:-1]
    return variable_name


def build_variable_dictionary(
    wave1: pd.DataFrame,
    wave3: pd.DataFrame,
    columns_config: dict,
    rules_config: dict,
) -> pd.DataFrame:
    all_variables = sorted(set(wave1.columns).union(set(wave3.columns)))
    rows: list[dict[str, object]] = []

    for variable_name in all_variables:
        wave1_available = variable_name in wave1.columns
        wave3_available = variable_name in wave3.columns
        wave1_missing_pct = None
        wave3_missing_pct = None
        data_type = ""

        if wave1_available:
            wave1_missing_pct = round(float(wave1[variable_name].isna().mean()), 6)
            data_type = infer_data_type(wave1[variable_name])
        elif wave3_available:
            wave3_missing_pct = round(float(wave3[variable_name].isna().mean()), 6)
            data_type = infer_data_type(wave3[variable_name])

        if wave3_available and wave3_missing_pct is None:
            wave3_missing_pct = round(float(wave3[variable_name].isna().mean()), 6)
        if wave1_available and wave1_missing_pct is None:
            wave1_missing_pct = round(float(wave1[variable_name].isna().mean()), 6)

        include_in_main_model = variable_name not in MAIN_MODEL_EXCLUSION_REASONS and role_for_variable(
            variable_name, columns_config
        ) in {"demographic", "anthropometric"}

        exclude_reason = ""
        if variable_name in MAIN_MODEL_EXCLUSION_REASONS:
            exclude_reason = MAIN_MODEL_EXCLUSION_REASONS[variable_name]
        elif not include_in_main_model:
            exclude_reason = "not retained in current main model candidate pool"

        notes = ""
        if variable_name == "wspeed (步行速度测试-均值)":
            notes = "敏感性分析候选，暂不进入主标签，也未纳入当前主模型候选池。"
        elif variable_name in columns_config.get("blood_biomarker_cols", {}).values():
            notes = "当前 labeled cohort 未保留该字段；如后续回接主分析数据，应补充波次支持与缺失结构。"
        elif variable_name in {
            "mheight (身高/m)",
            "mweight (体重/kg)",
            "bmi (BMI)",
            "mwaist (腰围/cm)",
        }:
            notes = "仅做缺失与异常值记录，不自动清洗。"

        rows.append(
            {
                "variable_name": variable_name,
                "module_group": module_group_for_variable(variable_name, columns_config),
                "human_readable_label": human_label(variable_name),
                "variable_role": role_for_variable(variable_name, columns_config),
                "data_type": data_type,
                "unit": unit_for_variable(variable_name, rules_config),
                "source_support": source_support_for_variable(variable_name, columns_config),
                "wave1_available": wave1_available,
                "wave3_available": wave3_available,
                "missing_pct_wave1": wave1_missing_pct,
                "missing_pct_wave3": wave3_missing_pct,
                "include_in_main_model": include_in_main_model,
                "exclude_reason": exclude_reason,
                "notes": notes,
            }
        )

    return pd.DataFrame(rows).sort_values(by=["module_group", "variable_name"]).reset_index(drop=True)


def build_main_model_candidate_variables(variable_dictionary: pd.DataFrame) -> pd.DataFrame:
    candidates = variable_dictionary.loc[variable_dictionary["include_in_main_model"]].copy()
    return candidates.reset_index(drop=True)


def build_main_model_exclusion_list(variable_dictionary: pd.DataFrame) -> pd.DataFrame:
    excluded = variable_dictionary.loc[
        variable_dictionary["variable_name"].isin(MAIN_MODEL_EXCLUSION_REASONS.keys())
    ][["variable_name", "variable_role", "exclude_reason", "notes"]].copy()
    return excluded.sort_values(by="variable_name").reset_index(drop=True)


def missingness_table(df: pd.DataFrame, wave_label: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        rows.append(
            {
                "wave": wave_label,
                "variable_name": column,
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": round(float(df[column].isna().mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(by="variable_name").reset_index(drop=True)


def stratified_missingness(df: pd.DataFrame, stratify_col: str, table_name: str) -> pd.DataFrame:
    rows = []
    for stratum_value in sorted(df[stratify_col].dropna().unique().tolist()):
        subset = df.loc[df[stratify_col] == stratum_value]
        for column in df.columns:
            rows.append(
                {
                    "table_name": table_name,
                    "stratify_col": stratify_col,
                    "stratum_value": stratum_value,
                    "variable_name": column,
                    "n_in_stratum": int(len(subset)),
                    "missing_count": int(subset[column].isna().sum()),
                    "missing_pct": round(float(subset[column].isna().mean()), 6),
                }
            )
    return pd.DataFrame(rows)


def predictor_group_rows() -> list[dict[str, str]]:
    return [
        {"module_group": "demographic_social", "variable_name": "age (年龄)", "candidate_group": "人口学/社会学"},
        {"module_group": "demographic_social", "variable_name": "ragender (性别)", "candidate_group": "人口学/社会学"},
        {"module_group": "anthropometric", "variable_name": "mheight (身高/m)", "candidate_group": "体格指标"},
        {"module_group": "anthropometric", "variable_name": "mweight (体重/kg)", "candidate_group": "体格指标"},
        {"module_group": "anthropometric", "variable_name": "bmi (BMI)", "candidate_group": "体格指标"},
        {"module_group": "anthropometric", "variable_name": "mwaist (腰围/cm)", "candidate_group": "体格指标"},
        {"module_group": "health_behavior", "variable_name": "", "candidate_group": "健康行为"},
        {"module_group": "comorbidity", "variable_name": "", "candidate_group": "共病"},
        {"module_group": "biomarker", "variable_name": "", "candidate_group": "血液指标"},
    ]


def build_notes_doc(variable_dictionary: pd.DataFrame, candidate_variables: pd.DataFrame) -> str:
    lines = [
        "# 第一部分变量字典与缺失审计说明",
        "",
        "## 当前范围",
        "- 仅基于已生成的 wave 1 / wave 3 labeled cohort 进行整理和审计",
        "- 当前不建模、不插补、不做特征选择",
        "- 当前不修改原始数据",
        "",
        "## 主模型候选池边界",
        "- 当前 labeled cohort 中真正可直接进入主模型候选池的变量，主要是人口学与体格指标：",
    ]

    for variable_name in candidate_variables["variable_name"].tolist():
        lines.append(f"- {variable_name}")

    lines.extend(
        [
            "",
            "## 明确排除项",
            "- 握力、5次起立及其衍生标签变量当前不得进入第一部分主模型",
            "- 这些变量已在 `04_main_model_exclusion_list.csv` 中冻结",
            "",
            "## 关于 wspeed",
            "- `wspeed` 继续标记为敏感性分析候选，暂不进入主标签，也不进入当前主模型候选池",
            "",
            "## 关于血液指标",
            "- 当前这两个 labeled cohort 文件未保留血液指标列，因此本轮缺失审计无法对其在正式 labeled cohort 中逐列统计",
            "- 变量字典中对此已保留说明，后续若回接更完整主分析数据，再补充血液指标缺失结构",
            "",
            "## 异常值处理",
            "- 当前阶段只记录异常值风险，不自动清洗",
            "",
            "## 变量字典规模",
            f"- 当前变量字典共 {len(variable_dictionary)} 个字段",
            f"- 当前主模型候选池共 {len(candidate_variables)} 个字段",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ensure_output_dirs()

    columns_config = load_json("config/first_part_columns_template.json")
    rules_config = load_json("config/first_part_rules_template.json")
    wave1, wave3 = load_labeled_cohorts()

    variable_dictionary = build_variable_dictionary(wave1, wave3, columns_config, rules_config)
    candidate_variables = build_main_model_candidate_variables(variable_dictionary)
    exclusion_list = build_main_model_exclusion_list(variable_dictionary)

    wave1_missingness = missingness_table(wave1, "wave1")
    wave3_missingness = missingness_table(wave3, "wave3")

    by_eligibility = pd.concat(
        [
            stratified_missingness(wave1, "label_eligible", "wave1_by_label_eligible"),
            stratified_missingness(wave3, "label_eligible", "wave3_by_label_eligible"),
        ],
        ignore_index=True,
    )

    wave1_eligible = wave1.loc[wave1["label_eligible"] == 1].copy()
    wave3_eligible = wave3.loc[wave3["label_eligible"] == 1].copy()
    by_possible = pd.concat(
        [
            stratified_missingness(
                wave1_eligible, "possible_sarcopenia", "wave1_eligible_by_possible_sarcopenia"
            ),
            stratified_missingness(
                wave3_eligible, "possible_sarcopenia", "wave3_eligible_by_possible_sarcopenia"
            ),
        ],
        ignore_index=True,
    )

    write_dataframe_csv(variable_dictionary, "output/tables/04_variable_dictionary.csv")
    write_dataframe_csv(candidate_variables, "output/tables/04_main_model_candidate_variables.csv")
    write_dataframe_csv(exclusion_list, "output/tables/04_main_model_exclusion_list.csv")
    write_dataframe_csv(wave1_missingness, "output/tables/04_missingness_wave1.csv")
    write_dataframe_csv(wave3_missingness, "output/tables/04_missingness_wave3.csv")
    write_dataframe_csv(by_eligibility, "output/tables/04_missingness_by_label_eligibility.csv")
    write_dataframe_csv(by_possible, "output/tables/04_missingness_by_possible_sarcopenia.csv")

    notes_text = build_notes_doc(variable_dictionary, candidate_variables)
    write_text("docs/04_variable_dictionary_and_missingness_notes.md", notes_text)

    unresolved_count = int((variable_dictionary["source_support"] == "unresolved").sum())
    summary_lines = [
        "第一部分 04_variable_dictionary_and_missingness_audit 摘要",
        "",
        f"wave1 labeled cohort 行数: {len(wave1)}",
        f"wave3 labeled cohort 行数: {len(wave3)}",
        f"变量字典字段数: {len(variable_dictionary)}",
        f"主模型候选变量数: {len(candidate_variables)}",
        f"主模型排除变量数: {len(exclusion_list)}",
        f"source_support = unresolved 的字段数: {unresolved_count}",
        "",
        "主模型候选池（当前可直接进入者）:",
    ]

    for variable_name in candidate_variables["variable_name"].tolist():
        summary_lines.append(f"- {variable_name}")

    summary_lines.extend(
        [
            "",
            "主要说明:",
            "- 当前 labeled cohort 只保留了标签生成所需关键字段与体格指标，因此健康行为、共病、血液指标未能在本轮 labeled cohort 中逐列展开缺失审计。",
            "- 标签构成变量、标签衍生变量已被冻结为主模型排除项。",
            "- wspeed 继续保留为敏感性分析候选，不进入主标签，也不进入当前主模型候选池。",
        ]
    )

    write_text("output/logs/04_variable_dictionary_and_missingness_audit_summary.txt", "\n".join(summary_lines))
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
