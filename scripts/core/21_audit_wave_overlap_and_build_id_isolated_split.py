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


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_flow(df: pd.DataFrame, wave_label: str, label_col: str, eligibility_col: str) -> list[dict[str, object]]:
    eligible = df.loc[df[eligibility_col].eq(1) & df[label_col].notna()].copy()
    ineligible = df.loc[~(df[eligibility_col].eq(1) & df[label_col].notna())].copy()
    return [
        {"dataset": wave_label, "metric": "age_ge_60_total", "value": int(len(df))},
        {"dataset": wave_label, "metric": "label_eligible_n", "value": int(len(eligible))},
        {"dataset": wave_label, "metric": "label_ineligible_n", "value": int(len(ineligible))},
        {
            "dataset": wave_label,
            "metric": "possible_sarcopenia_positive_n",
            "value": int((eligible[label_col] == 1).sum()),
        },
        {
            "dataset": wave_label,
            "metric": "possible_sarcopenia_negative_n",
            "value": int((eligible[label_col] == 0).sum()),
        },
    ]


def main() -> None:
    columns = load_json("config/first_part_columns_template.json")
    spec = load_json("config/model_matrix_spec.json")
    id_col = spec["id_col"]
    wave_col = spec["wave_col"]
    label_col = spec["label_col"]
    eligibility_col = spec["eligibility_col"]

    raw_df = read_csv(columns["input_csv"])
    wave1_labeled = read_csv("output/tables/03_wave1_development_cohort_labeled.csv")
    wave3_labeled = read_csv("output/tables/03_wave3_temporal_validation_cohort_labeled.csv")
    wave1_full = read_csv("output/tables/05_wave1_development_full_analysis_base.csv")
    wave3_full = read_csv("output/tables/05_wave3_temporal_validation_full_analysis_base.csv")

    raw_wave1_ids = set(raw_df.loc[raw_df[wave_col].eq(1), id_col].dropna().astype(str))
    raw_wave3_ids = set(raw_df.loc[raw_df[wave_col].eq(3), id_col].dropna().astype(str))
    labeled_wave1_ids = set(wave1_labeled[id_col].dropna().astype(str))
    labeled_wave3_ids = set(wave3_labeled[id_col].dropna().astype(str))
    full_wave1_ids = set(wave1_full[id_col].dropna().astype(str))
    full_wave3_ids = set(wave3_full[id_col].dropna().astype(str))

    raw_overlap = raw_wave1_ids & raw_wave3_ids
    labeled_overlap = labeled_wave1_ids & labeled_wave3_ids
    full_overlap = full_wave1_ids & full_wave3_ids

    audit_rows = [
        {
            "scope": "raw_charls_wave1_wave3",
            "wave1_unique_id_n": len(raw_wave1_ids),
            "wave3_unique_id_n": len(raw_wave3_ids),
            "overlap_id_n": len(raw_overlap),
            "overlap_fraction_of_wave1": pct(len(raw_overlap), len(raw_wave1_ids)),
            "overlap_fraction_of_wave3": pct(len(raw_overlap), len(raw_wave3_ids)),
            "has_id_overlap": len(raw_overlap) > 0,
        },
        {
            "scope": "03_labeled_cohort",
            "wave1_unique_id_n": len(labeled_wave1_ids),
            "wave3_unique_id_n": len(labeled_wave3_ids),
            "overlap_id_n": len(labeled_overlap),
            "overlap_fraction_of_wave1": pct(len(labeled_overlap), len(labeled_wave1_ids)),
            "overlap_fraction_of_wave3": pct(len(labeled_overlap), len(labeled_wave3_ids)),
            "has_id_overlap": len(labeled_overlap) > 0,
        },
        {
            "scope": "05_full_analysis_base_current_main_data",
            "wave1_unique_id_n": len(full_wave1_ids),
            "wave3_unique_id_n": len(full_wave3_ids),
            "overlap_id_n": len(full_overlap),
            "overlap_fraction_of_wave1": pct(len(full_overlap), len(full_wave1_ids)),
            "overlap_fraction_of_wave3": pct(len(full_overlap), len(full_wave3_ids)),
            "has_id_overlap": len(full_overlap) > 0,
        },
    ]
    audit_df = pd.DataFrame(audit_rows)
    write_dataframe_csv(audit_df, "output/tables/21_wave_overlap_audit_summary.csv")

    wave1_isolated = wave1_full.copy()
    wave3_isolated = wave3_full.loc[~wave3_full[id_col].astype(str).isin(full_wave1_ids)].copy()
    isolated_overlap = set(wave1_isolated[id_col].dropna().astype(str)) & set(wave3_isolated[id_col].dropna().astype(str))

    if isolated_overlap:
        raise RuntimeError("id_isolated 构建失败：wave 1 与 wave 3 仍存在 ID 重叠。")

    write_dataframe_csv(wave1_isolated, "output/tables/21_wave1_development_full_analysis_base_id_isolated.csv")
    write_dataframe_csv(wave3_isolated, "output/tables/21_wave3_temporal_validation_full_analysis_base_id_isolated.csv")

    sample_flow_rows = []
    sample_flow_rows.extend(summarize_flow(wave1_isolated, "wave1_id_isolated", label_col, eligibility_col))
    sample_flow_rows.extend(summarize_flow(wave3_isolated, "wave3_id_isolated", label_col, eligibility_col))
    sample_flow_rows.extend(
        [
            {
                "dataset": "id_isolation",
                "metric": "wave1_row_loss_vs_original_full_base",
                "value": int(len(wave1_full) - len(wave1_isolated)),
            },
            {
                "dataset": "id_isolation",
                "metric": "wave3_row_loss_vs_original_full_base",
                "value": int(len(wave3_full) - len(wave3_isolated)),
            },
            {
                "dataset": "id_isolation",
                "metric": "remaining_overlap_id_n",
                "value": int(len(isolated_overlap)),
            },
        ]
    )
    sample_flow_df = pd.DataFrame(sample_flow_rows)
    write_dataframe_csv(sample_flow_df, "output/tables/21_id_isolated_sample_flow.csv")

    current_has_overlap = len(full_overlap) > 0
    validation_label = (
        "within-cohort across-wave validation with participant overlap"
        if current_has_overlap
        else "participant-independent temporal validation"
    )
    wave3_loss = len(wave3_full) - len(wave3_isolated)
    wave3_loss_fraction = pct(wave3_loss, len(wave3_full))

    notes = "\n".join(
        [
            "# 第 21 阶段：wave 1 / wave 3 ID 重叠审计与 ID 隔离说明",
            "",
            "## 审计结论",
            f"- 当前 05 full analysis base 中 wave 1 unique ID 数量：{len(full_wave1_ids)}。",
            f"- 当前 05 full analysis base 中 wave 3 unique ID 数量：{len(full_wave3_ids)}。",
            f"- wave 1 与 wave 3 交集 ID 数量：{len(full_overlap)}。",
            f"- 交集占 wave 1 比例：{pct(len(full_overlap), len(full_wave1_ids)):.4f}。",
            f"- 交集占 wave 3 比例：{pct(len(full_overlap), len(full_wave3_ids)):.4f}。",
            f"- 当前已用于主分析的数据中是否存在 ID 重叠：{'是' if current_has_overlap else '否'}。",
            f"- 因此，当前 non-isolated 版本更准确的表述为：{validation_label}。",
            "",
            "## ID 隔离构建规则",
            "- 保留 wave 1 作为开发来源。",
            "- 从 wave 3 中剔除所有出现在 wave 1 full analysis base 中的 ID。",
            "- 生成的新数据文件均使用 `id_isolated` 命名。",
            "",
            "## 隔离后样本量",
            f"- 新版 wave 1 样本量：{len(wave1_isolated)}。",
            f"- 新版 wave 3 样本量：{len(wave3_isolated)}。",
            f"- 新版 wave 1 / wave 3 是否完全 ID 不重叠：{'是' if len(isolated_overlap) == 0 else '否'}。",
            f"- wave 1 损失样本数：{len(wave1_full) - len(wave1_isolated)}。",
            f"- wave 3 损失样本数：{wave3_loss}，占原 wave 3 full base 的 {wave3_loss_fraction:.4f}。",
            "- 是否明显影响主文可行性：需结合 21b/21c/21d/21e 的主文双模型重跑结果判断。",
            "",
        ]
    ) + "\n"
    write_text("docs/21_wave_overlap_notes.md", notes)

    payload = {
        "audit": audit_rows,
        "current_main_analysis_has_id_overlap": current_has_overlap,
        "recommended_current_validation_label": validation_label,
        "id_isolated_outputs": {
            "wave1": "output/tables/21_wave1_development_full_analysis_base_id_isolated.csv",
            "wave3": "output/tables/21_wave3_temporal_validation_full_analysis_base_id_isolated.csv",
            "sample_flow": "output/tables/21_id_isolated_sample_flow.csv",
        },
        "id_isolated_row_counts": {
            "wave1": int(len(wave1_isolated)),
            "wave3": int(len(wave3_isolated)),
            "wave1_loss": int(len(wave1_full) - len(wave1_isolated)),
            "wave3_loss": int(wave3_loss),
            "remaining_overlap_id_n": int(len(isolated_overlap)),
        },
    }
    write_text("output/logs/21_wave_overlap_audit_summary.txt", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
