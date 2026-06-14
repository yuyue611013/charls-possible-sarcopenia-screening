"""Generate main-text Table 2 for id-isolated model performance.

The table reports only the two main-text models:
- Low-missingness baseline path: logistic_A_only_id_isolated.
- Enhanced path with anthropometric variables: xgboost_A_plus_B_id_isolated.

Original non-isolated across-wave results are read only for audit context and
are not included in the main table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

LOGISTIC_CV = ROOT / "output/tables/21c_cv_metrics_logistic_A_only_id_isolated.csv"
XGB_CV = ROOT / "output/tables/21d_cv_metrics_xgboost_A_plus_B_id_isolated.csv"
VALIDATION = ROOT / "output/tables/21e_temporal_validation_metrics_id_isolated.csv"
ORIGINAL_COMPARISON = ROOT / "output/tables/21_compare_original_vs_id_isolated_main_models.csv"

TABLE_OUT = ROOT / "output/tables/Table2_main_model_performance_id_isolated.csv"
FOOTNOTE_OUT = ROOT / "docs/Table2_title_and_footnote_cn.md"
LOG_OUT = ROOT / "output/logs/Table2_generation_summary.txt"


MODEL_LABELS = {
    "logistic_A_only_id_isolated": {
        "display": "Logistic baseline path",
        "description": "Low-missingness baseline path",
        "short_name": "Logistic A-only",
    },
    "xgboost_A_plus_B_id_isolated": {
        "display": "XGBoost enhanced path",
        "description": "Enhanced path with anthropometric variables",
        "short_name": "XGBoost A+B",
    },
}


def fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def get_cv_mean(df: pd.DataFrame, model_name: str) -> pd.Series:
    row = df[df["fold"].astype(str).str.lower().eq("mean")]
    if row.empty:
        raise ValueError(f"Missing CV mean row for {model_name}")
    return row.iloc[0]


def get_validation_row(df: pd.DataFrame, model_name: str) -> pd.Series:
    row = df[df["model_name"] == model_name]
    if row.empty:
        raise ValueError(f"Missing validation row for {model_name}")
    return row.iloc[0]


def main() -> None:
    logistic_cv = read_required(LOGISTIC_CV)
    xgb_cv = read_required(XGB_CV)
    validation = read_required(VALIDATION)
    original_comparison = read_required(ORIGINAL_COMPARISON)

    specs = [
        ("logistic_A_only_id_isolated", logistic_cv),
        ("xgboost_A_plus_B_id_isolated", xgb_cv),
    ]

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for model_name, cv_df in specs:
        labels = MODEL_LABELS[model_name]
        cv = get_cv_mean(cv_df, model_name)
        val = get_validation_row(validation, model_name)
        rows.append(
            {
                "model": labels["display"],
                "model_description": labels["description"],
                "development_wave1_auroc": fmt(cv["auroc"]),
                "development_wave1_auprc": fmt(cv["auprc"]),
                "development_wave1_f1": fmt(cv["f1"]),
                "id_isolated_wave3_auroc": fmt(val["auroc"]),
                "id_isolated_wave3_auprc": fmt(val["auprc"]),
                "id_isolated_wave3_f1": fmt(val["f1"]),
                "id_isolated_wave3_accuracy": fmt(val["accuracy"]),
                "id_isolated_wave3_sensitivity": fmt(val["sensitivity"]),
                "id_isolated_wave3_specificity": fmt(val["specificity"]),
                "development_cv_source": str((LOGISTIC_CV if model_name.startswith("logistic") else XGB_CV).relative_to(ROOT)),
                "validation_source": str(VALIDATION.relative_to(ROOT)),
            }
        )
        comparison_name = model_name.replace("_id_isolated", "")
        comp = original_comparison[original_comparison["model_name"] == comparison_name]
        if not comp.empty:
            audit_rows.append(comp.iloc[0].to_dict())

    table = pd.DataFrame(rows)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_OUT, index=False)

    logistic = table.iloc[0]
    xgb = table.iloc[1]
    footnote = f"""# Table 2 表题与表注建议

**中文表题：** Table 2. 两个主文模型在 wave 1 开发集与 ID 隔离后 wave 3 验证集中的表现

**表注：** 开发集指标来自 wave 1 的 5-fold stratified cross-validation 平均值；更严格跨波次验证指标来自剔除所有 wave 1 重叠 ID 后的 id-isolated wave 3 验证集。本表仅报告当前主文两个模型：low-missingness baseline path（Logistic A-only）和 enhanced path with anthropometric variables（XGBoost A+B）。Original non-isolated across-wave 结果未纳入主文表格，可作为补充或方法学对照结果报告。

**正文总结建议：**
1. 在 wave 1 开发集交叉验证中，Logistic baseline path 的 AUROC/AUPRC/F1 为 {logistic['development_wave1_auroc']}/{logistic['development_wave1_auprc']}/{logistic['development_wave1_f1']}；XGBoost enhanced path 分别为 {xgb['development_wave1_auroc']}/{xgb['development_wave1_auprc']}/{xgb['development_wave1_f1']}。
2. 在 ID 隔离后的 wave 3 验证集中，Logistic baseline path 的 AUROC/AUPRC/F1 为 {logistic['id_isolated_wave3_auroc']}/{logistic['id_isolated_wave3_auprc']}/{logistic['id_isolated_wave3_f1']}；XGBoost enhanced path 分别为 {xgb['id_isolated_wave3_auroc']}/{xgb['id_isolated_wave3_auprc']}/{xgb['id_isolated_wave3_f1']}。
3. 与 original non-isolated 版本相比，id-isolated 验证更严格，主文表格应优先呈现本表结果。
"""
    FOOTNOTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    FOOTNOTE_OUT.write_text(footnote, encoding="utf-8")

    log_payload = {
        "stage": "Table 2 main model performance id-isolated generation",
        "inputs": {
            "logistic_cv": str(LOGISTIC_CV.relative_to(ROOT)),
            "xgboost_cv": str(XGB_CV.relative_to(ROOT)),
            "id_isolated_validation": str(VALIDATION.relative_to(ROOT)),
            "original_vs_id_isolated_comparison_audit_only": str(ORIGINAL_COMPARISON.relative_to(ROOT)),
        },
        "outputs": {
            "table2": str(TABLE_OUT.relative_to(ROOT)),
            "title_and_footnote": str(FOOTNOTE_OUT.relative_to(ROOT)),
        },
        "main_table_rows": table.to_dict(orient="records"),
        "original_non_isolated_not_included": True,
        "audit_original_vs_id_isolated_rows": audit_rows,
        "notes": [
            "No model retraining was performed.",
            "Original non-isolated performance metrics were not included in Table 2.",
            "All reported validation metrics come from id-isolated wave 3.",
        ],
    }
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {TABLE_OUT.relative_to(ROOT)}")
    print(f"Wrote {FOOTNOTE_OUT.relative_to(ROOT)}")
    print(f"Wrote {LOG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
