"""Generate main-text Table 1 for wave 1 and ID-isolated wave 3 cohorts.

The table compares baseline characteristics for:
- Wave 1 development cohort, age >=60 overall full analysis base.
- ID-isolated wave 3 validation cohort, age >=60 overall full analysis base.

This script writes new Table 1 assets only. It does not overwrite source data,
existing Table 1 preparations, models, or manuscripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

WAVE1_BASE = ROOT / "output/tables/05_wave1_development_full_analysis_base.csv"
WAVE1_READABLE_REFERENCE = ROOT / "output/tables/16_5_table1_wave1_readable.csv"
WAVE3_READABLE_OLD = ROOT / "output/tables/16_5_table1_wave3_readable.csv"
ID_ISOLATED_FLOW = ROOT / "output/tables/21_id_isolated_sample_flow.csv"
WAVE3_ID_ISOLATED_BASE = ROOT / "output/tables/21_wave3_temporal_validation_full_analysis_base_id_isolated.csv"

TABLE_OUT = ROOT / "output/tables/Table1_baseline_characteristics_id_isolated.csv"
FOOTNOTE_OUT = ROOT / "docs/Table1_title_and_footnote_cn.md"
LOG_OUT = ROOT / "output/logs/Table1_generation_summary.txt"


CONTINUOUS = ["age", "mheight", "mweight", "bmi", "mwaist"]
CATEGORICAL = [
    "ragender",
    "hrural",
    "marry",
    "raeduc_c",
    "hukou",
    "hibpe",
    "diabe",
    "hearte",
    "stroke",
    "arthre",
    "smoken",
    "drinkl",
]

VARIABLE_ORDER = [
    "age",
    "ragender",
    "hrural",
    "marry",
    "raeduc_c",
    "hukou",
    "hibpe",
    "diabe",
    "hearte",
    "stroke",
    "arthre",
    "smoken",
    "drinkl",
    "mheight",
    "mweight",
    "bmi",
    "mwaist",
]

LABELS = {
    "age": "Age, years",
    "ragender": "Sex",
    "hrural": "Residence",
    "marry": "Marital status",
    "raeduc_c": "Education",
    "hukou": "Hukou status",
    "hibpe": "Hypertension",
    "diabe": "Diabetes",
    "hearte": "Heart disease",
    "stroke": "Stroke",
    "arthre": "Arthritis",
    "smoken": "Current smoking",
    "drinkl": "Current drinking",
    "mheight": "Height, m",
    "mweight": "Weight, kg",
    "bmi": "BMI, kg/m2",
    "mwaist": "Waist circumference, cm",
}

CATEGORY_LABELS = {
    "ragender": {0: "Female", 1: "Male"},
    "hrural": {0: "Urban", 1: "Rural"},
    "marry": {
        1: "Married",
        2: "Married but living apart",
        4: "Separated",
        5: "Divorced",
        7: "Widowed",
        8: "Never married",
    },
    "raeduc_c": {
        1: "Less than primary school",
        2: "Primary school",
        3: "Middle school",
        4: "High school or above",
    },
    "hukou": {
        1: "Agricultural hukou",
        2: "Non-agricultural hukou",
        3: "Unified residence hukou",
        4: "No hukou",
    },
    "hibpe": {0: "No", 1: "Yes"},
    "diabe": {0: "No", 1: "Yes"},
    "hearte": {0: "No", 1: "Yes"},
    "stroke": {0: "No", 1: "Yes"},
    "arthre": {0: "No", 1: "Yes"},
    "smoken": {0: "No", 1: "Yes"},
    "drinkl": {0: "No", 1: "Yes"},
}


def find_col(df: pd.DataFrame, stem: str) -> str:
    exact = [c for c in df.columns if c == stem]
    if exact:
        return exact[0]
    matches = [c for c in df.columns if c.startswith(f"{stem} ") or c.startswith(f"{stem}(")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous matches for {stem}: {matches}")
    raise KeyError(f"Column for {stem} not found")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fmt_cont(series: pd.Series) -> tuple[str, dict[str, Any]]:
    x = clean_numeric(series).dropna()
    if x.empty:
        return "NA", {"n": 0, "missing": int(series.shape[0])}
    mean = x.mean()
    sd = x.std(ddof=1)
    median = x.median()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    cell = f"{mean:.2f} ± {sd:.2f}; {median:.2f} [{q1:.2f}, {q3:.2f}]"
    return cell, {
        "n": int(x.shape[0]),
        "missing": int(series.shape[0] - x.shape[0]),
        "mean": mean,
        "sd": sd,
        "median": median,
        "q1": q1,
        "q3": q3,
    }


def normalize_code(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fmt_cat(series: pd.Series, code: int) -> tuple[str, dict[str, Any]]:
    x = clean_numeric(series)
    nonmissing = x.dropna()
    denom = int(nonmissing.shape[0])
    count = int((nonmissing == code).sum())
    pct = 100 * count / denom if denom else np.nan
    cell = f"{count:,} ({pct:.1f}%)" if denom else "NA"
    return cell, {"count": count, "denominator": denom, "missing": int(series.shape[0] - denom), "pct": pct}


def sample_flow_lookup(flow: pd.DataFrame, dataset: str, metric: str) -> int:
    row = flow[(flow["dataset"] == dataset) & (flow["metric"] == metric)]
    if row.empty:
        raise ValueError(f"Missing sample flow metric: {dataset}/{metric}")
    return int(row["value"].iloc[0])


def main() -> None:
    for path in [
        WAVE1_BASE,
        WAVE1_READABLE_REFERENCE,
        WAVE3_READABLE_OLD,
        ID_ISOLATED_FLOW,
        WAVE3_ID_ISOLATED_BASE,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    wave1 = pd.read_csv(WAVE1_BASE)
    wave3_iso = pd.read_csv(WAVE3_ID_ISOLATED_BASE)
    flow = pd.read_csv(ID_ISOLATED_FLOW)

    n_wave1 = int(wave1.shape[0])
    n_wave3 = int(wave3_iso.shape[0])
    expected_w1 = sample_flow_lookup(flow, "wave1_id_isolated", "age_ge_60_total")
    expected_w3 = sample_flow_lookup(flow, "wave3_id_isolated", "age_ge_60_total")
    if n_wave1 != expected_w1:
        raise ValueError(f"Wave 1 row count {n_wave1} does not match sample flow {expected_w1}")
    if n_wave3 != expected_w3:
        raise ValueError(f"ID-isolated wave 3 row count {n_wave3} does not match sample flow {expected_w3}")

    col_map_w1 = {v: find_col(wave1, v) for v in VARIABLE_ORDER}
    col_map_w3 = {v: find_col(wave3_iso, v) for v in VARIABLE_ORDER}

    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "continuous": {},
        "categorical": {},
        "column_map_wave1": col_map_w1,
        "column_map_wave3_id_isolated": col_map_w3,
    }

    for variable in VARIABLE_ORDER:
        label = LABELS[variable]
        if variable in CONTINUOUS:
            w1_cell, w1_stats = fmt_cont(wave1[col_map_w1[variable]])
            w3_cell, w3_stats = fmt_cont(wave3_iso[col_map_w3[variable]])
            rows.append(
                {
                    "variable_name": variable,
                    "variable_label": label,
                    "category": "",
                    "wave1_development_cohort_n_7560": w1_cell,
                    "id_isolated_wave3_validation_cohort_n_4063": w3_cell,
                    "summary_format": "mean ± SD; median [Q1, Q3]",
                    "notes": "Continuous variable; percentages not applicable.",
                }
            )
            audit["continuous"][variable] = {"wave1": w1_stats, "wave3_id_isolated": w3_stats}
            continue

        mapping = CATEGORY_LABELS[variable]
        audit["categorical"][variable] = {}
        for code, cat_label in mapping.items():
            w1_cell, w1_stats = fmt_cat(wave1[col_map_w1[variable]], code)
            w3_cell, w3_stats = fmt_cat(wave3_iso[col_map_w3[variable]], code)
            rows.append(
                {
                    "variable_name": variable,
                    "variable_label": label,
                    "category": cat_label,
                    "wave1_development_cohort_n_7560": w1_cell,
                    "id_isolated_wave3_validation_cohort_n_4063": w3_cell,
                    "summary_format": "n (%)",
                    "notes": "Categorical percentage calculated among non-missing observations for the variable.",
                }
            )
            audit["categorical"][variable][code] = {"wave1": w1_stats, "wave3_id_isolated": w3_stats}

    table = pd.DataFrame(rows)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    FOOTNOTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_OUT, index=False)

    bmi_w1 = audit["continuous"]["bmi"]["wave1"]
    bmi_w3 = audit["continuous"]["bmi"]["wave3_id_isolated"]
    age_w1 = audit["continuous"]["age"]["wave1"]
    age_w3 = audit["continuous"]["age"]["wave3_id_isolated"]

    footnote = f"""# Table 1 表题、表注与正文总结建议

**中文表题：** Table 1. Wave 1 开发队列与 ID 隔离后 wave 3 验证队列的基线特征

**表注：** 表中连续变量以 mean ± SD；median [Q1, Q3] 表示，分类变量以 n (%) 表示。开发队列分母为 wave 1 年龄 >=60 岁 full analysis base（n = {n_wave1:,}）；验证队列分母为剔除所有 wave 1 重叠 ID 后的 id-isolated wave 3 年龄 >=60 岁 full analysis base（n = {n_wave3:,}）。分类变量百分比按该变量非缺失样本计算；不同变量因缺失情况不同，实际分母可能略有差异。BMI 等体格指标受极端值影响时，正文描述建议优先报告 median [Q1, Q3]，mean ± SD 作为表格补充信息。

**正文总结建议：**
1. Wave 1 开发队列共纳入 {n_wave1:,} 例年龄 >=60 岁受试者，ID 隔离后的 wave 3 验证队列共纳入 {n_wave3:,} 例，二者无受试者 ID 重叠。
2. ID 隔离后的 wave 3 验证队列年龄分布更年轻：wave 1 中位年龄为 {age_w1['median']:.2f} [{age_w1['q1']:.2f}, {age_w1['q3']:.2f}] 岁，id-isolated wave 3 为 {age_w3['median']:.2f} [{age_w3['q1']:.2f}, {age_w3['q3']:.2f}] 岁；这一点在解释更严格验证结果时需要同步说明。
3. BMI 的均值和标准差受极端值影响，正文中建议重点呈现中位数和四分位数：wave 1 为 {bmi_w1['median']:.2f} [{bmi_w1['q1']:.2f}, {bmi_w1['q3']:.2f}]，id-isolated wave 3 为 {bmi_w3['median']:.2f} [{bmi_w3['q1']:.2f}, {bmi_w3['q3']:.2f}]。
"""
    FOOTNOTE_OUT.write_text(footnote, encoding="utf-8")

    log = {
        "stage": "Table 1 baseline characteristics id-isolated generation",
        "inputs": {
            "wave1_base": str(WAVE1_BASE.relative_to(ROOT)),
            "wave1_readable_reference": str(WAVE1_READABLE_REFERENCE.relative_to(ROOT)),
            "old_original_wave3_readable_not_used_for_validation_column": str(WAVE3_READABLE_OLD.relative_to(ROOT)),
            "id_isolated_sample_flow": str(ID_ISOLATED_FLOW.relative_to(ROOT)),
            "wave3_id_isolated_base": str(WAVE3_ID_ISOLATED_BASE.relative_to(ROOT)),
        },
        "outputs": {
            "table": str(TABLE_OUT.relative_to(ROOT)),
            "title_and_footnote": str(FOOTNOTE_OUT.relative_to(ROOT)),
        },
        "denominators": {
            "wave1_development_cohort": n_wave1,
            "id_isolated_wave3_validation_cohort": n_wave3,
        },
        "variables": VARIABLE_ORDER,
        "audit": audit,
        "notes": [
            "The old 16_5_table1_wave3_readable.csv is original non-isolated wave 3 and was not used for the validation column.",
            "No source data, manuscripts, models, or existing result files were modified.",
        ],
    }
    LOG_OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {TABLE_OUT.relative_to(ROOT)}")
    print(f"Wrote {FOOTNOTE_OUT.relative_to(ROOT)}")
    print(f"Wrote {LOG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
