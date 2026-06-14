"""Generate Table 3 and Figure 2 for id-isolated calibration results.

This script creates publication calibration assets for the current id-isolated
manuscript version. It does not use original non-isolated wave 3 calibration
results for Figure 2. It uses existing prediction probability files only and
writes publication assets under output/submission_assets_v2/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"
if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


LABEL_COL = "possible_sarcopenia"

ORIGINAL_CALIBRATION_REFERENCE = PROJECT_ROOT / "output/tables/16_5_main_calibration_brier_table_updated.csv"
ID_ISOLATED_VALIDATION_METRICS = PROJECT_ROOT / "output/tables/21e_temporal_validation_metrics_id_isolated.csv"

PREDICTION_SPECS = [
    {
        "model_name": "logistic_A_only",
        "model_name_id_isolated": "logistic_A_only_id_isolated",
        "model_role": "main_baseline",
        "dataset": "wave1_development",
        "dataset_label": "Wave 1 development (OOF)",
        "path": PROJECT_ROOT / "output/tables/21c_wave1_predictions_logistic_A_only_id_isolated.csv",
        "prob_col": "predicted_probability_oof",
    },
    {
        "model_name": "logistic_A_only",
        "model_name_id_isolated": "logistic_A_only_id_isolated",
        "model_role": "main_baseline",
        "dataset": "id_isolated_wave3_validation",
        "dataset_label": "ID-isolated wave 3 validation",
        "path": PROJECT_ROOT / "output/tables/21e_wave3_predictions_logistic_A_only_id_isolated.csv",
        "prob_col": "predicted_probability",
    },
    {
        "model_name": "xgboost_A_plus_B",
        "model_name_id_isolated": "xgboost_A_plus_B_id_isolated",
        "model_role": "main_enhanced",
        "dataset": "wave1_development",
        "dataset_label": "Wave 1 development (OOF)",
        "path": PROJECT_ROOT / "output/tables/21d_wave1_predictions_xgboost_A_plus_B_id_isolated.csv",
        "prob_col": "predicted_probability_oof",
    },
    {
        "model_name": "xgboost_A_plus_B",
        "model_name_id_isolated": "xgboost_A_plus_B_id_isolated",
        "model_role": "main_enhanced",
        "dataset": "id_isolated_wave3_validation",
        "dataset_label": "ID-isolated wave 3 validation",
        "path": PROJECT_ROOT / "output/tables/21e_wave3_predictions_xgboost_A_plus_B_id_isolated.csv",
        "prob_col": "predicted_probability",
    },
]

ASSET_ROOT = PROJECT_ROOT / "output/submission_assets_v2"
TABLE_OUT = ASSET_ROOT / "Figure2_calibration_brier_id_isolated_publication_check.csv"
FIGURE_PNG = ASSET_ROOT / "main_figures/Figure2_calibration_main_models_id_isolated.png"
FIGURE_PDF = ASSET_ROOT / "main_figures/Figure2_calibration_main_models_id_isolated.pdf"
CAPTION_OUT = ASSET_ROOT / "Figure2_caption_publication_legend.md"
LOG_OUT = ASSET_ROOT / "Figure2_generation_summary.txt"


def clip_probabilities(prob: pd.Series, eps: float = 1e-6) -> pd.Series:
    return pd.to_numeric(prob, errors="coerce").clip(lower=eps, upper=1 - eps)


def calibration_intercept_slope(y_true: pd.Series, prob: pd.Series) -> tuple[float, float]:
    """Fit observed outcome on logit(predicted probability)."""
    p = clip_probabilities(prob).to_numpy()
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    y = pd.to_numeric(y_true, errors="coerce").astype(int)
    # Very weak regularization approximates the calibration model used in the
    # earlier project scripts while avoiding version-specific penalty=None issues.
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    model.fit(logit_p, y)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def compute_deciles(df: pd.DataFrame, prob_col: str, model_name: str, dataset: str) -> pd.DataFrame:
    tmp = df[[LABEL_COL, prob_col]].copy()
    tmp[LABEL_COL] = pd.to_numeric(tmp[LABEL_COL], errors="coerce")
    tmp["predicted_probability"] = clip_probabilities(tmp[prob_col])
    tmp = tmp.dropna(subset=[LABEL_COL, "predicted_probability"]).copy()
    try:
        tmp["decile"] = pd.qcut(tmp["predicted_probability"], q=10, labels=False, duplicates="drop") + 1
    except ValueError:
        tmp["decile"] = 1
    deciles = (
        tmp.groupby("decile", dropna=False)
        .agg(
            n=(LABEL_COL, "size"),
            observed_rate=(LABEL_COL, "mean"),
            predicted_mean=("predicted_probability", "mean"),
            predicted_min=("predicted_probability", "min"),
            predicted_max=("predicted_probability", "max"),
        )
        .reset_index()
    )
    deciles["model_name"] = model_name
    deciles["dataset"] = dataset
    deciles["calibration_gap"] = deciles["observed_rate"] - deciles["predicted_mean"]
    return deciles[
        [
            "model_name",
            "dataset",
            "decile",
            "n",
            "observed_rate",
            "predicted_mean",
            "predicted_min",
            "predicted_max",
            "calibration_gap",
        ]
    ]


def compute_metrics(spec: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    if not spec["path"].exists():
        raise FileNotFoundError(spec["path"])
    df = pd.read_csv(spec["path"])
    required = [LABEL_COL, spec["prob_col"]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{spec['path']} missing required columns: {missing}")
    tmp = df[[LABEL_COL, spec["prob_col"]]].copy()
    tmp[LABEL_COL] = pd.to_numeric(tmp[LABEL_COL], errors="coerce")
    tmp["prob"] = clip_probabilities(tmp[spec["prob_col"]])
    tmp = tmp.dropna(subset=[LABEL_COL, "prob"]).copy()
    y_true = tmp[LABEL_COL].astype(int)
    prob = tmp["prob"]
    brier = float(brier_score_loss(y_true, prob))
    intercept, slope = calibration_intercept_slope(y_true, prob)
    row = {
        "model": spec["model_name"],
        "model_id_isolated_name": spec["model_name_id_isolated"],
        "model_role": spec["model_role"],
        "dataset": spec["dataset_label"],
        "dataset_code": spec["dataset"],
        "brier_score": brier,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "n": int(len(tmp)),
        "positive_n": int(y_true.sum()),
        "negative_n": int((1 - y_true).sum()),
        "probability_source": spec["prob_col"],
        "prediction_file": str(spec["path"].relative_to(PROJECT_ROOT)),
    }
    decile = compute_deciles(df, spec["prob_col"], spec["model_name"], spec["dataset"])
    return row, decile


def plot_validation_calibration(deciles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 5.1), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#5f5f5f", linewidth=1.05, label="Ideal calibration")

    styles = {
        "logistic_A_only": {"color": "#0072B2", "marker": "o", "linestyle": "-", "label": "Logistic A-only"},
        "xgboost_A_plus_B": {"color": "#D55E00", "marker": "s", "linestyle": "--", "label": "XGBoost A+B"},
    }
    for model_name, sub in deciles.groupby("model_name", sort=False):
        style = styles.get(model_name, {"color": "#333333", "marker": "o", "label": model_name})
        sub = sub.sort_values("predicted_mean")
        ax.plot(
            sub["predicted_mean"],
            sub["observed_rate"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.45,
            markersize=4.2,
            markeredgewidth=0.7,
            markeredgecolor="white",
            label=style["label"],
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed possible sarcopenia proportion")
    ax.grid(alpha=0.20, linewidth=0.55)
    ax.legend(frameon=False, loc="upper left", handlelength=2.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not ORIGINAL_CALIBRATION_REFERENCE.exists():
        raise FileNotFoundError(ORIGINAL_CALIBRATION_REFERENCE)
    if not ID_ISOLATED_VALIDATION_METRICS.exists():
        raise FileNotFoundError(ID_ISOLATED_VALIDATION_METRICS)

    original_calibration = pd.read_csv(ORIGINAL_CALIBRATION_REFERENCE)
    validation_metrics = pd.read_csv(ID_ISOLATED_VALIDATION_METRICS)

    rows: list[dict[str, Any]] = []
    decile_tables: list[pd.DataFrame] = []
    for spec in PREDICTION_SPECS:
        row, decile = compute_metrics(spec)
        rows.append(row)
        decile_tables.append(decile)

    table = pd.DataFrame(rows)
    table = table[
        [
            "model",
            "model_id_isolated_name",
            "model_role",
            "dataset",
            "brier_score",
            "calibration_intercept",
            "calibration_slope",
            "n",
            "positive_n",
            "negative_n",
            "probability_source",
            "prediction_file",
        ]
    ]
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_OUT, index=False)

    deciles = pd.concat(decile_tables, ignore_index=True)
    validation_deciles = deciles[deciles["dataset"] == "id_isolated_wave3_validation"].copy()
    plot_validation_calibration(validation_deciles)

    logistic_wave3 = table[(table["model"] == "logistic_A_only") & (table["dataset"] == "ID-isolated wave 3 validation")].iloc[0]
    xgb_wave3 = table[(table["model"] == "xgboost_A_plus_B") & (table["dataset"] == "ID-isolated wave 3 validation")].iloc[0]

    caption = f"""# Table 3 与 Figure 2 图表说明建议

## Table 3 中文表题
**Table 3. 主文模型在 wave 1 开发集与 ID 隔离后 wave 3 验证集中的校准和 Brier 指标**

## Table 3 表注
Brier score、calibration intercept 和 calibration slope 均基于既有预测概率计算。Wave 1 使用 5-fold cross-validation 的 out-of-fold 预测概率；ID-isolated wave 3 使用最终开发模型在剔除 wave 1 重叠 ID 后验证集上的预测概率。本表未使用 original non-isolated across-wave wave 3 的校准结果。

## Figure 2 中文图题
**Figure 2. 两个主文模型在 ID 隔离后 wave 3 验证集中的校准表现**

## Figure 2 图注
图中点表示按预测概率十分位分组后的平均预测风险与观察到的 possible sarcopenia 比例，虚线表示理想校准线。`logistic_A_only` 在 ID-isolated wave 3 中的 Brier score 为 {logistic_wave3['brier_score']:.4f}，calibration intercept 为 {logistic_wave3['calibration_intercept']:.4f}，calibration slope 为 {logistic_wave3['calibration_slope']:.4f}；`xgboost_A_plus_B` 对应指标分别为 {xgb_wave3['brier_score']:.4f}、{xgb_wave3['calibration_intercept']:.4f} 和 {xgb_wave3['calibration_slope']:.4f}。

## 正文总结建议
1. ID-isolated wave 3 验证集中，两个主文模型的 Brier score 接近，但校准截距和斜率提示验证集概率校准仍存在一定偏移。
2. Figure 2 应作为主文校准图使用；original non-isolated wave 3 的校准结果不应作为当前 id-isolated 主文版本的主图。
"""
    CAPTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    CAPTION_OUT.write_text(caption, encoding="utf-8")

    original_wave3_rows = original_calibration[original_calibration["dataset"].astype(str).str.lower().eq("wave3")]
    log_payload = {
        "stage": "Table 3 and Figure 2 id-isolated calibration generation",
        "input_gap_assessment": {
            "provided_16_5_table_contains_original_non_isolated_wave3": True,
            "original_wave3_rows_not_used_for_main_id_isolated_table_or_figure": original_wave3_rows.to_dict(orient="records"),
            "id_isolated_calibration_assets_preexisting": False,
            "action_taken": "Recomputed Brier score, calibration intercept, and calibration slope from existing id-isolated prediction files.",
        },
        "inputs": {
            "original_calibration_reference_not_used_for_wave3_main": str(ORIGINAL_CALIBRATION_REFERENCE.relative_to(PROJECT_ROOT)),
            "id_isolated_validation_metrics": str(ID_ISOLATED_VALIDATION_METRICS.relative_to(PROJECT_ROOT)),
            "prediction_files": [str(spec["path"].relative_to(PROJECT_ROOT)) for spec in PREDICTION_SPECS],
        },
        "outputs": {
            "table3": str(TABLE_OUT.relative_to(PROJECT_ROOT)),
            "figure2_pdf": str(FIGURE_PDF.relative_to(PROJECT_ROOT)),
            "figure2_png": str(FIGURE_PNG.relative_to(PROJECT_ROOT)),
            "caption": str(CAPTION_OUT.relative_to(PROJECT_ROOT)),
        },
        "table3_rows": table.to_dict(orient="records"),
        "validation_metrics_context": validation_metrics.to_dict(orient="records"),
        "notes": [
            "No models were retrained.",
            "No original data, manuscripts, or previous result files were modified.",
            "Figure 2 includes only ID-isolated wave 3 calibration curves for the two main-text models.",
        ],
    }
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {TABLE_OUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {FIGURE_PDF.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {FIGURE_PNG.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {CAPTION_OUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {LOG_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
