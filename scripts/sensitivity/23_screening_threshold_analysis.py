from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_io import project_path, read_csv, write_dataframe_csv, write_text

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


LABEL_CANDIDATES = ["possible_sarcopenia", "label", "y_true"]
PROBABILITY_CANDIDATES = ["predicted_probability", "predicted_probability_full_model", "predicted_probability_oof"]
MODEL_CANDIDATES = ["model_name"]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
SENSITIVITY_TARGETS = [0.70, 0.75, 0.80]


INPUTS = {
    "logistic_A_only_id_isolated": "output/tables/21e_wave3_predictions_logistic_A_only_id_isolated.csv",
    "xgboost_A_plus_B_id_isolated": "output/tables/21e_wave3_predictions_xgboost_A_plus_B_id_isolated.csv",
}

PUBLICATION_ROOT = "output/submission_assets_v2"


def find_first_existing_column(df: pd.DataFrame, candidates: list[str], role: str) -> tuple[str, str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate, "matched_expected_name"
    raise RuntimeError(f"预测文件缺少 {role} 列；候选列名: {candidates}")


def load_predictions(model_name: str, path: str) -> tuple[pd.DataFrame, dict[str, str]]:
    df = read_csv(path)
    label_col, label_status = find_first_existing_column(df, LABEL_CANDIDATES, "true label")
    prob_col, prob_status = find_first_existing_column(df, PROBABILITY_CANDIDATES, "predicted probability")
    model_col = None
    model_status = "model_name_from_input_key"
    for candidate in MODEL_CANDIDATES:
        if candidate in df.columns:
            model_col = candidate
            model_status = "matched_expected_name"
            break

    out = pd.DataFrame(
        {
            "model_name": df[model_col].astype(str) if model_col else model_name,
            "y_true": pd.to_numeric(df[label_col], errors="coerce").astype(int),
            "predicted_probability": pd.to_numeric(df[prob_col], errors="coerce"),
        }
    )
    out["source_file"] = path
    if out["predicted_probability"].isna().any():
        raise RuntimeError(f"{path} 存在无法解析的 predicted_probability。")
    if not set(out["y_true"].unique()).issubset({0, 1}):
        raise RuntimeError(f"{path} 的真实标签不是二分类 0/1。")

    return out, {
        "model_name": model_name,
        "source_file": path,
        "label_col": label_col,
        "label_col_status": label_status,
        "probability_col": prob_col,
        "probability_col_status": prob_status,
        "model_col": model_col or "",
        "model_col_status": model_status,
    }


def metrics_at_threshold(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "false_negative_n": fn,
        "false_positive_n": fp,
        "true_positive_n": tp,
        "true_negative_n": tn,
    }


def sweep_model(model_name: str, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = predictions["y_true"].astype(int)
    y_prob = predictions["predicted_probability"]
    for threshold in THRESHOLDS:
        rows.append(
            {
                "model_name": model_name,
                "n": int(len(predictions)),
                "positive_n": int((y_true == 1).sum()),
                "negative_n": int((y_true == 0).sum()),
                **metrics_at_threshold(y_true, y_prob, float(threshold)),
            }
        )
    return pd.DataFrame(rows)


def select_target_threshold(grid: pd.DataFrame, target: float) -> pd.Series:
    eligible = grid.loc[grid["sensitivity"] >= target].copy()
    if not eligible.empty:
        eligible = eligible.sort_values(
            ["specificity", "threshold", "sensitivity"],
            ascending=[False, False, False],
        )
        selected = eligible.iloc[0].copy()
        selected["threshold_selection_status"] = "target_achieved"
        return selected
    nearest = grid.assign(abs_diff=(grid["sensitivity"] - target).abs()).sort_values(
        ["abs_diff", "specificity"], ascending=[True, False]
    ).iloc[0].copy()
    nearest["threshold_selection_status"] = "nearest_achievable"
    return nearest


def select_recommended_threshold(grid: pd.DataFrame) -> tuple[pd.Series, str]:
    # Screening-oriented default: target sensitivity >=0.75 while preserving the best specificity.
    selected = select_target_threshold(grid, 0.75).copy()
    rationale = (
        "推荐以 sensitivity >= 0.75 且 specificity 最高的阈值作为当前数据下的筛查导向可解释选择。"
        "该规则优先减少漏识别，同时避免 specificity 过度牺牲；并非唯一正确阈值。"
    )
    selected["recommendation_target"] = "sensitivity>=0.75_with_highest_specificity"
    selected["recommendation_rationale"] = rationale
    return selected, rationale


def build_summary_tables(grid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    recommendation_rows = []
    for model_name, sub in grid_df.groupby("model_name", sort=False):
        default = sub.loc[np.isclose(sub["threshold"], 0.50)].iloc[0].copy()
        default["scenario"] = "default_threshold_0.50"
        default["target_sensitivity"] = np.nan
        default["threshold_selection_status"] = "default"
        summary_rows.append(default)

        for target in SENSITIVITY_TARGETS:
            selected = select_target_threshold(sub, target)
            selected["scenario"] = f"screening_target_sensitivity_{target:.2f}"
            selected["target_sensitivity"] = target
            summary_rows.append(selected)

        recommended, _ = select_recommended_threshold(sub)
        recommendation_rows.append(recommended)

    summary_df = pd.DataFrame(summary_rows)
    cols_front = ["model_name", "scenario", "target_sensitivity", "threshold_selection_status"]
    summary_df = summary_df[cols_front + [col for col in summary_df.columns if col not in cols_front]]

    recommendation_df = pd.DataFrame(recommendation_rows)
    rec_cols_front = ["model_name", "recommendation_target", "recommendation_rationale", "threshold_selection_status"]
    recommendation_df = recommendation_df[rec_cols_front + [col for col in recommendation_df.columns if col not in rec_cols_front]]
    return summary_df, recommendation_df


def plot_sensitivity_specificity(grid_df: pd.DataFrame, model_name: str, output_path: str) -> None:
    sub = grid_df.loc[grid_df["model_name"] == model_name].copy().sort_values("threshold")
    fig, ax = plt.subplots(figsize=(6.7, 4), dpi=300)
    ax.plot(sub["threshold"], sub["sensitivity"], label="Sensitivity", linewidth=1.5, color="#0072B2")
    ax.plot(sub["threshold"], sub["specificity"], label="Specificity", linewidth=1.5, linestyle="--", color="#D55E00")
    ax.axvline(0.50, color="#555555", linestyle="-.", linewidth=1.0, label="Default 0.50")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.22, linewidth=0.55)
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def row_for(summary_df: pd.DataFrame, model_name: str, scenario: str) -> pd.Series:
    return summary_df.loc[(summary_df["model_name"] == model_name) & (summary_df["scenario"] == scenario)].iloc[0]


def build_notes(summary_df: pd.DataFrame, recommendation_df: pd.DataFrame, column_audit: list[dict[str, str]]) -> str:
    log_default = row_for(summary_df, "logistic_A_only_id_isolated", "default_threshold_0.50")
    xgb_default = row_for(summary_df, "xgboost_A_plus_B_id_isolated", "default_threshold_0.50")
    log_rec = recommendation_df.loc[recommendation_df["model_name"] == "logistic_A_only_id_isolated"].iloc[0]
    xgb_rec = recommendation_df.loc[recommendation_df["model_name"] == "xgboost_A_plus_B_id_isolated"].iloc[0]

    def fmt(row: pd.Series, col: str) -> str:
        return f"{float(row[col]):.4f}"

    def delta(rec: pd.Series, default: pd.Series, col: str) -> str:
        return f"{float(rec[col] - default[col]):.4f}"

    better_model = (
        "logistic_A_only_id_isolated"
        if int(log_rec["false_negative_n"]) <= int(xgb_rec["false_negative_n"])
        else "xgboost_A_plus_B_id_isolated"
    )

    results_paragraph = (
        "在 id-isolated wave 3 验证集中，补充性筛查导向阈值分析显示，"
        f"`logistic_A_only` 在默认 0.50 阈值下 sensitivity 为 {fmt(log_default, 'sensitivity')}，false negatives 为 {int(log_default['false_negative_n'])}；"
        f"采用推荐筛查阈值 {float(log_rec['threshold']):.2f} 后，sensitivity 提升至 {fmt(log_rec, 'sensitivity')}，false negatives 降至 {int(log_rec['false_negative_n'])}。"
        f"`xgboost_A_plus_B` 在默认 0.50 阈值下 sensitivity 为 {fmt(xgb_default, 'sensitivity')}，false negatives 为 {int(xgb_default['false_negative_n'])}；"
        f"采用推荐筛查阈值 {float(xgb_rec['threshold']):.2f} 后，sensitivity 提升至 {fmt(xgb_rec, 'sensitivity')}，false negatives 降至 {int(xgb_rec['false_negative_n'])}。"
        "上述结果提示，在 possible sarcopenia 筛查场景下，降低阈值可明显减少漏识别，但会以 specificity、PPV 和 accuracy 下降为代价。"
    )

    discussion_paragraph = (
        "由于本研究的结局定位为 possible sarcopenia 筛查而非 confirmed sarcopenia 确诊，阈值选择不应机械固定于 0.50。"
        "在社区筛查场景中，减少漏识别通常具有较高优先级，因此可以考虑在补充分析中报告更高敏感度目标下的阈值表现。"
        "不过，筛查导向阈值会增加假阳性并降低 specificity 和 PPV，因此其应用应结合后续确认评估资源、转诊能力和目标人群风险水平进行解释。"
        "本轮阈值分析为 exploratory supplementary analysis，不替代主分析默认阈值结果。"
    )

    return "\n".join(
        [
            "# 第 23 阶段：screening-oriented threshold analysis 说明",
            "",
            "## 分析定位",
            "- 本分析仅作为 supplementary / exploratory analysis。",
            "- 本分析仅基于 id-isolated wave 3 的既有预测概率。",
            "- 本分析未重训模型、未重新划分数据、未做 DCA、未做校准再校正。",
            "- 本分析不改变主分析默认阈值结果，也不替代主文模型选择。",
            "",
            "## 输入列检查",
            *[
                f"- {item['model_name']}: label={item['label_col']} ({item['label_col_status']}), probability={item['probability_col']} ({item['probability_col_status']}), model={item['model_col'] or 'input_key'} ({item['model_col_status']})。"
                for item in column_audit
            ],
            "",
            "## 默认阈值 0.50 表现",
            f"- logistic_A_only_id_isolated: sensitivity={fmt(log_default, 'sensitivity')}, false_negative_n={int(log_default['false_negative_n'])}, specificity={fmt(log_default, 'specificity')}, PPV={fmt(log_default, 'ppv')}, accuracy={fmt(log_default, 'accuracy')}, F1={fmt(log_default, 'f1')}。",
            f"- xgboost_A_plus_B_id_isolated: sensitivity={fmt(xgb_default, 'sensitivity')}, false_negative_n={int(xgb_default['false_negative_n'])}, specificity={fmt(xgb_default, 'specificity')}, PPV={fmt(xgb_default, 'ppv')}, accuracy={fmt(xgb_default, 'accuracy')}, F1={fmt(xgb_default, 'f1')}。",
            "",
            "## 推荐筛查导向阈值",
            f"- logistic_A_only_id_isolated: 推荐阈值={float(log_rec['threshold']):.2f}，sensitivity={fmt(log_rec, 'sensitivity')}（较默认 {delta(log_rec, log_default, 'sensitivity')}），false_negative_n={int(log_rec['false_negative_n'])}，specificity={fmt(log_rec, 'specificity')}（较默认 {delta(log_rec, log_default, 'specificity')}），PPV={fmt(log_rec, 'ppv')}（较默认 {delta(log_rec, log_default, 'ppv')}），accuracy={fmt(log_rec, 'accuracy')}（较默认 {delta(log_rec, log_default, 'accuracy')}），F1={fmt(log_rec, 'f1')}（较默认 {delta(log_rec, log_default, 'f1')}）。",
            f"- xgboost_A_plus_B_id_isolated: 推荐阈值={float(xgb_rec['threshold']):.2f}，sensitivity={fmt(xgb_rec, 'sensitivity')}（较默认 {delta(xgb_rec, xgb_default, 'sensitivity')}），false_negative_n={int(xgb_rec['false_negative_n'])}，specificity={fmt(xgb_rec, 'specificity')}（较默认 {delta(xgb_rec, xgb_default, 'specificity')}），PPV={fmt(xgb_rec, 'ppv')}（较默认 {delta(xgb_rec, xgb_default, 'ppv')}），accuracy={fmt(xgb_rec, 'accuracy')}（较默认 {delta(xgb_rec, xgb_default, 'accuracy')}），F1={fmt(xgb_rec, 'f1')}（较默认 {delta(xgb_rec, xgb_default, 'f1')}）。",
            "",
            "## 关键解释",
            f"- 在减少漏识别目标下更有优势的模型：{better_model}，依据为推荐阈值下 false negatives 更少或相当。",
            "- 该分析支持在论文中强调：possible sarcopenia 是筛查而非确诊，阈值选择应服务于筛查目标，而不是机械固定为 0.50。",
            "- 该分析建议放在补充材料；如主文空间允许，可在 Results 或 Discussion 中用 1-2 句话概括。",
            "",
            "## 可写入 Results 的简短段落",
            results_paragraph,
            "",
            "## 可写入 Discussion 的简短段落",
            discussion_paragraph,
            "",
        ]
    ) + "\n"


def main() -> None:
    grid_frames = []
    column_audit = []
    for model_name, path in INPUTS.items():
        predictions, audit = load_predictions(model_name, path)
        column_audit.append(audit)
        grid_frames.append(sweep_model(model_name, predictions))

    grid_df = pd.concat(grid_frames, ignore_index=True)
    summary_df, recommendation_df = build_summary_tables(grid_df)

    write_dataframe_csv(grid_df, f"{PUBLICATION_ROOT}/tables/23_threshold_grid_main_models_id_isolated.csv")
    write_dataframe_csv(summary_df, f"{PUBLICATION_ROOT}/tables/23_threshold_summary_main_models_id_isolated.csv")
    write_dataframe_csv(recommendation_df, f"{PUBLICATION_ROOT}/tables/23_threshold_recommendation_main_models_id_isolated.csv")

    plot_sensitivity_specificity(
        grid_df,
        "logistic_A_only_id_isolated",
        f"{PUBLICATION_ROOT}/supplementary_figures/23_threshold_sensitivity_specificity_logistic_A_only_id_isolated.png",
    )
    plot_sensitivity_specificity(
        grid_df,
        "xgboost_A_plus_B_id_isolated",
        f"{PUBLICATION_ROOT}/supplementary_figures/23_threshold_sensitivity_specificity_xgboost_A_plus_B_id_isolated.png",
    )

    notes = build_notes(summary_df, recommendation_df, column_audit)
    write_text(f"{PUBLICATION_ROOT}/23_screening_threshold_analysis_notes.md", notes)

    write_text(
        f"{PUBLICATION_ROOT}/23_screening_threshold_analysis_summary.txt",
        json.dumps(
            {
                "stage": "screening-oriented threshold analysis",
                "analysis_role": "supplementary_exploratory_only",
                "inputs": INPUTS,
                "threshold_range": {"min": 0.05, "max": 0.95, "step": 0.01},
                "column_audit": column_audit,
                "outputs": {
                    "grid": f"{PUBLICATION_ROOT}/tables/23_threshold_grid_main_models_id_isolated.csv",
                    "summary": f"{PUBLICATION_ROOT}/tables/23_threshold_summary_main_models_id_isolated.csv",
                    "recommendation": f"{PUBLICATION_ROOT}/tables/23_threshold_recommendation_main_models_id_isolated.csv",
                    "notes": f"{PUBLICATION_ROOT}/23_screening_threshold_analysis_notes.md",
                },
                "default_threshold_rows": summary_df.loc[
                    summary_df["scenario"].eq("default_threshold_0.50")
                ].to_dict(orient="records"),
                "recommended_threshold_rows": recommendation_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
