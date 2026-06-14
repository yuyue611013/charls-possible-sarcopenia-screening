"""Generate Supplementary Figure S2 for screening-oriented threshold analysis.

The figure displays sensitivity and specificity across thresholds for the two
main-text id-isolated models. It marks the default 0.50 threshold and each
model's recommended screening-oriented threshold.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

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

ROOT = Path(__file__).resolve().parents[2]

GRID_PATH = ROOT / "output/tables/23_threshold_grid_main_models_id_isolated.csv"
SUMMARY_PATH = ROOT / "output/tables/23_threshold_summary_main_models_id_isolated.csv"
RECOMMENDATION_PATH = ROOT / "output/tables/23_threshold_recommendation_main_models_id_isolated.csv"
NOTES_PATH = ROOT / "docs/23_screening_threshold_analysis_notes.md"

ASSET_ROOT = ROOT / "output/submission_assets_v2"
FIGURE_PNG = ASSET_ROOT / "supplementary_figures/SuppFigureS2_screening_threshold_tradeoffs.png"
FIGURE_PDF = ASSET_ROOT / "supplementary_figures/SuppFigureS2_screening_threshold_tradeoffs.pdf"
CAPTION_OUT = ASSET_ROOT / "SuppFigureS2_screening_threshold_tradeoffs_legend.md"
LOG_OUT = ASSET_ROOT / "SuppFigureS2_generation_summary.txt"

MODEL_LABELS = {
    "logistic_A_only_id_isolated": "A  Logistic A-only",
    "xgboost_A_plus_B_id_isolated": "B  XGBoost A+B",
}


def fmt(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    for path in [GRID_PATH, SUMMARY_PATH, RECOMMENDATION_PATH, NOTES_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    grid = pd.read_csv(GRID_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    recommendation = pd.read_csv(RECOMMENDATION_PATH)

    expected_models = list(MODEL_LABELS)
    missing_models = sorted(set(expected_models) - set(grid["model_name"].unique()))
    if missing_models:
        raise ValueError(f"Threshold grid missing models: {missing_models}")

    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    CAPTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.55), dpi=300, sharey=True)
    colors = {"sensitivity": "#0072B2", "specificity": "#D55E00"}

    plot_audit: list[dict[str, Any]] = []
    for ax, model_name in zip(axes, expected_models, strict=True):
        sub = grid[grid["model_name"] == model_name].sort_values("threshold")
        rec = recommendation[recommendation["model_name"] == model_name].iloc[0]
        default = summary[(summary["model_name"] == model_name) & (summary["scenario"] == "default_threshold_0.50")].iloc[0]

        ax.plot(sub["threshold"], sub["sensitivity"], color=colors["sensitivity"], lw=1.45, linestyle="-", label="Sensitivity")
        ax.plot(sub["threshold"], sub["specificity"], color=colors["specificity"], lw=1.45, linestyle="--", label="Specificity")
        ax.axvline(0.50, color="#666666", linestyle="-.", lw=1.05)
        ax.axvline(float(rec["threshold"]), color="#111111", linestyle=":", lw=1.25)
        ax.scatter([0.50], [float(default["sensitivity"])], color=colors["sensitivity"], s=28, zorder=4)
        ax.scatter([0.50], [float(default["specificity"])], color=colors["specificity"], s=28, zorder=4)
        ax.scatter([float(rec["threshold"])], [float(rec["sensitivity"])], color=colors["sensitivity"], s=40, marker="D", zorder=5)
        ax.scatter([float(rec["threshold"])], [float(rec["specificity"])], color=colors["specificity"], s=40, marker="D", zorder=5)

        ax.text(0.50, 0.08, "0.50", rotation=90, ha="right", va="bottom", fontsize=7.5, color="#555555")
        ax.text(float(rec["threshold"]), 0.08, f"{float(rec['threshold']):.2f}", rotation=90, ha="left", va="bottom", fontsize=7.5, color="#111111")
        ax.text(0.02, 0.96, MODEL_LABELS[model_name], transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        ax.set_xlabel("Threshold")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.18, linewidth=0.55)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        plot_audit.append(
            {
                "model_name": model_name,
                "default_threshold": 0.50,
                "default_sensitivity": float(default["sensitivity"]),
                "default_specificity": float(default["specificity"]),
                "default_false_negative_n": int(default["false_negative_n"]),
                "recommended_threshold": float(rec["threshold"]),
                "recommended_sensitivity": float(rec["sensitivity"]),
                "recommended_specificity": float(rec["specificity"]),
                "recommended_false_negative_n": int(rec["false_negative_n"]),
            }
        )

    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.4)
    fig.tight_layout(rect=[0, 0.08, 1, 1], w_pad=1.2)
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logistic = next(item for item in plot_audit if item["model_name"] == "logistic_A_only_id_isolated")
    xgb = next(item for item in plot_audit if item["model_name"] == "xgboost_A_plus_B_id_isolated")
    caption = f"""# Figure 3 图题与图注建议

**中文图题：** Supplementary Figure S2. 主文模型在 ID 隔离后 wave 3 验证集中的筛查导向阈值分析

**图注：** 图中展示两个主文模型在不同预测概率阈值下 sensitivity 与 specificity 的变化。虚线表示默认阈值 0.50，点线表示当前数据下的推荐筛查阈值：Logistic baseline path 为 {logistic['recommended_threshold']:.2f}，XGBoost enhanced path 为 {xgb['recommended_threshold']:.2f}。由于 possible sarcopenia 是筛查性结局而非确诊性结局，降低阈值可明显减少漏识别；例如 Logistic baseline path 的 sensitivity 从 {logistic['default_sensitivity']:.4f} 提高至 {logistic['recommended_sensitivity']:.4f}，false negatives 从 {logistic['default_false_negative_n']} 降至 {logistic['recommended_false_negative_n']}，但 specificity 从 {logistic['default_specificity']:.4f} 降至 {logistic['recommended_specificity']:.4f}。XGBoost enhanced path 也呈现类似权衡，sensitivity 从 {xgb['default_sensitivity']:.4f} 提高至 {xgb['recommended_sensitivity']:.4f}，specificity 从 {xgb['default_specificity']:.4f} 降至 {xgb['recommended_specificity']:.4f}。
"""
    CAPTION_OUT.write_text(caption, encoding="utf-8")

    log_payload = {
        "stage": "Figure 3 screening-oriented threshold generation",
        "inputs": {
            "threshold_grid": str(GRID_PATH.relative_to(ROOT)),
            "threshold_summary": str(SUMMARY_PATH.relative_to(ROOT)),
            "threshold_recommendation": str(RECOMMENDATION_PATH.relative_to(ROOT)),
            "threshold_notes": str(NOTES_PATH.relative_to(ROOT)),
        },
        "outputs": {
            "supplementary_figure_s2_pdf": str(FIGURE_PDF.relative_to(ROOT)),
            "supplementary_figure_s2_png": str(FIGURE_PNG.relative_to(ROOT)),
            "caption": str(CAPTION_OUT.relative_to(ROOT)),
        },
        "plot_audit": plot_audit,
        "notes": [
            "No model retraining or threshold recalculation was performed.",
            "Figure 3 uses only existing id-isolated wave 3 threshold analysis outputs.",
            "Default and recommended screening thresholds are annotated in the figure; detailed explanation is reserved for the caption.",
        ],
    }
    LOG_OUT.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {FIGURE_PDF.relative_to(ROOT)}")
    print(f"Wrote {FIGURE_PNG.relative_to(ROOT)}")
    print(f"Wrote {CAPTION_OUT.relative_to(ROOT)}")
    print(f"Wrote {LOG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
