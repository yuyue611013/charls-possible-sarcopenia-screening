from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"
if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import matplotlib.pyplot as plt
import pandas as pd

from _utils_v2 import (
    BLUE,
    GREY,
    MAIN_FIGURE_DIR,
    ORANGE,
    OUT_ROOT,
    SUPP_FIGURE_DIR,
    TABLE_DIR,
    ensure_dirs,
    read_csv,
    set_publication_rcparams,
    write_csv,
    write_text,
)


def copy_if_missing(src: Path, dst: Path) -> str:
    if not src.exists():
        return f"missing source: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return f"kept existing: {dst}"
    shutil.copy2(src, dst)
    return f"copied: {src} -> {dst}"


def generate_s4_original_vs_id_isolated() -> None:
    src = PROJECT_ROOT / "output/tables/21_compare_original_vs_id_isolated_main_models.csv"
    df = read_csv(src)
    write_csv(TABLE_DIR / "SuppFigureS4_original_vs_id_isolated_source_data.csv", df)
    if (SUPP_FIGURE_DIR / "SuppFigureS4_original_nonisolated_vs_idisolated_performance.pdf").exists():
        return
    models = ["logistic_A_only", "xgboost_A_plus_B"]
    labels = {"logistic_A_only": "Logistic A-only", "xgboost_A_plus_B": "XGBoost A+B"}
    colors = {"logistic_A_only": BLUE, "xgboost_A_plus_B": ORANGE}
    markers = {"logistic_A_only": "o", "xgboost_A_plus_B": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7), sharey=False)
    for ax, metric, title in [
        (axes[0], "auroc", "AUROC"),
        (axes[1], "auprc", "AUPRC"),
    ]:
        for model in models:
            row = df.loc[df["model_name"].eq(model)].iloc[0]
            y = [row[f"original_wave3_{metric}"], row[f"id_isolated_wave3_{metric}"]]
            ax.plot([0, 1], y, color=colors[model], marker=markers[model], linestyle="-" if model == "logistic_A_only" else "--", linewidth=1.5, label=labels[model])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Original\\nnon-isolated", "ID-isolated"], fontsize=8)
        ax.set_ylabel(title)
        ax.set_ylim(0.40, 0.75)
        ax.grid(axis="y", alpha=0.18)
    axes[0].text(-0.18, 1.06, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=11)
    axes[1].text(-0.18, 1.06, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=11)
    handles, legends = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, legends, frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(SUPP_FIGURE_DIR / "SuppFigureS4_original_nonisolated_vs_idisolated_performance.pdf", bbox_inches="tight")
    fig.savefig(SUPP_FIGURE_DIR / "SuppFigureS4_original_nonisolated_vs_idisolated_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_s6_learning_curves() -> None:
    src = PROJECT_ROOT / "output/tables/20_learning_curve_summary.csv"
    if not src.exists():
        return
    df = read_csv(src)
    write_csv(TABLE_DIR / "SuppFigureS6_learning_curve_source_data.csv", df)
    if (SUPP_FIGURE_DIR / "SuppFigureS6_learning_curves.pdf").exists():
        return
    plot = df.loc[df["model_name"].isin(["logistic_A_only", "xgboost_A_only", "xgboost_A_plus_B"])].copy()
    labels = {
        "logistic_A_only": "Logistic A-only",
        "xgboost_A_only": "XGBoost A-only",
        "xgboost_A_plus_B": "XGBoost A+B",
    }
    colors = {"logistic_A_only": BLUE, "xgboost_A_only": GREY, "xgboost_A_plus_B": ORANGE}
    styles = {"logistic_A_only": "-", "xgboost_A_only": ":", "xgboost_A_plus_B": "--"}
    markers = {"logistic_A_only": "o", "xgboost_A_only": "^", "xgboost_A_plus_B": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7))
    for ax, metric, ylabel in [
        (axes[0], "validation_auroc_mean", "Validation AUROC"),
        (axes[1], "validation_auprc_mean", "Validation AUPRC"),
    ]:
        for model, sub in plot.groupby("model_name"):
            sub = sub.sort_values("train_fraction_of_pool")
            ax.errorbar(
                sub["train_fraction_of_pool"] * 100,
                sub[metric],
                yerr=sub[metric.replace("_mean", "_sd")],
                color=colors[model],
                linestyle=styles[model],
                marker=markers[model],
                linewidth=1.3,
                markersize=3.6,
                capsize=2,
                label=labels[model],
            )
        ax.set_xlabel("Training fraction (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.grid(alpha=0.18)
    axes[0].text(-0.18, 1.06, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=11)
    axes[1].text(-0.18, 1.06, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=11)
    handles, legends = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, legends, frameon=False, loc="lower right", fontsize=7.4)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(SUPP_FIGURE_DIR / "SuppFigureS6_learning_curves.pdf", bbox_inches="tight")
    fig.savefig(SUPP_FIGURE_DIR / "SuppFigureS6_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_publication_tables() -> None:
    table_map = [
        (TABLE_DIR / "strict_label_flow_by_wave.csv", TABLE_DIR / "SuppTable_strict_label_flow.csv"),
        (TABLE_DIR / "strict_label_metrics_with_95CI.csv", TABLE_DIR / "SuppTable_strict_label_model_performance.csv"),
        (TABLE_DIR / "primary_vs_strict_label_comparison.csv", TABLE_DIR / "SuppTable_primary_vs_strict_label.csv"),
        (TABLE_DIR / "paired_model_differences_with_95CI.csv", TABLE_DIR / "SuppTable_paired_model_differences.csv"),
        (TABLE_DIR / "calibration_uncertainty_summary.csv", TABLE_DIR / "SuppTable_calibration_uncertainty.csv"),
        (TABLE_DIR / "subgroup_performance_with_95CI.csv", TABLE_DIR / "SuppTable_subgroup_performance.csv"),
    ]
    for src, dst in table_map:
        if not src.exists():
            raise RuntimeError(f"Required source table missing: {src}")
        if not dst.exists():
            shutil.copy2(src, dst)


def create_readme_and_text(copy_log: list[str]) -> None:
    readme = """# Submission Enhancement v2 Outputs

These outputs support the robustness and submission-readiness enhancement of the possible-sarcopenia manuscript.

Safety boundaries:
- Primary labels, primary prediction files, saved models, raw data, and existing primary outputs were not overwritten.
- Strict-label sensitivity outputs were generated separately under this v2 folder.
- Fixed predictions were used for paired comparison, subgroup robustness, calibration uncertainty, and figure generation.
- Wave 3 was not used for model tuning or new model selection.

Main figures:
1. Participant flow and ID-isolated cohort construction.
2. Primary ID-isolated common-subset ROC, precision-recall, and calibration.
3. Decision-curve analysis in the ID-isolated common subset.
4. Paired model differences and strict-label robustness.

Supplementary figures:
- S1 XGBoost normalised gain-based feature importance.
- S2 Screening-threshold trade-offs.
- S3 Wave 1 pooled OOF ROC, PR, and calibration.
- S4 Original non-isolated versus ID-isolated performance comparison.
- S5 Subgroup robustness.
- S6 Learning curves from existing verified output.
"""
    write_text(OUT_ROOT / "README_outputs.md", readme)

    manuscript_text = """# Manuscript-Ready Enhancement v2 Text

## Methods - strict-label sensitivity analysis
Because the primary available-component label classified participants as negative when one observed component was negative and the other component was missing, we performed a prespecified strict-label sensitivity analysis. In this sensitivity definition, a positive observed grip or chair-stand component remained sufficient for possible sarcopenia, but a negative label required both grip and chair-stand components to be observed and negative. Participants with one negative observed component and the other component missing were classified as uncertain and excluded from the strict-label sensitivity analysis. The primary outcome was not modified.

## Methods - paired bootstrap model comparison
For the primary ID-isolated validation comparison, Logistic A-only and XGBoost A+B predictions were compared on the common subset available to both models. Paired participant-level bootstrap resampling used identical resampled indices for both models, 2,000 requested replicates, seed 20260613, percentile 95% confidence intervals, and no model refitting.

## Methods - ROC, precision-recall, and calibration
Discrimination was summarized using ROC and precision-recall curves. Calibration used 10 equal-frequency bins with Wilson 95% confidence intervals for observed event proportions. Brier score, calibration intercept, and calibration slope were summarized with bootstrap 95% confidence intervals from fixed predictions.

## Methods - subgroup analysis
Subgroup analyses used fixed primary ID-isolated predictions and did not retrain models within subgroups. Prespecified subgroups included sex, age group, residence, and education. Subgroups with fewer than 50 events or 50 non-events were flagged as imprecise. No formal interaction test was performed.

## Results - strict-label sample sizes and exclusions
The strict-label definition excluded uncertain partial-negative cases while retaining participants with a positive observed component. Wave 1 had 313 uncertain partial-negative participants, and original Wave 3 had 278. After ID isolation, strict Wave 3 had 91 uncertain partial-negative participants before predictor complete-case filtering.

## Results - strict-label performance
Strict-label results remained in the modest-discrimination range. The Logistic A-only model retained slightly higher AUROC than XGBoost A+B in strict ID-isolated validation, and XGBoost did not show a clear practical advantage.

## Results - paired model differences
On the primary common ID-isolated validation subset, paired differences were small and uncertainty intervals crossed zero for AUROC and AUPRC. These results should not be interpreted as proof of equivalence or formal superiority.

## Results - subgroup findings
Subgroup estimates were broadly descriptive and should be interpreted cautiously, especially where event or non-event counts were limited. No causal or effect-modification claims should be made from these subgroup results alone.

## Results - calibration uncertainty
Calibration uncertainty summaries supported cautious interpretation of predicted probabilities. Calibration-in-the-large was defined as observed prevalence minus mean predicted probability, and expected calibration error used 10 equal-frequency bins.

## Discussion - robustness of the primary conclusion
The strict-label sensitivity analysis, paired bootstrap comparison, calibration uncertainty, and subgroup analyses support the main conclusion that both models show modest performance, with no stable evidence that the enhanced XGBoost A+B model materially outperforms the lower-burden Logistic A-only model.

## Limitations
The available-component outcome rule may classify some participants as negative when only one observed component is negative and the other OR component is missing. The strict-label sensitivity analysis addresses this concern descriptively but does not replace the prespecified primary analysis.

## Figure 2 legend
Figure 2. ROC, precision-recall, and calibration in the primary ID-isolated validation common subset. Panel A shows ROC curves, Panel B precision-recall curves, and Panel C calibration by 10 equal-frequency predicted-risk bins with Wilson 95% confidence intervals for observed event proportions.

## Figure 4 legend
Figure 4. Paired model differences and strict-label robustness. Panel A shows paired bootstrap differences on the primary ID-isolated common subset; positive values favour Logistic A-only. Panel B compares AUROC and AUPRC under the primary available-component label and strict-label sensitivity definition.

## Supplementary figure legends
Supplementary Figure S1. XGBoost A+B normalised gain-based feature importance. Importance reflects predictive contribution, not causality.

Supplementary Figure S2. Screening-threshold trade-offs for sensitivity and specificity. Vertical markers indicate the screening-oriented and default 0.50 thresholds.

Supplementary Figure S3. Wave 1 pooled OOF ROC, precision-recall, and calibration.

Supplementary Figure S4. Original non-isolated versus ID-isolated validation performance.

Supplementary Figure S5. Subgroup robustness in the primary ID-isolated validation common subset.

Supplementary Figure S6. Learning curves based on existing verified learning-curve outputs.
"""
    write_text(OUT_ROOT / "manuscript_support/manuscript_ready_enhancement_v2.md", manuscript_text)
    write_text(OUT_ROOT / "logs/final_figure_table_assembly_log.md", "# Final Figure/Table Assembly Log\n\n" + "\n".join(f"- {line}" for line in copy_log) + "\n")


def main() -> None:
    ensure_dirs()
    set_publication_rcparams()
    copy_log: list[str] = []
    asset_root = PROJECT_ROOT / "output/submission_assets_v2"
    copy_log.append(
        copy_if_missing(asset_root / "main_figures/Figure1_participant_flow_id_isolated.pdf", MAIN_FIGURE_DIR / "Figure1_participant_flow_id_isolated.pdf")
    )
    copy_log.append(
        copy_if_missing(asset_root / "main_figures/Figure1_participant_flow_id_isolated.png", MAIN_FIGURE_DIR / "Figure1_participant_flow_id_isolated.png")
    )
    copy_log.append(
        copy_if_missing(asset_root / "main_figures/Figure3_decision_curve_id_isolated_wave3.pdf", MAIN_FIGURE_DIR / "Figure3_decision_curve_id_isolated_wave3.pdf")
    )
    copy_log.append(
        copy_if_missing(asset_root / "main_figures/Figure3_decision_curve_id_isolated_wave3.png", MAIN_FIGURE_DIR / "Figure3_decision_curve_id_isolated_wave3.png")
    )
    copy_log.append(
        copy_if_missing(asset_root / "supplementary_figures/SuppFigureS1_xgboost_gain_importance.pdf", SUPP_FIGURE_DIR / "SuppFigureS1_xgboost_gain_importance.pdf")
    )
    copy_log.append(
        copy_if_missing(asset_root / "supplementary_figures/SuppFigureS1_xgboost_gain_importance.png", SUPP_FIGURE_DIR / "SuppFigureS1_xgboost_gain_importance.png")
    )
    copy_log.append(
        copy_if_missing(asset_root / "supplementary_figures/SuppFigureS2_screening_threshold_tradeoffs.pdf", SUPP_FIGURE_DIR / "SuppFigureS2_screening_threshold_tradeoffs.pdf")
    )
    copy_log.append(
        copy_if_missing(asset_root / "supplementary_figures/SuppFigureS2_screening_threshold_tradeoffs.png", SUPP_FIGURE_DIR / "SuppFigureS2_screening_threshold_tradeoffs.png")
    )

    generate_s4_original_vs_id_isolated()
    generate_s6_learning_curves()
    create_publication_tables()
    create_readme_and_text(copy_log)


if __name__ == "__main__":
    main()
