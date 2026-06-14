# Output Dictionary

The public example outputs support the final manuscript emphasis on participant overlap, ID-isolated evaluation, Logistic A-only versus XGBoost A+B comparison, accuracy-sensitivity discordance, calibration, decision curves, and strict-label robustness.

## Aggregate tables in `example_outputs/aggregate_tables/`

- `strict_label_flow_by_wave.csv`: aggregate strict-label eligibility and uncertain partial-negative counts by wave.
- `strict_label_metrics_with_95CI.csv`: strict-label model metrics with bootstrap 95% CIs.
- `primary_vs_strict_label_comparison.csv`: primary versus strict-label aggregate metric differences.
- `paired_model_differences_with_95CI.csv`: paired common-subset model differences with bootstrap 95% CIs.
- `calibration_uncertainty_summary.csv`: Brier score, calibration intercept/slope, calibration-in-the-large, and expected calibration error summaries.
- `subgroup_performance_with_95CI.csv`: aggregate subgroup AUROC, AUPRC, and Brier score estimates.
- `subgroup_sample_adequacy.csv`: subgroup N/event/non-event adequacy flags.
- `Figure2_ROC_curve_data.csv`: ROC curve coordinates for Figure 2.
- `Figure2_PR_curve_data.csv`: precision-recall curve coordinates for Figure 2.
- `Figure2_calibration_bin_data.csv`: 10-bin calibration data for Figure 2.
- `Figure2_calibration_summary.csv`: Figure 2 common-subset discrimination/calibration summary.
- `Figure4_plot_data.csv`: source data for Figure 4.
- `Table2_main_model_validation_concise.csv`: concise main Table 2 primary evaluation values. The filename retains historical `validation` wording.
- `SuppTable_wave1_oof_model_performance.csv`: supplementary Wave 1 pooled OOF model performance.

## Figures in `example_outputs/figures/`

- `Figure1_participant_flow_id_isolated.*`: participant flow, cross-wave overlap, and ID-isolated evaluation.
- `Figure2_discrimination_calibration_id_isolated.*`: ROC, PR, and calibration.
- `Figure3_decision_curve_id_isolated_wave3.*`: decision curves.
- `Figure4_model_comparison_robustness.*`: paired differences and strict-label robustness.
- `SuppFigureS1_xgboost_gain_importance.*`: XGBoost gain importance.
- `SuppFigureS2_screening_threshold_tradeoffs.*`: threshold trade-offs.
- `SuppFigureS3_wave1_oof_discrimination_calibration.*`: Wave 1 OOF discrimination/calibration.
- `SuppFigureS4_original_nonisolated_vs_idisolated_performance.*`: original non-isolated versus ID-isolated performance comparison.
- `SuppFigureS5_subgroup_robustness.*`: subgroup robustness.
- `SuppFigureS6_learning_curves.*`: learning curves.
