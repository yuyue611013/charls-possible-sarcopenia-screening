from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"
if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _utils_v2 import (
    BLUE,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    GREY,
    OOF_PRED_COL,
    ORANGE,
    PRED_COL,
    PRIMARY_LABEL_COL,
    STRICT_LABEL_COL,
    SUPP_FIGURE_DIR,
    TABLE_DIR,
    bootstrap_cis,
    calibration_bins,
    clip_prob,
    ensure_dirs,
    pr_curve_df,
    read_csv,
    roc_curve_df,
    set_publication_rcparams,
    write_csv,
)


def prediction_specs() -> list[dict[str, str]]:
    return [
        {
            "label_definition": "Primary available-component label",
            "model": "Logistic A-only",
            "dataset": "Wave 1 development OOF",
            "path": "output/tables/21c_wave1_predictions_logistic_A_only_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": OOF_PRED_COL,
        },
        {
            "label_definition": "Primary available-component label",
            "model": "XGBoost A+B",
            "dataset": "Wave 1 development OOF",
            "path": "output/tables/21d_wave1_predictions_xgboost_A_plus_B_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": OOF_PRED_COL,
        },
        {
            "label_definition": "Primary available-component label",
            "model": "Logistic A-only",
            "dataset": "ID-isolated Wave 3 validation",
            "path": "output/tables/21e_wave3_predictions_logistic_A_only_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": PRED_COL,
        },
        {
            "label_definition": "Primary available-component label",
            "model": "XGBoost A+B",
            "dataset": "ID-isolated Wave 3 validation",
            "path": "output/tables/21e_wave3_predictions_xgboost_A_plus_B_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": PRED_COL,
        },
        {
            "label_definition": "Strict-label sensitivity",
            "model": "Logistic A-only",
            "dataset": "Wave 1 development OOF",
            "path": TABLE_DIR / "strict_wave1_oof_predictions_logistic_A_only.csv",
            "label_col": STRICT_LABEL_COL,
            "prob_col": OOF_PRED_COL,
        },
        {
            "label_definition": "Strict-label sensitivity",
            "model": "XGBoost A+B",
            "dataset": "Wave 1 development OOF",
            "path": TABLE_DIR / "strict_wave1_oof_predictions_xgboost_A_plus_B.csv",
            "label_col": STRICT_LABEL_COL,
            "prob_col": OOF_PRED_COL,
        },
        {
            "label_definition": "Strict-label sensitivity",
            "model": "Logistic A-only",
            "dataset": "ID-isolated Wave 3 validation",
            "path": TABLE_DIR / "strict_wave3_predictions_logistic_A_only_id_isolated.csv",
            "label_col": STRICT_LABEL_COL,
            "prob_col": PRED_COL,
        },
        {
            "label_definition": "Strict-label sensitivity",
            "model": "XGBoost A+B",
            "dataset": "ID-isolated Wave 3 validation",
            "path": TABLE_DIR / "strict_wave3_predictions_xgboost_A_plus_B_id_isolated.csv",
            "label_col": STRICT_LABEL_COL,
            "prob_col": PRED_COL,
        },
    ]


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = calibration_bins(y, prob, "model", "dataset", n_bins=n_bins)
    gaps = (bins["observed_event_proportion"] - bins["mean_predicted_probability"]).abs()
    return float((bins["n"] / bins["n"].sum() * gaps).sum())


def main() -> None:
    ensure_dirs()
    set_publication_rcparams()
    plt.rcParams.update(
        {
            "font.size": 7.7,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.0,
        }
    )
    summary_rows = []
    bin_tables = []
    for spec in prediction_specs():
        df = read_csv(spec["path"])
        y = df[spec["label_col"]].astype(int).to_numpy()
        prob = clip_prob(df[spec["prob_col"]].astype(float).to_numpy())
        estimates, cis, valid = bootstrap_cis(
            y,
            prob,
            metrics=["Brier score", "calibration_intercept", "calibration_slope"],
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
        )
        observed_prevalence = float(y.mean())
        mean_predicted = float(prob.mean())
        ece = expected_calibration_error(y, prob)
        for metric in ["Brier score", "calibration_intercept", "calibration_slope"]:
            lo, hi = cis[metric]
            summary_rows.append(
                {
                    "label_definition": spec["label_definition"],
                    "model": spec["model"],
                    "dataset": spec["dataset"],
                    "N": int(len(df)),
                    "positive_N": int(y.sum()),
                    "negative_N": int((y == 0).sum()),
                    "observed_prevalence": observed_prevalence,
                    "mean_predicted_probability": mean_predicted,
                    "calibration_in_the_large_observed_minus_predicted": observed_prevalence - mean_predicted,
                    "expected_calibration_error_equal_frequency_10bin": ece,
                    "metric": metric,
                    "estimate": estimates[metric],
                    "ci_lower_95": lo,
                    "ci_upper_95": hi,
                    "bootstrap_replicates_requested": BOOTSTRAP_REPS,
                    "bootstrap_replicates_valid": valid,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "method_note": "Calibration-in-the-large is observed prevalence minus mean predicted probability; ECE uses 10 equal-frequency bins.",
                }
            )
        bins = calibration_bins(y, prob, spec["model"], spec["dataset"])
        bins.insert(0, "label_definition", spec["label_definition"])
        bin_tables.append(bins)

    summary = pd.DataFrame(summary_rows)
    bin_uncertainty = pd.concat(bin_tables, ignore_index=True)
    write_csv(TABLE_DIR / "calibration_uncertainty_summary.csv", summary)
    write_csv(TABLE_DIR / "calibration_bin_uncertainty.csv", bin_uncertainty)

    # Supplementary Figure S3: primary Wave 1 OOF discrimination/calibration.
    specs = [
        {
            "model": "Logistic A-only",
            "path": "output/tables/21c_wave1_predictions_logistic_A_only_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": OOF_PRED_COL,
            "color": BLUE,
            "marker": "o",
            "linestyle": "-",
        },
        {
            "model": "XGBoost A+B",
            "path": "output/tables/21d_wave1_predictions_xgboost_A_plus_B_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": OOF_PRED_COL,
            "color": ORANGE,
            "marker": "s",
            "linestyle": "--",
        },
    ]
    roc_tables = []
    pr_tables = []
    cal_tables = []
    metric_lookup = {}
    for spec in specs:
        df = read_csv(spec["path"])
        y = df[spec["label_col"]].astype(int).to_numpy()
        prob = clip_prob(df[spec["prob_col"]].astype(float).to_numpy())
        estimates, cis, _valid = bootstrap_cis(y, prob, metrics=["AUROC", "AUPRC"], reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)
        metric_lookup[spec["model"]] = (estimates, cis, float(y.mean()))
        roc_tables.append(roc_curve_df(y, prob, spec["model"], "Wave 1 development OOF"))
        pr_tables.append(pr_curve_df(y, prob, spec["model"], "Wave 1 development OOF"))
        cal_tables.append(calibration_bins(y, prob, spec["model"], "Wave 1 development OOF"))
    roc = pd.concat(roc_tables, ignore_index=True)
    pr = pd.concat(pr_tables, ignore_index=True)
    cal = pd.concat(cal_tables, ignore_index=True)
    write_csv(TABLE_DIR / "SuppFigureS3_wave1_oof_ROC_curve_data.csv", roc)
    write_csv(TABLE_DIR / "SuppFigureS3_wave1_oof_PR_curve_data.csv", pr)
    write_csv(TABLE_DIR / "SuppFigureS3_wave1_oof_calibration_bin_data.csv", cal)

    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.62))
    ax = axes[0]
    for spec in specs:
        data = roc.loc[roc["model"].eq(spec["model"])]
        estimates, cis, _prev = metric_lookup[spec["model"]]
        lo, hi = cis["AUROC"]
        ax.plot(data["fpr"], data["tpr"], color=spec["color"], linestyle=spec["linestyle"], linewidth=1.5, label=f"{spec['model']} {estimates['AUROC']:.3f} ({lo:.3f}-{hi:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(-0.16, 1.05, "A", transform=ax.transAxes, fontweight="bold", fontsize=10.5)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.02), frameon=False, fontsize=5.8, handlelength=2.0)
    ax.grid(alpha=0.18)

    ax = axes[1]
    for spec in specs:
        data = pr.loc[pr["model"].eq(spec["model"])]
        estimates, cis, prev = metric_lookup[spec["model"]]
        lo, hi = cis["AUPRC"]
        ax.plot(data["recall"], data["precision"], color=spec["color"], linestyle=spec["linestyle"], linewidth=1.5, label=f"{spec['model']} {estimates['AUPRC']:.3f} ({lo:.3f}-{hi:.3f})")
    prevalence = metric_lookup["Logistic A-only"][2]
    ax.axhline(prevalence, color=GREY, linestyle=":", linewidth=1, label=f"Prevalence {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(-0.16, 1.05, "B", transform=ax.transAxes, fontweight="bold", fontsize=10.5)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=5.8, handlelength=2.0)
    ax.grid(alpha=0.18)

    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1, label="Ideal")
    for spec in specs:
        data = cal.loc[cal["model"].eq(spec["model"])]
        yerr = [
            data["observed_event_proportion"] - data["observed_wilson_lower_95"],
            data["observed_wilson_upper_95"] - data["observed_event_proportion"],
        ]
        ax.errorbar(
            data["mean_predicted_probability"],
            data["observed_event_proportion"],
            yerr=yerr,
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            linewidth=1.3,
            markersize=3.2,
            capsize=1.8,
            label=spec["model"],
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.text(-0.16, 1.05, "C", transform=ax.transAxes, fontweight="bold", fontsize=10.5)
    ax.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.8)
    ax.grid(alpha=0.18)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.90, wspace=0.55)
    fig.savefig(SUPP_FIGURE_DIR / "SuppFigureS3_wave1_oof_discrimination_calibration.pdf", bbox_inches="tight")
    fig.savefig(SUPP_FIGURE_DIR / "SuppFigureS3_wave1_oof_discrimination_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
