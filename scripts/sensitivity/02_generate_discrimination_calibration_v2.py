from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"
if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from _utils_v2 import (
    BLUE,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    GREY,
    ID_COL,
    MAIN_FIGURE_DIR,
    ORANGE,
    PRED_COL,
    PRIMARY_LABEL_COL,
    TABLE_DIR,
    bootstrap_cis,
    calibration_bins,
    ensure_dirs,
    metric_values,
    pr_curve_df,
    read_csv,
    roc_curve_df,
    set_publication_rcparams,
    write_csv,
)


def common_primary_predictions() -> pd.DataFrame:
    logistic = read_csv("output/tables/21e_wave3_predictions_logistic_A_only_id_isolated.csv")
    xgboost = read_csv("output/tables/21e_wave3_predictions_xgboost_A_plus_B_id_isolated.csv")
    common = logistic[[ID_COL, PRIMARY_LABEL_COL, PRED_COL]].rename(
        columns={PRIMARY_LABEL_COL: "y_logistic", PRED_COL: "prob_logistic"}
    ).merge(
        xgboost[[ID_COL, PRIMARY_LABEL_COL, PRED_COL]].rename(
            columns={PRIMARY_LABEL_COL: "y_xgboost", PRED_COL: "prob_xgboost"}
        ),
        on=ID_COL,
        how="inner",
    )
    if len(common) != 2430:
        raise RuntimeError(f"Expected common ID-isolated subset N=2430, got {len(common)}")
    if not common["y_logistic"].astype(int).equals(common["y_xgboost"].astype(int)):
        raise RuntimeError("Common subset labels differ between Logistic and XGBoost prediction files.")
    common["possible_sarcopenia"] = common["y_logistic"].astype(int)
    return common


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
    common = common_primary_predictions()
    y = common["possible_sarcopenia"].astype(int).to_numpy()
    prevalence = float(y.mean())

    model_specs = [
        ("Logistic A-only", "prob_logistic", BLUE, "o", "-"),
        ("XGBoost A+B", "prob_xgboost", ORANGE, "s", "--"),
    ]

    roc_tables = []
    pr_tables = []
    bin_tables = []
    summary_rows = []
    for label, prob_col, *_style in model_specs:
        prob = common[prob_col].astype(float).to_numpy()
        estimates, cis, valid = bootstrap_cis(
            y,
            prob,
            metrics=["AUROC", "AUPRC", "Brier score", "calibration_intercept", "calibration_slope"],
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
        )
        for metric in ["AUROC", "AUPRC", "Brier score", "calibration_intercept", "calibration_slope"]:
            lo, hi = cis[metric]
            summary_rows.append(
                {
                    "model": label,
                    "dataset": "Primary ID-isolated Wave 3 common subset",
                    "n": int(len(common)),
                    "positive_n": int(y.sum()),
                    "negative_n": int((y == 0).sum()),
                    "prevalence": prevalence,
                    "metric": metric,
                    "estimate": estimates[metric],
                    "ci_lower_95": lo,
                    "ci_upper_95": hi,
                    "bootstrap_replicates_requested": BOOTSTRAP_REPS,
                    "bootstrap_replicates_valid": valid,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
        roc_tables.append(roc_curve_df(y, prob, label, "Primary ID-isolated Wave 3 common subset"))
        pr_tables.append(pr_curve_df(y, prob, label, "Primary ID-isolated Wave 3 common subset"))
        bin_tables.append(calibration_bins(y, prob, label, "Primary ID-isolated Wave 3 common subset"))

    roc_data = pd.concat(roc_tables, ignore_index=True)
    pr_data = pd.concat(pr_tables, ignore_index=True)
    bin_data = pd.concat(bin_tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    write_csv(TABLE_DIR / "Figure2_ROC_curve_data.csv", roc_data)
    write_csv(TABLE_DIR / "Figure2_PR_curve_data.csv", pr_data)
    write_csv(TABLE_DIR / "Figure2_calibration_bin_data.csv", bin_data)
    write_csv(TABLE_DIR / "Figure2_calibration_summary.csv", summary)

    metric_lookup = {
        (row["model"], row["metric"]): row
        for _, row in summary.iterrows()
    }

    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.95))

    # Panel A: ROC
    ax = axes[0]
    for label, _prob_col, color, marker, linestyle in model_specs:
        data = roc_data.loc[roc_data["model"].eq(label)]
        row = metric_lookup[(label, "AUROC")]
        ax.plot(data["fpr"], data["tpr"], color=color, linestyle=linestyle, linewidth=1.6)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(-0.16, 1.05, "A", transform=ax.transAxes, fontweight="bold", fontsize=10.5)
    roc_text = "; ".join(
        f"{label} {metric_lookup[(label, 'AUROC')]['estimate']:.3f} "
        f"({metric_lookup[(label, 'AUROC')]['ci_lower_95']:.3f}-{metric_lookup[(label, 'AUROC')]['ci_upper_95']:.3f})"
        for label, *_ in model_specs
    )
    ax.grid(alpha=0.18)

    # Panel B: PR
    ax = axes[1]
    for label, _prob_col, color, marker, linestyle in model_specs:
        data = pr_data.loc[pr_data["model"].eq(label)]
        row = metric_lookup[(label, "AUPRC")]
        ax.plot(data["recall"], data["precision"], color=color, linestyle=linestyle, linewidth=1.6)
    ax.axhline(prevalence, color=GREY, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(-0.16, 1.05, "B", transform=ax.transAxes, fontweight="bold", fontsize=10.5)
    pr_text = "; ".join(
        f"{label} {metric_lookup[(label, 'AUPRC')]['estimate']:.3f} "
        f"({metric_lookup[(label, 'AUPRC')]['ci_lower_95']:.3f}-{metric_lookup[(label, 'AUPRC')]['ci_upper_95']:.3f})"
        for label, *_ in model_specs
    )
    ax.grid(alpha=0.18)

    # Panel C: calibration
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1.0)
    for label, _prob_col, color, marker, linestyle in model_specs:
        data = bin_data.loc[bin_data["model"].eq(label)].sort_values("bin")
        yerr = [
            data["observed_event_proportion"] - data["observed_wilson_lower_95"],
            data["observed_wilson_upper_95"] - data["observed_event_proportion"],
        ]
        ax.errorbar(
            data["mean_predicted_probability"],
            data["observed_event_proportion"],
            yerr=yerr,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.4,
            markersize=3.2,
            capsize=1.8,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.text(-0.16, 1.05, "C", transform=ax.transAxes, fontweight="bold", fontsize=10.5)
    ax.grid(alpha=0.18)

    handles = [
        Line2D([0], [0], color=BLUE, linestyle="-", marker="o", linewidth=1.5, markersize=3.2, label="Logistic A-only"),
        Line2D([0], [0], color=ORANGE, linestyle="--", marker="s", linewidth=1.5, markersize=3.2, label="XGBoost A+B"),
        Line2D([0], [0], color=GREY, linestyle=":", linewidth=1.0, label="Reference / prevalence / ideal"),
    ]
    fig.text(0.5, 0.205, f"Panel A AUROC: {roc_text}", ha="center", fontsize=6.0)
    fig.text(0.5, 0.160, f"Panel B AUPRC: {pr_text}", ha="center", fontsize=6.0)
    fig.text(0.5, 0.115, f"Common subset: n={len(common)}; prevalence={prevalence:.3f}", ha="center", fontsize=6.4)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=False, fontsize=6.2, handlelength=2.2)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.40, top=0.90, wspace=0.55)
    fig.savefig(MAIN_FIGURE_DIR / "Figure2_discrimination_calibration_id_isolated.pdf", bbox_inches="tight")
    fig.savefig(MAIN_FIGURE_DIR / "Figure2_discrimination_calibration_id_isolated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
