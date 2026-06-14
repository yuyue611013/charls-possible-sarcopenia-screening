from __future__ import annotations

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
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    GREY,
    ID_COL,
    MAIN_FIGURE_DIR,
    ORANGE,
    PRED_COL,
    PRIMARY_LABEL_COL,
    STRICT_LABEL_COL,
    TABLE_DIR,
    bootstrap_cis,
    paired_bootstrap_differences,
    read_csv,
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
        raise RuntimeError(f"Expected common primary subset N=2430, got {len(common)}")
    if not common["y_logistic"].astype(int).equals(common["y_xgboost"].astype(int)):
        raise RuntimeError("Labels differ between primary prediction files in common subset.")
    common["possible_sarcopenia"] = common["y_logistic"].astype(int)
    return common


def robustness_plot_data() -> pd.DataFrame:
    primary = read_csv(TABLE_DIR / "Figure2_calibration_summary.csv")
    primary = primary.loc[primary["metric"].isin(["AUROC", "AUPRC"])].copy()
    primary["label_definition"] = "Primary available-component label"

    strict_logistic = read_csv(TABLE_DIR / "strict_wave3_predictions_logistic_A_only_id_isolated.csv")
    strict_xgboost = read_csv(TABLE_DIR / "strict_wave3_predictions_xgboost_A_plus_B_id_isolated.csv")
    strict_common = strict_logistic[[ID_COL, STRICT_LABEL_COL, PRED_COL]].rename(
        columns={STRICT_LABEL_COL: "y_logistic", PRED_COL: "prob_logistic"}
    ).merge(
        strict_xgboost[[ID_COL, STRICT_LABEL_COL, PRED_COL]].rename(
            columns={STRICT_LABEL_COL: "y_xgboost", PRED_COL: "prob_xgboost"}
        ),
        on=ID_COL,
        how="inner",
    )
    if len(strict_common) != 2373:
        raise RuntimeError(f"Expected strict-label common subset N=2373, got {len(strict_common)}")
    if not strict_common["y_logistic"].astype(int).equals(strict_common["y_xgboost"].astype(int)):
        raise RuntimeError("Strict-label common subset labels differ between model prediction files.")
    y_strict = strict_common["y_logistic"].astype(int).to_numpy()
    strict_rows = []
    for model, prob_col in [("Logistic A-only", "prob_logistic"), ("XGBoost A+B", "prob_xgboost")]:
        estimates, cis, valid = bootstrap_cis(
            y_strict,
            strict_common[prob_col].astype(float).to_numpy(),
            metrics=["AUROC", "AUPRC"],
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
        )
        for metric in ["AUROC", "AUPRC"]:
            strict_rows.append(
                {
                    "model": model,
                    "label_definition": "Strict-label sensitivity",
                    "metric": metric,
                    "estimate": estimates[metric],
                    "ci_lower_95": cis[metric][0],
                    "ci_upper_95": cis[metric][1],
                    "n": int(len(strict_common)),
                    "positive_n": int(y_strict.sum()),
                    "negative_n": int((y_strict == 0).sum()),
                    "bootstrap_replicates_valid": valid,
                    "sample_basis": "common paired strict-label subset",
                }
            )
    strict = pd.DataFrame(strict_rows)
    strict["label_definition"] = "Strict-label sensitivity"
    primary["sample_basis"] = "common paired primary subset"
    primary["bootstrap_replicates_valid"] = primary.get("bootstrap_replicates_valid")
    cols = [
        "model",
        "label_definition",
        "metric",
        "estimate",
        "ci_lower_95",
        "ci_upper_95",
        "n",
        "positive_n",
        "negative_n",
        "sample_basis",
        "bootstrap_replicates_valid",
    ]
    out = pd.concat([primary[cols], strict[cols]], ignore_index=True)
    out["panel"] = "B"
    return out


def main() -> None:
    set_publication_rcparams()
    common = common_primary_predictions()
    paired, valid = paired_bootstrap_differences(
        common["possible_sarcopenia"].astype(int).to_numpy(),
        common["prob_logistic"].astype(float).to_numpy(),
        common["prob_xgboost"].astype(float).to_numpy(),
        reps=BOOTSTRAP_REPS,
        seed=BOOTSTRAP_SEED,
    )
    paired["direction_note"] = paired["direction_note"].str.replace("Brier advantage", "Brier-score advantage", regex=False)
    paired.insert(0, "dataset", "Primary ID-isolated Wave 3 common subset")
    paired.insert(1, "common_sample_size", len(common))
    write_csv(TABLE_DIR / "paired_model_differences_with_95CI.csv", paired)

    robust = robustness_plot_data()
    plot_rows = []
    for _, row in paired.iterrows():
        plot_rows.append(
            {
                "panel": "A",
                "metric": row["metric"],
                "estimate": row["estimate"],
                "ci_lower_95": row["ci_lower_95"],
                "ci_upper_95": row["ci_upper_95"],
                "n": len(common),
                "note": row["direction_note"],
            }
        )
    fig4_data = pd.concat([pd.DataFrame(plot_rows), robust], ignore_index=True)
    write_csv(TABLE_DIR / "Figure4_plot_data.csv", fig4_data)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.35), gridspec_kw={"width_ratios": [1.05, 1.25]})

    # Panel A: paired differences.
    ax = axes[0]
    metric_order = [
        "AUROC advantage",
        "AUPRC advantage",
        "Brier advantage",
        "sensitivity difference",
        "specificity difference",
        "F1 difference",
    ]
    data = paired.set_index("metric").loc[metric_order].reset_index()
    ypos = list(range(len(data)))[::-1]
    ax.axvline(0, color=GREY, linewidth=1.0, linestyle=":")
    for y, (_, row) in zip(ypos, data.iterrows()):
        ax.errorbar(
            row["estimate"],
            y,
            xerr=[[row["estimate"] - row["ci_lower_95"]], [row["ci_upper_95"] - row["estimate"]]],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            elinewidth=1.3,
            capsize=2,
            markersize=4,
        )
    ax.set_yticks(ypos)
    metric_labels = {
        "AUROC advantage": "AUROC",
        "AUPRC advantage": "AUPRC",
        "Brier advantage": "Brier-score advantage",
        "sensitivity difference": "Sensitivity",
        "specificity difference": "Specificity",
        "F1 difference": "F1",
    }
    ax.set_yticklabels([metric_labels[m] for m in metric_order])
    ax.set_xlabel("Difference")
    ax.text(0.0, -0.29, "Positive values favour Logistic A-only", transform=ax.transAxes, fontsize=7.2)
    ax.text(-0.17, 1.04, "A", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.grid(axis="x", alpha=0.18)

    # Panel B: primary vs strict robustness.
    ax = axes[1]
    metrics = ["AUROC", "AUPRC"]
    label_defs = ["Primary available-component label", "Strict-label sensitivity"]
    models = ["Logistic A-only", "XGBoost A+B"]
    colors = {"Logistic A-only": BLUE, "XGBoost A+B": ORANGE}
    markers = {"Logistic A-only": "o", "XGBoost A+B": "s"}
    y_positions = {}
    labels = []
    pos = 0
    for metric in metrics:
        for label_def in label_defs:
            n_value = int(robust.loc[robust["label_definition"].eq(label_def), "n"].iloc[0])
            clean_label = "Primary" if label_def.startswith("Primary") else "Strict-label"
            labels.append(f"{metric}\n{clean_label}, n={n_value}")
            y_positions[(metric, label_def)] = pos
            pos += 1
        pos += 0.5
    for model in models:
        subset = robust.loc[robust["model"].eq(model)]
        offset = -0.08 if model == "Logistic A-only" else 0.08
        for _, row in subset.iterrows():
            y = y_positions[(row["metric"], row["label_definition"])] + offset
            ax.errorbar(
                row["estimate"],
                y,
                xerr=[[row["estimate"] - row["ci_lower_95"]], [row["ci_upper_95"] - row["estimate"]]],
                fmt=markers[model],
                color=colors[model],
                ecolor=colors[model],
                elinewidth=1.3,
                capsize=2,
                markersize=4,
                label=model,
            )
    handles, labels_seen = ax.get_legend_handles_labels()
    dedup = dict(zip(labels_seen, handles))
    ax.set_yticks([y_positions[(metric, label_def)] for metric in metrics for label_def in label_defs])
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlabel("Estimate with 95% CI")
    ax.set_xlim(0.40, 0.75)
    ax.invert_yaxis()
    ax.text(-0.12, 1.04, "B", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.legend(dedup.values(), dedup.keys(), loc="lower right", frameon=False, fontsize=7.6)
    ax.grid(axis="x", alpha=0.18)

    fig.tight_layout(w_pad=0.8)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(MAIN_FIGURE_DIR / "Figure4_model_comparison_robustness.pdf", bbox_inches="tight")
    fig.savefig(MAIN_FIGURE_DIR / "Figure4_model_comparison_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
