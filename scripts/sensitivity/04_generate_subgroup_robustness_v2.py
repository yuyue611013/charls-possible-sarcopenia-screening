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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from _utils_v2 import (
    BLUE,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    GREY,
    ID_COL,
    ORANGE,
    PRED_COL,
    PRIMARY_LABEL_COL,
    SUPP_FIGURE_DIR,
    TABLE_DIR,
    clip_prob,
    ensure_dirs,
    read_csv,
    set_publication_rcparams,
    write_csv,
)


AGE_COL = "age (年龄)"
SEX_COL = "ragender (性别)"
RESIDENCE_COL = "hrural (居住在农村或城市)"
EDUCATION_COL = "raeduc_c (教育)"


def common_predictions_with_subgroups() -> pd.DataFrame:
    logistic = read_csv("output/tables/21e_wave3_predictions_logistic_A_only_id_isolated.csv")
    xgboost = read_csv("output/tables/21e_wave3_predictions_xgboost_A_plus_B_id_isolated.csv")
    base = read_csv("output/tables/21_wave3_temporal_validation_full_analysis_base_id_isolated.csv")[
        [ID_COL, AGE_COL, SEX_COL, RESIDENCE_COL, EDUCATION_COL]
    ].copy()
    common = logistic[[ID_COL, PRIMARY_LABEL_COL, PRED_COL]].rename(
        columns={PRIMARY_LABEL_COL: "y_logistic", PRED_COL: "prob_logistic"}
    ).merge(
        xgboost[[ID_COL, PRIMARY_LABEL_COL, PRED_COL]].rename(
            columns={PRIMARY_LABEL_COL: "y_xgboost", PRED_COL: "prob_xgboost"}
        ),
        on=ID_COL,
        how="inner",
    ).merge(base, on=ID_COL, how="left")
    if len(common) != 2430:
        raise RuntimeError(f"Expected common primary subset N=2430, got {len(common)}")
    if not common["y_logistic"].astype(int).equals(common["y_xgboost"].astype(int)):
        raise RuntimeError("Common labels differ between model prediction files.")
    common["possible_sarcopenia"] = common["y_logistic"].astype(int)
    return common


def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sex_subgroup"] = out[SEX_COL].map({0.0: "Women", 1.0: "Men"})
    out["residence_subgroup"] = out[RESIDENCE_COL].map({0: "Urban", 1: "Rural"})
    age = out[AGE_COL].astype(float)
    out["age_subgroup"] = pd.cut(
        age,
        bins=[59, 69, 79, np.inf],
        labels=["60-69 years", "70-79 years", ">=80 years"],
        right=True,
    ).astype(str)
    out["education_subgroup"] = out[EDUCATION_COL].map(
        {
            1.0: "Less than primary school",
            2.0: "Primary school",
            3.0: "Middle school",
            4.0: "High school or above",
        }
    )
    return out


def subgroup_metric_estimates(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    yy = np.asarray(y, dtype=int)
    pp = clip_prob(prob)
    out = {"Brier score": float(brier_score_loss(yy, pp))}
    if len(np.unique(yy)) >= 2:
        out["AUROC"] = float(roc_auc_score(yy, pp))
        out["AUPRC"] = float(average_precision_score(yy, pp))
    else:
        out["AUROC"] = np.nan
        out["AUPRC"] = np.nan
    return out


def subgroup_bootstrap(y: np.ndarray, prob: np.ndarray) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    yy = np.asarray(y, dtype=int)
    pp = clip_prob(prob)
    estimates = subgroup_metric_estimates(yy, pp)
    values = {metric: [] for metric in ["AUROC", "AUPRC", "Brier score"]}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    valid = 0
    n = len(yy)
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)
        yb = yy[idx]
        if len(np.unique(yb)) < 2:
            continue
        pb = pp[idx]
        vals = subgroup_metric_estimates(yb, pb)
        for metric in values:
            values[metric].append(vals[metric])
        valid += 1
    cis = {
        metric: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        for metric, vals in values.items()
        if vals
    }
    return estimates, cis, valid


def main() -> None:
    ensure_dirs()
    set_publication_rcparams()
    df = add_subgroup_columns(common_predictions_with_subgroups())
    subgroup_specs = [
        ("Sex", "sex_subgroup"),
        ("Age", "age_subgroup"),
        ("Residence", "residence_subgroup"),
        ("Education", "education_subgroup"),
    ]
    model_specs = [
        ("Logistic A-only", "prob_logistic"),
        ("XGBoost A+B", "prob_xgboost"),
    ]
    adequacy_rows = []
    performance_rows = []
    for domain, col in subgroup_specs:
        for subgroup_label, group in df.dropna(subset=[col]).groupby(col, dropna=True):
            y = group["possible_sarcopenia"].astype(int).to_numpy()
            pos = int(y.sum())
            neg = int((y == 0).sum())
            adequate = pos >= 50 and neg >= 50
            one_class = len(np.unique(y)) < 2
            adequacy_rows.append(
                {
                    "subgroup_domain": domain,
                    "subgroup_label": subgroup_label,
                    "n": int(len(group)),
                    "positive_n": pos,
                    "negative_n": neg,
                    "prevalence": float(y.mean()) if len(group) else np.nan,
                    "adequacy_flag": "adequate" if adequate else "imprecise_small_subgroup",
                    "one_class_flag": bool(one_class),
                    "graph_included": bool(adequate and not one_class),
                }
            )
            for model, prob_col in model_specs:
                estimates, cis, valid = subgroup_bootstrap(y, group[prob_col].astype(float).to_numpy())
                for metric in ["AUROC", "AUPRC", "Brier score"]:
                    lo, hi = cis.get(metric, (np.nan, np.nan))
                    performance_rows.append(
                        {
                            "subgroup_domain": domain,
                            "subgroup_label": subgroup_label,
                            "model": model,
                            "N": int(len(group)),
                            "positive_N": pos,
                            "negative_N": neg,
                            "prevalence": float(y.mean()) if len(group) else np.nan,
                            "metric": metric,
                            "estimate": estimates[metric],
                            "ci_lower_95": lo,
                            "ci_upper_95": hi,
                            "bootstrap_replicates_requested": BOOTSTRAP_REPS,
                            "bootstrap_replicates_valid": valid,
                            "bootstrap_seed": BOOTSTRAP_SEED,
                            "adequacy_flag": "adequate" if adequate else "imprecise_small_subgroup",
                            "interpretation_note": "Subgroup estimates are descriptive; no formal interaction test was performed.",
                        }
                    )

    adequacy = pd.DataFrame(adequacy_rows)
    perf = pd.DataFrame(performance_rows)
    write_csv(TABLE_DIR / "subgroup_sample_adequacy.csv", adequacy)
    write_csv(TABLE_DIR / "subgroup_performance_with_95CI.csv", perf)

    plot = perf.merge(
        adequacy[["subgroup_domain", "subgroup_label", "graph_included"]],
        on=["subgroup_domain", "subgroup_label"],
        how="left",
    )
    plot = plot.loc[plot["metric"].eq("AUROC") & plot["graph_included"].eq(True)].copy()
    plot["label"] = plot["subgroup_domain"] + ": " + plot["subgroup_label"].astype(str)
    label_order = plot[["subgroup_domain", "subgroup_label", "label"]].drop_duplicates()["label"].tolist()
    ypos = {label: i for i, label in enumerate(label_order[::-1])}
    colors = {"Logistic A-only": BLUE, "XGBoost A+B": ORANGE}
    markers = {"Logistic A-only": "o", "XGBoost A+B": "s"}
    offsets = {"Logistic A-only": -0.09, "XGBoost A+B": 0.09}

    height = max(3.2, 0.34 * len(label_order) + 1.0)
    fig, ax = plt.subplots(figsize=(6.7, height))
    for _, row in plot.iterrows():
        y = ypos[row["label"]] + offsets[row["model"]]
        ax.errorbar(
            row["estimate"],
            y,
            xerr=[[row["estimate"] - row["ci_lower_95"]], [row["ci_upper_95"] - row["estimate"]]],
            fmt=markers[row["model"]],
            color=colors[row["model"]],
            ecolor=colors[row["model"]],
            elinewidth=1.2,
            capsize=2,
            markersize=4,
            label=row["model"],
        )
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.set_yticks([ypos[label] for label in label_order])
    ax.set_yticklabels(label_order, fontsize=7.5)
    ax.set_xlabel("AUROC with 95% CI")
    ax.set_xlim(0.45, 0.85)
    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1)
    ax.legend(dedup.values(), dedup.keys(), loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(SUPP_FIGURE_DIR / "SuppFigureS5_subgroup_robustness.pdf", bbox_inches="tight")
    fig.savefig(SUPP_FIGURE_DIR / "SuppFigureS5_subgroup_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
