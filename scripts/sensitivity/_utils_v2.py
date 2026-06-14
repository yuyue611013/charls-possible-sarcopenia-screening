from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"'penalty' was deprecated.*",
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "output/submission_enhancement_v2"
TABLE_DIR = OUT_ROOT / "tables"
LOG_DIR = OUT_ROOT / "logs"
MAIN_FIGURE_DIR = OUT_ROOT / "main_figures"
SUPP_FIGURE_DIR = OUT_ROOT / "supplementary_figures"
MANUSCRIPT_DIR = OUT_ROOT / "manuscript_support"
MANIFEST_DIR = OUT_ROOT / "manifests"

ID_COL = "ID (受访者编码)"
WAVE_COL = "wave (第几波调查)"
PRIMARY_LABEL_COL = "possible_sarcopenia"
STRICT_LABEL_COL = "possible_sarcopenia_strict"
PRED_COL = "predicted_probability"
OOF_PRED_COL = "predicted_probability_oof"
BOOTSTRAP_SEED = 20260613
BOOTSTRAP_REPS = 2000
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#666666"


def ensure_dirs() -> None:
    for directory in [TABLE_DIR, LOG_DIR, MAIN_FIGURE_DIR, SUPP_FIGURE_DIR, MANUSCRIPT_DIR, MANIFEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def read_csv(relative_path: str | Path) -> pd.DataFrame:
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return pd.read_csv(path, low_memory=False)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def set_publication_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )


def clip_prob(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)


def calibration_intercept_slope(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    yy = np.asarray(y, dtype=int)
    pp = clip_prob(prob)
    logit_p = np.log(pp / (1 - pp)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\\.")
        model.fit(logit_p, yy)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def metric_values(y: np.ndarray, prob: np.ndarray, threshold: float = 0.50) -> dict[str, float]:
    yy = np.asarray(y, dtype=int)
    pp = clip_prob(prob)
    pred = (pp >= threshold).astype(int)
    intercept, slope = calibration_intercept_slope(yy, pp)
    return {
        "AUROC": float(roc_auc_score(yy, pp)),
        "AUPRC": float(average_precision_score(yy, pp)),
        "Brier score": float(brier_score_loss(yy, pp)),
        "accuracy": float(accuracy_score(yy, pred)),
        "sensitivity": float(recall_score(yy, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(yy, pred, pos_label=0, zero_division=0)),
        "F1": float(f1_score(yy, pred, zero_division=0)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def bootstrap_cis(
    y: np.ndarray,
    prob: np.ndarray,
    metrics: list[str] | None = None,
    threshold: float = 0.50,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    yy = np.asarray(y, dtype=int)
    pp = clip_prob(prob)
    estimates_all = metric_values(yy, pp, threshold=threshold)
    if metrics is None:
        metrics = list(estimates_all.keys())
    estimates = {metric: estimates_all[metric] for metric in metrics}
    values = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    n = len(yy)
    valid = 0
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yb = yy[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boot_all = metric_values(yb, pp[idx], threshold=threshold)
        except Exception:
            continue
        for metric in metrics:
            values[metric].append(float(boot_all[metric]))
        valid += 1
    cis = {
        metric: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        for metric, vals in values.items()
        if vals
    }
    return estimates, cis, valid


def paired_bootstrap_differences(
    y: np.ndarray,
    prob_logistic: np.ndarray,
    prob_xgboost: np.ndarray,
    threshold: float = 0.50,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, int]:
    yy = np.asarray(y, dtype=int)
    pl = clip_prob(prob_logistic)
    px = clip_prob(prob_xgboost)

    def diffs(idx: np.ndarray | None = None) -> dict[str, float]:
        if idx is None:
            y0, l0, x0 = yy, pl, px
        else:
            y0, l0, x0 = yy[idx], pl[idx], px[idx]
        ml = metric_values(y0, l0, threshold=threshold)
        mx = metric_values(y0, x0, threshold=threshold)
        return {
            "AUROC advantage": ml["AUROC"] - mx["AUROC"],
            "AUPRC advantage": ml["AUPRC"] - mx["AUPRC"],
            "Brier advantage": mx["Brier score"] - ml["Brier score"],
            "sensitivity difference": ml["sensitivity"] - mx["sensitivity"],
            "specificity difference": ml["specificity"] - mx["specificity"],
            "F1 difference": ml["F1"] - mx["F1"],
        }

    estimate = diffs()
    values = {metric: [] for metric in estimate}
    rng = np.random.default_rng(seed)
    n = len(yy)
    valid = 0
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(yy[idx])) < 2:
            continue
        try:
            boot = diffs(idx)
        except Exception:
            continue
        for metric, value in boot.items():
            values[metric].append(float(value))
        valid += 1
    rows = []
    for metric, value in estimate.items():
        vals = values[metric]
        rows.append(
            {
                "metric": metric,
                "estimate": float(value),
                "ci_lower_95": float(np.percentile(vals, 2.5)),
                "ci_upper_95": float(np.percentile(vals, 97.5)),
                "bootstrap_replicates_requested": reps,
                "bootstrap_replicates_valid": valid,
                "bootstrap_seed": seed,
                "direction_note": "Positive values favour Logistic A-only; Brier advantage is XGBoost Brier minus Logistic Brier.",
            }
        )
    return pd.DataFrame(rows), valid


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def calibration_bins(
    y: np.ndarray,
    prob: np.ndarray,
    model: str,
    dataset: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": np.asarray(y, dtype=int), "prob": clip_prob(prob)})
    try:
        tmp["bin"] = pd.qcut(tmp["prob"], q=n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        tmp["bin"] = 1
    rows = []
    for bin_id, group in tmp.groupby("bin", dropna=False):
        n = int(len(group))
        events = int(group["y"].sum())
        lo, hi = wilson_ci(events, n)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "bin": int(bin_id),
                "n": n,
                "events": events,
                "mean_predicted_probability": float(group["prob"].mean()),
                "observed_event_proportion": float(group["y"].mean()),
                "observed_wilson_lower_95": lo,
                "observed_wilson_upper_95": hi,
            }
        )
    return pd.DataFrame(rows)


def roc_curve_df(y: np.ndarray, prob: np.ndarray, model: str, dataset: str) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(np.asarray(y, dtype=int), clip_prob(prob))
    return pd.DataFrame({"model": model, "dataset": dataset, "fpr": fpr, "tpr": tpr, "threshold": thresholds})


def pr_curve_df(y: np.ndarray, prob: np.ndarray, model: str, dataset: str) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(np.asarray(y, dtype=int), clip_prob(prob))
    threshold_values = np.r_[thresholds, np.nan]
    return pd.DataFrame(
        {
            "model": model,
            "dataset": dataset,
            "precision": precision,
            "recall": recall,
            "threshold": threshold_values,
        }
    )
