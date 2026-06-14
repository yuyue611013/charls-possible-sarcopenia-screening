"""Batch 3 exploratory decision curve analysis from existing predictions.

Safety notes:
- Reads existing ID-isolated wave 3 prediction files only.
- Does not retrain, refit, or update any model.
- Does not read raw CHARLS data.
- Writes only under output/submission_assets_v2/.
- Does not compute SHAP.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = ROOT / ".python_packages"
if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import numpy as np
import pandas as pd


OUT_ROOT = ROOT / "output" / "submission_assets_v2"
TABLE_DIR = OUT_ROOT / "tables"
FIGURE_DIR = OUT_ROOT / "main_figures"
LOG_DIR = OUT_ROOT / "logs"

LOGISTIC_PRED = ROOT / "output/tables/21e_wave3_predictions_logistic_A_only_id_isolated.csv"
XGB_PRED = ROOT / "output/tables/21e_wave3_predictions_xgboost_A_plus_B_id_isolated.csv"
LABEL_COL = "possible_sarcopenia"
PROB_COL = "predicted_probability"
ID_COL = "ID (受访者编码)"
WAVE_COL = "wave (第几波调查)"


def require_output_path(path: Path) -> None:
    resolved = path.resolve()
    allowed = OUT_ROOT.resolve()
    if not str(resolved).startswith(str(allowed) + "/") and resolved != allowed:
        raise RuntimeError(f"Unsafe output path outside Batch 3 folder: {path}")


def write_text(path: Path, text: str) -> None:
    require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def net_benefit(y: pd.Series, prob: pd.Series, threshold: float) -> tuple[float, int, int]:
    pred = prob >= threshold
    positive = y == 1
    tp = int((pred & positive).sum())
    fp = int((pred & ~positive).sum())
    n = len(y)
    nb = tp / n - fp / n * threshold / (1 - threshold)
    return float(nb), tp, fp


def build_common_subset(logistic: pd.DataFrame, xgb: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    required = {ID_COL, WAVE_COL, LABEL_COL, PROB_COL}
    if not required.issubset(logistic.columns) or not required.issubset(xgb.columns):
        return None, "required shared identifier or prediction columns missing"
    left = logistic[[ID_COL, WAVE_COL, LABEL_COL, PROB_COL]].rename(columns={PROB_COL: "prob_logistic", LABEL_COL: "label_logistic"})
    right = xgb[[ID_COL, WAVE_COL, LABEL_COL, PROB_COL]].rename(columns={PROB_COL: "prob_xgboost", LABEL_COL: "label_xgboost"})
    merged = left.merge(right, on=[ID_COL, WAVE_COL], how="inner")
    if merged.empty:
        return None, "shared ID/wave subset is empty"
    mismatch = merged["label_logistic"].astype(int) != merged["label_xgboost"].astype(int)
    if mismatch.any():
        return None, f"outcome labels mismatch for {int(mismatch.sum())} shared rows"
    merged["possible_sarcopenia"] = merged["label_logistic"].astype(int)
    return merged, "common subset constructed using shared ID and wave"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not LOGISTIC_PRED.exists() or not XGB_PRED.exists():
        missing = [str(p.relative_to(ROOT)) for p in [LOGISTIC_PRED, XGB_PRED] if not p.exists()]
        write_text(
            LOG_DIR / "decision_curve_methods_note.md",
            "# Decision Curve Methods Note\n\nDCA skipped because required prediction files were missing: "
            + ", ".join(missing)
            + "\n",
        )
        return

    logistic = pd.read_csv(LOGISTIC_PRED)
    xgb = pd.read_csv(XGB_PRED)
    common, reason = build_common_subset(logistic, xgb)
    rows = []
    thresholds = np.round(np.arange(0.05, 0.801, 0.01), 2)
    if common is not None:
        y = common["possible_sarcopenia"].astype(int)
        prevalence = float(y.mean())
        n = int(len(common))
        sample_mode = "common_subset_by_id_wave"
        for threshold in thresholds:
            treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
            for model_name, prob_col in [
                ("logistic_A_only_id_isolated", "prob_logistic"),
                ("xgboost_A_plus_B_id_isolated", "prob_xgboost"),
            ]:
                nb, tp, fp = net_benefit(y, common[prob_col], float(threshold))
                rows.append(
                    {
                        "sample_mode": sample_mode,
                        "model_name": model_name,
                        "threshold": float(threshold),
                        "n": n,
                        "prevalence": prevalence,
                        "net_benefit": nb,
                        "true_positive_n": tp,
                        "false_positive_n": fp,
                        "treat_all_net_benefit": treat_all,
                        "treat_none_net_benefit": 0.0,
                    }
                )
    else:
        sample_mode = "model_specific_complete_case_samples"
        for model_name, df in [
            ("logistic_A_only_id_isolated", logistic),
            ("xgboost_A_plus_B_id_isolated", xgb),
        ]:
            y = df[LABEL_COL].astype(int)
            prob = df[PROB_COL]
            prevalence = float(y.mean())
            n = int(len(df))
            for threshold in thresholds:
                treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
                nb, tp, fp = net_benefit(y, prob, float(threshold))
                rows.append(
                    {
                        "sample_mode": sample_mode,
                        "model_name": model_name,
                        "threshold": float(threshold),
                        "n": n,
                        "prevalence": prevalence,
                        "net_benefit": nb,
                        "true_positive_n": tp,
                        "false_positive_n": fp,
                        "treat_all_net_benefit": treat_all,
                        "treat_none_net_benefit": 0.0,
                    }
                )
    out = pd.DataFrame(rows)
    write_csv(TABLE_DIR / "SuppTable_decision_curve_id_isolated_wave3.csv", out)

    try:
        import matplotlib.pyplot as plt

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

        plot_data = out.loc[out["threshold"].between(0.05, 0.50)].copy()
        fig, ax = plt.subplots(figsize=(6.7, 4.25), dpi=300)
        for model_name, label in [
            ("logistic_A_only_id_isolated", "Logistic A-only"),
            ("xgboost_A_plus_B_id_isolated", "XGBoost A+B"),
        ]:
            sub = plot_data.loc[plot_data["model_name"] == model_name]
            style = {
                "logistic_A_only_id_isolated": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
                "xgboost_A_plus_B_id_isolated": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
            }[model_name]
            ax.plot(
                sub["threshold"],
                sub["net_benefit"],
                label=label,
                linewidth=1.65,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=2.7,
                markevery=4,
                color=style["color"],
            )
        ref = plot_data.drop_duplicates("threshold")
        ax.plot(ref["threshold"], ref["treat_all_net_benefit"], label="Treat all", linestyle="-.", linewidth=1.25, color="#555555")
        ax.axhline(0, label="Treat none", linestyle=":", linewidth=1.25, color="#111111")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_xlim(0.05, 0.50)
        ax.set_ylim(-0.05, 0.30)
        ax.grid(alpha=0.22, linewidth=0.55)
        ax.legend(frameon=False, loc="upper right", handlelength=2.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        png_path = FIGURE_DIR / "Figure3_decision_curve_id_isolated_wave3.png"
        pdf_path = FIGURE_DIR / "Figure3_decision_curve_id_isolated_wave3.pdf"
        require_output_path(png_path)
        require_output_path(pdf_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    except Exception as exc:
        write_text(LOG_DIR / "decision_curve_figure_skipped.md", f"# Decision Curve Figure Skipped\n\nFigure skipped: {exc}\n")

    note = [
        "# Decision Curve Methods Note",
        "",
        "- DCA was computed from existing ID-isolated wave 3 prediction files only.",
        "- No model was retrained or refit.",
        "- DCA is an exploratory screening-utility analysis.",
        "- DCA does not prove clinical effectiveness.",
        "- Threshold interpretation depends on downstream confirmatory assessment resources.",
        f"- Sample mode used: `{sample_mode}`.",
        f"- Common-subset assessment: {reason}.",
        f"- Logistic prediction rows: {len(logistic)}.",
        f"- XGBoost prediction rows: {len(xgb)}.",
    ]
    if common is not None:
        note.append(f"- Common subset N: {len(common)}.")
    else:
        note.append("- Model-specific samples were used because a safe common subset was not available.")
    write_text(LOG_DIR / "decision_curve_methods_note.md", "\n".join(note) + "\n")


if __name__ == "__main__":
    main()
