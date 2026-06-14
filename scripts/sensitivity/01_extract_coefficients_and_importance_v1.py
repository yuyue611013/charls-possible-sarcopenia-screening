"""Batch 3 supplementary coefficients and feature-importance extraction.

Safety notes:
- Reads existing saved models and existing model-input headers only.
- Does not retrain, refit, or update any model.
- Does not modify raw data, existing scripts, configs, models, or existing outputs.
- Does not compute SHAP.
- Writes only under output/submission_assets_v2/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = ROOT / ".python_packages"
if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import joblib
import pandas as pd


OUT_ROOT = ROOT / "output" / "submission_assets_v2"
TABLE_DIR = OUT_ROOT / "tables"
LOG_DIR = OUT_ROOT / "logs"
SUPP_FIGURE_DIR = OUT_ROOT / "supplementary_figures"

INPUTS = {
    "logistic_model": {
        "path": ROOT / "output/models/21c_logistic_A_only_id_isolated.joblib",
        "intended_use": "Extract saved Logistic A-only coefficients without refitting.",
    },
    "xgboost_model": {
        "path": ROOT / "output/models/21d_xgboost_A_plus_B_id_isolated.joblib",
        "intended_use": "Extract saved XGBoost A+B feature importance without retraining.",
    },
    "logistic_model_input": {
        "path": ROOT / "output/tables/21b_wave1_model_input_A_only_id_isolated.csv",
        "intended_use": "Confirm model-input headers for feature-name alignment.",
    },
    "xgboost_model_input": {
        "path": ROOT / "output/tables/21b_wave1_model_input_A_plus_B_id_isolated.csv",
        "intended_use": "Confirm model-input headers for feature-name alignment.",
    },
    "model_matrix_spec": {
        "path": ROOT / "config/model_matrix_spec.json",
        "intended_use": "Map transformed features back to A-core and B anthropometric/body-size groups.",
    },
}

ADDITIONAL_DISCOVERY_INPUTS = {
    "logistic_wave3_predictions": {
        "path": ROOT / "output/tables/21e_wave3_predictions_logistic_A_only_id_isolated.csv",
        "intended_use": "Decision curve analysis from existing ID-isolated Logistic predictions.",
        "used": True,
    },
    "xgboost_wave3_predictions": {
        "path": ROOT / "output/tables/21e_wave3_predictions_xgboost_A_plus_B_id_isolated.csv",
        "intended_use": "Decision curve analysis from existing ID-isolated XGBoost predictions.",
        "used": True,
    },
    "threshold_grid": {
        "path": ROOT / "output/tables/23_threshold_grid_main_models_id_isolated.csv",
        "intended_use": "Threshold sensitivity summary.",
        "used": True,
    },
    "threshold_summary": {
        "path": ROOT / "output/tables/23_threshold_summary_main_models_id_isolated.csv",
        "intended_use": "Threshold sensitivity summary.",
        "used": True,
    },
    "threshold_recommendation": {
        "path": ROOT / "output/tables/23_threshold_recommendation_main_models_id_isolated.csv",
        "intended_use": "Threshold sensitivity summary.",
        "used": True,
    },
    "original_vs_id_isolated": {
        "path": ROOT / "output/tables/21_compare_original_vs_id_isolated_main_models.csv",
        "intended_use": "Original non-isolated vs ID-isolated validation sensitivity summary.",
        "used": True,
    },
    "sample_flow": {
        "path": ROOT / "output/tables/07_sample_flow_summary.csv",
        "intended_use": "Eligible sample and participant-flow context.",
        "used": True,
    },
    "id_isolated_sample_flow": {
        "path": ROOT / "output/tables/21_id_isolated_sample_flow.csv",
        "intended_use": "ID-isolated eligible sample context.",
        "used": True,
    },
    "complete_case_scenarios": {
        "path": ROOT / "output/tables/08_complete_case_scenarios.csv",
        "intended_use": "Complete-case vs eligible sample compression context.",
        "used": True,
    },
    "matrix_sample_size_summary": {
        "path": ROOT / "output/tables/12b_model_matrix_sample_size_summary.csv",
        "intended_use": "Formal matrix sample-compression context.",
        "used": True,
    },
    "id_isolated_model_input_summary": {
        "path": ROOT / "output/logs/21b_prepare_model_input_id_isolated_summary.txt",
        "intended_use": "Main ID-isolated complete-case model-input summary.",
        "used": True,
    },
}


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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def safe_exp(value: float) -> float | None:
    try:
        if abs(value) > 700:
            return None
        return float(math.exp(value))
    except Exception:
        return None


def infer_original_variable(transformed_name: str, known_variables: list[str]) -> tuple[str, str]:
    for prefix in ("continuous__", "categorical__"):
        if transformed_name.startswith(prefix):
            remainder = transformed_name[len(prefix) :]
            for variable in sorted(known_variables, key=len, reverse=True):
                if remainder == variable or remainder.startswith(variable + "_"):
                    note = "mapped_by_prefix_to_known_predictor"
                    return variable, note
            return remainder, "uncertain_mapping_no_known_predictor_match"
    return transformed_name, "uncertain_mapping_unexpected_transform_prefix"


def infer_group(original_variable: str, a_vars: set[str], b_vars: set[str]) -> str:
    if original_variable in a_vars:
        return "A-core"
    if original_variable in b_vars:
        return "B-anthropometric/body-size"
    return "unclear/other"


READABLE_BASE_LABELS = {
    "age": "Age",
    "ragender": "Sex",
    "hrural": "Residence",
    "marry": "Marital status",
    "raeduc_c": "Education",
    "hukou": "Household registration",
    "hibpe": "Hypertension",
    "diabe": "Diabetes",
    "hearte": "Heart disease",
    "stroke": "Stroke",
    "arthre": "Arthritis",
    "drinkl": "Current drinking",
    "smoken": "Current smoking",
    "mheight": "Height",
    "mweight": "Body weight",
    "bmi": "Body mass index",
    "mwaist": "Waist circumference",
}


def readable_category(value: str) -> str:
    value = value.strip()
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return value


def readable_plot_label(transformed_name: str, original_variable: str) -> str:
    base_name = original_variable.split(" (", 1)[0]
    label = READABLE_BASE_LABELS.get(base_name, base_name)
    if transformed_name.startswith("categorical__"):
        prefix = f"categorical__{original_variable}_"
        if transformed_name.startswith(prefix):
            category = readable_category(transformed_name[len(prefix):])
            return f"{label} category {category}"
    return label


def update_input_discovery(
    records: list[dict[str, str]],
    name: str,
    path: Path,
    intended_use: str,
    used: bool,
    skipped_reason: str = "",
) -> None:
    records.append(
        {
            "input": name,
            "file_path": str(path.relative_to(ROOT)),
            "exists": "yes" if path.exists() else "no",
            "intended_use": intended_use,
            "used": "yes" if used else "no",
            "skipped_reason": skipped_reason,
        }
    )


def render_input_discovery(records: list[dict[str, str]]) -> str:
    lines = [
        "# Batch 3 Input Discovery Log",
        "",
        "This log records availability and use of inputs for Batch 3 supplementary submission-enhancement outputs.",
        "",
        "| Input | File path | Exists | Intended use | Used | If skipped, why |",
        "|---|---|---|---|---|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['input']} | `{row['file_path']}` | {row['exists']} | "
            f"{row['intended_use']} | {row['used']} | {row['skipped_reason']} |"
        )
    lines.append("")
    lines.append("Safety confirmation: no models were retrained; SHAP was not computed.")
    return "\n".join(lines) + "\n"


def add_additional_discovery(records: list[dict[str, str]]) -> None:
    seen = {row["input"] for row in records}
    for name, info in ADDITIONAL_DISCOVERY_INPUTS.items():
        if name in seen:
            continue
        path = info["path"]
        skipped_reason = "" if path.exists() else "missing required or contextual file"
        update_input_discovery(records, name, path, info["intended_use"], bool(info["used"] and path.exists()), skipped_reason)


def extract_logistic(records: list[dict[str, str]], spec: dict[str, Any]) -> None:
    model_info = INPUTS["logistic_model"]
    input_info = INPUTS["logistic_model_input"]
    if not model_info["path"].exists() or not input_info["path"].exists():
        update_input_discovery(records, "logistic_model", model_info["path"], model_info["intended_use"], False, "missing required file")
        update_input_discovery(records, "logistic_model_input", input_info["path"], input_info["intended_use"], False, "missing required file")
        write_text(
            LOG_DIR / "logistic_coefficients_methods_note.md",
            "# Logistic Coefficients Methods Note\n\nSkipped because a required saved model or model-input file was missing.\n",
        )
        return

    update_input_discovery(records, "logistic_model", model_info["path"], model_info["intended_use"], True)
    update_input_discovery(records, "logistic_model_input", input_info["path"], input_info["intended_use"], True)
    _ = load_header(input_info["path"])

    model = joblib.load(model_info["path"])
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    coefs = estimator.coef_[0]
    if len(feature_names) != len(coefs):
        raise RuntimeError("Logistic feature-name and coefficient lengths do not match.")

    a_vars = set(spec["matrix_sets"]["A_only"])
    b_vars = set(spec["matrix_sets"].get("B_only", []))
    known = list(a_vars | b_vars)
    rows = []
    for name, coef in zip(feature_names, coefs):
        original, mapping_note = infer_original_variable(name, known)
        rows.append(
            {
                "transformed_feature_name": name,
                "coefficient": float(coef),
                "absolute_coefficient": abs(float(coef)),
                "odds_ratio": safe_exp(float(coef)),
                "feature_group": infer_group(original, a_vars, b_vars),
                "original_variable": original,
                "mapping_note": mapping_note,
                "interpretation_note": "odds ratio is conditional on preprocessing/encoding and should not be interpreted causally",
            }
        )
    df = pd.DataFrame(rows).sort_values("absolute_coefficient", ascending=False)
    write_csv(TABLE_DIR / "SuppTable_logistic_A_only_coefficients.csv", df)
    write_text(
        LOG_DIR / "logistic_coefficients_methods_note.md",
        "\n".join(
            [
                "# Logistic Coefficients Methods Note",
                "",
                "- Coefficients were extracted from the saved model `output/models/21c_logistic_A_only_id_isolated.joblib`.",
                "- No model was refit or retrained.",
                "- Coefficients are model associations / predictive contributions, not causal effects.",
                "- Transformed features may reflect preprocessing and one-hot encoding steps.",
                "- Odds ratios are reported for transformed features and should be interpreted with caution.",
                "- SHAP was not computed in this batch.",
            ]
        )
        + "\n",
    )


def extract_xgboost(records: list[dict[str, str]], spec: dict[str, Any]) -> None:
    model_info = INPUTS["xgboost_model"]
    input_info = INPUTS["xgboost_model_input"]
    if not model_info["path"].exists() or not input_info["path"].exists():
        update_input_discovery(records, "xgboost_model", model_info["path"], model_info["intended_use"], False, "missing required file")
        update_input_discovery(records, "xgboost_model_input", input_info["path"], input_info["intended_use"], False, "missing required file")
        write_text(
            LOG_DIR / "xgboost_feature_importance_methods_note.md",
            "# XGBoost Feature Importance Methods Note\n\nSkipped because a required saved model or model-input file was missing.\n",
        )
        return

    update_input_discovery(records, "xgboost_model", model_info["path"], model_info["intended_use"], True)
    update_input_discovery(records, "xgboost_model_input", input_info["path"], input_info["intended_use"], True)
    _ = load_header(input_info["path"])

    model = joblib.load(model_info["path"])
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    importances = list(estimator.feature_importances_)
    if len(feature_names) != len(importances):
        raise RuntimeError("XGBoost feature-name and importance lengths do not match.")

    a_vars = set(spec["matrix_sets"]["A_only"])
    b_vars = set(spec["matrix_sets"].get("B_only", []))
    known = list(a_vars | b_vars)
    total = float(sum(importances))
    rows = []
    for name, importance in zip(feature_names, importances):
        original, mapping_note = infer_original_variable(name, known)
        normalized = float(importance) / total if total > 0 else 0.0
        rows.append(
            {
                "transformed_feature_name": name,
                "importance_value": float(importance),
                "normalized_importance": normalized,
                "inferred_original_variable": original,
                "feature_group": infer_group(original, a_vars, b_vars),
                "mapping_note": mapping_note,
                "interpretation_note": "feature importance reflects predictive contribution, not causality",
            }
        )
    df = pd.DataFrame(rows).sort_values("importance_value", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    write_csv(TABLE_DIR / "SuppTable_feature_importance_xgboost_A_plus_B.csv", df)

    group = (
        df.groupby("feature_group", as_index=False)
        .agg(number_of_transformed_features=("transformed_feature_name", "count"), total_importance=("importance_value", "sum"))
        .sort_values("total_importance", ascending=False)
    )
    total_group_importance = float(group["total_importance"].sum())
    group["percentage_of_total_importance"] = group["total_importance"] / total_group_importance * 100 if total_group_importance > 0 else 0.0
    group["interpretation_note"] = "grouped transformed-feature importance from saved XGBoost A+B model; predictive contribution only"
    write_csv(TABLE_DIR / "SuppTable_feature_importance_group_summary.csv", group)

    write_text(
        LOG_DIR / "xgboost_feature_importance_methods_note.md",
        "\n".join(
            [
                "# XGBoost Feature Importance Methods Note",
                "",
                "- Feature importance was extracted from the saved XGBoost A+B model `output/models/21d_xgboost_A_plus_B_id_isolated.joblib`.",
                "- No model was retrained.",
                "- Importance reflects predictive contribution, not causality.",
                "- A-core vs B-anthropometric grouping is intended to clarify whether the enhanced model relies mainly on core variables or added body-size variables.",
                "- Transformed features may reflect preprocessing and one-hot encoding steps.",
                "- SHAP was not computed in this batch.",
            ]
        )
        + "\n",
    )

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

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

        top = df.head(20).copy()
        top["plot_label"] = [
            readable_plot_label(row["transformed_feature_name"], row["inferred_original_variable"])
            for _, row in top.iterrows()
        ]
        top = top.sort_values("normalized_importance", ascending=True)
        color_map = {
            "A-core": "#0072B2",
            "B-anthropometric/body-size": "#D55E00",
            "unclear/other": "#999999",
        }
        colors = [color_map.get(group, "#999999") for group in top["feature_group"]]
        fig_height = max(4.9, 0.24 * len(top) + 1.1)
        fig, ax = plt.subplots(figsize=(6.7, fig_height), dpi=300)
        ax.barh(top["plot_label"], top["normalized_importance"], color=colors, edgecolor="white", linewidth=0.35)
        ax.set_xlabel("Normalised gain importance")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.22, linewidth=0.55)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        legend_handles = [
            Patch(facecolor="#0072B2", label="A-core"),
            Patch(facecolor="#D55E00", label="Anthropometric/body-size"),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="lower right")
        fig.tight_layout()
        png_path = SUPP_FIGURE_DIR / "SuppFigureS1_xgboost_gain_importance.png"
        pdf_path = SUPP_FIGURE_DIR / "SuppFigureS1_xgboost_gain_importance.pdf"
        require_output_path(png_path)
        require_output_path(pdf_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    except Exception as exc:
        write_text(
            LOG_DIR / "xgboost_feature_importance_figure_skipped.md",
            f"# XGBoost Feature Importance Figure Skipped\n\nFigure generation was skipped: {exc}\n",
        )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    spec_info = INPUTS["model_matrix_spec"]
    if not spec_info["path"].exists():
        update_input_discovery(records, "model_matrix_spec", spec_info["path"], spec_info["intended_use"], False, "missing required file")
        write_text(LOG_DIR / "input_discovery_log.md", render_input_discovery(records))
        raise RuntimeError("Missing config/model_matrix_spec.json")
    spec = read_json(spec_info["path"])
    update_input_discovery(records, "model_matrix_spec", spec_info["path"], spec_info["intended_use"], True)

    extract_logistic(records, spec)
    extract_xgboost(records, spec)
    add_additional_discovery(records)
    write_text(LOG_DIR / "input_discovery_log.md", render_input_discovery(records))


if __name__ == "__main__":
    main()
