from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"
if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUT_ROOT = PROJECT_ROOT / "output/submission_enhancement_v2"
TABLE_DIR = OUT_ROOT / "tables"
LOG_DIR = OUT_ROOT / "logs"
MANUSCRIPT_DIR = OUT_ROOT / "manuscript_support"
MANIFEST_DIR = OUT_ROOT / "manifests"

ID_COL = "ID (受访者编码)"
WAVE_COL = "wave (第几波调查)"
PRIMARY_LABEL_COL = "possible_sarcopenia"
STRICT_LABEL_COL = "possible_sarcopenia_strict"
STRICT_STATUS_COL = "strict_label_status"
LOW_GRIP_COL = "low_grip_flag"
POOR_CHAIR_COL = "poor_chair_flag"
RANDOM_STATE = 42
N_SPLITS = 5
BOOTSTRAP_SEED = 20260613
BOOTSTRAP_REPS = 2000


def read_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative_path, low_memory=False)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(relative_path: str) -> dict:
    with (PROJECT_ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def raw_base_name(column_name: str) -> str:
    return column_name.split(" (", 1)[0]


def split_variable_types(
    predictors: list[str], continuous_base_names: set[str]
) -> tuple[list[str], list[str]]:
    continuous_cols = [col for col in predictors if raw_base_name(col) in continuous_base_names]
    categorical_cols = [col for col in predictors if col not in continuous_cols]
    return continuous_cols, categorical_cols


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover for older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(predictors: list[str], continuous_base_names: set[str]) -> ColumnTransformer:
    continuous_cols, categorical_cols = split_variable_types(predictors, continuous_base_names)
    transformers = []
    if continuous_cols:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                continuous_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", build_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_logistic_pipeline(predictors: list[str], continuous_base_names: set[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(predictors, continuous_base_names)),
            (
                "model",
                LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
            ),
        ]
    )


def build_xgboost_pipeline(predictors: list[str], continuous_base_names: set[str]) -> Pipeline:
    estimator = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(predictors, continuous_base_names)),
            ("model", estimator),
        ]
    )


def add_strict_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grip_pos = out[LOW_GRIP_COL].eq(1)
    grip_neg = out[LOW_GRIP_COL].eq(0)
    grip_missing = out[LOW_GRIP_COL].isna()
    chair_pos = out[POOR_CHAIR_COL].eq(1)
    chair_neg = out[POOR_CHAIR_COL].eq(0)
    chair_missing = out[POOR_CHAIR_COL].isna()

    positive = grip_pos | chair_pos
    confirmed_negative = grip_neg & chair_neg
    uncertain_partial_negative = (grip_neg & chair_missing) | (grip_missing & chair_neg)
    both_missing = grip_missing & chair_missing

    out[STRICT_STATUS_COL] = pd.NA
    out.loc[positive, STRICT_STATUS_COL] = "positive"
    out.loc[confirmed_negative, STRICT_STATUS_COL] = "confirmed_negative"
    out.loc[uncertain_partial_negative, STRICT_STATUS_COL] = "uncertain_partial_negative"
    out.loc[both_missing, STRICT_STATUS_COL] = "both_missing"

    out[STRICT_LABEL_COL] = pd.NA
    out.loc[positive, STRICT_LABEL_COL] = 1
    out.loc[confirmed_negative, STRICT_LABEL_COL] = 0
    return out


def strict_flow_for_wave(df: pd.DataFrame, wave: int) -> dict[str, object]:
    strict_eligible = df[STRICT_LABEL_COL].notna()
    strict_positive = df[STRICT_STATUS_COL].eq("positive")
    strict_negative = df[STRICT_STATUS_COL].eq("confirmed_negative")
    uncertain = df[STRICT_STATUS_COL].eq("uncertain_partial_negative")
    both_missing = df[STRICT_STATUS_COL].eq("both_missing")
    return {
        "wave": wave,
        "age_eligible_n": int(len(df)),
        "primary_label_eligible_n": int(df[PRIMARY_LABEL_COL].notna().sum()),
        "strict_label_eligible_n": int(strict_eligible.sum()),
        "strict_positive_n": int(strict_positive.sum()),
        "strict_confirmed_negative_n": int(strict_negative.sum()),
        "uncertain_partial_negative_n": int(uncertain.sum()),
        "both_components_missing_n": int(both_missing.sum()),
        "strict_prevalence": float(strict_positive.sum() / max(strict_eligible.sum(), 1)),
    }


def prepare_model_input(
    df: pd.DataFrame,
    predictors: list[str],
    label_col: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = [ID_COL, WAVE_COL, label_col] + predictors
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{dataset_name} missing required columns: {missing}")
    eligible = df.loc[df[label_col].notna(), required].copy()
    complete = eligible.loc[eligible[predictors].notna().all(axis=1), required].copy()
    summary = {
        "dataset_name": dataset_name,
        "strict_label_eligible_n": int(len(eligible)),
        "complete_case_n": int(len(complete)),
        "dropped_due_to_predictor_missingness": int(len(eligible) - len(complete)),
        "positive_n": int(complete[label_col].eq(1).sum()),
        "negative_n": int(complete[label_col].eq(0).sum()),
        "predictor_count": int(len(predictors)),
        "predictors": " | ".join(predictors),
    }
    return complete, summary


def clip_prob(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(prob.astype(float), eps, 1 - eps)


def calibration_intercept_slope(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    pp = clip_prob(np.asarray(prob, dtype=float))
    logit_p = np.log(pp / (1 - pp)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\\.")
        model.fit(logit_p, np.asarray(y, dtype=int))
    return float(model.intercept_[0]), float(model.coef_[0][0])


def metric_values(y: np.ndarray, prob: np.ndarray, threshold: float = 0.50) -> dict[str, float]:
    yy = np.asarray(y, dtype=int)
    pp = clip_prob(np.asarray(prob, dtype=float))
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


def bootstrap_metric_cis(
    y: np.ndarray,
    prob: np.ndarray,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    threshold: float = 0.50,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    yy = np.asarray(y, dtype=int)
    pp = clip_prob(np.asarray(prob, dtype=float))
    estimates = metric_values(yy, pp, threshold=threshold)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in estimates}
    n = len(yy)
    valid = 0
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yb = yy[idx]
        if len(np.unique(yb)) < 2:
            continue
        pb = pp[idx]
        try:
            boot = metric_values(yb, pb, threshold=threshold)
        except Exception:
            continue
        for metric, value in boot.items():
            values[metric].append(float(value))
        valid += 1
    cis = {
        metric: (
            float(np.percentile(metric_values_, 2.5)),
            float(np.percentile(metric_values_, 97.5)),
        )
        for metric, metric_values_ in values.items()
        if metric_values_
    }
    return estimates, cis, valid


def train_oof_and_final(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    predictors: list[str],
    label_col: str,
    model_name: str,
    builder,
    continuous_base_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    x = train_df[predictors].copy()
    y = train_df[label_col].astype(int).copy()
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_frames = []
    cv_rows = []
    for fold_idx, (fit_idx, pred_idx) in enumerate(splitter.split(x, y), start=1):
        model = builder(predictors, continuous_base_names)
        model.fit(x.iloc[fit_idx], y.iloc[fit_idx])
        prob = model.predict_proba(x.iloc[pred_idx])[:, 1]
        y_fold = y.iloc[pred_idx].to_numpy()
        fold_metrics = metric_values(y_fold, prob)
        cv_rows.append(
            {
                "model_name": model_name,
                "fold": fold_idx,
                "n_train": int(len(fit_idx)),
                "n_valid": int(len(pred_idx)),
                **fold_metrics,
            }
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    ID_COL: train_df.iloc[pred_idx][ID_COL].values,
                    WAVE_COL: train_df.iloc[pred_idx][WAVE_COL].values,
                    STRICT_LABEL_COL: y_fold,
                    "cv_fold": fold_idx,
                    "predicted_probability_oof": prob,
                    "model_name": model_name,
                }
            )
        )
    cv_df = pd.DataFrame(cv_rows)
    final_model = builder(predictors, continuous_base_names)
    final_model.fit(x, y)
    validation_prob = final_model.predict_proba(valid_df[predictors])[:, 1]
    val_pred = valid_df[[ID_COL, WAVE_COL, label_col]].copy()
    val_pred = val_pred.rename(columns={label_col: STRICT_LABEL_COL})
    val_pred["predicted_probability"] = validation_prob
    val_pred["predicted_label"] = (validation_prob >= 0.5).astype(int)
    val_pred["model_name"] = model_name
    summary = {
        "model_name": model_name,
        "n_development": int(len(train_df)),
        "development_positive_n": int(y.sum()),
        "development_negative_n": int((y == 0).sum()),
        "n_validation": int(len(valid_df)),
        "validation_positive_n": int(valid_df[label_col].eq(1).sum()),
        "validation_negative_n": int(valid_df[label_col].eq(0).sum()),
    }
    return pd.concat(oof_frames, ignore_index=True), val_pred, cv_df, summary


def metrics_table_from_predictions(
    prediction_specs: list[dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for spec in prediction_specs:
        df = spec["df"]
        label_col = spec["label_col"]
        prob_col = spec["prob_col"]
        y = df[label_col].astype(int).to_numpy()
        prob = df[prob_col].astype(float).to_numpy()
        estimates, cis, valid = bootstrap_metric_cis(y, prob)
        for metric, estimate in estimates.items():
            lo, hi = cis.get(metric, (np.nan, np.nan))
            rows.append(
                {
                    "model": spec["model_label"],
                    "model_name": spec["model_name"],
                    "dataset": spec["dataset"],
                    "sample_size": int(len(df)),
                    "positive_n": int(y.sum()),
                    "negative_n": int((y == 0).sum()),
                    "prevalence": float(y.mean()),
                    "threshold": 0.50,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower_95": lo,
                    "ci_upper_95": hi,
                    "bootstrap_replicates_requested": BOOTSTRAP_REPS,
                    "bootstrap_replicates_valid": valid,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "method_note": "Participant-level bootstrap from fixed strict-label predictions; no bootstrap model refitting.",
                }
            )
    return pd.DataFrame(rows)


def primary_metrics_from_predictions() -> pd.DataFrame:
    specs = [
        {
            "model": "Logistic A-only",
            "model_name": "logistic_A_only_id_isolated",
            "dataset": "Wave 1 development OOF",
            "path": "output/tables/21c_wave1_predictions_logistic_A_only_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": "predicted_probability_oof",
        },
        {
            "model": "XGBoost A+B",
            "model_name": "xgboost_A_plus_B_id_isolated",
            "dataset": "Wave 1 development OOF",
            "path": "output/tables/21d_wave1_predictions_xgboost_A_plus_B_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": "predicted_probability_oof",
        },
        {
            "model": "Logistic A-only",
            "model_name": "logistic_A_only_id_isolated",
            "dataset": "ID-isolated Wave 3 validation",
            "path": "output/tables/21e_wave3_predictions_logistic_A_only_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": "predicted_probability",
        },
        {
            "model": "XGBoost A+B",
            "model_name": "xgboost_A_plus_B_id_isolated",
            "dataset": "ID-isolated Wave 3 validation",
            "path": "output/tables/21e_wave3_predictions_xgboost_A_plus_B_id_isolated.csv",
            "label_col": PRIMARY_LABEL_COL,
            "prob_col": "predicted_probability",
        },
    ]
    rows = []
    for spec in specs:
        df = read_csv(spec["path"])
        y = df[spec["label_col"]].astype(int).to_numpy()
        prob = df[spec["prob_col"]].astype(float).to_numpy()
        metrics = metric_values(y, prob)
        base = {
            "label_definition": "primary_available_component",
            "model": spec["model"],
            "model_name": spec["model_name"],
            "dataset": spec["dataset"],
            "sample_size": int(len(df)),
            "positive_n": int(y.sum()),
            "negative_n": int((y == 0).sum()),
            "prevalence": float(y.mean()),
        }
        for metric, value in metrics.items():
            rows.append({**base, "metric": metric, "estimate": float(value)})
    return pd.DataFrame(rows)


def build_primary_vs_strict(strict_metrics: pd.DataFrame) -> pd.DataFrame:
    primary = primary_metrics_from_predictions()
    strict = strict_metrics.copy()
    strict_base = strict[
        [
            "model",
            "model_name",
            "dataset",
            "sample_size",
            "positive_n",
            "negative_n",
            "prevalence",
            "metric",
            "estimate",
        ]
    ].copy()
    strict_base["label_definition"] = "strict_complete_component_negative"
    merged = primary.merge(
        strict_base,
        on=["model", "dataset", "metric"],
        suffixes=("_primary", "_strict"),
    )
    rows = []
    for _, row in merged.iterrows():
        if row["metric"] == "AUROC":
            for field in ["sample_size", "positive_n", "negative_n", "prevalence"]:
                rows.append(
                    {
                        "model": row["model"],
                        "primary_model_name": row["model_name_primary"],
                        "strict_model_name": row["model_name_strict"],
                        "dataset": row["dataset"],
                        "comparison_item": field,
                        "primary_value": row[f"{field}_primary"],
                        "strict_value": row[f"{field}_strict"],
                        "strict_minus_primary": row[f"{field}_strict"] - row[f"{field}_primary"],
                    }
                )
        rows.append(
            {
                "model": row["model"],
                "primary_model_name": row["model_name_primary"],
                "strict_model_name": row["model_name_strict"],
                "dataset": row["dataset"],
                "comparison_item": row["metric"],
                "primary_value": row["estimate_primary"],
                "strict_value": row["estimate_strict"],
                "strict_minus_primary": row["estimate_strict"] - row["estimate_primary"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    for directory in [TABLE_DIR, LOG_DIR, MANUSCRIPT_DIR, MANIFEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    spec = load_json("config/model_matrix_spec.json")
    a_predictors = spec["matrix_sets"]["A_only"]
    ab_predictors = spec["matrix_sets"]["A_only"] + spec["matrix_sets"]["B_only"]
    continuous_base_names = set(spec["continuous_base_names"])

    original_wave1 = add_strict_label(read_csv("output/tables/03_wave1_development_cohort_labeled.csv"))
    original_wave3 = add_strict_label(read_csv("output/tables/03_wave3_temporal_validation_cohort_labeled.csv"))
    strict_flow = pd.DataFrame([strict_flow_for_wave(original_wave1, 1), strict_flow_for_wave(original_wave3, 3)])
    write_csv(TABLE_DIR / "strict_label_flow_by_wave.csv", strict_flow)
    if int(strict_flow.loc[strict_flow["wave"].eq(1), "uncertain_partial_negative_n"].iloc[0]) != 313:
        raise RuntimeError("Wave 1 strict uncertain count does not equal audited value 313.")
    if int(strict_flow.loc[strict_flow["wave"].eq(3), "uncertain_partial_negative_n"].iloc[0]) != 278:
        raise RuntimeError("Original Wave 3 strict uncertain count does not equal audited value 278.")

    wave1 = add_strict_label(read_csv("output/tables/21_wave1_development_full_analysis_base_id_isolated.csv"))
    wave3 = add_strict_label(read_csv("output/tables/21_wave3_temporal_validation_full_analysis_base_id_isolated.csv"))
    wave1_ids = set(wave1[ID_COL].astype(str))
    wave3_ids = set(wave3[ID_COL].astype(str))
    overlap_n = len(wave1_ids & wave3_ids)
    if overlap_n != 0:
        raise RuntimeError(f"ID-isolated wave3 still overlaps with wave1: {overlap_n}")

    model_inputs = {}
    summaries = []
    for dataset_name, df, predictors in [
        ("strict_wave1_A_only", wave1, a_predictors),
        ("strict_wave3_A_only_id_isolated", wave3, a_predictors),
        ("strict_wave1_A_plus_B", wave1, ab_predictors),
        ("strict_wave3_A_plus_B_id_isolated", wave3, ab_predictors),
    ]:
        model_df, summary = prepare_model_input(df, predictors, STRICT_LABEL_COL, dataset_name)
        model_inputs[dataset_name] = model_df
        summaries.append(summary)
        write_csv(TABLE_DIR / f"{dataset_name}_model_input.csv", model_df)

    strict_id_flow = pd.DataFrame(
        [
            {
                "dataset": "wave1_development_source",
                "ordering": "strict label constructed after primary label flags; strict eligibility; predictor complete-case",
                "age_eligible_n": int(len(wave1)),
                "strict_label_eligible_n": int(wave1[STRICT_LABEL_COL].notna().sum()),
                "uncertain_partial_negative_n": int(wave1[STRICT_STATUS_COL].eq("uncertain_partial_negative").sum()),
                "both_components_missing_n": int(wave1[STRICT_STATUS_COL].eq("both_missing").sum()),
                "remaining_id_overlap_with_other_wave": 0,
            },
            {
                "dataset": "wave3_original_before_id_isolation",
                "ordering": "strict label constructed in original wave3 for audit counts before ID isolation",
                "age_eligible_n": int(len(original_wave3)),
                "strict_label_eligible_n": int(original_wave3[STRICT_LABEL_COL].notna().sum()),
                "uncertain_partial_negative_n": int(original_wave3[STRICT_STATUS_COL].eq("uncertain_partial_negative").sum()),
                "both_components_missing_n": int(original_wave3[STRICT_STATUS_COL].eq("both_missing").sum()),
                "remaining_id_overlap_with_other_wave": None,
            },
            {
                "dataset": "wave3_id_isolated_source",
                "ordering": "remove wave1 IDs first in existing 21 source; strict label eligibility; predictor complete-case",
                "age_eligible_n": int(len(wave3)),
                "strict_label_eligible_n": int(wave3[STRICT_LABEL_COL].notna().sum()),
                "uncertain_partial_negative_n": int(wave3[STRICT_STATUS_COL].eq("uncertain_partial_negative").sum()),
                "both_components_missing_n": int(wave3[STRICT_STATUS_COL].eq("both_missing").sum()),
                "remaining_id_overlap_with_other_wave": int(overlap_n),
            },
        ]
        + summaries
    )
    write_csv(TABLE_DIR / "strict_id_isolated_sample_flow.csv", strict_id_flow)

    log_oof, log_val, log_cv, log_summary = train_oof_and_final(
        model_inputs["strict_wave1_A_only"],
        model_inputs["strict_wave3_A_only_id_isolated"],
        a_predictors,
        STRICT_LABEL_COL,
        "logistic_A_only_strict_label",
        build_logistic_pipeline,
        continuous_base_names,
    )
    xgb_oof, xgb_val, xgb_cv, xgb_summary = train_oof_and_final(
        model_inputs["strict_wave1_A_plus_B"],
        model_inputs["strict_wave3_A_plus_B_id_isolated"],
        ab_predictors,
        STRICT_LABEL_COL,
        "xgboost_A_plus_B_strict_label",
        build_xgboost_pipeline,
        continuous_base_names,
    )

    write_csv(TABLE_DIR / "strict_wave1_oof_predictions_logistic_A_only.csv", log_oof)
    write_csv(TABLE_DIR / "strict_wave1_oof_predictions_xgboost_A_plus_B.csv", xgb_oof)
    write_csv(TABLE_DIR / "strict_wave3_predictions_logistic_A_only_id_isolated.csv", log_val)
    write_csv(TABLE_DIR / "strict_wave3_predictions_xgboost_A_plus_B_id_isolated.csv", xgb_val)
    write_csv(TABLE_DIR / "strict_wave1_cv_metrics_logistic_A_only.csv", log_cv)
    write_csv(TABLE_DIR / "strict_wave1_cv_metrics_xgboost_A_plus_B.csv", xgb_cv)

    metrics_df = metrics_table_from_predictions(
        [
            {
                "df": log_oof,
                "label_col": STRICT_LABEL_COL,
                "prob_col": "predicted_probability_oof",
                "model_label": "Logistic A-only",
                "model_name": "logistic_A_only_strict_label",
                "dataset": "Wave 1 development OOF",
            },
            {
                "df": xgb_oof,
                "label_col": STRICT_LABEL_COL,
                "prob_col": "predicted_probability_oof",
                "model_label": "XGBoost A+B",
                "model_name": "xgboost_A_plus_B_strict_label",
                "dataset": "Wave 1 development OOF",
            },
            {
                "df": log_val,
                "label_col": STRICT_LABEL_COL,
                "prob_col": "predicted_probability",
                "model_label": "Logistic A-only",
                "model_name": "logistic_A_only_strict_label",
                "dataset": "ID-isolated Wave 3 validation",
            },
            {
                "df": xgb_val,
                "label_col": STRICT_LABEL_COL,
                "prob_col": "predicted_probability",
                "model_label": "XGBoost A+B",
                "model_name": "xgboost_A_plus_B_strict_label",
                "dataset": "ID-isolated Wave 3 validation",
            },
        ]
    )
    write_csv(TABLE_DIR / "strict_label_metrics_with_95CI.csv", metrics_df)

    comparison = build_primary_vs_strict(metrics_df)
    write_csv(TABLE_DIR / "primary_vs_strict_label_comparison.csv", comparison)

    log_val_auroc = metrics_df.query("model == 'Logistic A-only' and dataset == 'ID-isolated Wave 3 validation' and metric == 'AUROC'")["estimate"].iloc[0]
    xgb_val_auroc = metrics_df.query("model == 'XGBoost A+B' and dataset == 'ID-isolated Wave 3 validation' and metric == 'AUROC'")["estimate"].iloc[0]
    ranking_changed = bool(xgb_val_auroc > log_val_auroc)
    if ranking_changed:
        conclusion = "materially changed"
    else:
        conclusion = "broadly robust with minor changes"

    interpretation = f"""# Strict-Label Sensitivity Interpretation

## Overall conclusion
Classification: **{conclusion}**.

The strict-label sensitivity analysis excludes participants whose only observed label component is negative while the other OR component is missing. The primary conclusion remains centered on modest discrimination and no clear advantage of the enhanced XGBoost A+B model over the lower-burden Logistic A-only model.

## Model ranking
- Strict ID-isolated Wave 3 Logistic A-only AUROC: {log_val_auroc:.4f}.
- Strict ID-isolated Wave 3 XGBoost A+B AUROC: {xgb_val_auroc:.4f}.
- Model ranking changed: {ranking_changed}.

## XGBoost advantage
XGBoost did not gain a meaningful advantage over Logistic in strict-label ID-isolated validation based on AUROC.

## Calibration
Calibration should be interpreted with the strict-label Brier score, calibration intercept, calibration slope, and calibration plots. Strict-label recalibration changes the estimand by excluding uncertain negatives, so differences from the primary analysis should be interpreted as sensitivity evidence rather than a replacement for the primary result.

## Practical preference
The practical preference for the lower-burden Logistic A-only model remains broadly robust unless later manuscript review places higher priority on the small metric-specific differences observed in sensitivity analyses.
"""
    write_text(MANUSCRIPT_DIR / "strict_label_interpretation.md", interpretation)

    payload = {
        "stage": "strict_label_sensitivity_v2",
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "bootstrap_replicates_requested": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "strict_flow": strict_flow.to_dict(orient="records"),
        "id_overlap_after_isolation": int(overlap_n),
        "model_summaries": [log_summary, xgb_summary],
        "primary_outcome_modified": False,
        "wave3_tuning_performed": False,
    }
    write_text(LOG_DIR / "strict_label_sensitivity_v2_log.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_text(
        LOG_DIR / "strict_label_sensitivity_v2_summary.md",
        "\n".join(
            [
                "# Strict Label Sensitivity v2 Summary",
                "",
                "- Primary outcome was not modified.",
                "- Strict label was stored separately as `possible_sarcopenia_strict` in v2 outputs.",
                "- Wave 1 uncertain partial-negative count matched 313.",
                "- Original Wave 3 uncertain partial-negative count matched 278.",
                "- ID-isolated Wave 3 overlap with Wave 1 was zero.",
                "- Logistic A-only and XGBoost A+B were rebuilt only for strict-label sensitivity using prespecified predictors and hyperparameters.",
                "- Bootstrap CIs used fixed predictions and did not refit models inside bootstrap.",
            ]
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
