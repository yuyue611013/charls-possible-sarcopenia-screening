from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils_io import load_json, read_csv, write_dataframe_csv, write_text


ID_COL = "ID (受访者编码)"
WAVE_COL = "wave (第几波调查)"
LABEL_COL = "possible_sarcopenia"
RANDOM_STATE = 42
N_SPLITS = 5


def raw_base_name(column_name: str) -> str:
    return column_name.split(" (", 1)[0]


def split_variable_types(predictors: list[str], continuous_base_names: set[str]) -> tuple[list[str], list[str]]:
    continuous_cols = [col for col in predictors if raw_base_name(col) in continuous_base_names]
    categorical_cols = [col for col in predictors if col not in continuous_cols]
    return continuous_cols, categorical_cols


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(predictors: list[str], continuous_base_names: set[str]) -> Pipeline:
    continuous_cols, categorical_cols = split_variable_types(predictors, continuous_base_names)
    transformers = []
    if continuous_cols:
        transformers.append(("continuous", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), continuous_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", build_one_hot_encoder())]), categorical_cols))
    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("model", LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)),
        ]
    )


def metrics(y_true: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def main() -> None:
    spec = load_json("config/model_matrix_spec.json")
    predictors = spec["matrix_sets"]["A_only"]
    continuous_base_names = set(spec["continuous_base_names"])
    model_name = "logistic_A_only_id_isolated"
    df = read_csv("output/tables/21b_wave1_model_input_A_only_id_isolated.csv")
    if df[predictors + [LABEL_COL]].isna().any().any():
        raise RuntimeError("logistic_A_only_id_isolated 输入数据存在缺失。")

    x = df[predictors].copy()
    y = df[LABEL_COL].astype(int).copy()
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    pred_frames = []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        model = build_pipeline(predictors, continuous_base_names)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        prob = pd.Series(model.predict_proba(x.iloc[valid_idx])[:, 1], index=y.iloc[valid_idx].index)
        rows.append({"model_name": model_name, "fold": fold_idx, "n_train": len(train_idx), "n_valid": len(valid_idx), **metrics(y.iloc[valid_idx], prob)})
        pred_frames.append(
            pd.DataFrame(
                {
                    ID_COL: df.iloc[valid_idx][ID_COL].values,
                    WAVE_COL: df.iloc[valid_idx][WAVE_COL].values,
                    LABEL_COL: y.iloc[valid_idx].values,
                    "cv_fold": fold_idx,
                    "predicted_probability_oof": prob.values,
                    "model_name": model_name,
                }
            )
        )

    cv_df = pd.DataFrame(rows)
    mean_row = {"model_name": model_name, "fold": "mean", "n_train": None, "n_valid": None}
    for metric in ["auroc", "auprc", "sensitivity", "specificity", "accuracy", "f1"]:
        mean_row[metric] = float(cv_df[metric].mean())
    cv_df = pd.concat([cv_df, pd.DataFrame([mean_row])], ignore_index=True)

    final_model = build_pipeline(predictors, continuous_base_names)
    final_model.fit(x, y)
    joblib.dump(final_model, PROJECT_ROOT / "output/models/21c_logistic_A_only_id_isolated.joblib")

    write_dataframe_csv(cv_df, "output/tables/21c_cv_metrics_logistic_A_only_id_isolated.csv")
    write_dataframe_csv(pd.concat(pred_frames, ignore_index=True), "output/tables/21c_wave1_predictions_logistic_A_only_id_isolated.csv")
    write_text(
        "output/logs/21c_model_logistic_id_isolated_summary.txt",
        json.dumps({"model_name": model_name, "n": len(df), "positive_n": int(y.sum()), "negative_n": int((y == 0).sum())}, ensure_ascii=False, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
