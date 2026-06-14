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
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score

from utils_io import read_csv, write_dataframe_csv, write_text


ID_COL = "ID (受访者编码)"
WAVE_COL = "wave (第几波调查)"
LABEL_COL = "possible_sarcopenia"


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


def evaluate(model_name: str, model_path: str, input_path: str, output_path: str) -> dict[str, object]:
    model = joblib.load(PROJECT_ROOT / model_path)
    df = read_csv(input_path)
    predictors = [col for col in df.columns if col not in {ID_COL, WAVE_COL, LABEL_COL}]
    probabilities = pd.Series(model.predict_proba(df[predictors])[:, 1], index=df.index)
    metric_values = metrics(df[LABEL_COL].astype(int), probabilities)
    predictions = df[[ID_COL, WAVE_COL, LABEL_COL]].copy()
    predictions["predicted_probability"] = probabilities
    predictions["predicted_label"] = (probabilities >= 0.5).astype(int)
    predictions["model_name"] = model_name
    write_dataframe_csv(predictions, output_path)
    return {
        "model_name": model_name,
        "input_path": input_path,
        "prediction_output": output_path,
        "n_validation": int(len(df)),
        "positive_n": int((df[LABEL_COL] == 1).sum()),
        "negative_n": int((df[LABEL_COL] == 0).sum()),
        **metric_values,
    }


def main() -> None:
    rows = [
        evaluate(
            "logistic_A_only_id_isolated",
            "output/models/21c_logistic_A_only_id_isolated.joblib",
            "output/tables/21b_wave3_model_input_A_only_id_isolated.csv",
            "output/tables/21e_wave3_predictions_logistic_A_only_id_isolated.csv",
        ),
        evaluate(
            "xgboost_A_plus_B_id_isolated",
            "output/models/21d_xgboost_A_plus_B_id_isolated.joblib",
            "output/tables/21b_wave3_model_input_A_plus_B_id_isolated.csv",
            "output/tables/21e_wave3_predictions_xgboost_A_plus_B_id_isolated.csv",
        ),
    ]
    validation_df = pd.DataFrame(rows)
    write_dataframe_csv(validation_df, "output/tables/21e_temporal_validation_metrics_id_isolated.csv")
    write_text(
        "output/logs/21e_temporal_validation_id_isolated_summary.txt",
        json.dumps({"stage": "21e_temporal_validation_id_isolated", "rows": rows}, ensure_ascii=False, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
