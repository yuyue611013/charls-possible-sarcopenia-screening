from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils_io import load_json, project_path, read_csv, write_dataframe_csv, write_text


ID_COL = "ID (受访者编码)"
WAVE_COL = "wave (第几波调查)"
LABEL_COL = "possible_sarcopenia"

RANDOM_STATE = 42
N_REPEATS = 10
VALIDATION_SIZE = 0.20
TRAIN_FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]


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


def build_preprocessor(predictors: list[str], continuous_base_names: set[str]) -> ColumnTransformer:
    continuous_cols, categorical_cols = split_variable_types(predictors, continuous_base_names)
    transformers = []
    if continuous_cols:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    steps=[
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
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", build_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model(model_type: str, predictors: list[str], continuous_base_names: set[str], random_state: int) -> Pipeline:
    preprocessor = build_preprocessor(predictors, continuous_base_names)
    if model_type == "logistic":
        estimator = LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state)
    elif model_type == "xgboost":
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
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def stratified_subsample_indices(y: pd.Series, fraction: float, random_state: int) -> list[int]:
    if fraction >= 1.0:
        return list(range(len(y)))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=random_state)
    sampled_idx, _ = next(splitter.split(pd.DataFrame(index=range(len(y))), y))
    return sampled_idx.tolist()


def calculate_auc_metrics(y_true: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def load_model_specs() -> list[dict[str, object]]:
    matrix_spec = load_json("config/model_matrix_spec.json")
    matrix_sets = matrix_spec["matrix_sets"]
    return [
        {
            "comparison_group": "same_input_method_comparison",
            "model_name": "logistic_A_only",
            "model_type": "logistic",
            "input_path": "output/tables/09b_wave1_model_input_A_only.csv",
            "predictors": matrix_sets["A_only"],
        },
        {
            "comparison_group": "same_input_method_comparison",
            "model_name": "xgboost_A_only",
            "model_type": "xgboost",
            "input_path": "output/tables/09b_wave1_model_input_A_only.csv",
            "predictors": matrix_sets["A_only"],
        },
        {
            "comparison_group": "main_text_dual_track_comparison",
            "model_name": "xgboost_A_plus_B",
            "model_type": "xgboost",
            "input_path": "output/tables/09_wave1_model_input_A_plus_B.csv",
            "predictors": matrix_sets["A_only"] + matrix_sets["B_only"],
        },
    ]


def run_learning_curve() -> tuple[pd.DataFrame, pd.DataFrame]:
    continuous_base_names = set(load_json("config/model_matrix_spec.json")["continuous_base_names"])
    model_specs = load_model_specs()
    rows = []

    for spec in model_specs:
        df = read_csv(str(spec["input_path"]))
        predictors = list(spec["predictors"])
        missing_predictors = [col for col in predictors if col not in df.columns]
        if missing_predictors:
            raise RuntimeError(f"{spec['model_name']} 缺少 predictors: {missing_predictors}")
        if df[predictors + [LABEL_COL]].isna().any().any():
            raise RuntimeError(f"{spec['model_name']} 输入数据仍存在缺失值，不符合 complete-case learning curve。")

        x = df[predictors].copy()
        y = df[LABEL_COL].astype(int).copy()
        splitter = StratifiedShuffleSplit(
            n_splits=N_REPEATS,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
        )

        for repeat_idx, (train_pool_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
            x_train_pool = x.iloc[train_pool_idx].reset_index(drop=True)
            y_train_pool = y.iloc[train_pool_idx].reset_index(drop=True)
            x_valid = x.iloc[valid_idx].copy()
            y_valid = y.iloc[valid_idx].copy()

            for fraction in TRAIN_FRACTIONS:
                seed = RANDOM_STATE + repeat_idx * 1000 + int(fraction * 100)
                sampled_positions = stratified_subsample_indices(y_train_pool, fraction, seed)
                x_train = x_train_pool.iloc[sampled_positions].copy()
                y_train = y_train_pool.iloc[sampled_positions].copy()

                model = build_model(str(spec["model_type"]), predictors, continuous_base_names, random_state=seed)
                model.fit(x_train, y_train)

                train_prob = pd.Series(model.predict_proba(x_train)[:, 1], index=y_train.index)
                valid_prob = pd.Series(model.predict_proba(x_valid)[:, 1], index=y_valid.index)
                train_metrics = calculate_auc_metrics(y_train, train_prob)
                valid_metrics = calculate_auc_metrics(y_valid, valid_prob)

                rows.append(
                    {
                        "comparison_group": spec["comparison_group"],
                        "model_name": spec["model_name"],
                        "model_type": spec["model_type"],
                        "input_path": spec["input_path"],
                        "train_fraction_of_pool": fraction,
                        "train_fraction_label": f"{int(fraction * 100)}%",
                        "repeat": repeat_idx,
                        "random_seed": seed,
                        "train_pool_n": int(len(train_pool_idx)),
                        "train_n": int(len(x_train)),
                        "validation_n": int(len(x_valid)),
                        "train_positive_n": int(y_train.sum()),
                        "train_negative_n": int((y_train == 0).sum()),
                        "validation_positive_n": int(y_valid.sum()),
                        "validation_negative_n": int((y_valid == 0).sum()),
                        "train_auroc": train_metrics["auroc"],
                        "train_auprc": train_metrics["auprc"],
                        "validation_auroc": valid_metrics["auroc"],
                        "validation_auprc": valid_metrics["auprc"],
                    }
                )

    by_repeat_df = pd.DataFrame(rows)
    summary_df = (
        by_repeat_df.groupby(
            [
                "comparison_group",
                "model_name",
                "model_type",
                "train_fraction_of_pool",
                "train_fraction_label",
            ],
            as_index=False,
        )
        .agg(
            train_n_mean=("train_n", "mean"),
            validation_n_mean=("validation_n", "mean"),
            validation_auroc_mean=("validation_auroc", "mean"),
            validation_auroc_sd=("validation_auroc", "std"),
            validation_auprc_mean=("validation_auprc", "mean"),
            validation_auprc_sd=("validation_auprc", "std"),
            train_auroc_mean=("train_auroc", "mean"),
            train_auroc_sd=("train_auroc", "std"),
            train_auprc_mean=("train_auprc", "mean"),
            train_auprc_sd=("train_auprc", "std"),
            repeat_count=("repeat", "nunique"),
        )
        .sort_values(["comparison_group", "model_name", "train_fraction_of_pool"])
        .reset_index(drop=True)
    )
    return summary_df, by_repeat_df


def slope_between(summary_df: pd.DataFrame, model_name: str, metric: str, start: float = 0.80, end: float = 1.00) -> float:
    sub = summary_df.loc[
        (summary_df["model_name"] == model_name)
        & (summary_df["train_fraction_of_pool"].isin([start, end]))
    ].set_index("train_fraction_of_pool")
    if start not in sub.index or end not in sub.index:
        return float("nan")
    return float(sub.loc[end, metric] - sub.loc[start, metric])


def best_value(summary_df: pd.DataFrame, model_name: str, metric: str, fraction: float = 1.00) -> float:
    row = summary_df.loc[
        (summary_df["model_name"] == model_name)
        & (summary_df["train_fraction_of_pool"] == fraction),
        metric,
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0])


def interpret_results(summary_df: pd.DataFrame) -> dict[str, object]:
    logistic_late_auroc_gain = slope_between(summary_df, "logistic_A_only", "validation_auroc_mean")
    logistic_late_auprc_gain = slope_between(summary_df, "logistic_A_only", "validation_auprc_mean")
    xgb_a_late_auroc_gain = slope_between(summary_df, "xgboost_A_only", "validation_auroc_mean")
    xgb_a_late_auprc_gain = slope_between(summary_df, "xgboost_A_only", "validation_auprc_mean")
    xgb_ab_late_auroc_gain = slope_between(summary_df, "xgboost_A_plus_B", "validation_auroc_mean")
    xgb_ab_late_auprc_gain = slope_between(summary_df, "xgboost_A_plus_B", "validation_auprc_mean")

    plateau_threshold = 0.005
    logistic_plateau = abs(logistic_late_auroc_gain) < plateau_threshold and abs(logistic_late_auprc_gain) < plateau_threshold
    xgb_a_continues = xgb_a_late_auroc_gain >= plateau_threshold or xgb_a_late_auprc_gain >= plateau_threshold
    xgb_ab_continues = xgb_ab_late_auroc_gain >= plateau_threshold or xgb_ab_late_auprc_gain >= plateau_threshold
    xgb_ab_more_sample_dependent = (
        max(xgb_ab_late_auroc_gain, xgb_ab_late_auprc_gain)
        > max(logistic_late_auroc_gain, logistic_late_auprc_gain) + plateau_threshold
    )

    return {
        "plateau_threshold": plateau_threshold,
        "logistic_A_only_late_auroc_gain_80_to_100": logistic_late_auroc_gain,
        "logistic_A_only_late_auprc_gain_80_to_100": logistic_late_auprc_gain,
        "xgboost_A_only_late_auroc_gain_80_to_100": xgb_a_late_auroc_gain,
        "xgboost_A_only_late_auprc_gain_80_to_100": xgb_a_late_auprc_gain,
        "xgboost_A_plus_B_late_auroc_gain_80_to_100": xgb_ab_late_auroc_gain,
        "xgboost_A_plus_B_late_auprc_gain_80_to_100": xgb_ab_late_auprc_gain,
        "logistic_A_only_near_plateau": logistic_plateau,
        "xgboost_A_only_continues_to_benefit": xgb_a_continues,
        "xgboost_A_plus_B_continues_to_benefit": xgb_ab_continues,
        "xgboost_A_plus_B_more_sample_dependent_than_logistic_A_only": xgb_ab_more_sample_dependent,
        "validation_auroc_at_100": {
            "logistic_A_only": best_value(summary_df, "logistic_A_only", "validation_auroc_mean"),
            "xgboost_A_only": best_value(summary_df, "xgboost_A_only", "validation_auroc_mean"),
            "xgboost_A_plus_B": best_value(summary_df, "xgboost_A_plus_B", "validation_auroc_mean"),
        },
        "validation_auprc_at_100": {
            "logistic_A_only": best_value(summary_df, "logistic_A_only", "validation_auprc_mean"),
            "xgboost_A_only": best_value(summary_df, "xgboost_A_only", "validation_auprc_mean"),
            "xgboost_A_plus_B": best_value(summary_df, "xgboost_A_plus_B", "validation_auprc_mean"),
        },
    }


def plot_learning_curve(summary_df: pd.DataFrame, metric: str, ylabel: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, sub in summary_df.groupby("model_name"):
        sub = sub.sort_values("train_fraction_of_pool")
        x_values = sub["train_fraction_of_pool"] * 100
        y_values = sub[f"validation_{metric}_mean"]
        y_err = sub[f"validation_{metric}_sd"].fillna(0)
        ax.errorbar(x_values, y_values, yerr=y_err, marker="o", capsize=3, label=model_name)
    ax.set_xlabel("Training sample fraction of wave 1 training pool (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Learning curve: validation {ylabel}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_notes(interpretation: dict[str, object]) -> str:
    logistic_plateau = "是" if interpretation["logistic_A_only_near_plateau"] else "否"
    xgb_a_gain = "是" if interpretation["xgboost_A_only_continues_to_benefit"] else "否"
    xgb_ab_gain = "是" if interpretation["xgboost_A_plus_B_continues_to_benefit"] else "否"
    xgb_ab_dependency = "是" if interpretation["xgboost_A_plus_B_more_sample_dependent_than_logistic_A_only"] else "否"

    return "\n".join(
        [
            "# 第 20 阶段：learning curve 补充分析说明",
            "",
            "## 分析定位",
            "- 本分析仅为 supplementary / exploratory analysis。",
            "- 本分析不替代主分析，不改变主文已冻结的模型选择。",
            "- 本分析仅使用 wave 1 开发集；wave 3 未参与训练、调参或本轮 learning curve。",
            "- 本分析不重新调参，不进行插补，不修改既有时间验证结果。",
            "",
            "## 设计概要",
            "- 训练比例：20%、40%、60%、80%、100%。",
            "- 每个比例重复 10 次。",
            "- 每次重复在 wave 1 内进行分层训练/验证拆分，其中验证侧仅用于本补充分析。",
            "- 输出指标：验证侧 AUROC、AUPRC；同时记录训练侧 AUROC、AUPRC 以辅助判断过拟合趋势。",
            "",
            "## 关键问题回答",
            f"1. `logistic_A_only` 是否已接近平台：{logistic_plateau}。80% 到 100% 训练比例的 AUROC 变化为 {interpretation['logistic_A_only_late_auroc_gain_80_to_100']:.4f}，AUPRC 变化为 {interpretation['logistic_A_only_late_auprc_gain_80_to_100']:.4f}。",
            f"2. `xgboost_A_only` 是否仍随样本量增加而持续获益：{xgb_a_gain}。80% 到 100% 训练比例的 AUROC 变化为 {interpretation['xgboost_A_only_late_auroc_gain_80_to_100']:.4f}，AUPRC 变化为 {interpretation['xgboost_A_only_late_auprc_gain_80_to_100']:.4f}。",
            f"3. `xgboost_A_plus_B` 是否显示出比 `logistic_A_only` 更强的样本量依赖：{xgb_ab_dependency}。`xgboost_A_plus_B` 80% 到 100% 训练比例的 AUROC 变化为 {interpretation['xgboost_A_plus_B_late_auroc_gain_80_to_100']:.4f}，AUPRC 变化为 {interpretation['xgboost_A_plus_B_late_auprc_gain_80_to_100']:.4f}；是否仍持续获益：{xgb_ab_gain}。",
            "4. 当前结果是否支持“为了复杂模型重做扩样”：不建议基于本轮补充分析单独提出重做扩样。该分析只能说明在当前 wave 1 数据内的样本量-性能趋势，不能替代真实外部扩样研究。",
            "5. 这一分析更适合放主文还是补充材料：补充材料。",
            "",
            "## 写作提醒",
            "- 不要把本分析写成主结果。",
            "- 不要用本分析重新定义主文模型。",
            "- 如需在 Discussion 中引用，应表述为“补充性 learning curve 分析提示……”。",
            "",
        ]
    )


def main() -> None:
    summary_df, by_repeat_df = run_learning_curve()
    interpretation = interpret_results(summary_df)

    write_dataframe_csv(summary_df, "output/tables/20_learning_curve_summary.csv")
    write_dataframe_csv(by_repeat_df, "output/tables/20_learning_curve_by_repeat.csv")
    plot_learning_curve(summary_df, "auroc", "AUROC", "output/figures/20_learning_curve_auroc.png")
    plot_learning_curve(summary_df, "auprc", "AUPRC", "output/figures/20_learning_curve_auprc.png")

    notes = build_notes(interpretation)
    write_text("docs/20_learning_curve_notes.md", notes)
    write_text(
        "output/logs/20_learning_curve_analysis_summary.txt",
        json.dumps(
            {
                "stage": "learning curve 补充分析",
                "analysis_role": "supplementary_exploratory_only",
                "wave3_used": False,
                "train_fractions": TRAIN_FRACTIONS,
                "n_repeats": N_REPEATS,
                "validation_size_within_wave1": VALIDATION_SIZE,
                "outputs": {
                    "summary": "output/tables/20_learning_curve_summary.csv",
                    "by_repeat": "output/tables/20_learning_curve_by_repeat.csv",
                    "auroc_plot": "output/figures/20_learning_curve_auroc.png",
                    "auprc_plot": "output/figures/20_learning_curve_auprc.png",
                    "notes": "docs/20_learning_curve_notes.md",
                },
                "interpretation": interpretation,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
