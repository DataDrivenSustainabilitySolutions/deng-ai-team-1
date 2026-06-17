"""Adversarial train/test distribution diagnostic for DengAI feature CSVs.

Example:
    python3 oliver/diagnostic/adversarial_validation.py
"""

import argparse
import math
import os
import random
from pathlib import Path


script_dir = Path(__file__).resolve().parent
default_output_dir = script_dir.parent / "outputs" / script_dir.name
default_output_dir_label = "oliver/outputs/diagnostic"


parser = argparse.ArgumentParser(
    description=(
        "Run city-specific adversarial validation to check whether raw train and "
        "test feature distributions are distinguishable. The diagnostic never "
        "loads labels and rejects total_cases columns."
    )
)
parser.add_argument(
    "--train-features-csv",
    default="data/dengue_features_train.csv",
    help="Path to the training feature CSV. Default: data/dengue_features_train.csv",
)
parser.add_argument(
    "--test-features-csv",
    default="data/dengue_features_test.csv",
    help="Path to the test feature CSV. Default: data/dengue_features_test.csv",
)
parser.add_argument(
    "--output-dir",
    default=default_output_dir,
    help=f"Directory for generated diagnostic outputs. Default: {default_output_dir_label}",
)
parser.add_argument(
    "--n-splits",
    type=int,
    default=5,
    help="Stratified folds for the domain classifier. Default: 5",
)
parser.add_argument(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducible folds and LightGBM. Default: 42",
)
parser.add_argument(
    "--n-estimators",
    type=int,
    default=300,
    help="Number of LightGBM classifier trees. Default: 300",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=0.03,
    help="LightGBM learning rate. Default: 0.03",
)
parser.add_argument(
    "--num-leaves",
    type=int,
    default=15,
    help="LightGBM tree leaves. Default: 15",
)
parser.add_argument(
    "--min-child-samples",
    type=int,
    default=20,
    help="Minimum samples per LightGBM leaf. Default: 20",
)
parser.add_argument(
    "--skip-plot",
    action="store_true",
    help="Skip the AUC summary PNG plot.",
)
args = parser.parse_args()


if args.n_splits < 2:
    raise ValueError("--n-splits must be at least 2.")

if args.n_estimators < 1:
    raise ValueError("--n-estimators must be at least 1.")

if args.learning_rate <= 0:
    raise ValueError("--learning-rate must be positive.")

if args.num_leaves < 2:
    raise ValueError("--num-leaves must be at least 2.")

if args.min_child_samples < 1:
    raise ValueError("--min-child-samples must be at least 1.")

os.environ["PYTHONHASHSEED"] = str(args.random_state)

train_features_path = Path(args.train_features_csv)
test_features_path = Path(args.test_features_csv)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

matplotlib_config_dir = output_dir / ".matplotlib"
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

random.seed(args.random_state)
np.random.seed(args.random_state)

scores_path = output_dir / "domain_classifier_scores.csv"
fold_scores_path = output_dir / "fold_scores.csv"
feature_importance_path = output_dir / "feature_importance.csv"
feature_shift_path = output_dir / "feature_shift_summary.csv"
missing_shift_path = output_dir / "missingness_shift_summary.csv"
prediction_path = output_dir / "adversarial_oof_predictions.csv"
interpretation_path = output_dir / "interpretation.md"
plot_path = output_dir / "auc_by_city_scenario.png"

merge_keys = ["city", "year", "weekofyear"]
identifier_columns = {"city", "year", "weekofyear", "week_start_date", "source", "domain_is_train"}
forbidden_substrings = ["total_cases"]


### Data import
train_features = pd.read_csv(train_features_path, parse_dates=["week_start_date"])
test_features = pd.read_csv(test_features_path, parse_dates=["week_start_date"])

if list(train_features.columns) != list(test_features.columns):
    raise ValueError("Train and test feature CSVs must have the same columns.")

for column in train_features.columns:
    if any(forbidden in column for forbidden in forbidden_substrings):
        raise ValueError(f"Forbidden target-derived column found: {column}")

if train_features[merge_keys].duplicated().any() or test_features[merge_keys].duplicated().any():
    raise ValueError("Feature rows must be unique by city, year, and weekofyear.")

train_features = train_features.copy()
test_features = test_features.copy()
train_features["source"] = "train"
test_features["source"] = "test"
train_features["domain_is_train"] = 1
test_features["domain_is_train"] = 0

raw_data = pd.concat([train_features, test_features], ignore_index=True, sort=False)
data = raw_data.copy()

weather_feature_columns = [
    column
    for column in train_features.columns
    if column not in identifier_columns and pd.api.types.is_numeric_dtype(train_features[column])
]

if not weather_feature_columns:
    raise ValueError("No numeric weather feature columns were found.")


### Preprocessing
# Interpolate weather features city-by-city so the city time series stay separate.
interpolated_city_data = []
for city, city_data in data.groupby("city", sort=False):
    city_data = city_data.sort_values("week_start_date").copy()
    city_data[weather_feature_columns] = city_data[weather_feature_columns].interpolate(
        method="linear",
        limit_direction="both",
    )
    interpolated_city_data.append(city_data)

data = pd.concat(interpolated_city_data, ignore_index=True)

week_angle = 2 * math.pi * data["weekofyear"] / 52.0
data["weekofyear_sin"] = np.sin(week_angle)
data["weekofyear_cos"] = np.cos(week_angle)
seasonality_columns = ["weekofyear_sin", "weekofyear_cos"]

scenarios = {
    "weather_only": weather_feature_columns,
    "weather_plus_seasonality": weather_feature_columns + seasonality_columns,
}


### Direct distribution summaries
shift_records = []
missing_records = []
summary_columns = weather_feature_columns + seasonality_columns

for city, city_data in data.groupby("city", sort=False):
    city_raw_data = raw_data[raw_data["city"] == city].copy()
    city_train = city_data[city_data["source"] == "train"]
    city_test = city_data[city_data["source"] == "test"]
    city_raw_train = city_raw_data[city_raw_data["source"] == "train"]
    city_raw_test = city_raw_data[city_raw_data["source"] == "test"]

    for feature in summary_columns:
        train_values = city_train[feature]
        test_values = city_test[feature]
        pooled_std = math.sqrt(
            (float(train_values.var(ddof=1)) + float(test_values.var(ddof=1))) / 2.0
        )
        standardized_mean_diff = (
            (float(test_values.mean()) - float(train_values.mean())) / pooled_std
            if pooled_std > 0
            else 0.0
        )
        shift_records.append(
            {
                "city": city,
                "feature": feature,
                "train_mean": float(train_values.mean()),
                "test_mean": float(test_values.mean()),
                "train_std": float(train_values.std(ddof=1)),
                "test_std": float(test_values.std(ddof=1)),
                "standardized_mean_diff": standardized_mean_diff,
                "abs_standardized_mean_diff": abs(standardized_mean_diff),
            }
        )

    for feature in weather_feature_columns:
        missing_records.append(
            {
                "city": city,
                "feature": feature,
                "train_missing_rate": float(city_raw_train[feature].isna().mean()),
                "test_missing_rate": float(city_raw_test[feature].isna().mean()),
                "missing_rate_difference": float(
                    city_raw_test[feature].isna().mean() - city_raw_train[feature].isna().mean()
                ),
                "abs_missing_rate_difference": abs(
                    float(city_raw_test[feature].isna().mean() - city_raw_train[feature].isna().mean())
                ),
            }
        )

feature_shift = pd.DataFrame(shift_records).sort_values(
    ["city", "abs_standardized_mean_diff"], ascending=[True, False]
)
missing_shift = pd.DataFrame(missing_records).sort_values(
    ["city", "abs_missing_rate_difference"], ascending=[True, False]
)
feature_shift.to_csv(feature_shift_path, index=False)
missing_shift.to_csv(missing_shift_path, index=False)


### Domain classifier
fold_score_records = []
score_records = []
feature_importance_records = []
prediction_frames = []

model_params = {
    "objective": "binary",
    "n_estimators": args.n_estimators,
    "learning_rate": args.learning_rate,
    "num_leaves": args.num_leaves,
    "min_child_samples": args.min_child_samples,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": args.random_state,
    "seed": args.random_state,
    "data_random_seed": args.random_state,
    "feature_fraction_seed": args.random_state,
    "bagging_seed": args.random_state,
    "drop_seed": args.random_state,
    "extra_seed": args.random_state,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbose": -1,
    "importance_type": "gain",
}

for city, city_data in data.groupby("city", sort=False):
    city_data = city_data.sort_values("week_start_date").reset_index(drop=True)
    target = city_data["domain_is_train"].astype(int)
    class_counts = target.value_counts()
    city_splits = min(args.n_splits, int(class_counts.min()))

    if city_splits < 2:
        raise ValueError(f"City {city} does not have enough rows in both domains for CV.")

    for scenario, feature_columns in scenarios.items():
        x_data = city_data[feature_columns]
        splitter = StratifiedKFold(
            n_splits=city_splits,
            shuffle=True,
            random_state=args.random_state,
        )
        scenario_predictions = city_data[["city", "year", "weekofyear", "week_start_date", "source"]].copy()
        scenario_predictions["scenario"] = scenario
        scenario_predictions["domain_is_train"] = target.to_numpy()
        scenario_predictions["train_probability"] = np.nan
        scenario_predictions["fold"] = -1
        importance_by_fold = []

        for fold_id, (fit_index, validate_index) in enumerate(splitter.split(x_data, target), start=1):
            model = LGBMClassifier(**model_params)
            model.fit(x_data.iloc[fit_index], target.iloc[fit_index])

            validation_probability = model.predict_proba(x_data.iloc[validate_index])[:, 1]
            validation_target = target.iloc[validate_index]
            validation_prediction = (validation_probability >= 0.5).astype(int)

            fold_auc = float(roc_auc_score(validation_target, validation_probability))
            fold_average_precision = float(
                average_precision_score(validation_target, validation_probability)
            )
            fold_balanced_accuracy = float(
                balanced_accuracy_score(validation_target, validation_prediction)
            )
            fold_log_loss = float(log_loss(validation_target, validation_probability))

            fold_score_records.append(
                {
                    "city": city,
                    "scenario": scenario,
                    "fold": fold_id,
                    "rows": int(len(validate_index)),
                    "train_rows": int(validation_target.sum()),
                    "test_rows": int(len(validation_target) - validation_target.sum()),
                    "roc_auc": fold_auc,
                    "average_precision": fold_average_precision,
                    "balanced_accuracy": fold_balanced_accuracy,
                    "log_loss": fold_log_loss,
                }
            )

            scenario_predictions.loc[validate_index, "train_probability"] = validation_probability
            scenario_predictions.loc[validate_index, "fold"] = fold_id
            importance_by_fold.append(model.feature_importances_)

        fold_scores_for_scenario = pd.DataFrame(
            [
                record
                for record in fold_score_records
                if record["city"] == city and record["scenario"] == scenario
            ]
        )
        mean_importance = np.mean(np.vstack(importance_by_fold), axis=0)
        total_importance = float(mean_importance.sum())

        score_records.append(
            {
                "city": city,
                "scenario": scenario,
                "rows": int(len(city_data)),
                "train_rows": int(class_counts.get(1, 0)),
                "test_rows": int(class_counts.get(0, 0)),
                "feature_count": int(len(feature_columns)),
                "folds": int(city_splits),
                "roc_auc_mean": float(fold_scores_for_scenario["roc_auc"].mean()),
                "roc_auc_std": float(fold_scores_for_scenario["roc_auc"].std(ddof=1)),
                "average_precision_mean": float(
                    fold_scores_for_scenario["average_precision"].mean()
                ),
                "balanced_accuracy_mean": float(
                    fold_scores_for_scenario["balanced_accuracy"].mean()
                ),
                "log_loss_mean": float(fold_scores_for_scenario["log_loss"].mean()),
            }
        )

        for feature, importance in zip(feature_columns, mean_importance):
            feature_importance_records.append(
                {
                    "city": city,
                    "scenario": scenario,
                    "feature": feature,
                    "mean_gain_importance": float(importance),
                    "importance_share": (
                        float(importance / total_importance) if total_importance > 0 else 0.0
                    ),
                }
            )

        prediction_frames.append(scenario_predictions)

scores = pd.DataFrame(score_records).sort_values(["city", "scenario"])
fold_scores = pd.DataFrame(fold_score_records).sort_values(["city", "scenario", "fold"])
feature_importance = pd.DataFrame(feature_importance_records).sort_values(
    ["city", "scenario", "mean_gain_importance"], ascending=[True, True, False]
)
predictions = pd.concat(prediction_frames, ignore_index=True)

scores.to_csv(scores_path, index=False)
fold_scores.to_csv(fold_scores_path, index=False)
feature_importance.to_csv(feature_importance_path, index=False)
predictions.to_csv(prediction_path, index=False)


### Plot
if not args.skip_plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = scores.copy()
    plot_data["label"] = plot_data["city"] + " | " + plot_data["scenario"]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(plot_data["label"], plot_data["roc_auc_mean"], color=["#4c78a8", "#72b7b2"] * 2)
    axis.axhline(0.5, color="#444444", linewidth=1, linestyle="--", label="random")
    axis.axhline(0.8, color="#b279a2", linewidth=1, linestyle=":", label="strong shift")
    axis.set_ylim(0.45, 1.0)
    axis.set_ylabel("Mean ROC AUC")
    axis.set_title("Adversarial Train/Test Distinguishability")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


### Interpretation report
interpretation_lines = [
    "# Adversarial Validation Interpretation",
    "",
    "This diagnostic trains city-specific classifiers to predict whether a row came from the original training features (`1`) or the competition test features (`0`). It reads only feature CSVs, never loads labels, and rejects columns containing `total_cases`.",
    "",
    "The default scenarios exclude absolute time fields (`year`, `week_start_date`, and raw `weekofyear`) from model inputs. `weather_only` uses only weather, climate, vegetation, and humidity columns. `weather_plus_seasonality` adds cyclic `weekofyear` sine/cosine features.",
    "",
    "Interpretation guide: ROC AUC near 0.50 means the classifier cannot separate train from test. Around 0.65-0.80 means meaningful covariate shift. Above 0.80 means the train/test feature distributions are highly distinguishable.",
    "",
    "## Results",
    "",
]

for _, score in scores.iterrows():
    if score["roc_auc_mean"] < 0.65:
        conclusion = "weak or negligible distinguishability"
    elif score["roc_auc_mean"] < 0.80:
        conclusion = "meaningful distinguishability"
    else:
        conclusion = "strong distinguishability"

    top_importance = feature_importance[
        (feature_importance["city"] == score["city"])
        & (feature_importance["scenario"] == score["scenario"])
    ].head(5)
    top_shift = feature_shift[feature_shift["city"] == score["city"]].head(5)

    interpretation_lines.extend(
        [
            f"### {score['city']} - {score['scenario']}",
            "",
            f"- ROC AUC: {score['roc_auc_mean']:.3f} +/- {score['roc_auc_std']:.3f} across {int(score['folds'])} folds ({conclusion}).",
            f"- Balanced accuracy: {score['balanced_accuracy_mean']:.3f}.",
            f"- Rows: {int(score['train_rows'])} train and {int(score['test_rows'])} test.",
            "- Top classifier features: "
            + ", ".join(
                f"{row.feature} ({row.importance_share:.1%})"
                for row in top_importance.itertuples(index=False)
            ),
            "- Largest direct mean shifts: "
            + ", ".join(
                f"{row.feature} (SMD {row.standardized_mean_diff:.2f})"
                for row in top_shift.itertuples(index=False)
            ),
            "",
        ]
    )

strong_rows = scores[scores["roc_auc_mean"] >= 0.80]
meaningful_rows = scores[(scores["roc_auc_mean"] >= 0.65) & (scores["roc_auc_mean"] < 0.80)]

interpretation_lines.extend(
    [
        "## Project Takeaway",
        "",
    ]
)

if len(strong_rows) > 0:
    interpretation_lines.append(
        "At least one city/scenario is strongly distinguishable, so train/test covariates are not IID. This does not make the competition pure gambling, but it does mean random validation would be misleading."
    )
elif len(meaningful_rows) > 0:
    interpretation_lines.append(
        "The diagnostic finds meaningful train/test shift. This supports strict forward-time validation and city-specific model selection rather than random validation."
    )
else:
    interpretation_lines.append(
        "The diagnostic does not find strong covariate shift from the included inputs. Future-time validation is still required because target dynamics can drift even when covariates look similar."
    )

interpretation_lines.extend(
    [
        "",
        "Use this as a diagnostic only. It should not replace forward-chaining dengue validation, and the classifier features should not be optimized directly unless they also improve future-holdout MAE.",
        "",
        "Generated files:",
        f"- `{scores_path}`",
        f"- `{fold_scores_path}`",
        f"- `{feature_importance_path}`",
        f"- `{feature_shift_path}`",
        f"- `{missing_shift_path}`",
        f"- `{prediction_path}`",
    ]
)

if not args.skip_plot:
    interpretation_lines.append(f"- `{plot_path}`")

interpretation_path.write_text("\n".join(interpretation_lines) + "\n")

print(f"Wrote scores to {scores_path}")
print(f"Wrote interpretation to {interpretation_path}")
