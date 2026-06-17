"""DengAI pipeline entrypoint from raw CSV import toward submission generation.

Example:
    python3 oliver/main_informed.py
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ["MPLCONFIGDIR"] = "/private/tmp/dengai_main_matplotlib"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd


script_name = Path(__file__).stem
default_output_dir = Path(__file__).resolve().parent / "outputs" / script_name
default_output_dir_label = f"oliver/outputs/{script_name}"


### CLI arguments
parser = argparse.ArgumentParser(
    description="Run the DengAI pipeline from data import through submission generation."
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
    "--labels-csv",
    default="data/dengue_labels_train.csv",
    help="Path to the training label CSV. Default: data/dengue_labels_train.csv",
)
parser.add_argument(
    "--submission-format-csv",
    default="data/submission_format.csv",
    help="Path to the submission format CSV. Default: data/submission_format.csv",
)
parser.add_argument(
    "--output-dir",
    default=default_output_dir,
    help=f"Directory for generated pipeline outputs. Default: {default_output_dir_label}",
)
parser.add_argument(
    "--preprocessed-train-output",
    default="preprocessed_train.csv",
    help="Filename for preprocessed and engineered training data. Default: preprocessed_train.csv",
)
parser.add_argument(
    "--preprocessed-test-output",
    default="preprocessed_test.csv",
    help="Filename for preprocessed and engineered test features. Default: preprocessed_test.csv",
)
parser.add_argument(
    "--missing-summary-output",
    default="missing_summary.csv",
    help="Filename for missing-value counts before and after preprocessing. Default: missing_summary.csv",
)
parser.add_argument(
    "--submission-output",
    default="submission.csv",
    help="Filename for final submission predictions. Default: submission.csv",
)
parser.add_argument(
    "--validation-output",
    default="validation_scores.csv",
    help="Filename for expanding-year validation scores. Default: validation_scores.csv",
)
parser.add_argument(
    "--validation-predictions-output",
    default="validation_predictions.csv",
    help="Filename for row-level validation predictions. Default: validation_predictions.csv",
)
parser.add_argument(
    "--feature-importance-output",
    default="feature_importance.csv",
    help="Filename for final city model feature importance. Default: feature_importance.csv",
)
parser.add_argument(
    "--loss-selection-output",
    default="loss_selection_scores.csv",
    help="Filename for city-level loss candidate validation scores. Default: loss_selection_scores.csv",
)
parser.add_argument(
    "--experiment-log-output",
    default="experiment_log.jsonl",
    help="Append-only JSONL log for run config and validation results. Default: experiment_log.jsonl",
)
parser.add_argument(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for LightGBM. Default: 42",
)
parser.add_argument(
    "--losses",
    nargs="+",
    choices=["tweedie", "poisson", "l1"],
    default=["tweedie", "poisson", "l1"],
    help="LightGBM loss candidates to evaluate per city. Default: tweedie poisson l1",
)
parser.add_argument(
    "--target-transform",
    choices=["none", "log1p"],
    default="none",
    help="Optional transform applied to total_cases during fitting. Default: none",
)
parser.add_argument(
    "--tweedie-variance-power",
    type=float,
    default=1.3,
    help="LightGBM Tweedie variance power. Default: 1.3",
)
parser.add_argument(
    "--tune-hyperparameters",
    action="store_true",
    help="Use Optuna to tune separate LightGBM hyperparameters for each city.",
)
parser.add_argument(
    "--optuna-trials",
    type=int,
    default=50,
    help="Optuna trials per city when tuning is enabled. Default: 50",
)
parser.add_argument(
    "--optuna-timeout",
    type=int,
    default=0,
    help="Optuna timeout in seconds per city. Use 0 for no timeout. Default: 0",
)
parser.add_argument(
    "--optuna-output",
    default="optuna_trials.csv",
    help="Filename for Optuna trial results. Default: optuna_trials.csv",
)
parser.add_argument(
    "--best-params-output",
    default="best_params.csv",
    help="Filename for best per-city model parameters. Default: best_params.csv",
)
parser.add_argument(
    "--skip-csv-previews",
    action="store_true",
    help="Skip PNG table previews for generated CSV outputs.",
)
parser.add_argument(
    "--csv-preview-dir",
    default="csv_previews",
    help="Subdirectory under the output dir for CSV table preview PNGs. Default: csv_previews",
)
parser.add_argument(
    "--csv-preview-max-rows",
    type=int,
    default=30,
    help="Maximum rows shown in each CSV preview image. Default: 30",
)
parser.add_argument(
    "--csv-preview-max-cols",
    type=int,
    default=12,
    help="Maximum columns shown in each CSV preview image. Default: 12",
)
parser.add_argument(
    "--csv-preview-dpi",
    type=int,
    default=150,
    help="Resolution for CSV preview images. Default: 150",
)
args = parser.parse_args()

try:
    from lightgbm import LGBMRegressor
except ImportError as error:
    raise ImportError(
        "LightGBM is required for the modeling pipeline. Install the project dependencies "
        "from the root pyproject.toml, for example: python3 -m pip install -e ."
    ) from error


### Configuration
train_features_path = Path(args.train_features_csv)
test_features_path = Path(args.test_features_csv)
labels_path = Path(args.labels_csv)
submission_format_path = Path(args.submission_format_csv)
output_dir = Path(args.output_dir)
preprocessed_train_path = output_dir / args.preprocessed_train_output
preprocessed_test_path = output_dir / args.preprocessed_test_output
missing_summary_path = output_dir / args.missing_summary_output
submission_path = output_dir / args.submission_output
validation_scores_path = output_dir / args.validation_output
validation_predictions_path = output_dir / args.validation_predictions_output
feature_importance_path = output_dir / args.feature_importance_output
loss_selection_path = output_dir / args.loss_selection_output
experiment_log_path = output_dir / args.experiment_log_output
optuna_trials_path = output_dir / args.optuna_output
best_params_path = output_dir / args.best_params_output
csv_preview_dir = output_dir / args.csv_preview_dir
merge_keys = ["city", "year", "weekofyear"]
identifier_columns = {"city", "year", "weekofyear", "week_start_date", "split", "total_cases"}
loss_objectives = {
    "tweedie": "tweedie",
    "poisson": "poisson",
    "l1": "regression_l1",
}
candidate_losses = list(dict.fromkeys(args.losses))
feature_variant = "iq_plos_informed_windows"
matplotlib_config_dir = output_dir / ".matplotlib"
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)

from sklearn.metrics import mean_absolute_error

if not 1.0 < args.tweedie_variance_power < 2.0:
    raise ValueError("--tweedie-variance-power must be between 1 and 2 for Tweedie regression.")

if args.optuna_trials < 1:
    raise ValueError("--optuna-trials must be at least 1.")

if args.optuna_timeout < 0:
    raise ValueError("--optuna-timeout must be non-negative.")

if args.csv_preview_max_rows < 1:
    raise ValueError("--csv-preview-max-rows must be at least 1.")

if args.csv_preview_max_cols < 1:
    raise ValueError("--csv-preview-max-cols must be at least 1.")

if args.csv_preview_dpi < 1:
    raise ValueError("--csv-preview-dpi must be at least 1.")

random.seed(args.random_state)
np.random.seed(args.random_state)
os.environ["PYTHONHASHSEED"] = str(args.random_state)

deterministic_lightgbm_params = {
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
}

if args.tune_hyperparameters:
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as error:
        raise ImportError(
            "Optuna is required when --tune-hyperparameters is used. Install the project "
            "dependencies from pyproject.toml, for example: python3 -m pip install -e ."
        ) from error
else:
    optuna = None


### Data import
train_features = pd.read_csv(train_features_path, parse_dates=["week_start_date"])
test_features = pd.read_csv(test_features_path, parse_dates=["week_start_date"])
labels = pd.read_csv(labels_path)
submission_format = pd.read_csv(submission_format_path)

if list(train_features.columns) != list(test_features.columns):
    raise ValueError("Train and test feature CSVs must have the same columns.")

if train_features[merge_keys].duplicated().any() or test_features[merge_keys].duplicated().any():
    raise ValueError("Feature rows must be unique by city, year, and weekofyear.")

if labels[merge_keys].duplicated().any():
    raise ValueError("Label rows must be unique by city, year, and weekofyear.")

if not submission_format[merge_keys].equals(test_features[merge_keys]):
    raise ValueError("Submission format rows must match the test feature index order.")

train_data = train_features.merge(labels, on=merge_keys, how="left", validate="one_to_one")
if train_data["total_cases"].isna().any():
    raise ValueError("Merged training data contains missing total_cases values.")

train_data["split"] = "train"
test_data = test_features.copy()
test_data["split"] = "test"

train_data["_row_order"] = range(len(train_data))
test_data["_row_order"] = range(len(test_data))
data = pd.concat([train_data, test_data], ignore_index=True, sort=False)


### 1. Preprocessing

### 1.1 Interpolation
# Linear interpolation is a transparent first baseline for weekly weather features.
# It is applied city by city so the two city time series stay completely separate.
numeric_feature_columns = [
    column
    for column in train_features.columns
    if column not in identifier_columns and pd.api.types.is_numeric_dtype(train_features[column])
]
missing_before = data[numeric_feature_columns].isna().sum()

interpolated_city_data = []
for city, city_data in data.groupby("city", sort=False):
    city_data = city_data.sort_values("week_start_date").copy()
    city_data[numeric_feature_columns] = city_data[numeric_feature_columns].interpolate(
        method="linear",
        limit_direction="both",
    )
    interpolated_city_data.append(city_data)

data = pd.concat(interpolated_city_data, ignore_index=True)
missing_after = data[numeric_feature_columns].isna().sum()


### 2. Feature engineering
engineered_feature_columns = []
engineered_feature_data = {}
iq_informed_feature_columns = []


### 2.1 Vegetation summary
# Mean NDVI keeps a compact vegetation proxy while preserving the quadrant columns.
ndvi_columns = ["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]
data["ndvi_mean"] = data[ndvi_columns].mean(axis=1)
engineered_feature_columns.append("ndvi_mean")
iq_informed_feature_columns.append("ndvi_mean")


### 2.2 Seasonality
# Weekly dengue counts are strongly seasonal; sine/cosine keep week 52 close to week 1.
week_angle = 2 * math.pi * data["weekofyear"] / 52.0
engineered_feature_data["weekofyear_sin"] = week_angle.map(math.sin)
engineered_feature_data["weekofyear_cos"] = week_angle.map(math.cos)
engineered_feature_columns.extend(["weekofyear_sin", "weekofyear_cos"])

month_angle = 2 * math.pi * data["week_start_date"].dt.month / 12.0
engineered_feature_data["month_sin"] = month_angle.map(math.sin)
engineered_feature_data["month_cos"] = month_angle.map(math.cos)
iq_season_weeks = (data["weekofyear"] >= 37) | (data["weekofyear"] <= 13)
engineered_feature_data["iq_is_dengue_season"] = iq_season_weeks.astype(int)
engineered_feature_data["iq_weeks_since_season_start"] = (data["weekofyear"] - 37) % 52
engineered_feature_data["iq_weeks_to_december_peak"] = np.minimum(
    (data["weekofyear"] - 52) % 52,
    (52 - data["weekofyear"]) % 52,
)
iq_informed_feature_columns.extend(
    [
        "month_sin",
        "month_cos",
        "iq_is_dengue_season",
        "iq_weeks_since_season_start",
        "iq_weeks_to_december_peak",
    ]
)
engineered_feature_columns.extend(
    [
        "month_sin",
        "month_cos",
        "iq_is_dengue_season",
        "iq_weeks_since_season_start",
        "iq_weeks_to_december_peak",
    ]
)


### 2.3 Lagged weather features
# Lag choices come from the lagged-feature correlation analysis:
# San Juan is strongest around 6-12 week delayed temperature/humidity.
# Iquitos is weaker and mostly short-lag humidity/dew-point/min-temperature.
weather_lag_plan = {
    "reanalysis_specific_humidity_g_per_kg": [1, 2, 3, 5, 6, 8, 10, 12],
    "reanalysis_dew_point_temp_k": [1, 2, 3, 5, 6, 8, 10, 12],
    "reanalysis_min_air_temp_k": [1, 2, 3, 5, 6, 8, 10, 12],
    "reanalysis_air_temp_k": [6, 8, 10, 12],
    "reanalysis_avg_temp_k": [6, 8, 10, 12],
    "reanalysis_max_air_temp_k": [6, 8, 10, 12],
    "station_avg_temp_c": [2, 4, 5, 6, 8, 10, 12],
    "station_min_temp_c": [1, 2, 3, 4, 6, 8, 10, 12],
    "station_max_temp_c": [4, 6, 8, 10, 12],
    "ndvi_sw": [10, 11, 14],
    "ndvi_nw": [10, 11, 14],
    "ndvi_mean": [10, 11, 14],
    "precipitation_amt_mm": [2, 3],
    "reanalysis_sat_precip_amt_mm": [2, 3],
}

for feature, lags in weather_lag_plan.items():
    for lag in lags:
        column = f"{feature}_lag_{lag}"
        engineered_feature_data[column] = data.groupby("city", sort=False)[feature].shift(lag)
        engineered_feature_columns.append(column)
        if feature == "ndvi_mean":
            iq_informed_feature_columns.append(column)


### 2.4 Rolling weather summaries
# Cumulative weekly rolling means summarize recent exposure over the last N available weeks.
weather_rolling_plan = {
    "reanalysis_specific_humidity_g_per_kg": [3, 4, 6, 8, 10, 12, 14],
    "reanalysis_dew_point_temp_k": [3, 4, 6, 8, 10, 12, 14],
    "reanalysis_min_air_temp_k": [3, 4, 5, 6, 7, 8, 10, 12, 14],
    "reanalysis_air_temp_k": [8, 10, 12, 14],
    "reanalysis_avg_temp_k": [8, 10, 12, 14],
    "reanalysis_max_air_temp_k": [8, 10, 12, 14],
    "station_avg_temp_c": [3, 4, 5, 6, 8, 10, 12, 14],
    "station_min_temp_c": [3, 4, 5, 6, 8, 10, 12, 14],
    "station_max_temp_c": [5, 6, 8, 10, 12, 14],
    "ndvi_mean": [4, 8, 12, 14],
    "precipitation_amt_mm": [4, 7, 10, 14],
    "reanalysis_sat_precip_amt_mm": [4, 7, 10, 14],
}

for feature, rolling_windows in weather_rolling_plan.items():
    for window in rolling_windows:
        column = f"{feature}_rolling_{window}_mean"
        engineered_feature_data[column] = (
            data.groupby("city", sort=False)[feature]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        engineered_feature_columns.append(column)
        if feature == "ndvi_mean":
            iq_informed_feature_columns.append(column)


### 2.5 Iquitos lag-window summaries
# Windowed summaries smear climate exposure over biologically plausible delays.
iq_lag_window_plan = {
    "station_min_temp_c": [(0, 2, ["mean"]), (2, 6, ["mean"]), (6, 12, ["mean"]), (12, 17, ["mean"])],
    "reanalysis_min_air_temp_k": [(0, 2, ["mean"]), (2, 6, ["mean"]), (6, 12, ["mean"]), (12, 17, ["mean"])],
    "reanalysis_dew_point_temp_k": [(0, 4, ["mean"]), (4, 8, ["mean"]), (8, 17, ["mean"])],
    "reanalysis_relative_humidity_percent": [(0, 4, ["mean"]), (4, 8, ["mean"]), (8, 17, ["mean"])],
    "reanalysis_specific_humidity_g_per_kg": [(0, 4, ["mean"]), (4, 8, ["mean"]), (8, 17, ["mean"])],
    "station_diur_temp_rng_c": [(0, 8, ["mean"]), (8, 17, ["mean"])],
    "reanalysis_tdtr_k": [(0, 8, ["mean"]), (8, 17, ["mean"])],
    "precipitation_amt_mm": [(2, 8, ["sum", "max"]), (8, 17, ["sum", "max"])],
    "reanalysis_sat_precip_amt_mm": [(2, 8, ["sum", "max"]), (8, 17, ["sum", "max"])],
    "reanalysis_precip_amt_kg_per_m2": [(2, 8, ["sum", "max"]), (8, 17, ["sum", "max"])],
}

iq_informed_feature_columns.extend(["station_diur_temp_rng_c", "reanalysis_tdtr_k"])
for feature, windows in iq_lag_window_plan.items():
    if feature not in data.columns:
        continue

    grouped_feature = data.groupby("city", sort=False)[feature]
    for start_lag, end_lag, aggregations in windows:
        shifted_feature = grouped_feature.shift(start_lag)
        rolling_window = end_lag - start_lag + 1
        rolling_shifted_feature = shifted_feature.groupby(data["city"], sort=False).rolling(
            window=rolling_window,
            min_periods=1,
        )

        if "mean" in aggregations:
            column = f"{feature}_lag_{start_lag}_{end_lag}_mean"
            engineered_feature_data[column] = rolling_shifted_feature.mean().reset_index(
                level=0,
                drop=True,
            )
            engineered_feature_columns.append(column)
            iq_informed_feature_columns.append(column)

        if "sum" in aggregations:
            column = f"{feature}_lag_{start_lag}_{end_lag}_sum"
            engineered_feature_data[column] = rolling_shifted_feature.sum().reset_index(
                level=0,
                drop=True,
            )
            engineered_feature_columns.append(column)
            iq_informed_feature_columns.append(column)

        if "max" in aggregations:
            column = f"{feature}_lag_{start_lag}_{end_lag}_max"
            engineered_feature_data[column] = rolling_shifted_feature.max().reset_index(
                level=0,
                drop=True,
            )
            engineered_feature_columns.append(column)
            iq_informed_feature_columns.append(column)


data = pd.concat([data, pd.DataFrame(engineered_feature_data, index=data.index)], axis=1)


### Output export
output_dir.mkdir(parents=True, exist_ok=True)

missing_summary = pd.DataFrame(
    {
        "feature": numeric_feature_columns,
        "missing_before": missing_before.reindex(numeric_feature_columns).to_numpy(),
        "missing_after": missing_after.reindex(numeric_feature_columns).to_numpy(),
    }
)
missing_summary.to_csv(missing_summary_path, index=False)

preprocessed_train = (
    data[data["split"] == "train"]
    .sort_values("_row_order")
    .drop(columns=["split", "_row_order"])
)
preprocessed_test = (
    data[data["split"] == "test"]
    .sort_values("_row_order")
    .drop(columns=["split", "_row_order", "total_cases"], errors="ignore")
)

preprocessed_train.to_csv(preprocessed_train_path, index=False)
preprocessed_test.to_csv(preprocessed_test_path, index=False)


### 3. Model feature sets
seasonality_feature_columns = ["weekofyear_sin", "weekofyear_cos"]

sj_weather_roots = [
    "reanalysis_air_temp_k",
    "reanalysis_avg_temp_k",
    "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k",
    "reanalysis_min_air_temp_k",
    "reanalysis_relative_humidity_percent",
    "reanalysis_specific_humidity_g_per_kg",
    "station_avg_temp_c",
    "station_max_temp_c",
    "station_min_temp_c",
]
iq_weather_roots = [
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "reanalysis_min_air_temp_k",
    "station_min_temp_c",
    "station_avg_temp_c",
    "station_max_temp_c",
    "ndvi_ne",
    "ndvi_nw",
    "ndvi_se",
    "ndvi_sw",
    "precipitation_amt_mm",
    "reanalysis_sat_precip_amt_mm",
    "reanalysis_precip_amt_kg_per_m2",
]

city_feature_columns = {}
for city in ["sj", "iq"]:
    weather_feature_columns = []

    if city == "sj":
        for feature in sj_weather_roots:
            if feature in data.columns:
                weather_feature_columns.append(feature)
            for column in engineered_feature_columns:
                if column.startswith(f"{feature}_lag_"):
                    lag_suffix = column.removeprefix(f"{feature}_lag_")
                    if lag_suffix.isdigit():
                        weather_feature_columns.append(column)
                elif column.startswith(f"{feature}_rolling_"):
                    weather_feature_columns.append(column)

    if city == "iq":
        short_lag_suffixes = ["lag_1", "lag_2", "lag_3", "lag_5", "lag_0_3_mean", "lag_1_3_mean"]
        station_suffixes = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_6", "lag_1_3_mean"]
        ndvi_suffixes = ["lag_10", "lag_11", "lag_14"]
        precip_suffixes = ["lag_2", "lag_3", "lag_0_3_mean", "lag_1_3_mean", "lag_2_5_mean"]
        short_rolling_suffixes = [
            "rolling_3_mean",
            "rolling_4_mean",
            "rolling_5_mean",
            "rolling_6_mean",
            "rolling_7_mean",
            "rolling_8_mean",
        ]
        station_rolling_suffixes = [
            "rolling_3_mean",
            "rolling_4_mean",
            "rolling_5_mean",
            "rolling_6_mean",
        ]
        precip_rolling_suffixes = [
            "rolling_4_mean",
            "rolling_7_mean",
            "rolling_10_mean",
            "rolling_14_mean",
        ]

        for feature in iq_weather_roots:
            if feature in data.columns:
                weather_feature_columns.append(feature)

            for column in engineered_feature_columns:
                if not column.startswith(f"{feature}_"):
                    continue

                suffix = column.removeprefix(f"{feature}_")
                if feature.startswith("ndvi_") and suffix in ndvi_suffixes:
                    weather_feature_columns.append(column)
                elif "precip" in feature and suffix in precip_suffixes:
                    weather_feature_columns.append(column)
                elif "precip" in feature and suffix in precip_rolling_suffixes:
                    weather_feature_columns.append(column)
                elif feature.startswith("station_") and suffix in station_suffixes:
                    weather_feature_columns.append(column)
                elif feature.startswith("station_") and suffix in station_rolling_suffixes:
                    weather_feature_columns.append(column)
                elif feature.startswith("reanalysis_") and suffix in short_lag_suffixes:
                    weather_feature_columns.append(column)
                elif feature.startswith("reanalysis_") and suffix in short_rolling_suffixes:
                    weather_feature_columns.append(column)

    selected_columns = seasonality_feature_columns + weather_feature_columns
    if city == "iq":
        selected_columns = selected_columns + [
            column
            for column in iq_informed_feature_columns
            if column in data.columns
        ]

    city_feature_columns[city] = list(dict.fromkeys(selected_columns))


### 4. City loss selection and optional Optuna hyperparameter tuning
base_hyperparameters = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

train_model_data = data[data["split"] == "train"].copy()
test_model_data = data[data["split"] == "test"].copy()
city_model_params = {}
city_model_losses = {}
loss_selection_records = []
optuna_trial_records = []
best_param_records = []

def make_model_params(loss_name, hyperparameters, tweedie_variance_power=None):
    model_params = {
        "objective": loss_objectives[loss_name],
        **hyperparameters,
        **deterministic_lightgbm_params,
    }
    if loss_name == "tweedie":
        model_params["tweedie_variance_power"] = (
            args.tweedie_variance_power
            if tweedie_variance_power is None
            else tweedie_variance_power
        )
    return model_params


def transform_target(target):
    if args.target_transform == "log1p":
        return np.log1p(target)
    return target


def inverse_transform_prediction(prediction):
    if args.target_transform == "log1p":
        return np.expm1(prediction)
    return prediction


def get_validation_years(city_train_data):
    year_counts = city_train_data.groupby("year").size()
    full_years = [year for year, rows in year_counts.items() if rows == 52]
    first_year = int(city_train_data["year"].min())
    first_full_year_after_start = min(year for year in full_years if year > first_year)
    return [
        year
        for year in full_years
        if year > first_full_year_after_start
    ]


def score_city_params(city_train_data, feature_columns, validation_years, model_params):
    fold_maes = []
    for validation_year in validation_years:
        fold_train_data = city_train_data[city_train_data["year"] < validation_year].copy()
        fold_validation_data = city_train_data[city_train_data["year"] == validation_year].copy()

        model = LGBMRegressor(**model_params)
        model.fit(
            fold_train_data[feature_columns],
            transform_target(fold_train_data["total_cases"]),
        )

        raw_predictions = model.predict(fold_validation_data[feature_columns])
        fold_predictions = [
            int(round(max(0.0, float(inverse_transform_prediction(raw_prediction)))))
            for raw_prediction in raw_predictions
        ]
        fold_maes.append(mean_absolute_error(fold_validation_data["total_cases"], fold_predictions))

    return float(np.mean(fold_maes))


for city, city_train_data in train_model_data.groupby("city", sort=False):
    city_train_data = city_train_data.sort_values("week_start_date").copy()
    validation_years = get_validation_years(city_train_data)
    feature_columns = city_feature_columns[city]

    if args.tune_hyperparameters:
        sampler = optuna.samplers.TPESampler(seed=args.random_state)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=f"{city}_lightgbm_loss_search",
        )

        def objective(trial):
            loss_name = trial.suggest_categorical("loss", candidate_losses)
            trial_hyperparameters = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            tweedie_variance_power = None
            if loss_name == "tweedie":
                tweedie_variance_power = trial.suggest_float("tweedie_variance_power", 1.1, 1.8)

            trial_params = make_model_params(
                loss_name,
                trial_hyperparameters,
                tweedie_variance_power,
            )
            return score_city_params(
                city_train_data,
                feature_columns,
                validation_years,
                trial_params,
            )

        study.optimize(
            objective,
            n_trials=args.optuna_trials,
            timeout=args.optuna_timeout or None,
            catch=(Exception,),
            n_jobs=1,
        )

        for trial in study.trials:
            trial_loss = trial.params.get("loss")
            optuna_trial_records.append(
                {
                    "city": city,
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "mae": trial.value if trial.value is not None else np.nan,
                    "loss": trial_loss,
                    "objective": loss_objectives.get(trial_loss, ""),
                    **{
                        key: value
                        for key, value in trial.params.items()
                        if key != "loss"
                    },
                    "subsample_freq": 1,
                    **deterministic_lightgbm_params,
                }
            )

        completed_trials = study.get_trials(
            deepcopy=False,
            states=[optuna.trial.TrialState.COMPLETE],
        )
        if completed_trials:
            best_search_params = study.best_params.copy()
            best_loss = best_search_params.pop("loss")
            best_hyperparameters = {
                key: best_search_params[key]
                for key in base_hyperparameters
                if key in best_search_params
            }
            best_hyperparameters["subsample_freq"] = 1
            best_tweedie_variance_power = best_search_params.get("tweedie_variance_power")
            best_params = make_model_params(
                best_loss,
                best_hyperparameters,
                best_tweedie_variance_power,
            )
            city_model_losses[city] = best_loss
            city_model_params[city] = best_params
            best_param_records.append(
                {
                    "city": city,
                    "best_mae": study.best_value,
                    "loss": best_loss,
                    **best_params,
                }
            )

            for loss_name in candidate_losses:
                loss_trials = [
                    trial
                    for trial in completed_trials
                    if trial.params.get("loss") == loss_name
                ]
                best_loss_mae = (
                    min(trial.value for trial in loss_trials)
                    if loss_trials
                    else np.nan
                )
                loss_selection_records.append(
                    {
                        "city": city,
                        "loss": loss_name,
                        "objective": loss_objectives[loss_name],
                        "selection_mode": "optuna",
                        "n_trials": len(loss_trials),
                        "mae": best_loss_mae,
                        "selected": loss_name == best_loss,
                    }
                )
    else:
        city_loss_records = []
        city_candidate_params = {}
        for loss_name in candidate_losses:
            model_params = make_model_params(loss_name, base_hyperparameters)
            candidate_mae = score_city_params(
                city_train_data,
                feature_columns,
                validation_years,
                model_params,
            )
            city_candidate_params[loss_name] = model_params
            city_loss_records.append(
                {
                    "city": city,
                    "loss": loss_name,
                    "objective": loss_objectives[loss_name],
                    "selection_mode": "default_params",
                    "n_trials": 0,
                    "mae": candidate_mae,
                    "selected": False,
                }
            )

        best_loss_record = min(city_loss_records, key=lambda record: record["mae"])
        best_loss = best_loss_record["loss"]
        city_model_losses[city] = best_loss
        city_model_params[city] = city_candidate_params[best_loss]

        for record in city_loss_records:
            record["selected"] = record["loss"] == best_loss
            loss_selection_records.append(record)

optuna_trials = pd.DataFrame(optuna_trial_records)
best_params = pd.DataFrame(best_param_records)
loss_selection_scores = pd.DataFrame(loss_selection_records)
if args.tune_hyperparameters:
    optuna_trials.to_csv(optuna_trials_path, index=False)
    best_params.to_csv(best_params_path, index=False)


### 5. Expanding full-year validation
validation_score_records = []
validation_prediction_records = []
all_validation_actuals = []
all_validation_predictions = []

for city, city_train_data in train_model_data.groupby("city", sort=False):
    city_train_data = city_train_data.sort_values("week_start_date").copy()
    validation_years = get_validation_years(city_train_data)
    feature_columns = city_feature_columns[city]
    selected_loss = city_model_losses[city]

    for validation_year in validation_years:
        fold_train_data = city_train_data[city_train_data["year"] < validation_year].copy()
        fold_validation_data = city_train_data[city_train_data["year"] == validation_year].copy()

        model = LGBMRegressor(**city_model_params[city])
        model.fit(
            fold_train_data[feature_columns],
            transform_target(fold_train_data["total_cases"]),
        )

        fold_predictions = []
        raw_predictions = model.predict(fold_validation_data[feature_columns])

        for row, raw_prediction in zip(fold_validation_data.itertuples(index=False), raw_predictions):
            raw_prediction = float(inverse_transform_prediction(raw_prediction))
            clipped_prediction = max(0.0, raw_prediction)
            integer_prediction = int(round(clipped_prediction))

            fold_predictions.append(integer_prediction)

            validation_prediction_records.append(
                {
                    "city": row.city,
                    "loss": selected_loss,
                    "year": row.year,
                    "weekofyear": row.weekofyear,
                    "week_start_date": row.week_start_date,
                    "validation_year": validation_year,
                    "actual_total_cases": row.total_cases,
                    "predicted_total_cases": integer_prediction,
                    "predicted_total_cases_raw": clipped_prediction,
                }
            )

        fold_mae = mean_absolute_error(fold_validation_data["total_cases"], fold_predictions)
        all_validation_actuals.extend(fold_validation_data["total_cases"].tolist())
        all_validation_predictions.extend(fold_predictions)
        validation_score_records.append(
            {
                "city": city,
                "loss": selected_loss,
                "validation_year": validation_year,
                "train_rows": len(fold_train_data),
                "validation_rows": len(fold_validation_data),
                "mae": fold_mae,
            }
        )

if all_validation_actuals:
    validation_score_records.append(
        {
            "city": "all",
            "loss": "selected_by_city",
            "validation_year": "all",
            "train_rows": "",
            "validation_rows": len(all_validation_actuals),
            "mae": mean_absolute_error(all_validation_actuals, all_validation_predictions),
        }
    )

validation_scores = pd.DataFrame(validation_score_records)
validation_predictions = pd.DataFrame(validation_prediction_records)


### 6. Final city models and submission
submission = submission_format.copy()
submission_predictions = {}
feature_importance_records = []

for city, city_train_data in train_model_data.groupby("city", sort=False):
    city_train_data = city_train_data.sort_values("week_start_date").copy()
    city_test_data = test_model_data[test_model_data["city"] == city].sort_values("week_start_date").copy()
    feature_columns = city_feature_columns[city]
    selected_loss = city_model_losses[city]

    model = LGBMRegressor(**city_model_params[city])
    model.fit(
        city_train_data[feature_columns],
        transform_target(city_train_data["total_cases"]),
    )

    gain_importance = model.booster_.feature_importance(importance_type="gain")
    split_importance = model.booster_.feature_importance(importance_type="split")
    for feature, gain, split in zip(feature_columns, gain_importance, split_importance):
        feature_importance_records.append(
            {
                "city": city,
                "loss": selected_loss,
                "feature": feature,
                "importance_gain": gain,
                "importance_split": split,
            }
        )

    raw_predictions = model.predict(city_test_data[feature_columns])

    for row, raw_prediction in zip(city_test_data.itertuples(index=False), raw_predictions):
        raw_prediction = float(inverse_transform_prediction(raw_prediction))
        clipped_prediction = max(0.0, raw_prediction)
        integer_prediction = int(round(clipped_prediction))

        submission_predictions[(row.city, row.year, row.weekofyear)] = integer_prediction

submission["total_cases"] = [
    submission_predictions[(row.city, row.year, row.weekofyear)]
    for row in submission.itertuples(index=False)
]

if list(submission.columns) != ["city", "year", "weekofyear", "total_cases"]:
    raise ValueError("Submission columns must be city, year, weekofyear, total_cases.")

if len(submission) != len(submission_format):
    raise ValueError("Submission row count must match the submission format.")

if not submission[merge_keys].equals(submission_format[merge_keys]):
    raise ValueError("Submission row order must match the submission format.")

if submission["total_cases"].isna().any():
    raise ValueError("Submission contains missing total_cases predictions.")

if (submission["total_cases"] < 0).any():
    raise ValueError("Submission contains negative total_cases predictions.")

submission["total_cases"] = submission["total_cases"].astype(int)
feature_importance = pd.DataFrame(feature_importance_records).sort_values(
    ["city", "importance_gain"],
    ascending=[True, False],
)

validation_scores.to_csv(validation_scores_path, index=False)
validation_predictions.to_csv(validation_predictions_path, index=False)
feature_importance.to_csv(feature_importance_path, index=False)
loss_selection_scores.to_csv(loss_selection_path, index=False)
submission.to_csv(submission_path, index=False)


### 7. Quick experiment log
overall_validation_mae = None
overall_validation_rows = None
overall_validation = validation_scores[validation_scores["city"] == "all"]
if not overall_validation.empty:
    overall_validation_mae = float(overall_validation["mae"].iloc[0])
    overall_validation_rows = int(overall_validation["validation_rows"].iloc[0])

city_validation_mae = {}
for city, city_scores in validation_scores[validation_scores["city"] != "all"].groupby("city", sort=False):
    city_validation_mae[city] = float(city_scores["mae"].mean())

experiment_log_record = {
    "logged_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "run_label": "pipeline_run",
    "script": str(Path(__file__)),
    "config": {
        "train_features_csv": str(train_features_path),
        "test_features_csv": str(test_features_path),
        "labels_csv": str(labels_path),
        "submission_format_csv": str(submission_format_path),
        "random_state": args.random_state,
        "losses": candidate_losses,
        "target_transform": args.target_transform,
        "feature_variant": feature_variant,
        "tweedie_variance_power": args.tweedie_variance_power,
        "tune_hyperparameters": args.tune_hyperparameters,
        "optuna_trials": args.optuna_trials if args.tune_hyperparameters else 0,
        "optuna_timeout": args.optuna_timeout if args.tune_hyperparameters else 0,
    },
    "results": {
        "overall_validation_mae": overall_validation_mae,
        "overall_validation_rows": overall_validation_rows,
        "city_validation_mae": city_validation_mae,
        "selected_losses": city_model_losses,
        "feature_counts": {
            city: len(feature_columns)
            for city, feature_columns in city_feature_columns.items()
        },
        "model_params": city_model_params,
    },
    "outputs": {
        "validation_scores": str(validation_scores_path),
        "validation_predictions": str(validation_predictions_path),
        "feature_importance": str(feature_importance_path),
        "loss_selection_scores": str(loss_selection_path),
        "submission": str(submission_path),
    },
}
with experiment_log_path.open("a", encoding="utf-8") as experiment_log_file:
    experiment_log_file.write(json.dumps(experiment_log_record, sort_keys=True, default=str) + "\n")


### 8. CSV content previews
csv_preview_paths = []
if not args.skip_csv_previews:
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
    csv_preview_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_preview_outputs = [
        ("missing_summary", missing_summary_path, missing_summary),
        ("preprocessed_train", preprocessed_train_path, preprocessed_train),
        ("preprocessed_test", preprocessed_test_path, preprocessed_test),
        ("validation_scores", validation_scores_path, validation_scores),
        ("validation_predictions", validation_predictions_path, validation_predictions),
        ("feature_importance", feature_importance_path, feature_importance),
        ("loss_selection_scores", loss_selection_path, loss_selection_scores),
        ("submission", submission_path, submission),
    ]
    if args.tune_hyperparameters:
        csv_preview_outputs.extend(
            [
                ("optuna_trials", optuna_trials_path, optuna_trials),
                ("best_params", best_params_path, best_params),
            ]
        )

    for output_name, csv_path, csv_data in csv_preview_outputs:
        preview_data = csv_data.copy()
        original_rows, original_columns = preview_data.shape
        preview_data = preview_data.iloc[
            : args.csv_preview_max_rows,
            : args.csv_preview_max_cols,
        ].copy()

        if preview_data.empty:
            preview_data = pd.DataFrame({"message": [f"{csv_path.name} is empty"]})
        else:
            if original_columns > args.csv_preview_max_cols:
                preview_data["..."] = f"+{original_columns - args.csv_preview_max_cols} columns"
            if original_rows > args.csv_preview_max_rows:
                overflow_row = {column: "" for column in preview_data.columns}
                overflow_row[preview_data.columns[0]] = (
                    f"... +{original_rows - args.csv_preview_max_rows} rows"
                )
                preview_data = pd.concat(
                    [preview_data, pd.DataFrame([overflow_row])],
                    ignore_index=True,
                )

        for column in preview_data.columns:
            if pd.api.types.is_float_dtype(preview_data[column]):
                preview_data[column] = preview_data[column].round(4)

        preview_data = preview_data.where(pd.notna(preview_data), "")
        preview_data = preview_data.astype(str)
        preview_data = preview_data.apply(
            lambda column: column.map(
                lambda value: value if len(value) <= 32 else f"{value[:29]}..."
            )
        )

        figure_width = min(24, max(8, len(preview_data.columns) * 1.45))
        figure_height = min(28, max(3.2, (len(preview_data) + 2) * 0.32))
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        axis.axis("off")
        axis.set_title(
            (
                f"{csv_path.name} ({original_rows} rows x {original_columns} columns)\n"
                f"Showing first {min(original_rows, args.csv_preview_max_rows)} rows "
                f"and first {min(original_columns, args.csv_preview_max_cols)} columns"
            ),
            fontsize=11,
            fontweight="bold",
            pad=12,
        )

        table = axis.table(
            cellText=preview_data.to_numpy(),
            colLabels=preview_data.columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.0, 1.18)

        for (row_index, _column_index), cell in table.get_celld().items():
            cell.set_edgecolor("#d8dee9")
            cell.set_linewidth(0.35)
            if row_index == 0:
                cell.set_facecolor("#eceff4")
                cell.set_text_props(weight="bold", color="#2e3440")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f8f9fb")

        figure.tight_layout()
        preview_path = csv_preview_dir / f"{output_name}.png"
        figure.savefig(preview_path, dpi=args.csv_preview_dpi)
        plt.close(figure)
        csv_preview_paths.append(preview_path)


print("Pipeline completed through preprocessing, feature engineering, validation, and submission.")
print(f"Imported train rows: {len(train_data)}")
print(f"Imported test rows: {len(test_data)}")
print(f"Numeric feature missing values before interpolation: {int(missing_before.sum())}")
print(f"Numeric feature missing values after interpolation: {int(missing_after.sum())}")
print(f"Engineered feature columns: {len(engineered_feature_columns)}")
print(f"Feature variant: {feature_variant}")
print(
    "Selected losses: "
    + ", ".join(f"{city}={loss}" for city, loss in sorted(city_model_losses.items()))
)
print(f"Validation folds: {len(validation_scores[validation_scores['city'] != 'all'])}")
if not validation_scores.empty:
    overall_mae = validation_scores.loc[validation_scores["city"] == "all", "mae"]
    if not overall_mae.empty:
        print(f"Overall validation MAE: {overall_mae.iloc[0]:.4f}")
print(f"Saved missing summary: {missing_summary_path}")
print(f"Saved preprocessed and engineered train data: {preprocessed_train_path}")
print(f"Saved preprocessed and engineered test data: {preprocessed_test_path}")
print(f"Saved validation scores: {validation_scores_path}")
print(f"Saved validation predictions: {validation_predictions_path}")
print(f"Saved feature importance: {feature_importance_path}")
print(f"Saved loss selection scores: {loss_selection_path}")
print(f"Appended experiment log: {experiment_log_path}")
print(f"Saved submission: {submission_path}")
if csv_preview_paths:
    print("Saved CSV preview images:")
    for csv_preview_path in csv_preview_paths:
        print(f"- {csv_preview_path}")
