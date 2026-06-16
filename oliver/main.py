"""DengAI pipeline entrypoint from raw CSV import toward submission generation.

Example:
    python3 oliver/main.py
"""

import argparse
import math
from pathlib import Path

import pandas as pd


script_name = Path(__file__).stem
default_output_dir = Path(__file__).resolve().parent / "outputs" / script_name
default_output_dir_label = f"oliver/outputs/{script_name}"


### CLI arguments
parser = argparse.ArgumentParser(
    description="Run the DengAI pipeline from data import through feature engineering."
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
    help=f"Directory for generated intermediate files. Default: {default_output_dir_label}",
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
    "--target-lags",
    type=int,
    nargs="+",
    default=[1, 2, 3, 4, 8, 12],
    help="Past total_cases lags to engineer. Default: 1 2 3 4 8 12",
)
args = parser.parse_args()


### Configuration
train_features_path = Path(args.train_features_csv)
test_features_path = Path(args.test_features_csv)
labels_path = Path(args.labels_csv)
submission_format_path = Path(args.submission_format_csv)
output_dir = Path(args.output_dir)
preprocessed_train_path = output_dir / args.preprocessed_train_output
preprocessed_test_path = output_dir / args.preprocessed_test_output
missing_summary_path = output_dir / args.missing_summary_output
merge_keys = ["city", "year", "weekofyear"]
identifier_columns = {"city", "year", "weekofyear", "week_start_date", "split", "total_cases"}

if any(target_lag < 1 for target_lag in args.target_lags):
    raise ValueError("--target-lags must contain positive integers.")


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


### 2.1 Seasonality
# Weekly dengue counts are strongly seasonal; sine/cosine keep week 52 close to week 1.
week_angle = 2 * math.pi * data["weekofyear"] / 52.0
engineered_feature_data["weekofyear_sin"] = week_angle.map(math.sin)
engineered_feature_data["weekofyear_cos"] = week_angle.map(math.cos)
engineered_feature_columns.extend(["weekofyear_sin", "weekofyear_cos"])


### 2.2 Lagged weather features
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
    "precipitation_amt_mm": [2, 3],
    "reanalysis_sat_precip_amt_mm": [2, 3],
}

for feature, lags in weather_lag_plan.items():
    for lag in lags:
        column = f"{feature}_lag_{lag}"
        engineered_feature_data[column] = data.groupby("city", sort=False)[feature].shift(lag)
        engineered_feature_columns.append(column)


### 2.3 Rolling weather summaries
# Rolling windows summarize delayed exposure instead of adding every single lag.
weather_rolling_plan = {
    "reanalysis_specific_humidity_g_per_kg": [(0, 4), (1, 3), (5, 4), (7, 4), (9, 4)],
    "reanalysis_dew_point_temp_k": [(0, 4), (1, 3), (5, 4), (7, 4), (9, 4)],
    "reanalysis_min_air_temp_k": [(0, 4), (1, 3), (5, 4), (7, 4), (9, 4)],
    "station_avg_temp_c": [(1, 3), (5, 4), (7, 4), (9, 4)],
    "station_min_temp_c": [(1, 3), (5, 4), (7, 4), (9, 4)],
    "station_max_temp_c": [(5, 4), (7, 4), (9, 4)],
    "precipitation_amt_mm": [(0, 4), (1, 3), (2, 4)],
    "reanalysis_sat_precip_amt_mm": [(0, 4), (1, 3), (2, 4)],
}

for feature, windows in weather_rolling_plan.items():
    for start_lag, window in windows:
        column = f"{feature}_lag_{start_lag}_{start_lag + window - 1}_mean"
        shifted_feature = data.groupby("city", sort=False)[feature].shift(start_lag)
        engineered_feature_data[column] = (
            shifted_feature.groupby(data["city"], sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        engineered_feature_columns.append(column)


### 2.4 Lagged target features
# Target history is the strongest linear signal. Test rows only get values where past
# labels are already known; later test target lags must be filled during recursive prediction.
for target_lag in sorted(set(args.target_lags)):
    column = f"total_cases_lag_{target_lag}"
    engineered_feature_data[column] = data.groupby("city", sort=False)["total_cases"].shift(target_lag)
    engineered_feature_columns.append(column)

for window in [2, 4, 8]:
    shifted_target = data.groupby("city", sort=False)["total_cases"].shift(1)
    mean_column = f"total_cases_lag_1_{window}_mean"
    max_column = f"total_cases_lag_1_{window}_max"
    engineered_feature_data[mean_column] = (
        shifted_target.groupby(data["city"], sort=False)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    engineered_feature_data[max_column] = (
        shifted_target.groupby(data["city"], sort=False)
        .rolling(window=window, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )
    engineered_feature_columns.extend([mean_column, max_column])

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

print("Pipeline completed through preprocessing and feature engineering.")
print(f"Imported train rows: {len(train_data)}")
print(f"Imported test rows: {len(test_data)}")
print(f"Numeric feature missing values before interpolation: {int(missing_before.sum())}")
print(f"Numeric feature missing values after interpolation: {int(missing_after.sum())}")
print(f"Engineered feature columns: {len(engineered_feature_columns)}")
print(f"Saved missing summary: {missing_summary_path}")
print(f"Saved preprocessed and engineered train data: {preprocessed_train_path}")
print(f"Saved preprocessed and engineered test data: {preprocessed_test_path}")
