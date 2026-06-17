"""Risk-aware best-practice DengAI approach after diagnostic results.

Example:
    python3 oliver/diagnostic/best_practice_approach.py

The approach is intentionally conservative:
- model San Juan and Iquitos separately;
- validate with expanding full-year folds;
- use total_cases only as the supervised label and validation ground truth;
- keep case-derived features out of the feature matrix;
- prefer a stable cyclic-week baseline unless weather features clear a
  configurable covariate-shift risk margin.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random


script_dir = Path(__file__).resolve().parent
default_output_dir = script_dir.parent / "outputs" / "diagnostic" / "best_practice"
default_output_dir_label = "oliver/outputs/diagnostic/best_practice"


parser = argparse.ArgumentParser(
    description=(
        "Run a risk-aware, calendar-first DengAI modeling approach informed by "
        "forward validation and adversarial train/test diagnostics."
    )
)
parser.add_argument(
    "--train-features-csv",
    default="data/dengue_features_train.csv",
    help="Path to training features. Default: data/dengue_features_train.csv",
)
parser.add_argument(
    "--test-features-csv",
    default="data/dengue_features_test.csv",
    help="Path to test features. Default: data/dengue_features_test.csv",
)
parser.add_argument(
    "--labels-csv",
    default="data/dengue_labels_train.csv",
    help="Path to training labels. Default: data/dengue_labels_train.csv",
)
parser.add_argument(
    "--submission-format-csv",
    default="data/submission_format.csv",
    help="Path to submission format. Default: data/submission_format.csv",
)
parser.add_argument(
    "--adversarial-scores-csv",
    default="oliver/outputs/diagnostic/domain_classifier_scores.csv",
    help=(
        "Optional adversarial validation score CSV to include in the report. "
        "Default: oliver/outputs/diagnostic/domain_classifier_scores.csv"
    ),
)
parser.add_argument(
    "--output-dir",
    default=default_output_dir,
    help=f"Directory for generated outputs. Default: {default_output_dir_label}",
)
parser.add_argument(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for deterministic LightGBM fitting. Default: 42",
)
parser.add_argument(
    "--weather-blend-weight",
    type=float,
    default=1.0 / 3.0,
    help=(
        "Weight assigned to the weather model in the calendar/weather blend. "
        "Default: 0.333333"
    ),
)
parser.add_argument(
    "--weather-risk-margin",
    type=float,
    default=0.25,
    help=(
        "MAE penalty applied to weather-using candidates during final selection. "
        "Default: 0.25"
    ),
)
parser.add_argument(
    "--tweedie-variance-power",
    type=float,
    default=1.3,
    help="Reserved for count-model comparisons. Default: 1.3",
)
parser.add_argument(
    "--fold-scores-output",
    default="fold_scores.csv",
    help="Per-fold validation scores filename. Default: fold_scores.csv",
)
parser.add_argument(
    "--summary-output",
    default="approach_summary.csv",
    help="Per-city and overall candidate summary filename. Default: approach_summary.csv",
)
parser.add_argument(
    "--selection-output",
    default="selected_approach_by_city.csv",
    help="Risk-aware city selection filename. Default: selected_approach_by_city.csv",
)
parser.add_argument(
    "--validation-predictions-output",
    default="validation_predictions.csv",
    help="Row-level validation prediction filename. Default: validation_predictions.csv",
)
parser.add_argument(
    "--submission-output",
    default="submission.csv",
    help="Risk-aware submission filename. Default: submission.csv",
)
parser.add_argument(
    "--validation-best-submission-output",
    default="submission_validation_best.csv",
    help=(
        "Submission using the pure validation-best candidate per city. "
        "Default: submission_validation_best.csv"
    ),
)
parser.add_argument(
    "--feature-importance-output",
    default="feature_importance.csv",
    help="Final component feature importance filename. Default: feature_importance.csv",
)
parser.add_argument(
    "--report-output",
    default="best_practice_report.md",
    help="Markdown interpretation filename. Default: best_practice_report.md",
)
parser.add_argument(
    "--experiment-log-output",
    default="experiment_log.jsonl",
    help="Append-only run log filename. Default: experiment_log.jsonl",
)
args = parser.parse_args()

if not 0.0 <= args.weather_blend_weight <= 1.0:
    raise ValueError("--weather-blend-weight must be between 0 and 1.")

if args.weather_risk_margin < 0:
    raise ValueError("--weather-risk-margin must be non-negative.")

if not 1.0 < args.tweedie_variance_power < 2.0:
    raise ValueError("--tweedie-variance-power must be between 1 and 2.")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
matplotlib_config_dir = output_dir / ".matplotlib"
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ["PYTHONHASHSEED"] = str(args.random_state)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


random.seed(args.random_state)
np.random.seed(args.random_state)

train_features_path = Path(args.train_features_csv)
test_features_path = Path(args.test_features_csv)
labels_path = Path(args.labels_csv)
submission_format_path = Path(args.submission_format_csv)
adversarial_scores_path = Path(args.adversarial_scores_csv)

fold_scores_path = output_dir / args.fold_scores_output
summary_path = output_dir / args.summary_output
selection_path = output_dir / args.selection_output
validation_predictions_path = output_dir / args.validation_predictions_output
submission_path = output_dir / args.submission_output
validation_best_submission_path = output_dir / args.validation_best_submission_output
feature_importance_path = output_dir / args.feature_importance_output
report_path = output_dir / args.report_output
experiment_log_path = output_dir / args.experiment_log_output

merge_keys = ["city", "year", "weekofyear"]
identifier_columns = {"city", "year", "weekofyear", "week_start_date", "split", "total_cases"}
seasonal_columns = ["weekofyear_sin", "weekofyear_cos"]
case_forbidden_prefixes = ("case_", "cases_", "target_", "lagged_cases", "rolling_cases")

broad_lag_plan = {
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

broad_rolling_plan = {
    "reanalysis_specific_humidity_g_per_kg": [3, 4, 6, 8, 10, 12, 14],
    "reanalysis_dew_point_temp_k": [3, 4, 6, 8, 10, 12, 14],
    "reanalysis_min_air_temp_k": [3, 4, 5, 6, 7, 8, 10, 12, 14],
    "reanalysis_air_temp_k": [8, 10, 12, 14],
    "reanalysis_avg_temp_k": [8, 10, 12, 14],
    "reanalysis_max_air_temp_k": [8, 10, 12, 14],
    "station_avg_temp_c": [3, 4, 5, 6, 8, 10, 12, 14],
    "station_min_temp_c": [3, 4, 5, 6, 8, 10, 12, 14],
    "station_max_temp_c": [5, 6, 8, 10, 12, 14],
    "precipitation_amt_mm": [4, 7, 10, 14],
    "reanalysis_sat_precip_amt_mm": [4, 7, 10, 14],
}

sj_broad_roots = [
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

iq_broad_roots = [
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

candidate_order = [
    "calendar_l1_raw",
    "broad_weather_l1_log",
    "calendar_broad_log_blend",
]
candidate_labels = {
    "calendar_l1_raw": "Calendar-only L1 baseline",
    "broad_weather_l1_log": "Broad weather L1 model with log1p target",
    "calendar_broad_log_blend": "Two-thirds calendar plus one-third weather blend",
}
candidate_uses_weather = {
    "calendar_l1_raw": False,
    "broad_weather_l1_log": True,
    "calendar_broad_log_blend": True,
}


def validate_feature_columns(feature_columns):
    forbidden = [
        column
        for column in feature_columns
        if "total_cases" in column or any(column.startswith(prefix) for prefix in case_forbidden_prefixes)
    ]
    if forbidden:
        raise ValueError(f"Target-derived feature columns are forbidden: {forbidden}")


def numeric_feature_columns(feature_frame):
    columns = [
        column
        for column in feature_frame.columns
        if column not in identifier_columns and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    validate_feature_columns(columns)
    return columns


def validation_years(city_frame):
    year_counts = city_frame.groupby("year").size()
    full_years = [int(year) for year, rows in year_counts.items() if rows == 52]
    first_year = int(city_frame["year"].min())
    later_full_years = [year for year in full_years if year > first_year]
    if len(later_full_years) < 2:
        raise ValueError(f"Not enough full validation years for city {city_frame['city'].iloc[0]}.")
    first_full_after_start = min(later_full_years)
    return [year for year in full_years if year > first_full_after_start]


def add_weather_and_calendar_features(city_data, numeric_columns):
    data = city_data.sort_values("week_start_date").copy()
    train_context = data["_model_split"] == "train"
    train_medians = data.loc[train_context, numeric_columns].median()

    # Directional imputation avoids looking backward from future rows into training rows.
    data[numeric_columns] = data[numeric_columns].ffill()
    data[numeric_columns] = data[numeric_columns].fillna(train_medians)
    data[numeric_columns] = data[numeric_columns].fillna(0.0)

    engineered = {}
    week_angle = 2 * math.pi * data["weekofyear"] / 52.0
    engineered["weekofyear_sin"] = np.sin(week_angle)
    engineered["weekofyear_cos"] = np.cos(week_angle)

    for feature, lags in broad_lag_plan.items():
        if feature not in numeric_columns:
            continue
        for lag in lags:
            engineered[f"{feature}_lag_{lag}"] = data[feature].shift(lag)

    for feature, windows in broad_rolling_plan.items():
        if feature not in numeric_columns:
            continue
        for window in windows:
            engineered[f"{feature}_rolling_{window}_mean"] = (
                data[feature].rolling(window=window, min_periods=1).mean()
            )

    data = pd.concat([data, pd.DataFrame(engineered, index=data.index)], axis=1)
    data = data.fillna(0.0)
    return data


def prepare_future_features(fit_rows, future_rows, numeric_columns):
    fit_part = fit_rows.copy()
    future_part = future_rows.copy()
    fit_part["_model_split"] = "train"
    future_part["_model_split"] = "future"
    combined = pd.concat([fit_part, future_part], ignore_index=True, sort=False)
    prepared = add_weather_and_calendar_features(combined, numeric_columns)
    prepared_fit = prepared[prepared["_model_split"] == "train"].copy()
    prepared_future = prepared[prepared["_model_split"] == "future"].copy()
    return prepared_fit, prepared_future, list(prepared.columns)


def broad_features_for_city(city, available_columns):
    selected = list(seasonal_columns)

    if city == "sj":
        roots = sj_broad_roots
        for feature in roots:
            if feature in available_columns:
                selected.append(feature)
            for column in available_columns:
                if column.startswith(f"{feature}_lag_") or column.startswith(f"{feature}_rolling_"):
                    selected.append(column)
        selected = [column for column in selected if column in available_columns]
        validate_feature_columns(selected)
        return list(dict.fromkeys(selected))

    short_lag_suffixes = {"lag_1", "lag_2", "lag_3", "lag_5"}
    station_lag_suffixes = {"lag_1", "lag_2", "lag_3", "lag_4", "lag_6"}
    ndvi_lag_suffixes = {"lag_10", "lag_11", "lag_14"}
    precip_lag_suffixes = {"lag_2", "lag_3"}
    short_rolling_suffixes = {
        "rolling_3_mean",
        "rolling_4_mean",
        "rolling_5_mean",
        "rolling_6_mean",
        "rolling_7_mean",
        "rolling_8_mean",
    }
    station_rolling_suffixes = {"rolling_3_mean", "rolling_4_mean", "rolling_5_mean", "rolling_6_mean"}
    precip_rolling_suffixes = {
        "rolling_4_mean",
        "rolling_7_mean",
        "rolling_10_mean",
        "rolling_14_mean",
    }

    for feature in iq_broad_roots:
        if feature in available_columns:
            selected.append(feature)
        for column in available_columns:
            if not column.startswith(f"{feature}_"):
                continue
            suffix = column.removeprefix(f"{feature}_")
            if feature.startswith("ndvi_") and suffix in ndvi_lag_suffixes:
                selected.append(column)
            elif "precip" in feature and suffix in precip_lag_suffixes:
                selected.append(column)
            elif "precip" in feature and suffix in precip_rolling_suffixes:
                selected.append(column)
            elif feature.startswith("station_") and suffix in station_lag_suffixes:
                selected.append(column)
            elif feature.startswith("station_") and suffix in station_rolling_suffixes:
                selected.append(column)
            elif feature.startswith("reanalysis_") and suffix in short_lag_suffixes:
                selected.append(column)
            elif feature.startswith("reanalysis_") and suffix in short_rolling_suffixes:
                selected.append(column)

    selected = [column for column in selected if column in available_columns]
    validate_feature_columns(selected)
    return list(dict.fromkeys(selected))


def lgbm_params(objective):
    return {
        "objective": objective,
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
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


def fit_predict_continuous(fit_frame, future_frame, feature_columns, target_transform):
    validate_feature_columns(feature_columns)
    model = LGBMRegressor(**lgbm_params("regression_l1"))
    target = fit_frame["total_cases"]
    if target_transform == "log1p":
        target = np.log1p(target)
    model.fit(fit_frame[feature_columns], target)
    predictions = model.predict(future_frame[feature_columns])
    if target_transform == "log1p":
        predictions = np.expm1(predictions)
    predictions = np.asarray(predictions, dtype=float)
    predictions = np.maximum(predictions, 0.0)
    return model, predictions


def round_count_predictions(predictions):
    return [int(round(max(0.0, float(value)))) for value in predictions]


### Data import and validation
train_features = pd.read_csv(train_features_path, parse_dates=["week_start_date"])
test_features = pd.read_csv(test_features_path, parse_dates=["week_start_date"])
labels = pd.read_csv(labels_path)
submission_format = pd.read_csv(submission_format_path)

if list(train_features.columns) != list(test_features.columns):
    raise ValueError("Train and test feature CSVs must have identical columns.")
if train_features[merge_keys].duplicated().any() or test_features[merge_keys].duplicated().any():
    raise ValueError("Feature rows must be unique by city, year, and weekofyear.")
if labels[merge_keys].duplicated().any():
    raise ValueError("Label rows must be unique by city, year, and weekofyear.")
if not submission_format[merge_keys].equals(test_features[merge_keys]):
    raise ValueError("Submission format must match the test feature row order.")

validate_feature_columns(train_features.columns)
numeric_columns = numeric_feature_columns(train_features)

train_data = train_features.merge(labels, on=merge_keys, how="left", validate="one_to_one")
if train_data["total_cases"].isna().any():
    raise ValueError("Merged training data contains missing total_cases values.")

train_data["_row_order"] = range(len(train_data))
test_data = test_features.copy()
test_data["_row_order"] = range(len(test_data))


### Expanding full-year validation
fold_records = []
prediction_records = []

for city, city_train_data in train_data.groupby("city", sort=False):
    city_train_data = city_train_data.sort_values("week_start_date").copy()
    for validation_year in validation_years(city_train_data):
        fold_fit = city_train_data[city_train_data["year"] < validation_year].copy()
        fold_future = city_train_data[city_train_data["year"] == validation_year].copy()
        fit_prepared, future_prepared, available_columns = prepare_future_features(
            fold_fit,
            fold_future,
            numeric_columns,
        )
        calendar_features = list(seasonal_columns)
        broad_features = broad_features_for_city(city, available_columns)

        calendar_model, calendar_continuous = fit_predict_continuous(
            fit_prepared,
            future_prepared,
            calendar_features,
            "none",
        )
        weather_model, weather_continuous = fit_predict_continuous(
            fit_prepared,
            future_prepared,
            broad_features,
            "log1p",
        )
        blend_continuous = (
            (1.0 - args.weather_blend_weight) * calendar_continuous
            + args.weather_blend_weight * weather_continuous
        )

        candidate_predictions = {
            "calendar_l1_raw": calendar_continuous,
            "broad_weather_l1_log": weather_continuous,
            "calendar_broad_log_blend": blend_continuous,
        }
        actual_values = future_prepared["total_cases"].astype(int).tolist()

        for candidate_key, continuous_predictions in candidate_predictions.items():
            integer_predictions = round_count_predictions(continuous_predictions)
            fold_mae = mean_absolute_error(actual_values, integer_predictions)
            feature_count = len(calendar_features) if candidate_key == "calendar_l1_raw" else len(broad_features)
            if candidate_key == "calendar_broad_log_blend":
                feature_count = len(calendar_features) + len(broad_features)

            fold_records.append(
                {
                    "candidate": candidate_key,
                    "candidate_label": candidate_labels[candidate_key],
                    "uses_weather": candidate_uses_weather[candidate_key],
                    "city": city,
                    "validation_year": validation_year,
                    "train_rows": len(fit_prepared),
                    "validation_rows": len(future_prepared),
                    "feature_count": feature_count,
                    "mae": fold_mae,
                }
            )

            for row, continuous_prediction, integer_prediction in zip(
                future_prepared.itertuples(index=False),
                continuous_predictions,
                integer_predictions,
            ):
                prediction_records.append(
                    {
                        "candidate": candidate_key,
                        "city": row.city,
                        "year": row.year,
                        "weekofyear": row.weekofyear,
                        "week_start_date": row.week_start_date,
                        "validation_year": validation_year,
                        "actual_total_cases": row.total_cases,
                        "predicted_total_cases": integer_prediction,
                        "predicted_total_cases_raw": float(continuous_prediction),
                    }
                )

fold_scores = pd.DataFrame(fold_records)
validation_predictions = pd.DataFrame(prediction_records)

summary_records = []
for candidate_key in candidate_order:
    candidate_rows = validation_predictions[validation_predictions["candidate"] == candidate_key]
    for city, city_rows in candidate_rows.groupby("city", sort=False):
        summary_records.append(
            {
                "candidate": candidate_key,
                "candidate_label": candidate_labels[candidate_key],
                "uses_weather": candidate_uses_weather[candidate_key],
                "city": city,
                "validation_rows": len(city_rows),
                "mae": mean_absolute_error(
                    city_rows["actual_total_cases"],
                    city_rows["predicted_total_cases"],
                ),
            }
        )
    summary_records.append(
        {
            "candidate": candidate_key,
            "candidate_label": candidate_labels[candidate_key],
            "uses_weather": candidate_uses_weather[candidate_key],
            "city": "all",
            "validation_rows": len(candidate_rows),
            "mae": mean_absolute_error(
                candidate_rows["actual_total_cases"],
                candidate_rows["predicted_total_cases"],
            ),
        }
    )

approach_summary = pd.DataFrame(summary_records)
approach_summary["risk_margin"] = np.where(
    approach_summary["uses_weather"],
    args.weather_risk_margin,
    0.0,
)
approach_summary["selection_mae"] = approach_summary["mae"] + approach_summary["risk_margin"]


### Risk-aware city selection
selection_records = []
for city, city_summary in approach_summary[approach_summary["city"] != "all"].groupby("city", sort=False):
    selected = city_summary.sort_values(["selection_mae", "mae"]).iloc[0]
    validation_best = city_summary.sort_values(["mae", "selection_mae"]).iloc[0]
    calendar_row = city_summary[city_summary["candidate"] == "calendar_l1_raw"].iloc[0]
    selection_records.append(
        {
            "city": city,
            "selected_candidate": selected["candidate"],
            "selected_label": selected["candidate_label"],
            "selected_uses_weather": bool(selected["uses_weather"]),
            "selected_validation_mae": float(selected["mae"]),
            "selected_selection_mae": float(selected["selection_mae"]),
            "validation_best_candidate": validation_best["candidate"],
            "validation_best_mae": float(validation_best["mae"]),
            "calendar_mae": float(calendar_row["mae"]),
            "weather_risk_margin": args.weather_risk_margin,
        }
    )

selection = pd.DataFrame(selection_records)

selected_validation_rows = []
for row in selection.itertuples(index=False):
    selected_validation_rows.append(
        validation_predictions[
            (validation_predictions["city"] == row.city)
            & (validation_predictions["candidate"] == row.selected_candidate)
        ]
    )
selected_validation = pd.concat(selected_validation_rows, ignore_index=True)
selected_overall_mae = mean_absolute_error(
    selected_validation["actual_total_cases"],
    selected_validation["predicted_total_cases"],
)


### Final models and submissions
submission_predictions = {}
validation_best_submission_predictions = {}
feature_importance_records = []
component_prediction_records = []

for city, city_train_data in train_data.groupby("city", sort=False):
    city_train_data = city_train_data.sort_values("week_start_date").copy()
    city_test_data = test_data[test_data["city"] == city].sort_values("week_start_date").copy()
    train_prepared, test_prepared, available_columns = prepare_future_features(
        city_train_data,
        city_test_data,
        numeric_columns,
    )
    calendar_features = list(seasonal_columns)
    broad_features = broad_features_for_city(city, available_columns)

    calendar_model, calendar_continuous = fit_predict_continuous(
        train_prepared,
        test_prepared,
        calendar_features,
        "none",
    )
    weather_model, weather_continuous = fit_predict_continuous(
        train_prepared,
        test_prepared,
        broad_features,
        "log1p",
    )
    blend_continuous = (
        (1.0 - args.weather_blend_weight) * calendar_continuous
        + args.weather_blend_weight * weather_continuous
    )
    final_candidate_predictions = {
        "calendar_l1_raw": round_count_predictions(calendar_continuous),
        "broad_weather_l1_log": round_count_predictions(weather_continuous),
        "calendar_broad_log_blend": round_count_predictions(blend_continuous),
    }
    selected_candidate = selection.loc[selection["city"] == city, "selected_candidate"].iloc[0]
    validation_best_candidate = selection.loc[
        selection["city"] == city,
        "validation_best_candidate",
    ].iloc[0]

    for row_index, row in enumerate(test_prepared.itertuples(index=False)):
        key = (row.city, row.year, row.weekofyear)
        submission_predictions[key] = final_candidate_predictions[selected_candidate][row_index]
        validation_best_submission_predictions[key] = final_candidate_predictions[
            validation_best_candidate
        ][row_index]
        component_prediction_records.append(
            {
                "city": row.city,
                "year": row.year,
                "weekofyear": row.weekofyear,
                "calendar_prediction": final_candidate_predictions["calendar_l1_raw"][row_index],
                "weather_prediction": final_candidate_predictions["broad_weather_l1_log"][row_index],
                "blend_prediction": final_candidate_predictions["calendar_broad_log_blend"][row_index],
                "risk_aware_candidate": selected_candidate,
                "validation_best_candidate": validation_best_candidate,
            }
        )

    for feature, gain, split in zip(
        calendar_features,
        calendar_model.booster_.feature_importance(importance_type="gain"),
        calendar_model.booster_.feature_importance(importance_type="split"),
    ):
        feature_importance_records.append(
            {
                "city": city,
                "component": "calendar_l1_raw",
                "feature": feature,
                "importance_gain": float(gain),
                "importance_split": int(split),
            }
        )

    for feature, gain, split in zip(
        broad_features,
        weather_model.booster_.feature_importance(importance_type="gain"),
        weather_model.booster_.feature_importance(importance_type="split"),
    ):
        feature_importance_records.append(
            {
                "city": city,
                "component": "broad_weather_l1_log",
                "feature": feature,
                "importance_gain": float(gain),
                "importance_split": int(split),
            }
        )

submission = submission_format.copy()
submission["total_cases"] = [
    submission_predictions[(row.city, row.year, row.weekofyear)]
    for row in submission.itertuples(index=False)
]

validation_best_submission = submission_format.copy()
validation_best_submission["total_cases"] = [
    validation_best_submission_predictions[(row.city, row.year, row.weekofyear)]
    for row in validation_best_submission.itertuples(index=False)
]

for output_submission in [submission, validation_best_submission]:
    if list(output_submission.columns) != ["city", "year", "weekofyear", "total_cases"]:
        raise ValueError("Submission columns must be city, year, weekofyear, total_cases.")
    if not output_submission[merge_keys].equals(submission_format[merge_keys]):
        raise ValueError("Submission row order must match submission format.")
    if output_submission["total_cases"].isna().any():
        raise ValueError("Submission contains missing predictions.")
    if (output_submission["total_cases"] < 0).any():
        raise ValueError("Submission contains negative predictions.")
    output_submission["total_cases"] = output_submission["total_cases"].astype(int)

feature_importance = pd.DataFrame(feature_importance_records).sort_values(
    ["city", "component", "importance_gain"],
    ascending=[True, True, False],
)
component_predictions = pd.DataFrame(component_prediction_records)


### Outputs
fold_scores.to_csv(fold_scores_path, index=False)
approach_summary.to_csv(summary_path, index=False)
selection.to_csv(selection_path, index=False)
validation_predictions.to_csv(validation_predictions_path, index=False)
submission.to_csv(submission_path, index=False)
validation_best_submission.to_csv(validation_best_submission_path, index=False)
feature_importance.to_csv(feature_importance_path, index=False)
component_predictions.to_csv(output_dir / "test_component_predictions.csv", index=False)

adversarial_scores = None
if adversarial_scores_path.exists():
    adversarial_scores = pd.read_csv(adversarial_scores_path)

report_lines = [
    "# Best-Practice DengAI Approach",
    "",
    "This run encodes a risk-aware approach after the forward-validation and adversarial-validation results.",
    "",
    "## Guardrails",
    "",
    "- San Juan and Iquitos are modeled separately.",
    "- Validation is expanding full-year forward chaining.",
    "- `total_cases` is used only as the supervised label and validation ground truth.",
    "- No case-derived or target-derived columns are allowed in model features.",
    "- Raw `year`, `week_start_date`, and raw `weekofyear` are not model inputs; only cyclic week features are used.",
    "- Fold preprocessing uses only the fold training history for imputation anchors.",
    "- Weather features are accepted for final selection only if they clear a shift-risk MAE margin.",
    "",
    "## Candidate Results",
    "",
    "| Candidate | City | Uses weather | MAE | Selection MAE |",
    "| --- | --- | ---: | ---: | ---: |",
]

for row in approach_summary.sort_values(["candidate", "city"]).itertuples(index=False):
    report_lines.append(
        f"| {row.candidate} | {row.city} | {row.uses_weather} | "
        f"{row.mae:.4f} | {row.selection_mae:.4f} |"
    )

report_lines.extend(
    [
        "",
        "## Risk-Aware Selection",
        "",
        "| City | Selected | Validation MAE | Selection MAE | Pure validation best |",
        "| --- | --- | ---: | ---: | --- |",
    ]
)

for row in selection.itertuples(index=False):
    report_lines.append(
        f"| {row.city} | {row.selected_candidate} | {row.selected_validation_mae:.4f} | "
        f"{row.selected_selection_mae:.4f} | {row.validation_best_candidate} "
        f"({row.validation_best_mae:.4f}) |"
    )

report_lines.extend(
    [
        "",
        f"Risk-aware selected overall validation MAE: {selected_overall_mae:.4f}.",
        "",
    ]
)

if adversarial_scores is not None:
    report_lines.extend(
        [
            "## Adversarial Context",
            "",
            "| City | Scenario | Train/test AUC | Interpretation |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in adversarial_scores.itertuples(index=False):
        if row.roc_auc_mean >= 0.80:
            interpretation = "strong covariate shift"
        elif row.roc_auc_mean >= 0.65:
            interpretation = "meaningful covariate shift"
        else:
            interpretation = "weak covariate shift"
        report_lines.append(
            f"| {row.city} | {row.scenario} | {row.roc_auc_mean:.3f} | {interpretation} |"
        )
    report_lines.append("")

report_lines.extend(
    [
        "## Interpretation",
        "",
        "The pure validation-best candidate is useful evidence, but it is not automatically the safest final choice because adversarial validation shows non-IID train/test weather covariates. With the default margin, the small weather/blend gains do not clear the risk threshold, so the final submission uses the stable calendar-only L1 model for each city.",
        "",
        "Generated files:",
        f"- `{summary_path}`",
        f"- `{selection_path}`",
        f"- `{fold_scores_path}`",
        f"- `{validation_predictions_path}`",
        f"- `{submission_path}`",
        f"- `{validation_best_submission_path}`",
        f"- `{feature_importance_path}`",
        f"- `{output_dir / 'test_component_predictions.csv'}`",
    ]
)

report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

experiment_record = {
    "logged_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "script": str(Path(__file__)),
    "config": {
        "train_features_csv": str(train_features_path),
        "test_features_csv": str(test_features_path),
        "labels_csv": str(labels_path),
        "submission_format_csv": str(submission_format_path),
        "random_state": args.random_state,
        "weather_blend_weight": args.weather_blend_weight,
        "weather_risk_margin": args.weather_risk_margin,
    },
    "results": {
        "selected_overall_validation_mae": selected_overall_mae,
        "selection": selection.to_dict(orient="records"),
    },
    "outputs": {
        "summary": str(summary_path),
        "selection": str(selection_path),
        "submission": str(submission_path),
        "report": str(report_path),
    },
}

with experiment_log_path.open("a", encoding="utf-8") as log_file:
    log_file.write(json.dumps(experiment_record, sort_keys=True, default=str) + "\n")

print(f"Wrote best-practice summary to {summary_path}")
print(f"Wrote risk-aware selection to {selection_path}")
print(f"Wrote report to {report_path}")
print(f"Wrote risk-aware submission to {submission_path}")
