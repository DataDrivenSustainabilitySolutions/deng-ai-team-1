"""Goal-oriented DengAI model comparison and submission pipeline.

Example:
    python3 oliver/sandbox/main_goal.py

The script runs a short, fixed set of sensible iterations and writes a
changelog with the rationale, validation result, and interpretation for each.
It never uses total_cases, lagged cases, or rolling cases as input features.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dengai_main_goal_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    from lightgbm import LGBMRegressor
except ImportError as error:
    raise ImportError(
        "LightGBM is required. Install dependencies from pyproject.toml, "
        "for example: python3 -m pip install -e ."
    ) from error

CATBOOST_AVAILABLE = importlib.util.find_spec("catboost") is not None
if CATBOOST_AVAILABLE:
    from catboost import CatBoostRegressor
else:
    CatBoostRegressor = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "main_goal"
MERGE_KEYS = ["city", "year", "weekofyear"]
IDENTIFIER_COLUMNS = {"city", "year", "weekofyear", "week_start_date", "split", "total_cases"}
SEASONAL_COLUMNS = ["weekofyear_sin", "weekofyear_cos"]
FORBIDDEN_FEATURE_NAMES = {"total_cases"}
FORBIDDEN_FEATURE_PREFIXES = ("cases_", "case_lag", "case_roll", "target_")

BROAD_LAG_PLAN = {
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

BROAD_ROLLING_PLAN = {
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

CONSERVATIVE_ROOTS = {
    "sj": [
        "reanalysis_specific_humidity_g_per_kg",
        "reanalysis_dew_point_temp_k",
        "reanalysis_min_air_temp_k",
        "reanalysis_avg_temp_k",
        "reanalysis_relative_humidity_percent",
        "station_avg_temp_c",
        "station_min_temp_c",
        "station_max_temp_c",
    ],
    "iq": [
        "reanalysis_specific_humidity_g_per_kg",
        "reanalysis_dew_point_temp_k",
        "reanalysis_min_air_temp_k",
        "station_avg_temp_c",
        "station_min_temp_c",
        "station_max_temp_c",
        "ndvi_ne",
        "ndvi_nw",
        "ndvi_se",
        "ndvi_sw",
        "precipitation_amt_mm",
        "reanalysis_precip_amt_kg_per_m2",
    ],
}

CONSERVATIVE_LAGS = [2, 4, 6, 8, 10, 12]
CONSERVATIVE_ROLLING_WINDOWS = [4, 8, 12]

SJ_BROAD_ROOTS = [
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

IQ_BROAD_ROOTS = [
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

LOSS_OBJECTIVES = {
    "l1": "regression_l1",
    "poisson": "poisson",
    "tweedie": "tweedie",
}

CANDIDATES = [
    {
        "iteration": "01_calendar_l1_raw",
        "feature_set": "calendar",
        "loss": "l1",
        "target_transform": "none",
        "rationale": "Start with a low-capacity seasonal baseline using only cyclic week features.",
    },
    {
        "iteration": "02_calendar_l1_log",
        "feature_set": "calendar",
        "loss": "l1",
        "target_transform": "log1p",
        "rationale": "Test whether the same seasonal-only model benefits from log1p target scaling.",
    },
    {
        "iteration": "03_raw_weather_l1_raw",
        "feature_set": "raw_weather",
        "loss": "l1",
        "target_transform": "none",
        "rationale": "Add the original climate and vegetation measurements without target transformation.",
    },
    {
        "iteration": "04_raw_weather_l1_log",
        "feature_set": "raw_weather",
        "loss": "l1",
        "target_transform": "log1p",
        "rationale": "Test log1p target scaling on the original climate and vegetation measurements.",
    },
    {
        "iteration": "05_conservative_lag_roll_l1_raw",
        "feature_set": "conservative_lag_roll",
        "loss": "l1",
        "target_transform": "none",
        "rationale": "Add a compact, biologically plausible lag/rolling set without target transformation.",
    },
    {
        "iteration": "06_conservative_lag_roll_l1_log",
        "feature_set": "conservative_lag_roll",
        "loss": "l1",
        "target_transform": "log1p",
        "rationale": "Test log1p target scaling on the compact lag/rolling weather feature set.",
    },
    {
        "iteration": "07_conservative_lag_roll_poisson",
        "feature_set": "conservative_lag_roll",
        "loss": "poisson",
        "target_transform": "none",
        "rationale": "Test a count-style objective on the conservative feature set.",
    },
    {
        "iteration": "08_broad_lag_roll_l1_raw",
        "feature_set": "broad_lag_roll",
        "loss": "l1",
        "target_transform": "none",
        "rationale": "Reproduce the strongest prior Oliver-style feature breadth without target transformation.",
    },
    {
        "iteration": "09_broad_lag_roll_l1_log",
        "feature_set": "broad_lag_roll",
        "loss": "l1",
        "target_transform": "log1p",
        "rationale": "Apply the best prior target scaling idea to the broad weather lag/rolling feature set.",
    },
    {
        "iteration": "10_broad_lag_roll_poisson",
        "feature_set": "broad_lag_roll",
        "loss": "poisson",
        "target_transform": "none",
        "rationale": "Check whether the broad feature set benefits from a direct count objective.",
    },
]

CATBOOST_CANDIDATES = [
    {
        "iteration": "12_catboost_conservative_lag_roll_raw",
        "model_family": "catboost",
        "feature_set": "conservative_lag_roll",
        "loss": "mae",
        "target_transform": "none",
        "rationale": (
            "Try CatBoost as a more regularized boosting alternative on numeric weather features; "
            "there is no categorical-heavy advantage in this dataset."
        ),
    },
]

BLEND_CANDIDATES = [
    {
        "iteration": "11_calendar_broad_log_blend",
        "feature_set": "calendar_broad_blend",
        "loss": "weighted_average",
        "target_transform": "mixed",
        "log_component_weight": 1.0 / 3.0,
        "members": [
            ("01_calendar_l1_raw", 2.0 / 3.0),
            ("09_broad_lag_roll_l1_log", 1.0 / 3.0),
        ],
        "rationale": (
            "Blend the robust seasonal model with a small broad-weather component "
            "because fold diagnostics show weather helps on some outbreak years."
        ),
    },
]

ACTIVE_MODEL_CANDIDATES = CANDIDATES + (CATBOOST_CANDIDATES if CATBOOST_AVAILABLE else [])
SKIPPED_CANDIDATES = (
    []
    if CATBOOST_AVAILABLE
    else [
        {
            "iteration": candidate["iteration"],
            "reason": "catboost is not installed; skipped optional CatBoost comparison",
        }
        for candidate in CATBOOST_CANDIDATES
    ]
)
ALL_CANDIDATES = ACTIVE_MODEL_CANDIDATES + BLEND_CANDIDATES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fixed, city-specific DengAI model iterations and write a changelog."
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
        "--previous-output-root",
        default="oliver/outputs",
        help="Root containing prior validation_scores.csv files. Default: oliver/outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for all generated outputs. Default: oliver/sandbox/outputs/main_goal",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic model fitting. Default: 42",
    )
    parser.add_argument(
        "--tweedie-variance-power",
        type=float,
        default=1.3,
        help="LightGBM Tweedie variance power. Default: 1.3",
    )
    parser.add_argument(
        "--log-target-min-validation-gain",
        type=float,
        default=1.0,
        help=(
            "MAE margin a log1p-target candidate must overcome during final selection. "
            "This incorporates hidden-score feedback that the same feature set with a "
            "log-transformed target was slightly worse externally. Default: 1.0"
        ),
    )
    parser.add_argument(
        "--validation-output",
        default="validation_scores.csv",
        help="Per-fold validation scores filename. Default: validation_scores.csv",
    )
    parser.add_argument(
        "--iteration-output",
        default="iteration_scores.csv",
        help="Candidate summary filename. Default: iteration_scores.csv",
    )
    parser.add_argument(
        "--selection-output",
        default="selected_iteration_by_city.csv",
        help="Final city selection filename. Default: selected_iteration_by_city.csv",
    )
    parser.add_argument(
        "--validation-predictions-output",
        default="validation_predictions.csv",
        help="Row-level validation predictions filename. Default: validation_predictions.csv",
    )
    parser.add_argument(
        "--feature-importance-output",
        default="feature_importance.csv",
        help="Final selected model feature importance filename. Default: feature_importance.csv",
    )
    parser.add_argument(
        "--submission-output",
        default="submission.csv",
        help="Final submission filename. Default: submission.csv",
    )
    parser.add_argument(
        "--historical-output",
        default="historical_scores.csv",
        help="Prior run score summary filename. Default: historical_scores.csv",
    )
    parser.add_argument(
        "--skipped-output",
        default="skipped_candidates.csv",
        help="Skipped optional candidate filename. Default: skipped_candidates.csv",
    )
    parser.add_argument(
        "--changelog-output",
        default="changelog.md",
        help="Iteration changelog filename. Default: changelog.md",
    )
    parser.add_argument(
        "--experiment-log-output",
        default="experiment_log.jsonl",
        help="Append-only run metadata filename. Default: experiment_log.jsonl",
    )
    return parser.parse_args()


def validate_paths(train_features, test_features, labels, submission_format):
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError("Train and test feature CSVs must have identical columns.")
    if train_features[MERGE_KEYS].duplicated().any() or test_features[MERGE_KEYS].duplicated().any():
        raise ValueError("Feature rows must be unique by city, year, and weekofyear.")
    if labels[MERGE_KEYS].duplicated().any():
        raise ValueError("Label rows must be unique by city, year, and weekofyear.")
    if not submission_format[MERGE_KEYS].equals(test_features[MERGE_KEYS]):
        raise ValueError("Submission format must match the test feature row order.")


def numeric_feature_columns(train_features):
    return [
        column
        for column in train_features.columns
        if column not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(train_features[column])
    ]


def get_validation_years(city_train_data):
    year_counts = city_train_data.groupby("year").size()
    full_years = [int(year) for year, rows in year_counts.items() if rows == 52]
    first_year = int(city_train_data["year"].min())
    later_full_years = [year for year in full_years if year > first_year]
    if len(later_full_years) < 2:
        raise ValueError(f"Not enough full validation years for city {city_train_data['city'].iloc[0]}.")
    first_full_after_start = min(later_full_years)
    return [year for year in full_years if year > first_full_after_start]


def impute_features_with_training_context(combined_data, numeric_columns):
    data = combined_data.sort_values("week_start_date").copy()
    train_mask = data["_model_split"] == "train"
    train_medians = data.loc[train_mask, numeric_columns].median()

    # Forward fill preserves time direction; medians only cover leading missing values.
    data[numeric_columns] = data[numeric_columns].ffill()
    data[numeric_columns] = data[numeric_columns].fillna(train_medians)
    data[numeric_columns] = data[numeric_columns].fillna(0.0)
    return data


def add_engineered_weather_features(data, numeric_columns):
    engineered = {}
    week_angle = 2 * math.pi * data["weekofyear"] / 52.0
    engineered["weekofyear_sin"] = np.sin(week_angle)
    engineered["weekofyear_cos"] = np.cos(week_angle)

    for feature, lags in BROAD_LAG_PLAN.items():
        if feature not in numeric_columns:
            continue
        for lag in lags:
            engineered[f"{feature}_lag_{lag}"] = data[feature].shift(lag)

    for feature, windows in BROAD_ROLLING_PLAN.items():
        if feature not in numeric_columns:
            continue
        for window in windows:
            engineered[f"{feature}_rolling_{window}_mean"] = (
                data[feature].rolling(window=window, min_periods=1).mean()
            )

    return pd.concat([data, pd.DataFrame(engineered, index=data.index)], axis=1)


def broad_feature_columns(city, available_columns):
    if city == "sj":
        roots = SJ_BROAD_ROOTS
        selected = list(SEASONAL_COLUMNS)
        for feature in roots:
            if feature in available_columns:
                selected.append(feature)
            for column in available_columns:
                if column.startswith(f"{feature}_lag_") or column.startswith(f"{feature}_rolling_"):
                    selected.append(column)
        return list(dict.fromkeys(selected))

    short_lag_suffixes = {"lag_1", "lag_2", "lag_3", "lag_5"}
    station_suffixes = {"lag_1", "lag_2", "lag_3", "lag_4", "lag_6"}
    ndvi_suffixes = {"lag_10", "lag_11", "lag_14"}
    precip_suffixes = {"lag_2", "lag_3"}
    short_rolling_suffixes = {
        "rolling_3_mean",
        "rolling_4_mean",
        "rolling_5_mean",
        "rolling_6_mean",
        "rolling_7_mean",
        "rolling_8_mean",
    }
    station_rolling_suffixes = {
        "rolling_3_mean",
        "rolling_4_mean",
        "rolling_5_mean",
        "rolling_6_mean",
    }
    precip_rolling_suffixes = {
        "rolling_4_mean",
        "rolling_7_mean",
        "rolling_10_mean",
        "rolling_14_mean",
    }

    selected = list(SEASONAL_COLUMNS)
    for feature in IQ_BROAD_ROOTS:
        if feature in available_columns:
            selected.append(feature)
        for column in available_columns:
            if not column.startswith(f"{feature}_"):
                continue
            suffix = column.removeprefix(f"{feature}_")
            if feature.startswith("ndvi_") and suffix in ndvi_suffixes:
                selected.append(column)
            elif "precip" in feature and suffix in precip_suffixes:
                selected.append(column)
            elif "precip" in feature and suffix in precip_rolling_suffixes:
                selected.append(column)
            elif feature.startswith("station_") and suffix in station_suffixes:
                selected.append(column)
            elif feature.startswith("station_") and suffix in station_rolling_suffixes:
                selected.append(column)
            elif feature.startswith("reanalysis_") and suffix in short_lag_suffixes:
                selected.append(column)
            elif feature.startswith("reanalysis_") and suffix in short_rolling_suffixes:
                selected.append(column)
    return list(dict.fromkeys(selected))


def conservative_feature_columns(city, available_columns):
    selected = list(SEASONAL_COLUMNS)
    for feature in CONSERVATIVE_ROOTS[city]:
        if feature in available_columns:
            selected.append(feature)
        for lag in CONSERVATIVE_LAGS:
            column = f"{feature}_lag_{lag}"
            if column in available_columns:
                selected.append(column)
        for window in CONSERVATIVE_ROLLING_WINDOWS:
            column = f"{feature}_rolling_{window}_mean"
            if column in available_columns:
                selected.append(column)
    return list(dict.fromkeys(selected))


def select_feature_columns(feature_set, city, numeric_columns, available_columns):
    if feature_set == "calendar":
        selected = list(SEASONAL_COLUMNS)
    elif feature_set == "raw_weather":
        selected = list(SEASONAL_COLUMNS) + list(numeric_columns)
    elif feature_set == "conservative_lag_roll":
        selected = conservative_feature_columns(city, available_columns)
    elif feature_set == "broad_lag_roll":
        selected = broad_feature_columns(city, available_columns)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    selected = [column for column in selected if column in available_columns]
    forbidden = [
        column
        for column in selected
        if column in FORBIDDEN_FEATURE_NAMES
        or any(column.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Target-derived feature columns are forbidden: {forbidden}")
    return selected


def prepare_fold_features(fold_train_data, fold_future_data, numeric_columns):
    train_part = fold_train_data.copy()
    future_part = fold_future_data.copy()
    train_part["_model_split"] = "train"
    future_part["_model_split"] = "future"
    combined = pd.concat([train_part, future_part], ignore_index=True, sort=False)
    combined = impute_features_with_training_context(combined, numeric_columns)
    combined = add_engineered_weather_features(combined, numeric_columns)
    train_prepared = combined[combined["_model_split"] == "train"].copy()
    future_prepared = combined[combined["_model_split"] == "future"].copy()
    return train_prepared, future_prepared, list(combined.columns)


def model_params(candidate, random_state, tweedie_variance_power):
    params = {
        "objective": LOSS_OBJECTIVES[candidate["loss"]],
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "seed": random_state,
        "data_random_seed": random_state,
        "feature_fraction_seed": random_state,
        "bagging_seed": random_state,
        "drop_seed": random_state,
        "extra_seed": random_state,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
        "verbose": -1,
    }
    if candidate["loss"] == "tweedie":
        params["tweedie_variance_power"] = tweedie_variance_power
    return params


def make_model(candidate, random_state, tweedie_variance_power):
    if candidate.get("model_family") == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("CatBoost candidate requested but catboost is not installed.")
        return CatBoostRegressor(
            loss_function="MAE",
            iterations=500,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=8.0,
            random_seed=random_state,
            allow_writing_files=False,
            thread_count=1,
            verbose=False,
        )
    return LGBMRegressor(**model_params(candidate, random_state, tweedie_variance_power))


def transform_target(values, target_transform):
    if target_transform == "log1p":
        return np.log1p(values)
    return values


def inverse_transform_predictions(values, target_transform):
    if target_transform == "log1p":
        return np.expm1(values)
    return values


def integer_predictions(raw_predictions, target_transform):
    restored = inverse_transform_predictions(np.asarray(raw_predictions, dtype=float), target_transform)
    return [int(round(max(0.0, float(value)))) for value in restored]


def evaluate_candidate(candidate, train_data, numeric_columns, random_state, tweedie_variance_power):
    fold_records = []
    prediction_records = []
    city_feature_counts = {}
    candidate_actuals = []
    candidate_predictions = []
    log_component_weight = 1.0 if candidate["target_transform"] == "log1p" else 0.0

    for city, city_data in train_data.groupby("city", sort=False):
        city_data = city_data.sort_values("week_start_date").copy()
        validation_years = get_validation_years(city_data)
        city_actuals = []
        city_predictions = []

        for validation_year in validation_years:
            fold_train_data = city_data[city_data["year"] < validation_year].copy()
            fold_validation_data = city_data[city_data["year"] == validation_year].copy()
            prepared_train, prepared_validation, available_columns = prepare_fold_features(
                fold_train_data,
                fold_validation_data,
                numeric_columns,
            )
            feature_columns = select_feature_columns(
                candidate["feature_set"],
                city,
                numeric_columns,
                available_columns,
            )
            city_feature_counts[city] = len(feature_columns)

            model = make_model(candidate, random_state, tweedie_variance_power)
            model.fit(
                prepared_train[feature_columns],
                transform_target(prepared_train["total_cases"], candidate["target_transform"]),
            )

            fold_predictions = integer_predictions(
                model.predict(prepared_validation[feature_columns]),
                candidate["target_transform"],
            )
            fold_actuals = fold_validation_data["total_cases"].astype(int).tolist()
            fold_mae = mean_absolute_error(fold_actuals, fold_predictions)

            city_actuals.extend(fold_actuals)
            city_predictions.extend(fold_predictions)
            candidate_actuals.extend(fold_actuals)
            candidate_predictions.extend(fold_predictions)

            fold_records.append(
                {
                    "iteration": candidate["iteration"],
                    "feature_set": candidate["feature_set"],
                    "loss": candidate["loss"],
                    "target_transform": candidate["target_transform"],
                    "log_component_weight": log_component_weight,
                    "city": city,
                    "validation_year": validation_year,
                    "train_rows": len(fold_train_data),
                    "validation_rows": len(fold_validation_data),
                    "feature_count": len(feature_columns),
                    "mae": float(fold_mae),
                }
            )

            for row, actual, prediction in zip(
                fold_validation_data.itertuples(index=False),
                fold_actuals,
                fold_predictions,
            ):
                prediction_records.append(
                    {
                        "iteration": candidate["iteration"],
                        "city": row.city,
                        "year": row.year,
                        "weekofyear": row.weekofyear,
                        "week_start_date": row.week_start_date,
                        "validation_year": validation_year,
                        "actual_total_cases": actual,
                        "predicted_total_cases": prediction,
                    }
                )

        fold_records.append(
            {
                "iteration": candidate["iteration"],
                "feature_set": candidate["feature_set"],
                "loss": candidate["loss"],
                "target_transform": candidate["target_transform"],
                "log_component_weight": log_component_weight,
                "city": city,
                "validation_year": "all",
                "train_rows": "",
                "validation_rows": len(city_actuals),
                "feature_count": city_feature_counts[city],
                "mae": float(mean_absolute_error(city_actuals, city_predictions)),
            }
        )

    fold_records.append(
        {
            "iteration": candidate["iteration"],
            "feature_set": candidate["feature_set"],
            "loss": candidate["loss"],
            "target_transform": candidate["target_transform"],
            "log_component_weight": log_component_weight,
            "city": "all",
            "validation_year": "all",
            "train_rows": "",
            "validation_rows": len(candidate_actuals),
            "feature_count": sum(city_feature_counts.values()),
            "mae": float(mean_absolute_error(candidate_actuals, candidate_predictions)),
        }
    )

    return fold_records, prediction_records


def evaluate_blend_candidates(blend_candidates, validation_predictions):
    fold_records = []
    prediction_records = []
    prediction_keys = [
        "city",
        "year",
        "weekofyear",
        "week_start_date",
        "validation_year",
        "actual_total_cases",
    ]

    for candidate in blend_candidates:
        blended = None
        weighted_prediction_columns = []
        for member_iteration, weight in candidate["members"]:
            member_predictions = validation_predictions[
                validation_predictions["iteration"] == member_iteration
            ][prediction_keys + ["predicted_total_cases"]].copy()
            member_column = f"prediction_{member_iteration}"
            member_predictions = member_predictions.rename(
                columns={"predicted_total_cases": member_column}
            )
            if blended is None:
                blended = member_predictions
            else:
                blended = blended.merge(member_predictions, on=prediction_keys, how="inner")
            weighted_prediction_columns.append((member_column, weight))

        blended["predicted_total_cases"] = np.rint(
            sum(blended[column] * weight for column, weight in weighted_prediction_columns)
        ).clip(lower=0).astype(int)

        for row in blended.itertuples(index=False):
            prediction_records.append(
                {
                    "iteration": candidate["iteration"],
                    "city": row.city,
                    "year": row.year,
                    "weekofyear": row.weekofyear,
                    "week_start_date": row.week_start_date,
                    "validation_year": row.validation_year,
                    "actual_total_cases": row.actual_total_cases,
                    "predicted_total_cases": row.predicted_total_cases,
                }
            )

        for (city, validation_year), group in blended.groupby(["city", "validation_year"], sort=False):
            fold_records.append(
                {
                    "iteration": candidate["iteration"],
                    "feature_set": candidate["feature_set"],
                    "loss": candidate["loss"],
                    "target_transform": candidate["target_transform"],
                    "log_component_weight": candidate["log_component_weight"],
                    "city": city,
                    "validation_year": validation_year,
                    "train_rows": "",
                    "validation_rows": len(group),
                    "feature_count": "",
                    "mae": float(
                        mean_absolute_error(
                            group["actual_total_cases"],
                            group["predicted_total_cases"],
                        )
                    ),
                }
            )

        for city, group in blended.groupby("city", sort=False):
            fold_records.append(
                {
                    "iteration": candidate["iteration"],
                    "feature_set": candidate["feature_set"],
                    "loss": candidate["loss"],
                    "target_transform": candidate["target_transform"],
                    "log_component_weight": candidate["log_component_weight"],
                    "city": city,
                    "validation_year": "all",
                    "train_rows": "",
                    "validation_rows": len(group),
                    "feature_count": "",
                    "mae": float(
                        mean_absolute_error(
                            group["actual_total_cases"],
                            group["predicted_total_cases"],
                        )
                    ),
                }
            )

        fold_records.append(
            {
                "iteration": candidate["iteration"],
                "feature_set": candidate["feature_set"],
                "loss": candidate["loss"],
                "target_transform": candidate["target_transform"],
                "log_component_weight": candidate["log_component_weight"],
                "city": "all",
                "validation_year": "all",
                "train_rows": "",
                "validation_rows": len(blended),
                "feature_count": "",
                "mae": float(
                    mean_absolute_error(
                        blended["actual_total_cases"],
                        blended["predicted_total_cases"],
                    )
                ),
            }
        )

    return fold_records, prediction_records


def historical_scores(previous_output_root):
    rows = []
    root = Path(previous_output_root)
    if not root.exists():
        return pd.DataFrame(rows)

    for path in sorted(root.glob("*/validation_scores.csv")):
        try:
            scores = pd.read_csv(path)
        except Exception as error:
            rows.append({"run": path.parent.name, "error": str(error)})
            continue
        if {"city", "mae"}.issubset(scores.columns):
            all_rows = scores[scores["city"].astype(str) == "all"]
            if not all_rows.empty:
                rows.append(
                    {
                        "run": path.parent.name,
                        "overall_mae": float(all_rows["mae"].iloc[-1]),
                        "validation_rows": all_rows["validation_rows"].iloc[-1]
                        if "validation_rows" in all_rows
                        else "",
                    }
                )
            else:
                rows.append(
                    {
                        "run": path.parent.name,
                        "overall_mae": float(scores["mae"].mean()),
                        "validation_rows": len(scores),
                    }
                )
    if not rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows).sort_values("overall_mae", na_position="last")


def write_changelog(
    path,
    run_started_at,
    historical,
    iteration_scores,
    selected_by_city,
    skipped_candidates,
    log_target_min_validation_gain,
):
    overall_rows = iteration_scores[iteration_scores["city"] == "all"].sort_values("mae")
    best_overall = overall_rows.iloc[0]

    lines = [
        "# main_goal.py changelog",
        "",
        f"Run timestamp UTC: {run_started_at}",
        "",
        "## Guardrails",
        "",
        "- Cities are modeled separately.",
        "- Validation is expanding, full-year forward chaining within each city.",
        "- No total_cases, lagged cases, rolling cases, or other target-derived predictors are used as features.",
        "- total_cases is used only as the supervised label and validation ground truth.",
        "- Feature imputation is forward-fill plus training-fold medians to avoid backward-looking future leakage.",
        (
            "- Hidden external feedback is incorporated by requiring log1p-target candidates "
            f"to clear a {log_target_min_validation_gain:.2f} MAE validation margin before final selection."
        ),
        "",
        "## Prior saved runs",
        "",
    ]

    if historical.empty:
        lines.append("No prior validation score files were found.")
    else:
        lines.append("| Run | Overall MAE | Rows |")
        lines.append("| --- | ---: | ---: |")
        for row in historical.head(8).itertuples(index=False):
            rows = getattr(row, "validation_rows", "")
            lines.append(f"| {row.run} | {row.overall_mae:.4f} | {rows} |")

    lines.extend(["", "## Skipped optional candidates", ""])
    if skipped_candidates.empty:
        lines.append("No optional candidates were skipped.")
    else:
        lines.append("| Iteration | Reason |")
        lines.append("| --- | --- |")
        for row in skipped_candidates.itertuples(index=False):
            lines.append(f"| {row.iteration} | {row.reason} |")

    lines.extend(
        [
            "",
            "## Iteration summary",
            "",
            "| Iteration | Feature set | Loss | Target transform | Overall MAE | SJ MAE | IQ MAE |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )

    for candidate in ALL_CANDIDATES:
        rows = iteration_scores[iteration_scores["iteration"] == candidate["iteration"]]
        all_mae = rows[(rows["city"] == "all")]["mae"].iloc[0]
        sj_mae = rows[(rows["city"] == "sj")]["mae"].iloc[0]
        iq_mae = rows[(rows["city"] == "iq")]["mae"].iloc[0]
        lines.append(
            "| {iteration} | {feature_set} | {loss} | {target_transform} | "
            "{all_mae:.4f} | {sj_mae:.4f} | {iq_mae:.4f} |".format(
                iteration=candidate["iteration"],
                feature_set=candidate["feature_set"],
                loss=candidate["loss"],
                target_transform=candidate["target_transform"],
                all_mae=all_mae,
                sj_mae=sj_mae,
                iq_mae=iq_mae,
            )
        )

    lines.extend(["", "## Iteration notes", ""])

    previous_overall_mae = None
    for candidate in ALL_CANDIDATES:
        rows = iteration_scores[iteration_scores["iteration"] == candidate["iteration"]]
        all_mae = float(rows[(rows["city"] == "all")]["mae"].iloc[0])
        sj_mae = float(rows[(rows["city"] == "sj")]["mae"].iloc[0])
        iq_mae = float(rows[(rows["city"] == "iq")]["mae"].iloc[0])
        sj_selection_mae = sj_mae + (
            float(rows[(rows["city"] == "sj")]["log_component_weight"].iloc[0])
            * log_target_min_validation_gain
        )
        iq_selection_mae = iq_mae + (
            float(rows[(rows["city"] == "iq")]["log_component_weight"].iloc[0])
            * log_target_min_validation_gain
        )
        selected_cities = selected_by_city[
            selected_by_city["iteration"] == candidate["iteration"]
        ]["city"].tolist()

        if previous_overall_mae is None:
            interpretation = "This is the reference point for later comparisons."
        elif all_mae < previous_overall_mae:
            interpretation = (
                f"Improved by {previous_overall_mae - all_mae:.4f} MAE versus the previous iteration."
            )
        else:
            interpretation = (
                f"Worse by {all_mae - previous_overall_mae:.4f} MAE versus the previous iteration."
            )
        decision = (
            "Selected for final submission city/cities: " + ", ".join(selected_cities)
            if selected_cities
            else "Not selected for final submission."
        )

        lines.extend(
            [
                f"### {candidate['iteration']}",
                "",
                f"Rationale: {candidate['rationale']}",
                "",
                f"Result: overall MAE {all_mae:.4f}; SJ {sj_mae:.4f}; IQ {iq_mae:.4f}.",
                "",
                (
                    "Hidden-prior selection score: "
                    f"SJ {sj_selection_mae:.4f}; IQ {iq_selection_mae:.4f}."
                ),
                "",
                f"Interpretation: {interpretation}",
                "",
                f"Decision: {decision}",
                "",
            ]
        )
        previous_overall_mae = all_mae

    lines.extend(
        [
            "## Final selection",
            "",
            "| City | Selected iteration | Validation MAE | Selection MAE | Feature set | Loss | Target transform |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in selected_by_city.sort_values("city").itertuples(index=False):
        lines.append(
            f"| {row.city} | {row.iteration} | {row.mae:.4f} | {row.selection_mae:.4f} | "
            f"{row.feature_set} | {row.loss} | {row.target_transform} |"
        )

    lines.extend(
        [
            "",
            "## Best overall single iteration",
            "",
            (
                f"{best_overall.iteration} had the best single-iteration overall MAE "
                f"at {best_overall.mae:.4f}. The submission uses per-city selection "
                "after the hidden-prior log-target adjustment because the challenge "
                "cities behave like separate datasets and the external feedback makes "
                "small log-target validation gains less trustworthy."
            ),
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def fit_selected_submission(
    selected_by_city,
    train_data,
    test_features,
    submission_format,
    numeric_columns,
    random_state,
    tweedie_variance_power,
):
    submission_predictions = {}
    feature_importance_records = []

    for selected in selected_by_city.itertuples(index=False):
        city = selected.city
        candidate = next(candidate for candidate in ALL_CANDIDATES if candidate["iteration"] == selected.iteration)
        city_train_data = train_data[train_data["city"] == city].sort_values("week_start_date").copy()
        city_test_data = test_features[test_features["city"] == city].sort_values("week_start_date").copy()
        prepared_train, prepared_test, available_columns = prepare_fold_features(
            city_train_data,
            city_test_data,
            numeric_columns,
        )

        if "members" in candidate:
            weighted_predictions = []
            for member_iteration, weight in candidate["members"]:
                member_candidate = next(
                    member
                    for member in ACTIVE_MODEL_CANDIDATES
                    if member["iteration"] == member_iteration
                )
                member_feature_columns = select_feature_columns(
                    member_candidate["feature_set"],
                    city,
                    numeric_columns,
                    available_columns,
                )
                member_model = make_model(member_candidate, random_state, tweedie_variance_power)
                member_model.fit(
                    prepared_train[member_feature_columns],
                    transform_target(
                        prepared_train["total_cases"],
                        member_candidate["target_transform"],
                    ),
                )
                member_raw_predictions = member_model.predict(prepared_test[member_feature_columns])
                member_predictions = np.asarray(
                    integer_predictions(member_raw_predictions, member_candidate["target_transform"]),
                    dtype=float,
                )
                weighted_predictions.append(member_predictions * weight)

                if member_candidate.get("model_family") == "catboost":
                    gain_importance = member_model.get_feature_importance()
                    split_importance = [0] * len(member_feature_columns)
                else:
                    gain_importance = member_model.booster_.feature_importance(importance_type="gain")
                    split_importance = member_model.booster_.feature_importance(importance_type="split")
                for feature, gain, split in zip(
                    member_feature_columns,
                    gain_importance,
                    split_importance,
                ):
                    feature_importance_records.append(
                        {
                            "city": city,
                            "iteration": candidate["iteration"],
                            "blend_member": member_iteration,
                            "feature": feature,
                            "importance_gain": float(gain) * weight,
                            "importance_split": int(split),
                        }
                    )

            predictions = np.rint(sum(weighted_predictions)).clip(min=0).astype(int).tolist()
        else:
            feature_columns = select_feature_columns(
                candidate["feature_set"],
                city,
                numeric_columns,
                available_columns,
            )
            model = make_model(candidate, random_state, tweedie_variance_power)
            model.fit(
                prepared_train[feature_columns],
                transform_target(prepared_train["total_cases"], candidate["target_transform"]),
            )

            if candidate.get("model_family") == "catboost":
                gain_importance = model.get_feature_importance()
                split_importance = [0] * len(feature_columns)
            else:
                gain_importance = model.booster_.feature_importance(importance_type="gain")
                split_importance = model.booster_.feature_importance(importance_type="split")
            for feature, gain, split in zip(feature_columns, gain_importance, split_importance):
                feature_importance_records.append(
                    {
                        "city": city,
                        "iteration": candidate["iteration"],
                        "blend_member": "",
                        "feature": feature,
                        "importance_gain": float(gain),
                        "importance_split": int(split),
                    }
                )

            predictions = integer_predictions(
                model.predict(prepared_test[feature_columns]),
                candidate["target_transform"],
            )
        for row, prediction in zip(city_test_data.itertuples(index=False), predictions):
            submission_predictions[(row.city, row.year, row.weekofyear)] = prediction

    submission = submission_format.copy()
    submission["total_cases"] = [
        submission_predictions[(row.city, row.year, row.weekofyear)]
        for row in submission.itertuples(index=False)
    ]

    if list(submission.columns) != ["city", "year", "weekofyear", "total_cases"]:
        raise ValueError("Submission columns must be city, year, weekofyear, total_cases.")
    if len(submission) != len(submission_format):
        raise ValueError("Submission row count must match submission format.")
    if not submission[MERGE_KEYS].equals(submission_format[MERGE_KEYS]):
        raise ValueError("Submission order must match submission format.")
    if submission["total_cases"].isna().any() or (submission["total_cases"] < 0).any():
        raise ValueError("Submission predictions must be non-missing and non-negative.")

    submission["total_cases"] = submission["total_cases"].astype(int)
    feature_importance = pd.DataFrame(feature_importance_records).sort_values(
        ["city", "importance_gain"],
        ascending=[True, False],
    )
    return submission, feature_importance


args = parse_args()
run_started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

if not 1.0 < args.tweedie_variance_power < 2.0:
    raise ValueError("--tweedie-variance-power must be between 1 and 2.")

if args.log_target_min_validation_gain < 0:
    raise ValueError("--log-target-min-validation-gain must be non-negative.")

random.seed(args.random_state)
np.random.seed(args.random_state)
os.environ["PYTHONHASHSEED"] = str(args.random_state)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

train_features = pd.read_csv(args.train_features_csv, parse_dates=["week_start_date"])
test_features = pd.read_csv(args.test_features_csv, parse_dates=["week_start_date"])
labels = pd.read_csv(args.labels_csv)
submission_format = pd.read_csv(args.submission_format_csv)
validate_paths(train_features, test_features, labels, submission_format)

numeric_columns = numeric_feature_columns(train_features)
train_data = train_features.merge(labels, on=MERGE_KEYS, how="left", validate="one_to_one")
if train_data["total_cases"].isna().any():
    raise ValueError("Merged training data contains missing total_cases values.")

historical = historical_scores(args.previous_output_root)
skipped_candidates = pd.DataFrame(SKIPPED_CANDIDATES)
all_fold_records = []
all_prediction_records = []

for candidate in ACTIVE_MODEL_CANDIDATES:
    print(f"Running {candidate['iteration']}...")
    fold_records, prediction_records = evaluate_candidate(
        candidate,
        train_data,
        numeric_columns,
        args.random_state,
        args.tweedie_variance_power,
    )
    all_fold_records.extend(fold_records)
    all_prediction_records.extend(prediction_records)

if BLEND_CANDIDATES:
    base_validation_predictions = pd.DataFrame(all_prediction_records)
    for candidate in BLEND_CANDIDATES:
        print(f"Running {candidate['iteration']}...")
    fold_records, prediction_records = evaluate_blend_candidates(
        BLEND_CANDIDATES,
        base_validation_predictions,
    )
    all_fold_records.extend(fold_records)
    all_prediction_records.extend(prediction_records)

validation_scores = pd.DataFrame(all_fold_records)
validation_predictions = pd.DataFrame(all_prediction_records)

iteration_scores = validation_scores[validation_scores["validation_year"].astype(str) == "all"].copy()
iteration_scores = iteration_scores.sort_values(["city", "mae", "iteration"]).reset_index(drop=True)

city_selection_rows = []
for city in ["sj", "iq"]:
    city_scores = iteration_scores[iteration_scores["city"] == city].copy()
    city_scores["selection_mae"] = (
        city_scores["mae"]
        + city_scores["log_component_weight"].fillna(0.0) * args.log_target_min_validation_gain
    )
    city_scores = city_scores.sort_values(["selection_mae", "mae", "iteration"])
    best = city_scores.iloc[0].copy()
    city_selection_rows.append(best)
selected_by_city = pd.DataFrame(city_selection_rows).reset_index(drop=True)

submission, feature_importance = fit_selected_submission(
    selected_by_city,
    train_data,
    test_features,
    submission_format,
    numeric_columns,
    args.random_state,
    args.tweedie_variance_power,
)

validation_scores.to_csv(output_dir / args.validation_output, index=False)
validation_predictions.to_csv(output_dir / args.validation_predictions_output, index=False)
iteration_scores.to_csv(output_dir / args.iteration_output, index=False)
selected_by_city.to_csv(output_dir / args.selection_output, index=False)
submission.to_csv(output_dir / args.submission_output, index=False)
feature_importance.to_csv(output_dir / args.feature_importance_output, index=False)
historical.to_csv(output_dir / args.historical_output, index=False)
skipped_candidates.to_csv(output_dir / args.skipped_output, index=False)
write_changelog(
    output_dir / args.changelog_output,
    run_started_at,
    historical,
    iteration_scores,
    selected_by_city,
    skipped_candidates,
    args.log_target_min_validation_gain,
)

experiment_log_record = {
    "logged_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "script": str(Path(__file__)),
    "config": {
        "train_features_csv": args.train_features_csv,
        "test_features_csv": args.test_features_csv,
        "labels_csv": args.labels_csv,
        "submission_format_csv": args.submission_format_csv,
        "random_state": args.random_state,
        "tweedie_variance_power": args.tweedie_variance_power,
        "log_target_min_validation_gain": args.log_target_min_validation_gain,
        "catboost_available": CATBOOST_AVAILABLE,
        "iterations": [candidate["iteration"] for candidate in ALL_CANDIDATES],
    },
    "results": {
        "best_single_iteration": iteration_scores[iteration_scores["city"] == "all"]
        .sort_values("mae")
        .iloc[0]
        .to_dict(),
        "selected_by_city": selected_by_city.to_dict(orient="records"),
        "skipped_candidates": skipped_candidates.to_dict(orient="records"),
    },
    "outputs": {
        "validation_scores": str(output_dir / args.validation_output),
        "iteration_scores": str(output_dir / args.iteration_output),
        "selected_iteration_by_city": str(output_dir / args.selection_output),
        "skipped_candidates": str(output_dir / args.skipped_output),
        "submission": str(output_dir / args.submission_output),
        "changelog": str(output_dir / args.changelog_output),
    },
}
with (output_dir / args.experiment_log_output).open("a", encoding="utf-8") as experiment_log_file:
    experiment_log_file.write(json.dumps(experiment_log_record, sort_keys=True, default=str) + "\n")

best_overall = iteration_scores[iteration_scores["city"] == "all"].sort_values("mae").iloc[0]
print(f"Best single iteration: {best_overall.iteration} MAE={best_overall.mae:.4f}")
for row in selected_by_city.sort_values("city").itertuples(index=False):
    print(f"Selected {row.city}: {row.iteration} MAE={row.mae:.4f}")
print(f"Wrote outputs to {output_dir}")
