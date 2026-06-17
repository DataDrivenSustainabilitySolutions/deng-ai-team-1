"""Additive no-log DengAI iteration after the main_goal run.

Example:
    python3 oliver/sandbox/main_goal_additive.py

This script keeps earlier sandbox results intact by writing each run to a
timestamped directory under oliver/sandbox/outputs/main_goal_additive/.
It only tests no-log, non-target-derived seasonal candidates.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dengai_main_goal_additive_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

try:
    from lightgbm import LGBMRegressor
except ImportError as error:
    raise ImportError(
        "LightGBM is required. Install dependencies from pyproject.toml, "
        "for example: python3 -m pip install -e ."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "main_goal_additive"
DEFAULT_MAIN_GOAL_DIR = SCRIPT_DIR / "outputs" / "main_goal"
MERGE_KEYS = ["city", "year", "weekofyear"]
CALENDAR_COLUMNS = ["weekofyear_sin", "weekofyear_cos"]
FIXED_RANDOM_STATE = 42

BASE_CANDIDATES = [
    {
        "iteration": "a01_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar",
        "rationale": "Re-run the previous selected no-log calendar LightGBM as the additive baseline.",
    },
    {
        "iteration": "a02_calendar_lgbm_l1_recent_weighted",
        "model_type": "lgbm_calendar_recent_weighted",
        "weight_floor": 0.5,
        "weight_power": 1.0,
        "rationale": "Favor recent years modestly because the test set is a future holdout.",
    },
    {
        "iteration": "a03_calendar_lgbm_l1_with_year_index",
        "model_type": "lgbm_calendar_year_index",
        "rationale": "Check whether a simple non-target time trend helps without adding weather complexity.",
    },
    {
        "iteration": "a04_week_median_profile",
        "model_type": "week_profile",
        "stat": "median",
        "window": 0,
        "recent_years": 0,
        "rationale": "Use the direct week-of-year median, the natural MAE baseline for a seasonal count pattern.",
    },
    {
        "iteration": "a05_week_mean_profile",
        "model_type": "week_profile",
        "stat": "mean",
        "window": 0,
        "recent_years": 0,
        "rationale": "Compare the week mean against the median to see whether outbreak years should pull predictions up.",
    },
    {
        "iteration": "a06_week_median_profile_pm1",
        "model_type": "week_profile",
        "stat": "median",
        "window": 1,
        "recent_years": 0,
        "rationale": "Smooth the weekly median over neighboring weeks to reduce noisy week-specific estimates.",
    },
    {
        "iteration": "a07_week_median_profile_pm2",
        "model_type": "week_profile",
        "stat": "median",
        "window": 2,
        "recent_years": 0,
        "rationale": "Try a slightly smoother seasonal median while staying transparent and low capacity.",
    },
    {
        "iteration": "a08_recent5_week_median_profile",
        "model_type": "week_profile",
        "stat": "median",
        "window": 0,
        "recent_years": 5,
        "rationale": "Use only the latest five training years for the seasonal median to test recency bias.",
    },
    {
        "iteration": "a10_calendar_lgbm_l1_recent_weighted_soft",
        "model_type": "lgbm_calendar_recent_weighted",
        "weight_floor": 0.75,
        "weight_power": 1.0,
        "rationale": "Use a softer recency weight to see whether a smaller future bias keeps all-year stability.",
    },
    {
        "iteration": "a11_calendar_lgbm_l1_recent_weighted_strong",
        "model_type": "lgbm_calendar_recent_weighted",
        "weight_floor": 0.25,
        "weight_power": 1.0,
        "rationale": "Use a stronger recency weight after SJ improved on the latest validation years.",
    },
    {
        "iteration": "a12_calendar_lgbm_l1_recent_weighted_curved",
        "model_type": "lgbm_calendar_recent_weighted",
        "weight_floor": 0.4,
        "weight_power": 2.0,
        "rationale": "Concentrate extra weight on the newest years without fully discarding older seasons.",
    },
    {
        "iteration": "a13_recent8_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 8,
        "rationale": "Fit the same calendar model using only the latest eight training years as a recency stress test.",
    },
    {
        "iteration": "a14_recent6_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 6,
        "rationale": "Check whether a shorter six-year calendar window improves future alignment without too little data.",
    },
    {
        "iteration": "a15_recent7_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 7,
        "rationale": "Test the neighboring seven-year window around the current eight-year winner.",
    },
    {
        "iteration": "a16_recent9_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 9,
        "rationale": "Test the neighboring nine-year window around the current eight-year winner.",
    },
    {
        "iteration": "a17_recent10_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 10,
        "rationale": "Check whether a ten-year window keeps the recency benefit with more seasonal history.",
    },
    {
        "iteration": "a18_recent12_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 12,
        "rationale": "Check whether a wider twelve-year window remains better than using all years.",
    },
    {
        "iteration": "a19_calendar_lgbm_l1_harmonic2",
        "model_type": "lgbm_calendar",
        "max_harmonic": 2,
        "rationale": "Add second-harmonic seasonal terms to capture asymmetry while staying calendar-only.",
    },
    {
        "iteration": "a20_recent8_calendar_lgbm_l1_harmonic2",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 8,
        "max_harmonic": 2,
        "rationale": "Combine the best recent-window idea with second-harmonic seasonal terms.",
    },
    {
        "iteration": "a21_recent4_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 4,
        "rationale": "Test a shorter four-year recent window after IQ improved with six recent years.",
    },
    {
        "iteration": "a22_recent5_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 5,
        "rationale": "Test a five-year recent calendar model between the direct recent profile and six-year winner.",
    },
    {
        "iteration": "a23_recent6_calendar_lgbm_l1_harmonic2",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 6,
        "max_harmonic": 2,
        "rationale": "Check whether the IQ-favored six-year window benefits from a richer seasonal shape.",
    },
    {
        "iteration": "a24_recent7_calendar_lgbm_l1_harmonic2",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 7,
        "max_harmonic": 2,
        "rationale": "Check whether the SJ-favored seven-year window benefits from a richer seasonal shape.",
    },
    {
        "iteration": "a25_recent2_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 2,
        "rationale": "Test a very short two-year window to bound the IQ recency effect.",
    },
    {
        "iteration": "a26_recent3_calendar_lgbm_l1_raw",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 3,
        "rationale": "Test a three-year window just below the current IQ four-year winner.",
    },
    {
        "iteration": "a27_recent4_calendar_lgbm_l1_harmonic2",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 4,
        "max_harmonic": 2,
        "rationale": "Check whether the IQ-favored four-year window benefits from second-harmonic seasonality.",
    },
    {
        "iteration": "a28_recent7_calendar_lgbm_l1_harmonic3",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 7,
        "max_harmonic": 3,
        "rationale": "Test one extra harmonic on the current SJ seven-year harmonic winner.",
    },
    {
        "iteration": "a29_recent2_calendar_lgbm_l1_harmonic2",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 2,
        "max_harmonic": 2,
        "rationale": "Check whether the IQ all-year-strong two-year window benefits from second-harmonic seasonality.",
    },
    {
        "iteration": "a30_recent3_calendar_lgbm_l1_harmonic2",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 3,
        "max_harmonic": 2,
        "rationale": "Check whether the three-year window gains enough shape from second-harmonic seasonality to compete with the four-year winner.",
    },
    {
        "iteration": "a31_recent4_calendar_lgbm_l1_harmonic3",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 4,
        "max_harmonic": 3,
        "rationale": "Test whether one extra harmonic improves the current IQ four-year harmonic winner or starts overfitting.",
    },
    {
        "iteration": "a32_recent7_calendar_lgbm_l1_harmonic2_regularized",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 7,
        "max_harmonic": 2,
        "lgbm_overrides": {
            "num_leaves": 7,
            "min_child_samples": 30,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
        },
        "rationale": "Retest the current SJ seven-year harmonic winner with a smaller, more regularized tree shape.",
    },
    {
        "iteration": "a33_recent4_calendar_lgbm_l1_harmonic2_regularized",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 4,
        "max_harmonic": 2,
        "lgbm_overrides": {
            "num_leaves": 7,
            "min_child_samples": 30,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
        },
        "rationale": "Retest the current IQ four-year harmonic winner with a smaller, more regularized tree shape.",
    },
    {
        "iteration": "a34_recent7_calendar_rf_abs_harmonic2",
        "model_type": "rf_calendar_recent_window",
        "recent_train_years": 7,
        "max_harmonic": 2,
        "rationale": "Mirror the SJ seven-year harmonic setup with an absolute-error Random Forest to test whether bagging beats boosting here.",
    },
    {
        "iteration": "a35_recent4_calendar_rf_abs_harmonic2",
        "model_type": "rf_calendar_recent_window",
        "recent_train_years": 4,
        "max_harmonic": 2,
        "rationale": "Mirror the IQ four-year harmonic setup with an absolute-error Random Forest to test model-family sensitivity.",
    },
    {
        "iteration": "a36_recent2_calendar_rf_abs_harmonic2",
        "model_type": "rf_calendar_recent_window",
        "recent_train_years": 2,
        "max_harmonic": 2,
        "rationale": "Test whether Random Forest stabilizes the very short IQ-favored two-year harmonic window.",
    },
    {
        "iteration": "a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 7,
        "max_harmonic": 2,
        "lgbm_overrides": {
            "num_leaves": 5,
            "min_child_samples": 40,
            "reg_alpha": 0.4,
            "reg_lambda": 4.0,
        },
        "rationale": "Check whether the SJ seven-year harmonic winner improves with one more step of shrinkage.",
    },
    {
        "iteration": "a38_recent4_calendar_lgbm_l1_harmonic2_more_regularized",
        "model_type": "lgbm_calendar_recent_window",
        "recent_train_years": 4,
        "max_harmonic": 2,
        "lgbm_overrides": {
            "num_leaves": 5,
            "min_child_samples": 40,
            "reg_alpha": 0.4,
            "reg_lambda": 4.0,
        },
        "rationale": "Check whether the IQ four-year harmonic winner improves with one more step of shrinkage.",
    },
    {
        "iteration": "a39_recent7_calendar_quantile_harmonic2",
        "model_type": "quantile_calendar_recent_window",
        "recent_train_years": 7,
        "max_harmonic": 2,
        "rationale": "Use a simple median linear seasonal curve for the SJ seven-year harmonic setup.",
    },
    {
        "iteration": "a40_recent4_calendar_quantile_harmonic2",
        "model_type": "quantile_calendar_recent_window",
        "recent_train_years": 4,
        "max_harmonic": 2,
        "rationale": "Use a simple median linear seasonal curve for the IQ four-year harmonic setup.",
    },
    {
        "iteration": "a41_recent7_calendar_poisson_glm_harmonic2",
        "model_type": "glm_poisson_calendar_recent_window",
        "recent_train_years": 7,
        "max_harmonic": 2,
        "rationale": "Fit a statistical Poisson count GLM for the SJ seven-year harmonic setup using raw counts.",
    },
    {
        "iteration": "a42_recent4_calendar_poisson_glm_harmonic2",
        "model_type": "glm_poisson_calendar_recent_window",
        "recent_train_years": 4,
        "max_harmonic": 2,
        "rationale": "Fit a statistical Poisson count GLM for the IQ four-year harmonic setup using raw counts.",
    },
    {
        "iteration": "a43_recent7_calendar_negbin_glm_harmonic2",
        "model_type": "glm_negbin_calendar_recent_window",
        "recent_train_years": 7,
        "max_harmonic": 2,
        "rationale": "Fit a statistical negative-binomial count GLM for the SJ seven-year harmonic setup to allow overdispersion.",
    },
    {
        "iteration": "a44_recent4_calendar_negbin_glm_harmonic2",
        "model_type": "glm_negbin_calendar_recent_window",
        "recent_train_years": 4,
        "max_harmonic": 2,
        "rationale": "Fit a statistical negative-binomial count GLM for the IQ four-year harmonic setup to allow overdispersion.",
    },
]

BLEND_CANDIDATES = [
    {
        "iteration": "a09_calendar_lgbm_pm1_median_blend",
        "model_type": "blend",
        "members": [
            ("a01_calendar_lgbm_l1_raw", 0.5),
            ("a06_week_median_profile_pm1", 0.5),
        ],
        "rationale": "Average the best flexible calendar learner with the smoothed median profile.",
    },
    {
        "iteration": "a45_iq_lgbm_frontier_blend",
        "model_type": "blend",
        "members": [
            ("a29_recent2_calendar_lgbm_l1_harmonic2", 0.5),
            ("a33_recent4_calendar_lgbm_l1_harmonic2_regularized", 0.5),
        ],
        "rationale": "Blend the two IQ models that split the selection-sensitivity grid to reduce selection-rule brittleness.",
    },
    {
        "iteration": "a46_iq_lgbm_rf_frontier_blend",
        "model_type": "blend",
        "members": [
            ("a29_recent2_calendar_lgbm_l1_harmonic2", 0.3333333333333333),
            ("a33_recent4_calendar_lgbm_l1_harmonic2_regularized", 0.3333333333333333),
            ("a35_recent4_calendar_rf_abs_harmonic2", 0.3333333333333333),
        ],
        "rationale": "Add the close IQ Random Forest check to the frontier blend as a small model-family hedge.",
    },
    {
        "iteration": "a47_sj_quantile_lgbm_frontier_blend",
        "model_type": "blend",
        "members": [
            ("a39_recent7_calendar_quantile_harmonic2", 0.5),
            ("a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized", 0.5),
        ],
        "rationale": "Blend the two strongest broad-stability SJ seasonal models to test whether averaging improves the selected quantile model.",
    },
    {
        "iteration": "a48_iq_lgbm_rf_frontier_blend_recent_tilt",
        "model_type": "blend",
        "members": [
            ("a29_recent2_calendar_lgbm_l1_harmonic2", 0.2),
            ("a33_recent4_calendar_lgbm_l1_harmonic2_regularized", 0.5),
            ("a35_recent4_calendar_rf_abs_harmonic2", 0.3),
        ],
        "rationale": "Tilt the IQ frontier blend toward the regularized recent-stable model while keeping the short-window and RF hedge.",
    },
    {
        "iteration": "a49_iq_lgbm_rf_frontier_blend_short_tilt",
        "model_type": "blend",
        "members": [
            ("a29_recent2_calendar_lgbm_l1_harmonic2", 0.5),
            ("a33_recent4_calendar_lgbm_l1_harmonic2_regularized", 0.25),
            ("a35_recent4_calendar_rf_abs_harmonic2", 0.25),
        ],
        "rationale": "Tilt the IQ frontier blend toward the all-year-strong short-window model without relying on it alone.",
    },
    {
        "iteration": "a50_sj_quantile_lgbm_frontier_blend_quantile_tilt",
        "model_type": "blend",
        "members": [
            ("a39_recent7_calendar_quantile_harmonic2", 0.75),
            ("a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized", 0.25),
        ],
        "rationale": "Tilt the SJ frontier blend toward the simpler quantile seasonal model.",
    },
    {
        "iteration": "a51_sj_quantile_lgbm_frontier_blend_lgbm_tilt",
        "model_type": "blend",
        "members": [
            ("a39_recent7_calendar_quantile_harmonic2", 0.25),
            ("a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized", 0.75),
        ],
        "rationale": "Tilt the SJ frontier blend toward the regularized tree seasonal model.",
    },
]

ALL_CANDIDATES = BASE_CANDIDATES + BLEND_CANDIDATES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run additive no-log seasonal DengAI iterations without overwriting older results."
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
        "--main-goal-output-dir",
        default=DEFAULT_MAIN_GOAL_DIR,
        help="Prior main_goal output directory for comparison. Default: oliver/sandbox/outputs/main_goal",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for timestamped additive outputs. Default: oliver/sandbox/outputs/main_goal_additive",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional run directory name. Default: UTC timestamp.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=FIXED_RANDOM_STATE,
        help="Fixed random seed. Must remain 42; seed tuning is not allowed.",
    )
    parser.add_argument(
        "--recent-validation-years",
        type=int,
        default=4,
        help="Number of latest validation years per city used in future-oriented scoring. Default: 4",
    )
    parser.add_argument(
        "--recent-score-weight",
        type=float,
        default=0.3,
        help="Weight for recent-year MAE in final additive selection score. Default: 0.3",
    )
    parser.add_argument(
        "--max-city-all-year-mae-over-best",
        type=float,
        default=0.35,
        help=(
            "City-level anti-overfit guardrail: select only among candidates within this "
            "MAE of the city's best all-year MAE. Default: 0.35"
        ),
    )
    parser.add_argument(
        "--sensitivity-recent-score-weights",
        default="0,0.15,0.3,0.45,0.6",
        help=(
            "Comma-separated recent-score weights used only for selection sensitivity reporting. "
            "Default: 0,0.15,0.3,0.45,0.6"
        ),
    )
    parser.add_argument(
        "--sensitivity-all-year-guardrails",
        default="0,0.2,0.35,0.75,inf",
        help=(
            "Comma-separated all-year guardrails used only for selection sensitivity reporting; "
            "use inf for no guardrail. Default: 0,0.2,0.35,0.75,inf"
        ),
    )
    return parser.parse_args()


def validate_inputs(train_features, test_features, labels, submission_format):
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError("Train and test features must have identical columns.")
    if train_features[MERGE_KEYS].duplicated().any() or test_features[MERGE_KEYS].duplicated().any():
        raise ValueError("Feature rows must be unique by city, year, and weekofyear.")
    if labels[MERGE_KEYS].duplicated().any():
        raise ValueError("Label rows must be unique by city, year, and weekofyear.")
    if not submission_format[MERGE_KEYS].equals(test_features[MERGE_KEYS]):
        raise ValueError("Submission format rows must match test feature order.")


def calendar_feature_names(max_harmonic):
    columns = []
    for harmonic in range(1, max_harmonic + 1):
        if harmonic == 1:
            columns.extend(["weekofyear_sin", "weekofyear_cos"])
        else:
            columns.extend(
                [
                    f"weekofyear_sin_{harmonic}",
                    f"weekofyear_cos_{harmonic}",
                ]
            )
    return columns


def add_calendar_features(data, max_harmonic=1):
    output = data.copy()
    base_angle = 2 * math.pi * output["weekofyear"] / 52.0
    for harmonic in range(1, max_harmonic + 1):
        week_angle = harmonic * base_angle
        if harmonic == 1:
            output["weekofyear_sin"] = np.sin(week_angle)
            output["weekofyear_cos"] = np.cos(week_angle)
        else:
            output[f"weekofyear_sin_{harmonic}"] = np.sin(week_angle)
            output[f"weekofyear_cos_{harmonic}"] = np.cos(week_angle)
    return output


def validation_years(city_data):
    year_counts = city_data.groupby("year").size()
    full_years = [int(year) for year, rows in year_counts.items() if rows == 52]
    first_year = int(city_data["year"].min())
    later_full_years = [year for year in full_years if year > first_year]
    first_full_after_start = min(later_full_years)
    return [year for year in full_years if year > first_full_after_start]


def lgbm_params(random_state, overrides=None):
    params = {
        "objective": "regression_l1",
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
    if overrides:
        params.update(overrides)
    return params


def rf_params(random_state, overrides=None):
    params = {
        "n_estimators": 500,
        "criterion": "absolute_error",
        "max_depth": 4,
        "min_samples_leaf": 8,
        "min_samples_split": 16,
        "max_features": 1.0,
        "bootstrap": True,
        "random_state": random_state,
        "n_jobs": 1,
    }
    if overrides:
        params.update(overrides)
    return params


def quantile_params(overrides=None):
    params = {
        "quantile": 0.5,
        "alpha": 0.01,
        "solver": "highs",
    }
    if overrides:
        params.update(overrides)
    return params


def integerize(predictions):
    return [int(round(max(0.0, float(prediction)))) for prediction in predictions]


def recent_weights(train_rows, weight_floor, weight_power):
    ordered_years = sorted(train_rows["year"].unique())
    year_rank = {year: rank for rank, year in enumerate(ordered_years)}
    max_rank = max(year_rank.values()) if year_rank else 0
    if max_rank == 0:
        return np.ones(len(train_rows))
    ranks = train_rows["year"].map(year_rank).to_numpy(dtype=float)
    scaled_rank = (ranks / max_rank) ** weight_power
    return weight_floor + (1.0 - weight_floor) * scaled_rank


def lgbm_training_rows(train_rows, candidate):
    recent_train_years = int(candidate.get("recent_train_years", 0))
    if recent_train_years <= 0:
        return train_rows
    years = sorted(train_rows["year"].unique())
    keep_years = set(years[-recent_train_years:])
    return train_rows[train_rows["year"].isin(keep_years)].copy()


def profile_training_rows(train_rows, recent_years):
    if recent_years <= 0:
        return train_rows
    years = sorted(train_rows["year"].unique())
    keep_years = set(years[-recent_years:])
    return train_rows[train_rows["year"].isin(keep_years)].copy()


def wrapped_weeks(week, window):
    weeks = []
    for offset in range(-window, window + 1):
        wrapped = ((int(week) + offset - 1) % 52) + 1
        weeks.append(wrapped)
    return weeks


def predict_week_profile(train_rows, future_rows, candidate):
    profile_rows = profile_training_rows(train_rows, candidate.get("recent_years", 0))
    fallback = float(profile_rows["total_cases"].median())
    predictions = []

    for week in future_rows["weekofyear"]:
        candidate_weeks = wrapped_weeks(week, candidate.get("window", 0))
        values = profile_rows[profile_rows["weekofyear"].isin(candidate_weeks)]["total_cases"]
        if values.empty:
            prediction = fallback
        elif candidate["stat"] == "mean":
            prediction = float(values.mean())
        else:
            prediction = float(values.median())
        predictions.append(prediction)

    return integerize(predictions)


def predict_lgbm(train_rows, future_rows, candidate, random_state):
    max_harmonic = int(candidate.get("max_harmonic", 1))
    train_model = add_calendar_features(lgbm_training_rows(train_rows, candidate), max_harmonic)
    future_model = add_calendar_features(future_rows, max_harmonic)
    feature_columns = calendar_feature_names(max_harmonic)

    if candidate["model_type"] == "lgbm_calendar_year_index":
        first_year = int(train_model["year"].min())
        train_model["year_index"] = train_model["year"] - first_year
        future_model["year_index"] = future_model["year"] - first_year
        feature_columns.append("year_index")

    sample_weight = None
    if candidate["model_type"] == "lgbm_calendar_recent_weighted":
        sample_weight = recent_weights(
            train_model,
            candidate.get("weight_floor", 0.5),
            candidate.get("weight_power", 1.0),
        )

    model = LGBMRegressor(**lgbm_params(random_state, candidate.get("lgbm_overrides")))
    model.fit(
        train_model[feature_columns],
        train_model["total_cases"],
        sample_weight=sample_weight,
    )
    return integerize(model.predict(future_model[feature_columns]))


def predict_rf(train_rows, future_rows, candidate, random_state):
    max_harmonic = int(candidate.get("max_harmonic", 1))
    train_model = add_calendar_features(lgbm_training_rows(train_rows, candidate), max_harmonic)
    future_model = add_calendar_features(future_rows, max_harmonic)
    feature_columns = calendar_feature_names(max_harmonic)

    model = RandomForestRegressor(**rf_params(random_state, candidate.get("rf_overrides")))
    model.fit(train_model[feature_columns], train_model["total_cases"])
    return integerize(model.predict(future_model[feature_columns]))


def predict_quantile(train_rows, future_rows, candidate):
    max_harmonic = int(candidate.get("max_harmonic", 1))
    train_model = add_calendar_features(lgbm_training_rows(train_rows, candidate), max_harmonic)
    future_model = add_calendar_features(future_rows, max_harmonic)
    feature_columns = calendar_feature_names(max_harmonic)

    model = QuantileRegressor(**quantile_params(candidate.get("quantile_overrides")))
    model.fit(train_model[feature_columns], train_model["total_cases"])
    return integerize(model.predict(future_model[feature_columns]))


def predict_glm_count(train_rows, future_rows, candidate):
    max_harmonic = int(candidate.get("max_harmonic", 1))
    train_model = add_calendar_features(lgbm_training_rows(train_rows, candidate), max_harmonic)
    future_model = add_calendar_features(future_rows, max_harmonic)
    feature_columns = calendar_feature_names(max_harmonic)
    train_matrix = sm.add_constant(train_model[feature_columns], has_constant="add")
    future_matrix = sm.add_constant(future_model[feature_columns], has_constant="add")

    if candidate["model_type"].startswith("glm_poisson"):
        family = sm.families.Poisson()
    elif candidate["model_type"].startswith("glm_negbin"):
        family = sm.families.NegativeBinomial(alpha=candidate.get("alpha", 1.0))
    else:
        raise ValueError(f"Unsupported GLM candidate: {candidate['iteration']}")

    model = sm.GLM(train_model["total_cases"], train_matrix, family=family)
    results = model.fit(maxiter=100, disp=0)
    return integerize(results.predict(future_matrix))


def predict_candidate(train_rows, future_rows, candidate, random_state):
    if candidate["model_type"].startswith("lgbm_calendar"):
        return predict_lgbm(train_rows, future_rows, candidate, random_state)
    if candidate["model_type"].startswith("rf_calendar"):
        return predict_rf(train_rows, future_rows, candidate, random_state)
    if candidate["model_type"].startswith("quantile_calendar"):
        return predict_quantile(train_rows, future_rows, candidate)
    if candidate["model_type"].startswith("glm_"):
        return predict_glm_count(train_rows, future_rows, candidate)
    if candidate["model_type"] == "week_profile":
        return predict_week_profile(train_rows, future_rows, candidate)
    raise ValueError(f"Unsupported base candidate: {candidate['iteration']}")


def evaluate_base_candidates(train_data, random_state):
    fold_records = []
    prediction_records = []

    for candidate in BASE_CANDIDATES:
        print(f"Running {candidate['iteration']}...")
        candidate_actuals = []
        candidate_predictions = []

        for city, city_data in train_data.groupby("city", sort=False):
            city_data = city_data.sort_values("week_start_date").copy()
            city_actuals = []
            city_predictions = []

            for validation_year in validation_years(city_data):
                train_rows = city_data[city_data["year"] < validation_year].copy()
                validation_rows = city_data[city_data["year"] == validation_year].copy()
                predictions = predict_candidate(train_rows, validation_rows, candidate, random_state)
                actuals = validation_rows["total_cases"].astype(int).tolist()

                fold_mae = mean_absolute_error(actuals, predictions)
                fold_records.append(
                    {
                        "iteration": candidate["iteration"],
                        "model_type": candidate["model_type"],
                        "city": city,
                        "validation_year": validation_year,
                        "train_rows": len(train_rows),
                        "validation_rows": len(validation_rows),
                        "mae": float(fold_mae),
                    }
                )

                for row, actual, prediction in zip(
                    validation_rows.itertuples(index=False),
                    actuals,
                    predictions,
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

                city_actuals.extend(actuals)
                city_predictions.extend(predictions)
                candidate_actuals.extend(actuals)
                candidate_predictions.extend(predictions)

            fold_records.append(
                {
                    "iteration": candidate["iteration"],
                    "model_type": candidate["model_type"],
                    "city": city,
                    "validation_year": "all",
                    "train_rows": "",
                    "validation_rows": len(city_actuals),
                    "mae": float(mean_absolute_error(city_actuals, city_predictions)),
                }
            )

        fold_records.append(
            {
                "iteration": candidate["iteration"],
                "model_type": candidate["model_type"],
                "city": "all",
                "validation_year": "all",
                "train_rows": "",
                "validation_rows": len(candidate_actuals),
                "mae": float(mean_absolute_error(candidate_actuals, candidate_predictions)),
            }
        )

    return fold_records, prediction_records


def evaluate_blends(base_prediction_records):
    fold_records = []
    prediction_records = []
    base_predictions = pd.DataFrame(base_prediction_records)
    prediction_keys = [
        "city",
        "year",
        "weekofyear",
        "week_start_date",
        "validation_year",
        "actual_total_cases",
    ]

    for candidate in BLEND_CANDIDATES:
        print(f"Running {candidate['iteration']}...")
        blended = None
        weighted_columns = []

        for member_iteration, weight in candidate["members"]:
            member = base_predictions[base_predictions["iteration"] == member_iteration][
                prediction_keys + ["predicted_total_cases"]
            ].copy()
            member_column = f"prediction_{member_iteration}"
            member = member.rename(columns={"predicted_total_cases": member_column})
            blended = member if blended is None else blended.merge(member, on=prediction_keys)
            weighted_columns.append((member_column, weight))

        blended["predicted_total_cases"] = np.rint(
            sum(blended[column] * weight for column, weight in weighted_columns)
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
                    "model_type": candidate["model_type"],
                    "city": city,
                    "validation_year": validation_year,
                    "train_rows": "",
                    "validation_rows": len(group),
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
                    "model_type": candidate["model_type"],
                    "city": city,
                    "validation_year": "all",
                    "train_rows": "",
                    "validation_rows": len(group),
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
                "model_type": candidate["model_type"],
                "city": "all",
                "validation_year": "all",
                "train_rows": "",
                "validation_rows": len(blended),
                "mae": float(
                    mean_absolute_error(
                        blended["actual_total_cases"],
                        blended["predicted_total_cases"],
                    )
                ),
            }
        )

    return fold_records, prediction_records


def predict_full_submission(selected_by_city, train_data, test_features, submission_format, random_state):
    submission_predictions = {}

    for selected in selected_by_city.itertuples(index=False):
        city = selected.city
        candidate = next(
            candidate
            for candidate in ALL_CANDIDATES
            if candidate["iteration"] == selected.iteration
        )
        city_train = train_data[train_data["city"] == city].sort_values("week_start_date").copy()
        city_test = test_features[test_features["city"] == city].sort_values("week_start_date").copy()

        if candidate["model_type"] == "blend":
            weighted_predictions = []
            for member_iteration, weight in candidate["members"]:
                member_candidate = next(
                    member
                    for member in BASE_CANDIDATES
                    if member["iteration"] == member_iteration
                )
                member_predictions = np.asarray(
                    predict_candidate(city_train, city_test, member_candidate, random_state),
                    dtype=float,
                )
                weighted_predictions.append(member_predictions * weight)
            predictions = np.rint(sum(weighted_predictions)).clip(min=0).astype(int).tolist()
        else:
            predictions = predict_candidate(city_train, city_test, candidate, random_state)

        for row, prediction in zip(city_test.itertuples(index=False), predictions):
            submission_predictions[(row.city, row.year, row.weekofyear)] = int(prediction)

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
    return submission


def read_main_goal_scores(main_goal_output_dir):
    path = Path(main_goal_output_dir) / "iteration_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    scores = pd.read_csv(path)
    return scores[scores["validation_year"].astype(str) == "all"].copy()


def parse_float_grid(raw_value, argument_name):
    values = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"inf", "infinity", "none"}:
            values.append(float("inf"))
        else:
            values.append(float(item))
    if not values:
        raise ValueError(f"{argument_name} must contain at least one value.")
    return values


def build_selection_scores(validation_scores, iteration_scores, recent_validation_years, recent_score_weight):
    fold_scores = validation_scores[validation_scores["validation_year"].astype(str) != "all"].copy()
    fold_scores["validation_year"] = fold_scores["validation_year"].astype(int)
    recent_rows = []

    for (iteration, city), city_scores in fold_scores.groupby(["iteration", "city"], sort=False):
        recent = city_scores.sort_values("validation_year").tail(recent_validation_years)
        recent_rows.append(
            {
                "iteration": iteration,
                "city": city,
                "recent_validation_years": recent_validation_years,
                "recent_mae": float(
                    (recent["mae"] * recent["validation_rows"]).sum()
                    / recent["validation_rows"].sum()
                ),
            }
        )

    all_rows = []
    for iteration, iteration_recent in pd.DataFrame(recent_rows).groupby("iteration", sort=False):
        all_rows.append(
            {
                "iteration": iteration,
                "city": "all",
                "recent_validation_years": recent_validation_years,
                "recent_mae": float(iteration_recent["recent_mae"].mean()),
            }
        )

    recent_scores = pd.concat(
        [pd.DataFrame(recent_rows), pd.DataFrame(all_rows)],
        ignore_index=True,
    )
    selection_scores = iteration_scores.merge(
        recent_scores,
        on=["iteration", "city"],
        how="left",
    ).rename(columns={"mae": "all_year_mae"})
    selection_scores["recent_score_weight"] = recent_score_weight
    selection_scores["selection_mae"] = (
        (1.0 - recent_score_weight) * selection_scores["all_year_mae"]
        + recent_score_weight * selection_scores["recent_mae"]
    )
    return selection_scores.sort_values(["city", "selection_mae", "iteration"]).reset_index(drop=True)


def select_by_city(selection_scores, max_city_all_year_mae_over_best):
    selected_rows = []
    for city in ["sj", "iq"]:
        city_scores = selection_scores[selection_scores["city"] == city].copy()
        best_city_all_year_mae = city_scores["all_year_mae"].min()
        if math.isfinite(max_city_all_year_mae_over_best):
            city_scores = city_scores[
                city_scores["all_year_mae"]
                <= best_city_all_year_mae + max_city_all_year_mae_over_best
            ]
        selected_rows.append(
            city_scores.sort_values(["selection_mae", "all_year_mae", "iteration"]).iloc[0]
        )
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def build_selection_sensitivity(
    selection_scores,
    recent_score_weights,
    all_year_guardrails,
):
    base_scores = selection_scores.copy()
    sensitivity_rows = []

    for recent_score_weight in recent_score_weights:
        scenario_scores = base_scores.copy()
        scenario_scores["recent_score_weight"] = recent_score_weight
        scenario_scores["selection_mae"] = (
            (1.0 - recent_score_weight) * scenario_scores["all_year_mae"]
            + recent_score_weight * scenario_scores["recent_mae"]
        )

        for guardrail in all_year_guardrails:
            selected = select_by_city(scenario_scores, guardrail)
            for row in selected.itertuples(index=False):
                city_scores = scenario_scores[scenario_scores["city"] == row.city].copy()
                best_city_all_year_mae = city_scores["all_year_mae"].min()
                allowed_count = len(city_scores)
                if math.isfinite(guardrail):
                    allowed_count = int(
                        (
                            city_scores["all_year_mae"]
                            <= best_city_all_year_mae + guardrail
                        ).sum()
                    )
                sensitivity_rows.append(
                    {
                        "city": row.city,
                        "recent_score_weight": recent_score_weight,
                        "max_city_all_year_mae_over_best": guardrail,
                        "selected_iteration": row.iteration,
                        "model_type": row.model_type,
                        "all_year_mae": row.all_year_mae,
                        "recent_mae": row.recent_mae,
                        "selection_mae": row.selection_mae,
                        "best_city_all_year_mae": best_city_all_year_mae,
                        "all_year_mae_gap": row.all_year_mae - best_city_all_year_mae,
                        "allowed_candidate_count": allowed_count,
                    }
                )

    return pd.DataFrame(sensitivity_rows).sort_values(
        ["city", "max_city_all_year_mae_over_best", "recent_score_weight"]
    )


def write_changelog(
    path,
    run_started_at,
    main_goal_scores,
    iteration_scores,
    selection_scores,
    selected_by_city,
    recent_validation_years,
    recent_score_weight,
    max_city_all_year_mae_over_best,
    selection_sensitivity,
):
    best_overall = iteration_scores[iteration_scores["city"] == "all"].sort_values("mae").iloc[0]
    lines = [
        "# main_goal_additive.py changelog",
        "",
        f"Run timestamp UTC: {run_started_at}",
        "",
        "## Purpose",
        "",
        "This is an additive no-log iteration after the first main_goal run.",
        "It keeps prior outputs intact and tests only shallow seasonal candidates.",
        "",
        "## Guardrails",
        "",
        "- Cities are modeled separately.",
        "- Validation is expanding, full-year forward chaining within each city.",
        "- No log target transform is used in any candidate.",
        "- No total_cases-derived predictors, lagged cases, or rolling cases are used as inputs.",
        "- total_cases is used only as the supervised label and validation ground truth.",
        (
            f"- Final additive selection uses {1.0 - recent_score_weight:.0%} all-year MAE "
            f"and {recent_score_weight:.0%} latest-{recent_validation_years}-validation-year MAE."
        ),
        (
            "- Final city selection is restricted to candidates within "
            f"{max_city_all_year_mae_over_best:.2f} MAE of that city's best all-year MAE "
            "to avoid chasing a narrow recent-year validation fluctuation."
        ),
        "",
        "## Previous main_goal reference",
        "",
    ]

    if main_goal_scores.empty:
        lines.append("No previous main_goal iteration_scores.csv was found.")
    else:
        prior = main_goal_scores[main_goal_scores["city"] == "all"].sort_values("mae").head(5)
        lines.append("| Prior iteration | Overall MAE |")
        lines.append("| --- | ---: |")
        for row in prior.itertuples(index=False):
            lines.append(f"| {row.iteration} | {row.mae:.4f} |")

    lines.extend(
        [
            "",
            "## Additive iteration summary",
            "",
            "| Iteration | Model type | Overall MAE | SJ MAE | IQ MAE |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )

    for candidate in ALL_CANDIDATES:
        rows = iteration_scores[iteration_scores["iteration"] == candidate["iteration"]]
        all_mae = rows[rows["city"] == "all"]["mae"].iloc[0]
        sj_mae = rows[rows["city"] == "sj"]["mae"].iloc[0]
        iq_mae = rows[rows["city"] == "iq"]["mae"].iloc[0]
        lines.append(
            f"| {candidate['iteration']} | {candidate['model_type']} | "
            f"{all_mae:.4f} | {sj_mae:.4f} | {iq_mae:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Selection score summary",
            "",
            "| Iteration | City | All-year MAE | Recent MAE | Selection MAE |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in selection_scores[selection_scores["city"] != "all"].itertuples(index=False):
        lines.append(
            f"| {row.iteration} | {row.city} | {row.all_year_mae:.4f} | "
            f"{row.recent_mae:.4f} | {row.selection_mae:.4f} |"
        )

    lines.extend(["", "## Iteration notes", ""])
    for candidate in ALL_CANDIDATES:
        rows = iteration_scores[iteration_scores["iteration"] == candidate["iteration"]]
        score_rows = selection_scores[selection_scores["iteration"] == candidate["iteration"]]
        all_mae = rows[rows["city"] == "all"]["mae"].iloc[0]
        sj_mae = rows[rows["city"] == "sj"]["mae"].iloc[0]
        iq_mae = rows[rows["city"] == "iq"]["mae"].iloc[0]
        sj_score = score_rows[score_rows["city"] == "sj"]["selection_mae"].iloc[0]
        iq_score = score_rows[score_rows["city"] == "iq"]["selection_mae"].iloc[0]
        selected_cities = selected_by_city[
            selected_by_city["iteration"] == candidate["iteration"]
        ]["city"].tolist()
        decision = (
            "Selected for final additive submission city/cities: " + ", ".join(selected_cities)
            if selected_cities
            else "Not selected for the additive submission."
        )
        lines.extend(
            [
                f"### {candidate['iteration']}",
                "",
                f"Rationale: {candidate['rationale']}",
                "",
                f"Result: overall MAE {all_mae:.4f}; SJ {sj_mae:.4f}; IQ {iq_mae:.4f}.",
                "",
                f"Selection score: SJ {sj_score:.4f}; IQ {iq_score:.4f}.",
                "",
                f"Decision: {decision}",
                "",
            ]
        )

    lines.extend(
        [
            "## Final additive selection",
            "",
            "| City | Selected iteration | All-year MAE | Recent MAE | Selection MAE | Model type |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in selected_by_city.sort_values("city").itertuples(index=False):
        lines.append(
            f"| {row.city} | {row.iteration} | {row.all_year_mae:.4f} | "
            f"{row.recent_mae:.4f} | {row.selection_mae:.4f} | {row.model_type} |"
        )

    lines.extend(
        [
            "",
            "## Selection robustness snapshot",
            "",
            "This reports how often each iteration is selected across the sensitivity grid "
            "in `selection_sensitivity.csv`.",
            "",
            "| City | Selected iteration | Scenarios selected |",
            "| --- | --- | ---: |",
        ]
    )
    robustness_counts = (
        selection_sensitivity.groupby(["city", "selected_iteration"])
        .size()
        .reset_index(name="scenarios_selected")
        .sort_values(["city", "scenarios_selected", "selected_iteration"], ascending=[True, False, True])
    )
    for row in robustness_counts.itertuples(index=False):
        lines.append(f"| {row.city} | {row.selected_iteration} | {row.scenarios_selected} |")

    lines.extend(
        [
            "",
            "## Best additive single iteration",
            "",
            f"{best_overall.iteration} had the best additive overall MAE at {best_overall.mae:.4f}.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


args = parse_args()
run_started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
run_label = args.run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_root = Path(args.output_root)
output_dir = output_root / run_label
output_dir.mkdir(parents=True, exist_ok=False)

if args.recent_validation_years < 1:
    raise ValueError("--recent-validation-years must be at least 1.")
if not 0 <= args.recent_score_weight <= 1:
    raise ValueError("--recent-score-weight must be between 0 and 1.")
if args.max_city_all_year_mae_over_best < 0:
    raise ValueError("--max-city-all-year-mae-over-best must be non-negative.")
if args.random_state != FIXED_RANDOM_STATE:
    raise ValueError(
        f"--random-state must remain fixed at {FIXED_RANDOM_STATE}; seed tuning is not allowed."
    )
sensitivity_recent_score_weights = parse_float_grid(
    args.sensitivity_recent_score_weights,
    "--sensitivity-recent-score-weights",
)
sensitivity_all_year_guardrails = parse_float_grid(
    args.sensitivity_all_year_guardrails,
    "--sensitivity-all-year-guardrails",
)
if any(weight < 0 or weight > 1 for weight in sensitivity_recent_score_weights):
    raise ValueError("--sensitivity-recent-score-weights values must be between 0 and 1.")
if any(guardrail < 0 for guardrail in sensitivity_all_year_guardrails):
    raise ValueError("--sensitivity-all-year-guardrails values must be non-negative.")

random.seed(args.random_state)
np.random.seed(args.random_state)
os.environ["PYTHONHASHSEED"] = str(args.random_state)

train_features = pd.read_csv(args.train_features_csv, parse_dates=["week_start_date"])
test_features = pd.read_csv(args.test_features_csv, parse_dates=["week_start_date"])
labels = pd.read_csv(args.labels_csv)
submission_format = pd.read_csv(args.submission_format_csv)
validate_inputs(train_features, test_features, labels, submission_format)

train_data = train_features.merge(labels, on=MERGE_KEYS, how="left", validate="one_to_one")
if train_data["total_cases"].isna().any():
    raise ValueError("Merged training data contains missing total_cases values.")

base_fold_records, base_prediction_records = evaluate_base_candidates(train_data, args.random_state)
blend_fold_records, blend_prediction_records = evaluate_blends(base_prediction_records)

validation_scores = pd.DataFrame(base_fold_records + blend_fold_records)
validation_predictions = pd.DataFrame(base_prediction_records + blend_prediction_records)
iteration_scores = (
    validation_scores[validation_scores["validation_year"].astype(str) == "all"]
    .sort_values(["city", "mae", "iteration"])
    .reset_index(drop=True)
)
selection_scores = build_selection_scores(
    validation_scores,
    iteration_scores,
    args.recent_validation_years,
    args.recent_score_weight,
)

selected_by_city = select_by_city(selection_scores, args.max_city_all_year_mae_over_best)
selection_sensitivity = build_selection_sensitivity(
    selection_scores,
    sensitivity_recent_score_weights,
    sensitivity_all_year_guardrails,
)

submission = predict_full_submission(
    selected_by_city,
    train_data,
    test_features,
    submission_format,
    args.random_state,
)
main_goal_scores = read_main_goal_scores(args.main_goal_output_dir)

validation_scores.to_csv(output_dir / "validation_scores.csv", index=False)
validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
iteration_scores.to_csv(output_dir / "iteration_scores.csv", index=False)
selection_scores.to_csv(output_dir / "selection_scores.csv", index=False)
selection_sensitivity.to_csv(output_dir / "selection_sensitivity.csv", index=False)
selected_by_city.to_csv(output_dir / "selected_iteration_by_city.csv", index=False)
submission.to_csv(output_dir / "submission.csv", index=False)
main_goal_scores.to_csv(output_dir / "main_goal_reference_scores.csv", index=False)
write_changelog(
    output_dir / "changelog.md",
    run_started_at,
    main_goal_scores,
    iteration_scores,
    selection_scores,
    selected_by_city,
    args.recent_validation_years,
    args.recent_score_weight,
    args.max_city_all_year_mae_over_best,
    selection_sensitivity,
)

experiment_log_record = {
    "logged_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "script": str(Path(__file__)),
    "config": {
        "random_state": args.random_state,
        "train_features_csv": args.train_features_csv,
        "test_features_csv": args.test_features_csv,
        "labels_csv": args.labels_csv,
        "submission_format_csv": args.submission_format_csv,
        "main_goal_output_dir": str(args.main_goal_output_dir),
        "run_label": run_label,
        "no_log_target": True,
        "recent_validation_years": args.recent_validation_years,
        "recent_score_weight": args.recent_score_weight,
        "max_city_all_year_mae_over_best": args.max_city_all_year_mae_over_best,
        "sensitivity_recent_score_weights": sensitivity_recent_score_weights,
        "sensitivity_all_year_guardrails": sensitivity_all_year_guardrails,
    },
    "results": {
        "best_single_iteration": iteration_scores[iteration_scores["city"] == "all"]
        .sort_values("mae")
        .iloc[0]
        .to_dict(),
        "selected_by_city": selected_by_city.to_dict(orient="records"),
    },
    "outputs": {
        "output_dir": str(output_dir),
        "validation_scores": str(output_dir / "validation_scores.csv"),
        "iteration_scores": str(output_dir / "iteration_scores.csv"),
        "selection_scores": str(output_dir / "selection_scores.csv"),
        "selection_sensitivity": str(output_dir / "selection_sensitivity.csv"),
        "selected_iteration_by_city": str(output_dir / "selected_iteration_by_city.csv"),
        "submission": str(output_dir / "submission.csv"),
        "changelog": str(output_dir / "changelog.md"),
    },
}
with (output_dir / "experiment_log.jsonl").open("a", encoding="utf-8") as experiment_log_file:
    experiment_log_file.write(json.dumps(experiment_log_record, sort_keys=True, default=str) + "\n")

best_overall = iteration_scores[iteration_scores["city"] == "all"].sort_values("mae").iloc[0]
print(f"Best additive single iteration: {best_overall.iteration} MAE={best_overall.mae:.4f}")
for row in selected_by_city.sort_values("city").itertuples(index=False):
    print(
        f"Selected {row.city}: {row.iteration} "
        f"all_year_MAE={row.all_year_mae:.4f} selection_MAE={row.selection_mae:.4f}"
    )
print(f"Wrote additive outputs to {output_dir}")
