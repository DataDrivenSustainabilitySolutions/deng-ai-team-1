"""DengAI Random Forest pipeline using individual lag/rolling features.

Example:
    python3 oliver/main_rf.py --skip-csv-previews
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ["MPLCONFIGDIR"] = "/private/tmp/dengai_main_rf_matplotlib"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline


script_name = Path(__file__).stem
script_dir = Path(__file__).resolve().parent
default_output_dir = script_dir / "outputs" / script_name
default_output_dir_label = f"oliver/outputs/{script_name}"
default_feature_plan_path = (
    script_dir
    / "outputs"
    / "lag_rolling_features_individual"
    / "individual_feature_plan.csv"
)
default_monthly_lag_correlations_path = (
    script_dir
    / "outputs"
    / "lagged_features_monthly"
    / "monthly_lagged_correlations.csv"
)


### CLI arguments
parser = argparse.ArgumentParser(
    description=(
        "Run the DengAI pipeline with Random Forest models and individual "
        "lag/rolling feature choices."
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
    "--feature-plan-csv",
    default=default_feature_plan_path,
    help=(
        "Path to the individual feature plan CSV from lag_rolling_features_individual.py. "
        "Default: oliver/outputs/lag_rolling_features_individual/individual_feature_plan.csv"
    ),
)
parser.add_argument(
    "--monthly-lag-correlations-csv",
    default=default_monthly_lag_correlations_path,
    help=(
        "Path to monthly lag correlations from lagged_features_monthly.py. "
        "Default: oliver/outputs/lagged_features_monthly/monthly_lagged_correlations.csv"
    ),
)
parser.add_argument(
    "--disable-monthly-lag-features",
    action="store_true",
    help="Disable monthly-correlation-derived extra lag features.",
)
parser.add_argument(
    "--monthly-min-abs-corr",
    type=float,
    default=0.10,
    help="Fallback monthly absolute-correlation threshold for extra features. Default: 0.10",
)
parser.add_argument(
    "--monthly-city-min-abs-corr",
    nargs="*",
    default=["sj=0.10", "iq=0.10"],
    help=(
        "Optional city-specific monthly thresholds as city=value. "
        "Default: sj=0.10 iq=0.10"
    ),
)
parser.add_argument(
    "--monthly-near-best-ratio",
    type=float,
    default=0.90,
    help=(
        "Keep monthly lags whose absolute correlation is at least this share "
        "of the feature's best monthly lag. Default: 0.90"
    ),
)
parser.add_argument(
    "--monthly-max-lags-per-feature",
    type=int,
    default=3,
    help="Maximum monthly lag rows to translate per city-feature. Default: 3",
)
parser.add_argument(
    "--monthly-week-lag-offsets",
    type=int,
    nargs="+",
    default=[-1, 0, 1],
    help=(
        "Weekly offsets around month_lag*4 for monthly-derived lags. "
        "Default: -1 0 1"
    ),
)
parser.add_argument(
    "--monthly-max-week-lag",
    type=int,
    default=60,
    help="Maximum weekly lag allowed from monthly-derived features. Default: 60",
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
    "--used-feature-plan-output",
    default="used_individual_feature_plan.csv",
    help="Filename for the individual feature plan used by this run.",
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
    "--rf-grid-output",
    default="rf_grid_scores.csv",
    help="Filename for city-level Random Forest grid scores. Default: rf_grid_scores.csv",
)
parser.add_argument(
    "--best-params-output",
    default="best_params.csv",
    help="Filename for best per-city Random Forest parameters. Default: best_params.csv",
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
    help="Random seed for the Random Forest. Default: 42",
)
parser.add_argument(
    "--target-transform",
    choices=["none", "log1p"],
    default="none",
    help="Optional transform applied to total_cases during fitting. Default: none",
)
parser.add_argument(
    "--n-jobs",
    type=int,
    default=1,
    help="Parallel jobs for Random Forest fitting. Default: 1",
)
parser.add_argument(
    "--rf-grid-json",
    default=None,
    help=(
        "Optional JSON dict or list of dicts passed to sklearn ParameterGrid. "
        "Default is a small depth/leaf/max_features grid."
    ),
)
parser.add_argument(
    "--tune-hyperparameters",
    action="store_true",
    help="Use Optuna to tune separate Random Forest hyperparameters for each city.",
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


### Configuration
train_features_path = Path(args.train_features_csv)
test_features_path = Path(args.test_features_csv)
labels_path = Path(args.labels_csv)
submission_format_path = Path(args.submission_format_csv)
feature_plan_path = Path(args.feature_plan_csv)
monthly_lag_correlations_path = Path(args.monthly_lag_correlations_csv)
output_dir = Path(args.output_dir)
preprocessed_train_path = output_dir / args.preprocessed_train_output
preprocessed_test_path = output_dir / args.preprocessed_test_output
missing_summary_path = output_dir / args.missing_summary_output
used_feature_plan_path = output_dir / args.used_feature_plan_output
submission_path = output_dir / args.submission_output
validation_scores_path = output_dir / args.validation_output
validation_predictions_path = output_dir / args.validation_predictions_output
feature_importance_path = output_dir / args.feature_importance_output
rf_grid_scores_path = output_dir / args.rf_grid_output
best_params_path = output_dir / args.best_params_output
experiment_log_path = output_dir / args.experiment_log_output
optuna_trials_path = output_dir / args.optuna_output
csv_preview_dir = output_dir / args.csv_preview_dir
merge_keys = ["city", "year", "weekofyear"]
identifier_columns = {"city", "year", "weekofyear", "week_start_date", "split", "total_cases"}
seasonality_feature_columns = ["weekofyear_sin", "weekofyear_cos"]
feature_plan_required_columns = {
    "city",
    "transform",
    "feature",
    "generated_feature",
    "parameter_weeks",
}
monthly_lag_required_columns = {
    "city",
    "feature_group",
    "feature",
    "lag_months",
    "pearson_corr",
    "abs_pearson_corr",
}
default_rf_param_grid = {
    "n_estimators": [300],
    "max_depth": [8, None],
    "min_samples_leaf": [1, 3],
    "max_features": ["sqrt", 0.7],
    "bootstrap": [True],
}
rf_optuna_search_space = {
    "n_estimators": {"low": 100, "high": 700, "step": 100},
    "max_depth": [None, 4, 6, 8, 12, 16],
    "min_samples_leaf": {"low": 1, "high": 8},
    "min_samples_split": {"low": 2, "high": 20},
    "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
    "bootstrap": [True, False],
}

if args.n_jobs == 0:
    raise ValueError("--n-jobs cannot be 0.")

if args.optuna_trials < 1:
    raise ValueError("--optuna-trials must be at least 1.")

if args.optuna_timeout < 0:
    raise ValueError("--optuna-timeout must be non-negative.")

if args.monthly_min_abs_corr < 0:
    raise ValueError("--monthly-min-abs-corr must be non-negative.")

if not 0 < args.monthly_near_best_ratio <= 1:
    raise ValueError("--monthly-near-best-ratio must be greater than 0 and at most 1.")

if args.monthly_max_lags_per_feature < 1:
    raise ValueError("--monthly-max-lags-per-feature must be at least 1.")

if not args.monthly_week_lag_offsets:
    raise ValueError("--monthly-week-lag-offsets must contain at least one integer.")

if args.monthly_max_week_lag < 1:
    raise ValueError("--monthly-max-week-lag must be at least 1.")

if args.csv_preview_max_rows < 1:
    raise ValueError("--csv-preview-max-rows must be at least 1.")

if args.csv_preview_max_cols < 1:
    raise ValueError("--csv-preview-max-cols must be at least 1.")

if args.csv_preview_dpi < 1:
    raise ValueError("--csv-preview-dpi must be at least 1.")

if not feature_plan_path.exists():
    raise FileNotFoundError(
        f"Feature plan not found: {feature_plan_path}. "
        "Run python3 oliver/lag_rolling_features_individual.py first or pass --feature-plan-csv."
    )

if not args.disable_monthly_lag_features and not monthly_lag_correlations_path.exists():
    raise FileNotFoundError(
        f"Monthly lag correlations not found: {monthly_lag_correlations_path}. "
        "Run python3 oliver/lagged_features_monthly.py first or pass "
        "--disable-monthly-lag-features."
    )

monthly_city_abs_corr_thresholds = {}
for city_threshold in args.monthly_city_min_abs_corr:
    if "=" not in city_threshold:
        raise ValueError("--monthly-city-min-abs-corr values must use city=value format.")
    city, value = city_threshold.split("=", maxsplit=1)
    city = city.strip()
    if not city:
        raise ValueError("--monthly-city-min-abs-corr contains an empty city value.")
    monthly_city_abs_corr_thresholds[city] = float(value)
    if monthly_city_abs_corr_thresholds[city] < 0:
        raise ValueError("--monthly-city-min-abs-corr thresholds must be non-negative.")

if args.rf_grid_json:
    rf_param_grid = json.loads(args.rf_grid_json)
    if isinstance(rf_param_grid, dict):
        rf_param_grid = [rf_param_grid]
    if not isinstance(rf_param_grid, list):
        raise ValueError("--rf-grid-json must decode to a JSON dict or list of dicts.")
else:
    rf_param_grid = [default_rf_param_grid]

rf_grid = list(ParameterGrid(rf_param_grid))
if not args.tune_hyperparameters and not rf_grid:
    raise ValueError("Random Forest parameter grid cannot be empty.")

random.seed(args.random_state)
np.random.seed(args.random_state)
os.environ["PYTHONHASHSEED"] = str(args.random_state)

matplotlib_config_dir = output_dir / ".matplotlib"
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)

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
feature_plan = pd.read_csv(feature_plan_path)
monthly_lag_correlations = None
if not args.disable_monthly_lag_features:
    monthly_lag_correlations = pd.read_csv(monthly_lag_correlations_path)

missing_plan_columns = feature_plan_required_columns - set(feature_plan.columns)
if missing_plan_columns:
    raise ValueError(f"Feature plan is missing columns: {sorted(missing_plan_columns)}")

if monthly_lag_correlations is not None:
    missing_monthly_columns = monthly_lag_required_columns - set(monthly_lag_correlations.columns)
    if missing_monthly_columns:
        raise ValueError(
            f"Monthly lag correlations are missing columns: {sorted(missing_monthly_columns)}"
        )

if list(train_features.columns) != list(test_features.columns):
    raise ValueError("Train and test feature CSVs must have the same columns.")

if train_features[merge_keys].duplicated().any() or test_features[merge_keys].duplicated().any():
    raise ValueError("Feature rows must be unique by city, year, and weekofyear.")

if labels[merge_keys].duplicated().any():
    raise ValueError("Label rows must be unique by city, year, and weekofyear.")

if not submission_format[merge_keys].equals(test_features[merge_keys]):
    raise ValueError("Submission format rows must match the test feature index order.")

if (
    (feature_plan["feature"] == "total_cases").any()
    or (feature_plan["generated_feature"] == "total_cases").any()
):
    raise ValueError("Feature plan must not include total_cases as an input feature.")

original_feature_plan_rows = len(feature_plan)
monthly_feature_plan_rows = 0
monthly_new_feature_rows = 0
feature_variant = "individual_lag_rolling"

if monthly_lag_correlations is not None:
    feature_variant = "individual_lag_rolling_plus_monthly_lag_liberal"
    monthly_feature_records = []
    valid_monthly_correlations = monthly_lag_correlations.dropna(subset=["pearson_corr"]).copy()

    for (city, feature), feature_correlations in valid_monthly_correlations.groupby(
        ["city", "feature"],
        sort=False,
    ):
        if feature == "total_cases":
            raise ValueError("Monthly lag correlations must not include total_cases as a feature.")

        feature_correlations = feature_correlations.sort_values(
            ["abs_pearson_corr", "lag_months"],
            ascending=[False, True],
        ).copy()
        best_abs_corr = float(feature_correlations.iloc[0]["abs_pearson_corr"])
        city_threshold = monthly_city_abs_corr_thresholds.get(city, args.monthly_min_abs_corr)
        if best_abs_corr < city_threshold:
            continue

        selection_cutoff = max(city_threshold, best_abs_corr * args.monthly_near_best_ratio)
        selected_monthly_lags = feature_correlations[
            feature_correlations["abs_pearson_corr"] >= selection_cutoff
        ].head(args.monthly_max_lags_per_feature)

        for rank, row in enumerate(selected_monthly_lags.itertuples(index=False), start=1):
            lag_months = int(row.lag_months)
            if lag_months == 0:
                weekly_lags = [0]
            else:
                weekly_lags = sorted(
                    {
                        lag_months * 4 + offset
                        for offset in args.monthly_week_lag_offsets
                        if 1 <= lag_months * 4 + offset <= args.monthly_max_week_lag
                    }
                )

            for lag_weeks in weekly_lags:
                transform = "raw" if lag_weeks == 0 else "lag"
                generated_feature = feature if lag_weeks == 0 else f"{feature}_lag_{lag_weeks}"
                monthly_feature_records.append(
                    {
                        "city": city,
                        "transform": transform,
                        "feature_group": row.feature_group,
                        "feature": feature,
                        "generated_feature": generated_feature,
                        "parameter_weeks": lag_weeks,
                        "source_analysis_type": "monthly_lag",
                        "pearson_corr": row.pearson_corr,
                        "abs_pearson_corr": row.abs_pearson_corr,
                        "best_abs_pearson_corr": best_abs_corr,
                        "relative_to_best": float(row.abs_pearson_corr) / best_abs_corr,
                        "city_abs_corr_threshold": city_threshold,
                        "selection_cutoff_abs_corr": selection_cutoff,
                        "monthly_lag_months": lag_months,
                        "monthly_selection_rank_within_feature": rank,
                    }
                )

    monthly_feature_plan = pd.DataFrame(monthly_feature_records)
    monthly_feature_plan_rows = len(monthly_feature_plan)

    if not monthly_feature_plan.empty:
        existing_city_features = set(zip(feature_plan["city"], feature_plan["generated_feature"]))
        monthly_new_feature_rows = len(
            {
                (row.city, row.generated_feature)
                for row in monthly_feature_plan.itertuples(index=False)
                if (row.city, row.generated_feature) not in existing_city_features
            }
        )

        feature_plan["_source_priority"] = 0
        monthly_feature_plan["_source_priority"] = 1
        feature_plan = (
            pd.concat([feature_plan, monthly_feature_plan], ignore_index=True, sort=False)
            .sort_values(["city", "generated_feature", "_source_priority"])
            .drop_duplicates(["city", "generated_feature"], keep="first")
            .drop(columns=["_source_priority"])
            .reset_index(drop=True)
        )

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
# Linear interpolation is applied city by city so the two time series remain separate.
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

week_angle = 2 * math.pi * data["weekofyear"] / 52.0
engineered_feature_data["weekofyear_sin"] = week_angle.map(math.sin)
engineered_feature_data["weekofyear_cos"] = week_angle.map(math.cos)
engineered_feature_columns.extend(seasonality_feature_columns)

plan_base_features = set(feature_plan["feature"])
missing_base_features = sorted(feature for feature in plan_base_features if feature not in data.columns)
if missing_base_features:
    raise ValueError(f"Feature plan references unknown base features: {missing_base_features}")

for row in feature_plan.itertuples(index=False):
    transform = row.transform
    feature = row.feature
    generated_feature = row.generated_feature
    parameter_weeks = int(row.parameter_weeks)

    if transform == "raw":
        if generated_feature != feature:
            raise ValueError(f"Raw feature row has mismatched generated feature: {generated_feature}")
        continue

    if transform == "lag":
        expected_feature = feature if parameter_weeks == 0 else f"{feature}_lag_{parameter_weeks}"
        if generated_feature != expected_feature:
            raise ValueError(f"Lag feature row has unexpected generated feature: {generated_feature}")
        if parameter_weeks == 0:
            continue
        if generated_feature not in engineered_feature_data:
            engineered_feature_data[generated_feature] = data.groupby("city", sort=False)[feature].shift(
                parameter_weeks
            )
            engineered_feature_columns.append(generated_feature)
        continue

    if transform == "rolling_mean":
        expected_feature = feature if parameter_weeks == 1 else f"{feature}_rolling_{parameter_weeks}_mean"
        if generated_feature != expected_feature:
            raise ValueError(f"Rolling feature row has unexpected generated feature: {generated_feature}")
        if parameter_weeks == 1:
            continue
        if generated_feature not in engineered_feature_data:
            engineered_feature_data[generated_feature] = (
                data.groupby("city", sort=False)[feature]
                .rolling(window=parameter_weeks, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            engineered_feature_columns.append(generated_feature)
        continue

    raise ValueError(f"Unsupported feature transform: {transform}")

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
feature_plan.to_csv(used_feature_plan_path, index=False)

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
city_feature_columns = {}
for city in ["sj", "iq"]:
    city_plan = feature_plan[feature_plan["city"] == city]
    selected_columns = seasonality_feature_columns + city_plan["generated_feature"].tolist()
    selected_columns = list(dict.fromkeys(selected_columns))
    missing_selected_columns = [column for column in selected_columns if column not in data.columns]
    if missing_selected_columns:
        raise ValueError(f"Missing selected columns for {city}: {missing_selected_columns}")
    city_feature_columns[city] = selected_columns

if not all(city_feature_columns.values()):
    raise ValueError("Each city must have at least one selected feature.")


### 4. Random Forest grid selection or Optuna tuning
train_model_data = data[data["split"] == "train"].copy()
test_model_data = data[data["split"] == "test"].copy()
city_model_params = {}
city_selection_mode = {}
city_selection_id = {}
city_best_grid_index = {}
city_best_trial_number = {}
grid_score_records = []
optuna_trial_records = []
best_param_records = []


def transform_target(target):
    if args.target_transform == "log1p":
        return np.log1p(target)
    return target


def inverse_transform_prediction(prediction):
    if args.target_transform == "log1p":
        return np.expm1(prediction)
    return prediction


def make_integer_predictions(raw_predictions):
    return [
        int(round(max(0.0, float(inverse_transform_prediction(raw_prediction)))))
        for raw_prediction in raw_predictions
    ]


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


def make_model(model_params):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(**model_params)),
        ]
    )


def make_model_params(search_params):
    return {
        **search_params,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
    }


def suggest_rf_hyperparameters(trial):
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            **rf_optuna_search_space["n_estimators"],
        ),
        "max_depth": trial.suggest_categorical(
            "max_depth",
            rf_optuna_search_space["max_depth"],
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            **rf_optuna_search_space["min_samples_leaf"],
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            **rf_optuna_search_space["min_samples_split"],
        ),
        "max_features": trial.suggest_categorical(
            "max_features",
            rf_optuna_search_space["max_features"],
        ),
        "bootstrap": trial.suggest_categorical(
            "bootstrap",
            rf_optuna_search_space["bootstrap"],
        ),
    }


def score_city_params(city_train_data, feature_columns, validation_years, model_params):
    fold_maes = []
    for validation_year in validation_years:
        fold_train_data = city_train_data[city_train_data["year"] < validation_year].copy()
        fold_validation_data = city_train_data[city_train_data["year"] == validation_year].copy()

        model = make_model(model_params)
        model.fit(
            fold_train_data[feature_columns],
            transform_target(fold_train_data["total_cases"]),
        )

        raw_predictions = model.predict(fold_validation_data[feature_columns])
        fold_predictions = make_integer_predictions(raw_predictions)
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
            study_name=f"{city}_random_forest_search",
        )

        def objective(trial):
            trial_params = make_model_params(suggest_rf_hyperparameters(trial))
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

        completed_trials = study.get_trials(
            deepcopy=False,
            states=[optuna.trial.TrialState.COMPLETE],
        )
        if not completed_trials:
            raise RuntimeError(f"Optuna completed no successful trials for city {city}.")

        best_trial_number = int(study.best_trial.number)
        city_model_params[city] = make_model_params(study.best_params)
        city_selection_mode[city] = "optuna"
        city_selection_id[city] = best_trial_number
        city_best_trial_number[city] = best_trial_number

        for trial in study.trials:
            optuna_trial_records.append(
                {
                    "city": city,
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "mae": trial.value if trial.value is not None else np.nan,
                    "selected": trial.number == best_trial_number,
                    **trial.params,
                    "random_state": args.random_state,
                    "n_jobs": args.n_jobs,
                }
            )

        best_param_records.append(
            {
                "city": city,
                "selection_mode": "optuna",
                "selection_id": best_trial_number,
                "best_trial_number": best_trial_number,
                "best_mae": float(study.best_value),
                **city_model_params[city],
            }
        )
    else:
        city_grid_records = []
        for grid_index, grid_params in enumerate(rf_grid):
            model_params = make_model_params(grid_params)
            candidate_mae = score_city_params(
                city_train_data,
                feature_columns,
                validation_years,
                model_params,
            )
            record = {
                "city": city,
                "grid_index": grid_index,
                "mae": candidate_mae,
                "selected": False,
                **model_params,
            }
            city_grid_records.append(record)

        best_record = min(city_grid_records, key=lambda record: record["mae"])
        best_record["selected"] = True
        city_model_params[city] = {
            key: best_record[key]
            for key in best_record
            if key not in {"city", "grid_index", "mae", "selected"}
        }
        city_best_grid_index[city] = int(best_record["grid_index"])
        city_selection_mode[city] = "grid"
        city_selection_id[city] = int(best_record["grid_index"])
        best_param_records.append(
            {
                "city": city,
                "selection_mode": "grid",
                "selection_id": int(best_record["grid_index"]),
                "grid_index": int(best_record["grid_index"]),
                "best_mae": float(best_record["mae"]),
                **city_model_params[city],
            }
        )
        grid_score_records.extend(city_grid_records)

rf_grid_scores = pd.DataFrame(grid_score_records)
optuna_trials = pd.DataFrame(optuna_trial_records)
best_params = pd.DataFrame(best_param_records)
if args.tune_hyperparameters:
    optuna_trials.to_csv(optuna_trials_path, index=False)
else:
    rf_grid_scores.to_csv(rf_grid_scores_path, index=False)
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

    for validation_year in validation_years:
        fold_train_data = city_train_data[city_train_data["year"] < validation_year].copy()
        fold_validation_data = city_train_data[city_train_data["year"] == validation_year].copy()

        model = make_model(city_model_params[city])
        model.fit(
            fold_train_data[feature_columns],
            transform_target(fold_train_data["total_cases"]),
        )

        fold_predictions = []
        raw_predictions = model.predict(fold_validation_data[feature_columns])

        for row, raw_prediction in zip(fold_validation_data.itertuples(index=False), raw_predictions):
            clipped_prediction = max(0.0, float(inverse_transform_prediction(raw_prediction)))
            integer_prediction = int(round(clipped_prediction))

            fold_predictions.append(integer_prediction)

            validation_prediction_records.append(
                {
                    "city": row.city,
                    "model": "random_forest",
                    "selection_mode": city_selection_mode[city],
                    "selection_id": city_selection_id[city],
                    "grid_index": city_best_grid_index.get(city, ""),
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
                "model": "random_forest",
                "selection_mode": city_selection_mode[city],
                "selection_id": city_selection_id[city],
                "grid_index": city_best_grid_index.get(city, ""),
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
            "model": "random_forest",
            "selection_mode": "selected_by_city",
            "selection_id": "selected_by_city",
            "grid_index": "selected_by_city",
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

    model = make_model(city_model_params[city])
    model.fit(
        city_train_data[feature_columns],
        transform_target(city_train_data["total_cases"]),
    )

    forest = model.named_steps["model"]
    for feature, importance in zip(feature_columns, forest.feature_importances_):
        feature_importance_records.append(
            {
                "city": city,
                "model": "random_forest",
                "selection_mode": city_selection_mode[city],
                "selection_id": city_selection_id[city],
                "grid_index": city_best_grid_index.get(city, ""),
                "feature": feature,
                "importance": importance,
            }
        )

    raw_predictions = model.predict(city_test_data[feature_columns])

    for row, raw_prediction in zip(city_test_data.itertuples(index=False), raw_predictions):
        clipped_prediction = max(0.0, float(inverse_transform_prediction(raw_prediction)))
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
    ["city", "importance"],
    ascending=[True, False],
)

validation_scores.to_csv(validation_scores_path, index=False)
validation_predictions.to_csv(validation_predictions_path, index=False)
feature_importance.to_csv(feature_importance_path, index=False)
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
        "feature_plan_csv": str(feature_plan_path),
        "monthly_lag_correlations_csv": str(monthly_lag_correlations_path),
        "feature_variant": feature_variant,
        "model": "random_forest",
        "random_state": args.random_state,
        "target_transform": args.target_transform,
        "n_jobs": args.n_jobs,
        "tune_hyperparameters": args.tune_hyperparameters,
        "optuna_trials": args.optuna_trials if args.tune_hyperparameters else 0,
        "optuna_timeout": args.optuna_timeout if args.tune_hyperparameters else 0,
        "monthly_lag_features_enabled": monthly_lag_correlations is not None,
        "monthly_min_abs_corr": args.monthly_min_abs_corr,
        "monthly_city_min_abs_corr": monthly_city_abs_corr_thresholds,
        "monthly_near_best_ratio": args.monthly_near_best_ratio,
        "monthly_max_lags_per_feature": args.monthly_max_lags_per_feature,
        "monthly_week_lag_offsets": args.monthly_week_lag_offsets,
        "monthly_max_week_lag": args.monthly_max_week_lag,
        "rf_grid": rf_param_grid if not args.tune_hyperparameters else None,
        "rf_optuna_search_space": rf_optuna_search_space if args.tune_hyperparameters else None,
    },
    "results": {
        "overall_validation_mae": overall_validation_mae,
        "overall_validation_rows": overall_validation_rows,
        "city_validation_mae": city_validation_mae,
        "feature_counts": {
            city: len(feature_columns)
            for city, feature_columns in city_feature_columns.items()
        },
        "original_feature_plan_rows": original_feature_plan_rows,
        "monthly_feature_plan_rows_before_dedup": monthly_feature_plan_rows,
        "monthly_new_generated_features": monthly_new_feature_rows,
        "final_feature_plan_rows": len(feature_plan),
        "selection_mode": city_selection_mode,
        "selection_id": city_selection_id,
        "best_grid_index": city_best_grid_index,
        "best_trial_number": city_best_trial_number,
        "model_params": city_model_params,
    },
    "outputs": {
        "used_feature_plan": str(used_feature_plan_path),
        "validation_scores": str(validation_scores_path),
        "validation_predictions": str(validation_predictions_path),
        "feature_importance": str(feature_importance_path),
        "rf_grid_scores": str(rf_grid_scores_path) if not args.tune_hyperparameters else None,
        "optuna_trials": str(optuna_trials_path) if args.tune_hyperparameters else None,
        "best_params": str(best_params_path),
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
        ("used_individual_feature_plan", used_feature_plan_path, feature_plan),
        ("preprocessed_train", preprocessed_train_path, preprocessed_train),
        ("preprocessed_test", preprocessed_test_path, preprocessed_test),
        ("best_params", best_params_path, best_params),
        ("validation_scores", validation_scores_path, validation_scores),
        ("validation_predictions", validation_predictions_path, validation_predictions),
        ("feature_importance", feature_importance_path, feature_importance),
        ("submission", submission_path, submission),
    ]
    if args.tune_hyperparameters:
        csv_preview_outputs.insert(4, ("optuna_trials", optuna_trials_path, optuna_trials))
    else:
        csv_preview_outputs.insert(4, ("rf_grid_scores", rf_grid_scores_path, rf_grid_scores))

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


print("Random Forest pipeline completed through preprocessing, feature engineering, validation, and submission.")
print(f"Imported train rows: {len(train_data)}")
print(f"Imported test rows: {len(test_data)}")
print(f"Feature variant: {feature_variant}")
print(f"Original feature plan rows: {original_feature_plan_rows}")
print(f"Monthly-derived candidate rows before de-duplication: {monthly_feature_plan_rows}")
print(f"Monthly-derived new generated features: {monthly_new_feature_rows}")
print(f"Feature plan rows used after de-duplication: {len(feature_plan)}")
print(f"Numeric feature missing values before interpolation: {int(missing_before.sum())}")
print(f"Numeric feature missing values after interpolation: {int(missing_after.sum())}")
print(f"Engineered feature columns: {len(engineered_feature_columns)}")
print(
    "Selected RF search ids: "
    + ", ".join(
        f"{city}={city_selection_mode[city]}:{city_selection_id[city]}"
        for city in sorted(city_selection_id)
    )
)
print(f"Validation folds: {len(validation_scores[validation_scores['city'] != 'all'])}")
if not validation_scores.empty:
    overall_mae = validation_scores.loc[validation_scores["city"] == "all", "mae"]
    if not overall_mae.empty:
        print(f"Overall validation MAE: {overall_mae.iloc[0]:.4f}")
print(f"Saved missing summary: {missing_summary_path}")
print(f"Saved used feature plan: {used_feature_plan_path}")
print(f"Saved preprocessed and engineered train data: {preprocessed_train_path}")
print(f"Saved preprocessed and engineered test data: {preprocessed_test_path}")
if args.tune_hyperparameters:
    print(f"Saved Optuna trials: {optuna_trials_path}")
else:
    print(f"Saved RF grid scores: {rf_grid_scores_path}")
print(f"Saved best params: {best_params_path}")
print(f"Saved validation scores: {validation_scores_path}")
print(f"Saved validation predictions: {validation_predictions_path}")
print(f"Saved feature importance: {feature_importance_path}")
print(f"Appended experiment log: {experiment_log_path}")
print(f"Saved submission: {submission_path}")
if csv_preview_paths:
    print("Saved CSV preview images:")
    for csv_preview_path in csv_preview_paths:
        print(f"- {csv_preview_path}")
