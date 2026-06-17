"""Standalone no-log DengAI submission for the current best sandbox solution.

Example:
    python3 oliver/sandbox/main_goal_best.py

This script intentionally keeps only the selected final city-specific blends
from the additive run. It uses calendar harmonics only, trains on raw
``total_cases`` labels, and writes a submission plus minimal run metadata.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dengai_main_goal_best_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor

try:
    from lightgbm import LGBMRegressor
except ImportError as error:
    raise ImportError(
        "LightGBM is required. Install dependencies from pyproject.toml, "
        "for example: python3 -m pip install -e ."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "main_goal_best"
REFERENCE_SUBMISSION = (
    SCRIPT_DIR
    / "outputs"
    / "main_goal_additive"
    / "20260617T141841Z"
    / "submission.csv"
)
MERGE_KEYS = ["city", "year", "weekofyear"]
FIXED_RANDOM_STATE = 42

BASE_MODELS = {
    "a29_recent2_calendar_lgbm_l1_harmonic2": {
        "model_type": "lgbm",
        "recent_train_years": 2,
        "max_harmonic": 2,
        "lgbm_overrides": {},
    },
    "a33_recent4_calendar_lgbm_l1_harmonic2_regularized": {
        "model_type": "lgbm",
        "recent_train_years": 4,
        "max_harmonic": 2,
        "lgbm_overrides": {
            "num_leaves": 7,
            "min_child_samples": 30,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
        },
    },
    "a35_recent4_calendar_rf_abs_harmonic2": {
        "model_type": "rf",
        "recent_train_years": 4,
        "max_harmonic": 2,
    },
    "a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized": {
        "model_type": "lgbm",
        "recent_train_years": 7,
        "max_harmonic": 2,
        "lgbm_overrides": {
            "num_leaves": 5,
            "min_child_samples": 40,
            "reg_alpha": 0.4,
            "reg_lambda": 4.0,
        },
    },
    "a39_recent7_calendar_quantile_harmonic2": {
        "model_type": "quantile",
        "recent_train_years": 7,
        "max_harmonic": 2,
    },
}

CITY_BLENDS = {
    "iq": {
        "iteration": "a46_iq_lgbm_rf_frontier_blend",
        "members": [
            ("a29_recent2_calendar_lgbm_l1_harmonic2", 1.0 / 3.0),
            ("a33_recent4_calendar_lgbm_l1_harmonic2_regularized", 1.0 / 3.0),
            ("a35_recent4_calendar_rf_abs_harmonic2", 1.0 / 3.0),
        ],
    },
    "sj": {
        "iteration": "a47_sj_quantile_lgbm_frontier_blend",
        "members": [
            ("a39_recent7_calendar_quantile_harmonic2", 0.5),
            ("a37_recent7_calendar_lgbm_l1_harmonic2_more_regularized", 0.5),
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the standalone current-best no-log DengAI sandbox solution."
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
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for standalone best-solution outputs. Default: oliver/sandbox/outputs/main_goal_best",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional run directory name. Default: UTC timestamp.",
    )
    return parser.parse_args()


def validate_inputs(train_features, test_features, labels, submission_format):
    if "total_cases" in train_features.columns or "total_cases" in test_features.columns:
        raise ValueError("Feature files must not contain total_cases.")
    if "total_cases" not in labels.columns:
        raise ValueError("Labels file must contain total_cases.")
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
            columns.extend([f"weekofyear_sin_{harmonic}", f"weekofyear_cos_{harmonic}"])
    return columns


def add_calendar_features(data, max_harmonic):
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


def lgbm_params(overrides=None):
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
        "random_state": FIXED_RANDOM_STATE,
        "seed": FIXED_RANDOM_STATE,
        "data_random_seed": FIXED_RANDOM_STATE,
        "feature_fraction_seed": FIXED_RANDOM_STATE,
        "bagging_seed": FIXED_RANDOM_STATE,
        "drop_seed": FIXED_RANDOM_STATE,
        "extra_seed": FIXED_RANDOM_STATE,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
        "verbose": -1,
    }
    if overrides:
        params.update(overrides)
    return params


def rf_params():
    return {
        "n_estimators": 500,
        "criterion": "absolute_error",
        "max_depth": 4,
        "min_samples_leaf": 8,
        "min_samples_split": 16,
        "max_features": 1.0,
        "bootstrap": True,
        "random_state": FIXED_RANDOM_STATE,
        "n_jobs": 1,
    }


def quantile_params():
    return {
        "quantile": 0.5,
        "alpha": 0.01,
        "solver": "highs",
    }


def integerize(predictions):
    return [int(round(max(0.0, float(prediction)))) for prediction in predictions]


def recent_training_rows(train_rows, recent_train_years):
    years = sorted(train_rows["year"].unique())
    keep_years = set(years[-int(recent_train_years) :])
    return train_rows[train_rows["year"].isin(keep_years)].copy()


def predict_lgbm(train_rows, future_rows, model_config):
    max_harmonic = int(model_config["max_harmonic"])
    train_model = add_calendar_features(
        recent_training_rows(train_rows, model_config["recent_train_years"]),
        max_harmonic,
    )
    future_model = add_calendar_features(future_rows, max_harmonic)
    feature_columns = calendar_feature_names(max_harmonic)

    model = LGBMRegressor(**lgbm_params(model_config.get("lgbm_overrides")))
    model.fit(train_model[feature_columns], train_model["total_cases"])
    return integerize(model.predict(future_model[feature_columns]))


def predict_rf(train_rows, future_rows, model_config):
    max_harmonic = int(model_config["max_harmonic"])
    train_model = add_calendar_features(
        recent_training_rows(train_rows, model_config["recent_train_years"]),
        max_harmonic,
    )
    future_model = add_calendar_features(future_rows, max_harmonic)
    feature_columns = calendar_feature_names(max_harmonic)

    model = RandomForestRegressor(**rf_params())
    model.fit(train_model[feature_columns], train_model["total_cases"])
    return integerize(model.predict(future_model[feature_columns]))


def predict_quantile(train_rows, future_rows, model_config):
    max_harmonic = int(model_config["max_harmonic"])
    train_model = add_calendar_features(
        recent_training_rows(train_rows, model_config["recent_train_years"]),
        max_harmonic,
    )
    future_model = add_calendar_features(future_rows, max_harmonic)
    feature_columns = calendar_feature_names(max_harmonic)

    model = QuantileRegressor(**quantile_params())
    model.fit(train_model[feature_columns], train_model["total_cases"])
    return integerize(model.predict(future_model[feature_columns]))


def predict_base_model(train_rows, future_rows, model_name):
    model_config = BASE_MODELS[model_name]
    if model_config["model_type"] == "lgbm":
        return predict_lgbm(train_rows, future_rows, model_config)
    if model_config["model_type"] == "rf":
        return predict_rf(train_rows, future_rows, model_config)
    if model_config["model_type"] == "quantile":
        return predict_quantile(train_rows, future_rows, model_config)
    raise ValueError(f"Unsupported model type for {model_name}: {model_config['model_type']}")


def predict_city_blend(city, train_data, test_features):
    blend_config = CITY_BLENDS[city]
    city_train = train_data[train_data["city"] == city].sort_values("week_start_date").copy()
    city_test = test_features[test_features["city"] == city].sort_values("week_start_date").copy()

    weighted_predictions = []
    member_summary = []
    for model_name, weight in blend_config["members"]:
        member_predictions = np.asarray(
            predict_base_model(city_train, city_test, model_name),
            dtype=float,
        )
        weighted_predictions.append(member_predictions * weight)
        member_summary.append(
            {
                "model": model_name,
                "weight": weight,
                "min_prediction": int(member_predictions.min()),
                "max_prediction": int(member_predictions.max()),
                "mean_prediction": float(member_predictions.mean()),
            }
        )

    predictions = np.rint(sum(weighted_predictions)).clip(min=0).astype(int)
    return city_test, predictions.tolist(), member_summary


def build_submission(train_data, test_features, submission_format):
    submission_predictions = {}
    city_summaries = []

    for city in ["sj", "iq"]:
        city_test, predictions, member_summary = predict_city_blend(city, train_data, test_features)
        for row, prediction in zip(city_test.itertuples(index=False), predictions):
            submission_predictions[(row.city, row.year, row.weekofyear)] = int(prediction)
        city_summaries.append(
            {
                "city": city,
                "iteration": CITY_BLENDS[city]["iteration"],
                "rows": len(predictions),
                "min_prediction": int(np.min(predictions)),
                "max_prediction": int(np.max(predictions)),
                "mean_prediction": float(np.mean(predictions)),
                "members": member_summary,
            }
        )

    submission = submission_format.copy()
    submission["total_cases"] = [
        submission_predictions[(row.city, row.year, row.weekofyear)]
        for row in submission.itertuples(index=False)
    ]
    validate_submission(submission, submission_format)
    return submission, city_summaries


def validate_submission(submission, submission_format):
    if list(submission.columns) != ["city", "year", "weekofyear", "total_cases"]:
        raise ValueError("Submission columns must be city, year, weekofyear, total_cases.")
    if len(submission) != len(submission_format):
        raise ValueError("Submission row count must match submission format.")
    if not submission[MERGE_KEYS].equals(submission_format[MERGE_KEYS]):
        raise ValueError("Submission order must match submission format.")
    if submission["total_cases"].isna().any() or (submission["total_cases"] < 0).any():
        raise ValueError("Submission predictions must be non-missing and non-negative.")
    if not pd.api.types.is_integer_dtype(submission["total_cases"]):
        raise ValueError("Submission predictions must be integers.")


def compare_with_reference(submission):
    if not REFERENCE_SUBMISSION.exists():
        return {
            "reference_submission": str(REFERENCE_SUBMISSION),
            "reference_found": False,
            "exact_match": None,
        }

    reference = pd.read_csv(REFERENCE_SUBMISSION)
    if list(reference.columns) != list(submission.columns):
        raise ValueError("Reference submission columns do not match generated submission.")
    if not reference[MERGE_KEYS].equals(submission[MERGE_KEYS]):
        raise ValueError("Reference submission row order does not match generated submission.")

    differences = reference["total_cases"].to_numpy() - submission["total_cases"].to_numpy()
    changed = differences != 0
    changed_count = int(changed.sum())
    return {
        "reference_submission": str(REFERENCE_SUBMISSION),
        "reference_found": True,
        "exact_match": changed_count == 0,
        "changed_rows": changed_count,
        "max_abs_difference": int(np.abs(differences).max()),
        "mean_abs_difference_on_changed_rows": (
            float(np.abs(differences[changed]).mean()) if changed_count else 0.0
        ),
    }


def write_changelog(output_dir, summary):
    lines = [
        "# Current Best Standalone Run",
        "",
        f"Run label: `{summary['run_label']}`",
        "",
        "This run implements only the selected no-log sandbox solution from the additive search.",
        "It uses raw `total_cases` labels for supervised training, but no target-derived input features.",
        "The only model inputs are week-of-year harmonic calendar features.",
        "",
        "## City models",
        "",
        "| City | Selected iteration | Members |",
        "| --- | --- | --- |",
    ]
    for city_summary in summary["city_summaries"]:
        members = ", ".join(
            f"{member['weight']:.4g} x `{member['model']}`"
            for member in city_summary["members"]
        )
        lines.append(
            f"| {city_summary['city']} | `{city_summary['iteration']}` | {members} |"
        )

    comparison = summary["reference_comparison"]
    lines.extend(
        [
            "",
            "## Reference comparison",
            "",
            f"Reference: `{comparison['reference_submission']}`",
        ]
    )
    if comparison["reference_found"]:
        lines.append(f"Exact match: `{comparison['exact_match']}`")
        lines.append(f"Changed rows: `{comparison['changed_rows']}`")
        lines.append(f"Max absolute difference: `{comparison['max_abs_difference']}`")
        lines.append(
            "Mean absolute difference on changed rows: "
            f"`{comparison['mean_abs_difference_on_changed_rows']}`"
        )
    else:
        lines.append("Reference file was not found, so no comparison was run.")

    output_dir.joinpath("changelog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


args = parse_args()

run_label = args.run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_dir = Path(args.output_root) / run_label
output_dir.mkdir(parents=True, exist_ok=False)

random.seed(FIXED_RANDOM_STATE)
np.random.seed(FIXED_RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(FIXED_RANDOM_STATE)

train_features = pd.read_csv(args.train_features_csv, parse_dates=["week_start_date"])
test_features = pd.read_csv(args.test_features_csv, parse_dates=["week_start_date"])
labels = pd.read_csv(args.labels_csv)
submission_format = pd.read_csv(args.submission_format_csv)
validate_inputs(train_features, test_features, labels, submission_format)

train_data = train_features.merge(labels, on=MERGE_KEYS, how="left", validate="one_to_one")
if train_data["total_cases"].isna().any():
    raise ValueError("Merged training data contains missing total_cases values.")

submission, city_summaries = build_submission(train_data, test_features, submission_format)
submission_path = output_dir / "submission.csv"
submission.to_csv(submission_path, index=False)

reference_comparison = compare_with_reference(submission)
summary = {
    "run_label": run_label,
    "fixed_random_state": FIXED_RANDOM_STATE,
    "raw_target": True,
    "log_target": False,
    "feature_policy": "calendar harmonics only; no total_cases-derived input features",
    "input_paths": {
        "train_features_csv": args.train_features_csv,
        "test_features_csv": args.test_features_csv,
        "labels_csv": args.labels_csv,
        "submission_format_csv": args.submission_format_csv,
    },
    "output_paths": {
        "output_dir": str(output_dir),
        "submission_csv": str(submission_path),
        "run_summary_json": str(output_dir / "run_summary.json"),
        "changelog_md": str(output_dir / "changelog.md"),
    },
    "city_summaries": city_summaries,
    "reference_comparison": reference_comparison,
}

(output_dir / "run_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
write_changelog(output_dir, summary)

print(f"Wrote standalone best submission to {submission_path}")
print(f"Reference exact match: {reference_comparison['exact_match']}")
