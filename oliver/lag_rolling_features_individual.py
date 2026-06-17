"""Select individual lag and rolling feature candidates from correlations.

Example:
    python3 oliver/lag_rolling_features_individual.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd


script_name = Path(__file__).stem
default_output_dir = Path(__file__).resolve().parent / "outputs" / script_name
default_output_dir_label = f"oliver/outputs/{script_name}"


### CLI arguments
parser = argparse.ArgumentParser(
    description=(
        "Compute city-specific lag/rolling correlations and select exact feature-window "
        "candidates for a later modelling script."
    )
)
parser.add_argument(
    "--train-features-csv",
    default="data/dengue_features_train.csv",
    help="Path to the training feature CSV. Default: data/dengue_features_train.csv",
)
parser.add_argument(
    "--labels-csv",
    default="data/dengue_labels_train.csv",
    help="Path to the training label CSV. Default: data/dengue_labels_train.csv",
)
parser.add_argument(
    "--output-dir",
    default=default_output_dir,
    help=f"Directory for generated analysis outputs. Default: {default_output_dir_label}",
)
parser.add_argument(
    "--max-lag",
    type=int,
    default=14,
    help="Maximum weekly lag to test. Lag 0 is the raw same-week feature. Default: 14",
)
parser.add_argument(
    "--max-window",
    type=int,
    default=14,
    help="Maximum rolling window when --windows is omitted. Default: 14",
)
parser.add_argument(
    "--windows",
    type=int,
    nargs="+",
    default=None,
    help="Rolling window sizes in weeks. Default: 1 through --max-window.",
)
parser.add_argument(
    "--min-window-periods",
    type=int,
    default=1,
    help="Minimum non-missing values required inside each rolling window. Default: 1",
)
parser.add_argument(
    "--min-pairs",
    type=int,
    default=20,
    help="Minimum target-feature pairs required for a correlation. Default: 20",
)
parser.add_argument(
    "--min-abs-corr",
    type=float,
    default=0.15,
    help="Fallback absolute-correlation threshold for selecting a feature. Default: 0.15",
)
parser.add_argument(
    "--city-min-abs-corr",
    nargs="*",
    default=["sj=0.20", "iq=0.15"],
    help=(
        "Optional city-specific thresholds as city=value. "
        "Default: sj=0.20 iq=0.15"
    ),
)
parser.add_argument(
    "--near-best-ratio",
    type=float,
    default=0.95,
    help=(
        "Keep additional lags/windows whose absolute correlation is at least this "
        "share of that feature's best value. Default: 0.95"
    ),
)
parser.add_argument(
    "--max-lags-per-feature",
    type=int,
    default=3,
    help="Maximum selected lag rows per city-feature. Default: 3",
)
parser.add_argument(
    "--max-windows-per-feature",
    type=int,
    default=3,
    help="Maximum selected rolling rows per city-feature. Default: 3",
)
parser.add_argument(
    "--interpolation-method",
    choices=["linear", "none"],
    default="linear",
    help="Numeric feature interpolation before correlation analysis. Default: linear",
)
parser.add_argument(
    "--all-correlations-output",
    default="individual_lag_rolling_correlations.csv",
    help="Filename for all lag and rolling correlations.",
)
parser.add_argument(
    "--selected-features-output",
    default="selected_individual_features.csv",
    help="Filename for selected lag/rolling rows before raw de-duplication.",
)
parser.add_argument(
    "--selected-lags-output",
    default="selected_lag_plan.csv",
    help="Filename for selected lag candidates.",
)
parser.add_argument(
    "--selected-rollings-output",
    default="selected_rolling_plan.csv",
    help="Filename for selected rolling-window candidates.",
)
parser.add_argument(
    "--feature-plan-output",
    default="individual_feature_plan.csv",
    help="Filename for de-duplicated generated features.",
)
parser.add_argument(
    "--feature-plan-json-output",
    default="individual_feature_plan.json",
    help="Filename for the de-duplicated feature plan in JSON.",
)
parser.add_argument(
    "--summary-output",
    default="individual_feature_plan.md",
    help="Filename for the readable feature-selection summary.",
)
parser.add_argument(
    "--missing-summary-output",
    default="missing_summary.csv",
    help="Filename for missing-value counts before and after interpolation.",
)
args = parser.parse_args()


### Configuration
if args.max_lag < 0:
    raise ValueError("--max-lag must be non-negative.")

if args.max_window < 1:
    raise ValueError("--max-window must be at least 1.")

window_sizes = sorted(set(args.windows if args.windows else range(1, args.max_window + 1)))
if any(window < 1 for window in window_sizes):
    raise ValueError("--windows must contain positive integers.")

if args.min_window_periods < 1:
    raise ValueError("--min-window-periods must be at least 1.")

if args.min_window_periods > min(window_sizes):
    raise ValueError("--min-window-periods cannot exceed the smallest requested rolling window.")

if args.min_pairs < 2:
    raise ValueError("--min-pairs must be at least 2.")

if args.min_abs_corr < 0:
    raise ValueError("--min-abs-corr must be non-negative.")

if not 0 < args.near_best_ratio <= 1:
    raise ValueError("--near-best-ratio must be greater than 0 and at most 1.")

if args.max_lags_per_feature < 1:
    raise ValueError("--max-lags-per-feature must be at least 1.")

if args.max_windows_per_feature < 1:
    raise ValueError("--max-windows-per-feature must be at least 1.")

city_abs_corr_thresholds = {}
for city_threshold in args.city_min_abs_corr:
    if "=" not in city_threshold:
        raise ValueError("--city-min-abs-corr values must use city=value format.")
    city, value = city_threshold.split("=", maxsplit=1)
    city = city.strip()
    if not city:
        raise ValueError("--city-min-abs-corr contains an empty city value.")
    city_abs_corr_thresholds[city] = float(value)
    if city_abs_corr_thresholds[city] < 0:
        raise ValueError("--city-min-abs-corr thresholds must be non-negative.")

train_features_path = Path(args.train_features_csv)
labels_path = Path(args.labels_csv)
output_dir = Path(args.output_dir)
all_correlations_path = output_dir / args.all_correlations_output
selected_features_path = output_dir / args.selected_features_output
selected_lags_path = output_dir / args.selected_lags_output
selected_rollings_path = output_dir / args.selected_rollings_output
feature_plan_path = output_dir / args.feature_plan_output
feature_plan_json_path = output_dir / args.feature_plan_json_output
summary_path = output_dir / args.summary_output
missing_summary_path = output_dir / args.missing_summary_output
merge_keys = ["city", "year", "weekofyear"]
identifier_columns = {"city", "year", "weekofyear", "week_start_date", "total_cases"}


### Data import
train_features = pd.read_csv(train_features_path, parse_dates=["week_start_date"])
labels = pd.read_csv(labels_path)

if train_features[merge_keys].duplicated().any():
    raise ValueError("Feature rows must be unique by city, year, and weekofyear.")

if labels[merge_keys].duplicated().any():
    raise ValueError("Label rows must be unique by city, year, and weekofyear.")

data = train_features.merge(labels, on=merge_keys, how="left", validate="one_to_one")
if data["total_cases"].isna().any():
    raise ValueError("Merged training data contains missing total_cases values.")

data = data.sort_values(["city", "week_start_date"]).reset_index(drop=True)
numeric_feature_columns = [
    column
    for column in train_features.columns
    if column not in identifier_columns and pd.api.types.is_numeric_dtype(train_features[column])
]
feature_groups = {
    feature: feature.split("_", maxsplit=1)[0]
    for feature in numeric_feature_columns
}


### 1. Preprocessing
# Interpolate within each city only. The two city time series are never blended.
missing_before = data[numeric_feature_columns].isna().sum()

city_frames = []
for city, city_data in data.groupby("city", sort=False):
    city_data = city_data.sort_values("week_start_date").copy()
    if args.interpolation_method == "linear":
        city_data[numeric_feature_columns] = city_data[numeric_feature_columns].interpolate(
            method="linear",
            limit_direction="both",
        )
    city_frames.append(city_data)

data = pd.concat(city_frames, ignore_index=True)
missing_after = data[numeric_feature_columns].isna().sum()


### 2. Lag and rolling correlations
correlation_records = []
for city, city_data in data.groupby("city", sort=False):
    city_data = city_data.sort_values("week_start_date").reset_index(drop=True)
    target = city_data["total_cases"]

    for feature in numeric_feature_columns:
        for lag in range(args.max_lag + 1):
            lagged_feature = city_data[feature].shift(lag)
            pair_data = pd.DataFrame(
                {
                    "target": target,
                    "candidate": lagged_feature,
                }
            ).dropna()

            pearson_corr = float("nan")
            if (
                len(pair_data) >= args.min_pairs
                and pair_data["target"].nunique() > 1
                and pair_data["candidate"].nunique() > 1
            ):
                pearson_corr = pair_data["target"].corr(pair_data["candidate"])

            generated_feature = feature if lag == 0 else f"{feature}_lag_{lag}"
            correlation_records.append(
                {
                    "city": city,
                    "analysis_type": "lag",
                    "feature_group": feature_groups[feature],
                    "feature": feature,
                    "generated_feature": generated_feature,
                    "parameter_weeks": lag,
                    "n_pairs": len(pair_data),
                    "pearson_corr": pearson_corr,
                    "abs_pearson_corr": abs(pearson_corr),
                }
            )

        for window in window_sizes:
            rolling_feature = city_data[feature].rolling(
                window=window,
                min_periods=args.min_window_periods,
            ).mean()
            pair_data = pd.DataFrame(
                {
                    "target": target,
                    "candidate": rolling_feature,
                }
            ).dropna()

            pearson_corr = float("nan")
            if (
                len(pair_data) >= args.min_pairs
                and pair_data["target"].nunique() > 1
                and pair_data["candidate"].nunique() > 1
            ):
                pearson_corr = pair_data["target"].corr(pair_data["candidate"])

            generated_feature = feature if window == 1 else f"{feature}_rolling_{window}_mean"
            correlation_records.append(
                {
                    "city": city,
                    "analysis_type": "rolling",
                    "feature_group": feature_groups[feature],
                    "feature": feature,
                    "generated_feature": generated_feature,
                    "parameter_weeks": window,
                    "n_pairs": len(pair_data),
                    "pearson_corr": pearson_corr,
                    "abs_pearson_corr": abs(pearson_corr),
                }
            )

all_correlations = pd.DataFrame(correlation_records)
valid_correlations = all_correlations.dropna(subset=["pearson_corr"]).copy()


### 3. Individual feature selection
# Keep exact lag/window choices close to each feature's own best correlation.
selected_records = []
for (city, analysis_type, feature), feature_correlations in valid_correlations.groupby(
    ["city", "analysis_type", "feature"],
    sort=False,
):
    feature_correlations = feature_correlations.sort_values(
        ["abs_pearson_corr", "parameter_weeks"],
        ascending=[False, True],
    ).copy()
    best_abs_corr = float(feature_correlations.iloc[0]["abs_pearson_corr"])
    city_threshold = city_abs_corr_thresholds.get(city, args.min_abs_corr)
    if best_abs_corr < city_threshold:
        continue

    selection_cutoff = max(city_threshold, best_abs_corr * args.near_best_ratio)
    max_selected_rows = (
        args.max_lags_per_feature if analysis_type == "lag" else args.max_windows_per_feature
    )
    selected_rows = feature_correlations[
        feature_correlations["abs_pearson_corr"] >= selection_cutoff
    ].head(max_selected_rows)

    for rank, row in enumerate(selected_rows.itertuples(index=False), start=1):
        selected_records.append(
            {
                "city": row.city,
                "analysis_type": row.analysis_type,
                "feature_group": row.feature_group,
                "feature": row.feature,
                "generated_feature": row.generated_feature,
                "parameter_weeks": int(row.parameter_weeks),
                "n_pairs": int(row.n_pairs),
                "pearson_corr": float(row.pearson_corr),
                "abs_pearson_corr": float(row.abs_pearson_corr),
                "best_abs_pearson_corr": best_abs_corr,
                "relative_to_best": float(row.abs_pearson_corr) / best_abs_corr,
                "city_abs_corr_threshold": city_threshold,
                "selection_cutoff_abs_corr": selection_cutoff,
                "selection_rank_within_feature": rank,
                "selection_reason": (
                    "best feature correlation clears city threshold and this "
                    "candidate is near that feature's best value"
                ),
            }
        )

selected_columns = [
    "city",
    "analysis_type",
    "feature_group",
    "feature",
    "generated_feature",
    "parameter_weeks",
    "n_pairs",
    "pearson_corr",
    "abs_pearson_corr",
    "best_abs_pearson_corr",
    "relative_to_best",
    "city_abs_corr_threshold",
    "selection_cutoff_abs_corr",
    "selection_rank_within_feature",
    "selection_reason",
]
selected_features = pd.DataFrame(selected_records, columns=selected_columns)

if selected_features.empty:
    feature_plan = pd.DataFrame(
        columns=[
            "city",
            "transform",
            "feature_group",
            "feature",
            "generated_feature",
            "parameter_weeks",
            "source_analysis_type",
            "pearson_corr",
            "abs_pearson_corr",
            "best_abs_pearson_corr",
            "relative_to_best",
            "city_abs_corr_threshold",
            "selection_cutoff_abs_corr",
        ]
    )
else:
    feature_plan = selected_features.copy()
    feature_plan["transform"] = feature_plan["analysis_type"].replace(
        {"lag": "lag", "rolling": "rolling_mean"}
    )
    feature_plan.loc[
        feature_plan["generated_feature"] == feature_plan["feature"],
        "transform",
    ] = "raw"
    feature_plan["source_analysis_type"] = feature_plan["analysis_type"]
    feature_plan["transform_order"] = feature_plan["transform"].map(
        {"raw": 0, "lag": 1, "rolling_mean": 2}
    )
    feature_plan = (
        feature_plan.sort_values(
            ["city", "generated_feature", "abs_pearson_corr", "transform_order"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["city", "generated_feature"], keep="first")
        .sort_values(
            ["city", "transform_order", "feature_group", "feature", "parameter_weeks"],
            ascending=[True, True, True, True, True],
        )
    )
    feature_plan = feature_plan[
        [
            "city",
            "transform",
            "feature_group",
            "feature",
            "generated_feature",
            "parameter_weeks",
            "source_analysis_type",
            "pearson_corr",
            "abs_pearson_corr",
            "best_abs_pearson_corr",
            "relative_to_best",
            "city_abs_corr_threshold",
            "selection_cutoff_abs_corr",
        ]
    ].reset_index(drop=True)

selected_lags = selected_features[selected_features["analysis_type"] == "lag"].copy()
selected_rollings = selected_features[selected_features["analysis_type"] == "rolling"].copy()

if not selected_lags.empty:
    selected_lags = selected_lags.rename(columns={"parameter_weeks": "lag_weeks"})

if not selected_rollings.empty:
    selected_rollings = selected_rollings.rename(columns={"parameter_weeks": "window_weeks"})


### 4. Readable summary and machine-readable plan
plan_payload = {
    "selection_rule": {
        "max_lag": args.max_lag,
        "windows": window_sizes,
        "min_pairs": args.min_pairs,
        "min_abs_corr": args.min_abs_corr,
        "city_min_abs_corr": city_abs_corr_thresholds,
        "near_best_ratio": args.near_best_ratio,
        "max_lags_per_feature": args.max_lags_per_feature,
        "max_windows_per_feature": args.max_windows_per_feature,
        "interpolation_method": args.interpolation_method,
    },
    "features": feature_plan.to_dict(orient="records"),
}

summary_lines = [
    "# Individual Lag/Rolling Feature Selection",
    "",
    "This is an analysis-only output for a later modelling variant.",
    "`total_cases` is used only as the target for correlations, never as an input feature.",
    "",
    "## Selection Rule",
    "",
    (
        f"- Default absolute-correlation threshold is {args.min_abs_corr:.3f}; "
        + ", ".join(
            f"{city} uses {threshold:.3f}"
            for city, threshold in sorted(city_abs_corr_thresholds.items())
        )
        + "."
    ),
    (
        f"- For each city, feature, and transform type, keep up to "
        f"{args.max_lags_per_feature} lags or {args.max_windows_per_feature} rolling "
        f"windows within {args.near_best_ratio:.0%} of that feature's best absolute correlation."
    ),
    "- Lag 0 and rolling window 1 are de-duplicated into the raw feature in the final plan.",
    "",
    "## Selected Feature Plan",
    "",
]

for city in sorted(data["city"].unique()):
    city_plan = feature_plan[feature_plan["city"] == city]
    summary_lines.extend(
        [
            f"### {city}",
            "",
            f"- Selected generated features: {len(city_plan)}",
        ]
    )
    for transform in ["raw", "lag", "rolling_mean"]:
        transform_plan = city_plan[city_plan["transform"] == transform]
        if transform_plan.empty:
            continue

        summary_lines.append(f"- {transform}: {len(transform_plan)}")
        for feature, feature_rows in transform_plan.groupby("feature", sort=True):
            if transform == "raw":
                best_corr = feature_rows["abs_pearson_corr"].max()
                summary_lines.append(f"  - {feature}: raw, abs corr {best_corr:.3f}")
            elif transform == "lag":
                lags = ", ".join(str(int(value)) for value in feature_rows["parameter_weeks"])
                best_corr = feature_rows["abs_pearson_corr"].max()
                summary_lines.append(f"  - {feature}: lags {lags}, best abs corr {best_corr:.3f}")
            else:
                windows = ", ".join(str(int(value)) for value in feature_rows["parameter_weeks"])
                best_corr = feature_rows["abs_pearson_corr"].max()
                summary_lines.append(
                    f"  - {feature}: rolling windows {windows}, best abs corr {best_corr:.3f}"
                )
        summary_lines.append("")

missing_summary = pd.DataFrame(
    {
        "feature": numeric_feature_columns,
        "missing_before": missing_before.reindex(numeric_feature_columns).to_numpy(),
        "missing_after": missing_after.reindex(numeric_feature_columns).to_numpy(),
    }
)


### Output export
output_dir.mkdir(parents=True, exist_ok=True)
all_correlations.to_csv(all_correlations_path, index=False)
selected_features.to_csv(selected_features_path, index=False)
selected_lags.to_csv(selected_lags_path, index=False)
selected_rollings.to_csv(selected_rollings_path, index=False)
feature_plan.to_csv(feature_plan_path, index=False)
missing_summary.to_csv(missing_summary_path, index=False)
feature_plan_json_path.write_text(
    json.dumps(plan_payload, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

print("Individual lag/rolling feature analysis completed.")
print(f"Cities analyzed: {', '.join(data['city'].drop_duplicates())}")
print(f"Features analyzed: {len(numeric_feature_columns)}")
print(f"Lags analyzed: 0 through {args.max_lag}")
print(f"Rolling windows analyzed: {', '.join(str(window) for window in window_sizes)}")
print(f"Numeric feature missing values before interpolation: {int(missing_before.sum())}")
print(f"Numeric feature missing values after interpolation: {int(missing_after.sum())}")
print(f"Selected rows before raw de-duplication: {len(selected_features)}")
print(f"Selected generated features after raw de-duplication: {len(feature_plan)}")
print(f"Saved all correlations: {all_correlations_path}")
print(f"Saved selected rows: {selected_features_path}")
print(f"Saved feature plan: {feature_plan_path}")
print(f"Saved readable summary: {summary_path}")
