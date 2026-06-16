"""Analyze city-specific target correlations with monthly lagged DengAI features.

Example:
    python3 oliver/lagged_features_monthly.py
"""

import argparse
import os
from pathlib import Path

import pandas as pd


script_name = Path(__file__).stem
default_output_dir = Path(__file__).resolve().parent / "outputs" / script_name
default_output_dir_label = f"oliver/outputs/{script_name}"


### CLI arguments
parser = argparse.ArgumentParser(
    description="Compute per-city Pearson correlations between monthly total_cases and monthly lagged features."
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
    help="Maximum lag in months. Lag 0 is contemporaneous, lag 14 is 14 months past. Default: 14",
)
parser.add_argument(
    "--min-pairs",
    type=int,
    default=20,
    help="Minimum non-missing target-feature pairs required for a correlation. Default: 20",
)
parser.add_argument(
    "--abs-corr-threshold",
    type=float,
    default=0.10,
    help="Absolute-correlation threshold used in lag summaries. Default: 0.10",
)
parser.add_argument(
    "--top-n-per-city",
    type=int,
    default=25,
    help="Rows per city to keep in the top monthly lagged-correlation summary. Default: 25",
)
parser.add_argument(
    "--skip-plots",
    action="store_true",
    help="Skip PNG plot generation and only write numerical CSV summaries.",
)
parser.add_argument(
    "--plot-dpi",
    type=int,
    default=150,
    help="Resolution for saved plots. Default: 150",
)
parser.add_argument(
    "--top-plot-n",
    type=int,
    default=15,
    help="Feature-lag pairs per city to show in top-correlation bar plots. Default: 15",
)
parser.add_argument(
    "--interpolation-method",
    choices=["linear", "none"],
    default="linear",
    help="Weekly numeric feature interpolation before monthly aggregation. Default: linear",
)
parser.add_argument(
    "--correlations-output",
    default="monthly_lagged_correlations.csv",
    help="Filename for all feature-lag correlations. Default: monthly_lagged_correlations.csv",
)
parser.add_argument(
    "--best-lags-output",
    default="best_month_lags_by_feature.csv",
    help="Filename for each feature's best monthly lag. Default: best_month_lags_by_feature.csv",
)
parser.add_argument(
    "--top-correlations-output",
    default="top_monthly_lagged_correlations_by_city.csv",
    help="Filename for top monthly feature-lag correlations per city. Default: top_monthly_lagged_correlations_by_city.csv",
)
parser.add_argument(
    "--lag-summary-output",
    default="month_lag_summary.csv",
    help="Filename for per-city month-lag aggregate summaries. Default: month_lag_summary.csv",
)
parser.add_argument(
    "--feature-group-summary-output",
    default="feature_group_month_lag_summary.csv",
    help="Filename for per-city feature-group monthly lag summaries. Default: feature_group_month_lag_summary.csv",
)
parser.add_argument(
    "--monthly-data-output",
    default="monthly_aggregated_data.csv",
    help="Filename for the aggregated monthly data used in the analysis. Default: monthly_aggregated_data.csv",
)
parser.add_argument(
    "--missing-summary-output",
    default="missing_summary.csv",
    help="Filename for missing-value counts before and after interpolation. Default: missing_summary.csv",
)
args = parser.parse_args()


### Configuration
if args.max_lag < 0:
    raise ValueError("--max-lag must be non-negative.")

if args.min_pairs < 2:
    raise ValueError("--min-pairs must be at least 2.")

if args.top_n_per_city < 1:
    raise ValueError("--top-n-per-city must be at least 1.")

if args.plot_dpi < 1:
    raise ValueError("--plot-dpi must be at least 1.")

if args.top_plot_n < 1:
    raise ValueError("--top-plot-n must be at least 1.")

train_features_path = Path(args.train_features_csv)
labels_path = Path(args.labels_csv)
output_dir = Path(args.output_dir)
correlations_path = output_dir / args.correlations_output
best_lags_path = output_dir / args.best_lags_output
top_correlations_path = output_dir / args.top_correlations_output
lag_summary_path = output_dir / args.lag_summary_output
feature_group_summary_path = output_dir / args.feature_group_summary_output
monthly_data_path = output_dir / args.monthly_data_output
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
precipitation_feature_columns = [
    column for column in numeric_feature_columns if "precip" in column
]
mean_feature_columns = [
    column for column in numeric_feature_columns if column not in precipitation_feature_columns
]
feature_groups = {
    feature: feature.split("_", maxsplit=1)[0]
    for feature in numeric_feature_columns
}


### 1. Preprocessing

### 1.1 Interpolation
# Interpolate weekly inputs within each city before aggregating to monthly rows.
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


### 1.2 Monthly aggregation
# Monthly cases are summed. Precipitation amount features are summed; other predictors are averaged.
data["month_start"] = data["week_start_date"].dt.to_period("M").dt.to_timestamp()
monthly_aggregation = {
    "week_start_date": "size",
    "total_cases": "sum",
}
for feature in precipitation_feature_columns:
    monthly_aggregation[feature] = "sum"
for feature in mean_feature_columns:
    monthly_aggregation[feature] = "mean"

monthly_data = (
    data.groupby(["city", "month_start"], as_index=False)
    .agg(monthly_aggregation)
    .rename(columns={"week_start_date": "weeks_in_month"})
)
monthly_data["year"] = monthly_data["month_start"].dt.year
monthly_data["month"] = monthly_data["month_start"].dt.month
monthly_data = monthly_data[
    ["city", "month_start", "year", "month", "weeks_in_month", "total_cases"]
    + numeric_feature_columns
].sort_values(["city", "month_start"]).reset_index(drop=True)


### 2. Monthly lagged feature correlations
# Lag k means monthly total_cases at month t is compared with the feature value from month t-k.
correlation_records = []
for city, city_data in monthly_data.groupby("city", sort=False):
    city_data = city_data.sort_values("month_start").reset_index(drop=True)
    target = city_data["total_cases"]

    for feature in numeric_feature_columns:
        for lag in range(args.max_lag + 1):
            lagged_feature = city_data[feature].shift(lag)
            pair_data = pd.DataFrame(
                {
                    "target": target,
                    "lagged_feature": lagged_feature,
                }
            ).dropna()

            pearson_corr = float("nan")
            if (
                len(pair_data) >= args.min_pairs
                and pair_data["target"].nunique() > 1
                and pair_data["lagged_feature"].nunique() > 1
            ):
                pearson_corr = pair_data["target"].corr(pair_data["lagged_feature"])

            correlation_records.append(
                {
                    "city": city,
                    "feature_group": feature_groups[feature],
                    "feature": feature,
                    "lag_months": lag,
                    "n_pairs": len(pair_data),
                    "pearson_corr": pearson_corr,
                    "abs_pearson_corr": abs(pearson_corr),
                }
            )

correlations = pd.DataFrame(correlation_records)
valid_correlations = correlations.dropna(subset=["pearson_corr"]).copy()
top_correlations = (
    valid_correlations.sort_values(
        ["city", "abs_pearson_corr", "lag_months"],
        ascending=[True, False, True],
    )
    .groupby("city", group_keys=False)
    .head(args.top_n_per_city)
)


### 3. Feature-level best monthly lags
best_lags = (
    valid_correlations.sort_values(
        ["city", "feature", "abs_pearson_corr", "lag_months"],
        ascending=[True, True, False, True],
    )
    .drop_duplicates(["city", "feature"], keep="first")
    .rename(
        columns={
            "lag_months": "best_lag_months",
            "n_pairs": "best_n_pairs",
            "pearson_corr": "best_pearson_corr",
            "abs_pearson_corr": "best_abs_pearson_corr",
        }
    )
)

lag0_correlations = valid_correlations[valid_correlations["lag_months"] == 0][
    ["city", "feature", "pearson_corr", "abs_pearson_corr"]
].rename(
    columns={
        "pearson_corr": "lag0_pearson_corr",
        "abs_pearson_corr": "lag0_abs_pearson_corr",
    }
)
best_lags = best_lags.merge(lag0_correlations, on=["city", "feature"], how="left")
best_lags["abs_corr_gain_over_lag0"] = (
    best_lags["best_abs_pearson_corr"] - best_lags["lag0_abs_pearson_corr"]
)
best_lags = best_lags[
    [
        "city",
        "feature_group",
        "feature",
        "best_lag_months",
        "best_n_pairs",
        "best_pearson_corr",
        "best_abs_pearson_corr",
        "lag0_pearson_corr",
        "lag0_abs_pearson_corr",
        "abs_corr_gain_over_lag0",
    ]
].sort_values(["city", "best_abs_pearson_corr"], ascending=[True, False])


### 4. Month-lag summaries
threshold_counts = (
    valid_correlations.assign(
        abs_corr_at_or_above_threshold=lambda frame: (
            frame["abs_pearson_corr"] >= args.abs_corr_threshold
        )
    )
    .groupby(["city", "lag_months"], as_index=False)
    .agg(features_at_or_above_threshold=("abs_corr_at_or_above_threshold", "sum"))
)
lag_summary = (
    valid_correlations.groupby(["city", "lag_months"], as_index=False)
    .agg(
        n_features=("feature", "nunique"),
        mean_abs_pearson_corr=("abs_pearson_corr", "mean"),
        median_abs_pearson_corr=("abs_pearson_corr", "median"),
        max_abs_pearson_corr=("abs_pearson_corr", "max"),
    )
    .merge(threshold_counts, on=["city", "lag_months"], how="left")
)
top_feature_by_lag = (
    valid_correlations.sort_values(
        ["city", "lag_months", "abs_pearson_corr"],
        ascending=[True, True, False],
    )
    .drop_duplicates(["city", "lag_months"], keep="first")
    [["city", "lag_months", "feature", "pearson_corr", "abs_pearson_corr"]]
    .rename(
        columns={
            "feature": "top_feature",
            "pearson_corr": "top_feature_pearson_corr",
            "abs_pearson_corr": "top_feature_abs_pearson_corr",
        }
    )
)
lag_summary = lag_summary.merge(top_feature_by_lag, on=["city", "lag_months"], how="left")


### 5. Feature-group month-lag summaries
feature_group_summary = (
    valid_correlations.groupby(["city", "feature_group", "lag_months"], as_index=False)
    .agg(
        n_features=("feature", "nunique"),
        mean_abs_pearson_corr=("abs_pearson_corr", "mean"),
        median_abs_pearson_corr=("abs_pearson_corr", "median"),
        max_abs_pearson_corr=("abs_pearson_corr", "max"),
    )
)
top_feature_by_group_lag = (
    valid_correlations.sort_values(
        ["city", "feature_group", "lag_months", "abs_pearson_corr"],
        ascending=[True, True, True, False],
    )
    .drop_duplicates(["city", "feature_group", "lag_months"], keep="first")
    [["city", "feature_group", "lag_months", "feature", "pearson_corr", "abs_pearson_corr"]]
    .rename(
        columns={
            "feature": "top_feature",
            "pearson_corr": "top_feature_pearson_corr",
            "abs_pearson_corr": "top_feature_abs_pearson_corr",
        }
    )
)
feature_group_summary = feature_group_summary.merge(
    top_feature_by_group_lag,
    on=["city", "feature_group", "lag_months"],
    how="left",
)


missing_summary = pd.DataFrame(
    {
        "feature": numeric_feature_columns,
        "monthly_aggregation": [
            "sum" if feature in precipitation_feature_columns else "mean"
            for feature in numeric_feature_columns
        ],
        "missing_before": missing_before.reindex(numeric_feature_columns).to_numpy(),
        "missing_after": missing_after.reindex(numeric_feature_columns).to_numpy(),
    }
)


### Output export
output_dir.mkdir(parents=True, exist_ok=True)
monthly_data.to_csv(monthly_data_path, index=False)
correlations.to_csv(correlations_path, index=False)
best_lags.to_csv(best_lags_path, index=False)
top_correlations.to_csv(top_correlations_path, index=False)
lag_summary.to_csv(lag_summary_path, index=False)
feature_group_summary.to_csv(feature_group_summary_path, index=False)
missing_summary.to_csv(missing_summary_path, index=False)


### 7. Visualization
saved_plot_paths = []
if not args.skip_plots:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    feature_order = sorted(
        numeric_feature_columns,
        key=lambda feature: (feature_groups[feature], numeric_feature_columns.index(feature)),
    )
    lag_columns = list(range(args.max_lag + 1))

    for city, city_correlations in valid_correlations.groupby("city", sort=False):
        heatmap_data = (
            city_correlations.pivot(
                index="feature",
                columns="lag_months",
                values="pearson_corr",
            )
            .reindex(index=feature_order, columns=lag_columns)
        )
        max_abs_corr = heatmap_data.abs().max().max()
        if pd.isna(max_abs_corr) or max_abs_corr == 0:
            max_abs_corr = 1.0

        figure, axis = plt.subplots(figsize=(10, max(6, len(feature_order) * 0.32)))
        image = axis.imshow(
            heatmap_data.to_numpy(),
            aspect="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-max_abs_corr, vcenter=0.0, vmax=max_abs_corr),
        )
        axis.set_title(f"{city} monthly feature-target correlation by lag")
        axis.set_xlabel("Lag months")
        axis.set_ylabel("Feature")
        axis.set_xticks(range(len(lag_columns)))
        axis.set_xticklabels(lag_columns)
        axis.set_yticks(range(len(feature_order)))
        axis.set_yticklabels([feature.replace("_", " ") for feature in feature_order], fontsize=7)
        figure.colorbar(image, ax=axis, label="Pearson correlation")
        figure.tight_layout()

        plot_path = output_dir / f"monthly_lagged_correlation_heatmap_{city}.png"
        figure.savefig(plot_path, dpi=args.plot_dpi)
        plt.close(figure)
        saved_plot_paths.append(plot_path)

        city_lag_summary = lag_summary[lag_summary["city"] == city].sort_values("lag_months")
        figure, axis = plt.subplots(figsize=(9, 5))
        for column in [
            "mean_abs_pearson_corr",
            "median_abs_pearson_corr",
            "max_abs_pearson_corr",
        ]:
            axis.plot(
                city_lag_summary["lag_months"],
                city_lag_summary[column],
                marker="o",
                linewidth=1.4,
                label=column.replace("_", " "),
            )
        axis.set_title(f"{city} monthly absolute correlation summary by lag")
        axis.set_xlabel("Lag months")
        axis.set_ylabel("Absolute Pearson correlation")
        axis.set_xticks(lag_columns)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()

        plot_path = output_dir / f"month_lag_summary_{city}.png"
        figure.savefig(plot_path, dpi=args.plot_dpi)
        plt.close(figure)
        saved_plot_paths.append(plot_path)

        city_group_summary = feature_group_summary[
            feature_group_summary["city"] == city
        ].sort_values(["feature_group", "lag_months"])
        figure, axis = plt.subplots(figsize=(9, 5))
        for feature_group, group_data in city_group_summary.groupby("feature_group", sort=True):
            axis.plot(
                group_data["lag_months"],
                group_data["mean_abs_pearson_corr"],
                marker="o",
                linewidth=1.4,
                label=feature_group,
            )
        axis.set_title(f"{city} feature-group monthly mean absolute correlation by lag")
        axis.set_xlabel("Lag months")
        axis.set_ylabel("Mean absolute Pearson correlation")
        axis.set_xticks(lag_columns)
        axis.grid(True, alpha=0.25)
        axis.legend(title="feature group")
        figure.tight_layout()

        plot_path = output_dir / f"feature_group_month_lag_summary_{city}.png"
        figure.savefig(plot_path, dpi=args.plot_dpi)
        plt.close(figure)
        saved_plot_paths.append(plot_path)

        city_top_correlations = (
            city_correlations.sort_values(
                ["abs_pearson_corr", "lag_months"],
                ascending=[False, True],
            )
            .head(args.top_plot_n)
            .sort_values("abs_pearson_corr", ascending=True)
        )
        labels = [
            f"{row.feature} lag {row.lag_months} (r={row.pearson_corr:.3f})"
            for row in city_top_correlations.itertuples()
        ]
        colors = [
            "#4c78a8" if pearson_corr >= 0 else "#e45756"
            for pearson_corr in city_top_correlations["pearson_corr"]
        ]

        figure, axis = plt.subplots(figsize=(11, max(5, len(city_top_correlations) * 0.36)))
        axis.barh(labels, city_top_correlations["abs_pearson_corr"], color=colors, alpha=0.9)
        axis.set_title(f"{city} top monthly lagged feature correlations")
        axis.set_xlabel("Absolute Pearson correlation")
        axis.grid(True, axis="x", alpha=0.25)
        axis.tick_params(axis="y", labelsize=7)
        figure.tight_layout()

        plot_path = output_dir / f"top_monthly_lagged_correlations_{city}.png"
        figure.savefig(plot_path, dpi=args.plot_dpi)
        plt.close(figure)
        saved_plot_paths.append(plot_path)

print("Monthly lagged feature analysis completed.")
print(f"Cities analyzed: {', '.join(monthly_data['city'].drop_duplicates())}")
print(f"Features analyzed: {len(numeric_feature_columns)}")
print(f"Lags analyzed: 0 through {args.max_lag} months")
for city, row_count in monthly_data.groupby("city").size().items():
    print(f"Monthly rows for {city}: {row_count}")
print(f"Numeric feature missing values before interpolation: {int(missing_before.sum())}")
print(f"Numeric feature missing values after interpolation: {int(missing_after.sum())}")
print(f"Saved monthly aggregated data: {monthly_data_path}")
print(f"Saved monthly lagged correlations: {correlations_path}")
print(f"Saved best month lags by feature: {best_lags_path}")
print(f"Saved top monthly lagged correlations by city: {top_correlations_path}")
print(f"Saved month lag summary: {lag_summary_path}")
print(f"Saved feature-group month lag summary: {feature_group_summary_path}")
if saved_plot_paths:
    print("Saved plots:")
    for plot_path in saved_plot_paths:
        print(f"- {plot_path}")
