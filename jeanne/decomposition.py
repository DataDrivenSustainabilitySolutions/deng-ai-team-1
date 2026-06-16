import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR / "decomposition_outputs"
INDEX_COLUMNS = {"year", "weekofyear"}


def safe_path_name(value: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe_name or "feature"


def load_city_feature_data(city_id: str) -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")
    features = pd.read_csv(DATA_DIR / "dengue_features_train.csv")

    data = labels.merge(features, on=["city", "year", "weekofyear"], how="left")
    data = data[data["city"] == city_id].copy()
    data["week_start_date"] = pd.to_datetime(data["week_start_date"])
    data = data.sort_values("week_start_date").set_index("week_start_date")
    feature_columns = [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in INDEX_COLUMNS
    ]

    return data[["year", "weekofyear", *feature_columns]]


def make_weekly_series(data: pd.DataFrame, feature: str) -> pd.Series:
    series = data[feature].astype(float)

    if series.isna().all():
        raise ValueError(f"{feature} has no non-missing values.")

    if series.isna().any():
        series = series.interpolate(method="time").ffill().bfill()

    return series


def decompose_series(series: pd.Series, period: int = 52):
    if len(series) < period * 2:
        raise ValueError(
            f"Need at least {period * 2} observations for a stable decomposition, got {len(series)}."
        )

    return STL(series, period=period, robust=True).fit()


def decomposition_components(result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_start_date": result.observed.index,
            "observed": result.observed.to_numpy(),
            "trend": result.trend.to_numpy(),
            "seasonal": result.seasonal.to_numpy(),
            "residual": result.resid.to_numpy(),
        }
    )


def seasonal_profile(result, data: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame(
        {
            "weekofyear": data.loc[result.seasonal.index, "weekofyear"].astype(int).to_numpy(),
            "seasonal": result.seasonal.to_numpy(),
        }
    )

    profile = (
        profile.groupby("weekofyear", as_index=False)["seasonal"]
        .mean()
        .rename(columns={"seasonal": "seasonal_mean"})
        .sort_values("weekofyear")
    )
    profile["seasonal_abs_mean"] = profile["seasonal_mean"].abs()

    return profile


def plot_decomposition(city_id: str, feature: str, result, output_dir: Path) -> Path:
    output_path = output_dir / "stl_decomposition.png"

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(result.observed, color="#1f2937")
    axes[0].set_ylabel("Observed")
    axes[1].plot(result.trend, color="#2563eb")
    axes[1].set_ylabel("Trend")
    axes[2].plot(result.seasonal, color="#16a34a")
    axes[2].set_ylabel("Seasonal")
    axes[3].plot(result.resid, color="#dc2626")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")

    fig.suptitle(f"STL decomposition for {city_id} / {feature}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def plot_seasonal_weeks(city_id: str, feature: str, profile: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "seasonal_week_profile.png"

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(profile["weekofyear"], profile["seasonal_mean"], color="#0f766e")
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title(f"Average seasonal effect by week for {city_id} / {feature}")
    ax.set_xlabel("Week of year")
    ax.set_ylabel("Seasonal effect")
    ax.set_xlim(0.5, 53.5)
    ax.set_xticks(range(1, 54, 4))

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def plot_city_decomposition_overview(city_id: str, feature_results: list[tuple[str, object]], output_dir: Path) -> Path:
    city_dir = output_dir / city_id
    output_path = city_dir / "all_stl_decompositions.png"
    component_specs = [
        ("Observed", "observed", "#1f2937"),
        ("Trend", "trend", "#2563eb"),
        ("Seasonal", "seasonal", "#16a34a"),
        ("Residual", "resid", "#dc2626"),
    ]
    fig, axes = plt.subplots(
        len(feature_results),
        len(component_specs),
        figsize=(24, max(6, len(feature_results) * 1.45)),
        sharex="col",
    )

    if len(feature_results) == 1:
        axes = axes.reshape(1, -1)

    for row, (feature, result) in enumerate(feature_results):
        for col, (title, attr_name, color) in enumerate(component_specs):
            ax = axes[row, col]
            ax.plot(getattr(result, attr_name), color=color, linewidth=0.8)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(feature, rotation=0, ha="right", va="center", labelpad=95)
            ax.tick_params(axis="both", labelsize=7)

    fig.suptitle(f"{city_id}: STL decompositions by feature", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    city_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def print_season_summary(city_id: str, feature: str, profile: pd.DataFrame, top_n: int) -> None:
    if top_n <= 0:
        return

    strongest_positive = profile.nlargest(top_n, "seasonal_mean")
    strongest_negative = profile.nsmallest(top_n, "seasonal_mean")

    print(f"\nCity: {city_id} | Feature: {feature}")
    print("Weeks with strongest positive seasonal effect:")
    print(strongest_positive[["weekofyear", "seasonal_mean"]].to_string(index=False))
    print("\nWeeks with strongest negative seasonal effect:")
    print(strongest_negative[["weekofyear", "seasonal_mean"]].to_string(index=False))


def run_feature(
    city_id: str,
    feature: str,
    data: pd.DataFrame,
    period: int,
    output_dir: Path,
    top_weeks: int,
) -> tuple[list[Path], object]:
    feature_dir = output_dir / city_id / safe_path_name(feature)
    series = make_weekly_series(data, feature)
    result = decompose_series(series, period=period)
    profile = seasonal_profile(result, data)
    components = decomposition_components(result)

    feature_dir.mkdir(parents=True, exist_ok=True)
    components_path = feature_dir / "stl_components.csv"
    profile_path = feature_dir / "seasonal_week_profile.csv"
    components.to_csv(components_path, index=False)
    profile.to_csv(profile_path, index=False)
    decomposition_path = plot_decomposition(city_id, feature, result, feature_dir)
    seasonal_path = plot_seasonal_weeks(city_id, feature, profile, feature_dir)

    print_season_summary(city_id, feature, profile, top_weeks)
    return [components_path, profile_path, decomposition_path, seasonal_path], result


def run_city(
    city_id: str,
    feature: str,
    period: int,
    output_dir: Path,
    top_weeks: int,
) -> list[Path]:
    data = load_city_feature_data(city_id)
    features = [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in INDEX_COLUMNS
    ]

    if feature != "all":
        if feature not in features:
            raise ValueError(f"Unknown feature for {city_id}: {feature}")
        features = [feature]

    saved_paths = []
    decomposition_results = []
    for feature_name in features:
        feature_paths, result = run_feature(city_id, feature_name, data, period, output_dir, top_weeks)
        saved_paths.extend(feature_paths)
        decomposition_results.append((feature_name, result))

    saved_paths.append(plot_city_decomposition_overview(city_id, decomposition_results, output_dir))

    print(f"\nSaved {len(saved_paths)} files for {city_id}:")
    for path in saved_paths:
        print(path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STL decomposition for each dengue city and numeric feature."
    )
    parser.add_argument(
        "--city",
        choices=["sj", "iq", "all"],
        default="all",
        help="City to decompose. Use 'all' for San Juan and Iquitos.",
    )
    parser.add_argument(
        "--feature",
        default="all",
        help="Feature to decompose. Use 'all' for every numeric feature plus total_cases.",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=52,
        help="Season length in weeks. Use 52 for yearly seasonality.",
    )
    parser.add_argument(
        "--top-weeks",
        type=int,
        default=8,
        help="Number of strongest seasonal weeks to print.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for plots and CSV outputs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city_ids = ["sj", "iq"] if args.city == "all" else [args.city]

    for city_id in city_ids:
        run_city(city_id, args.feature, args.period, args.output_dir, args.top_weeks)


if __name__ == "__main__":
    main()
