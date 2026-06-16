import argparse
import os
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


def load_city_cases(city_id: str) -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")
    features = pd.read_csv(
        DATA_DIR / "dengue_features_train.csv",
        usecols=["city", "year", "weekofyear", "week_start_date"],
    )

    data = labels.merge(features, on=["city", "year", "weekofyear"], how="left")
    data = data[data["city"] == city_id].copy()
    data["week_start_date"] = pd.to_datetime(data["week_start_date"])
    data = data.sort_values("week_start_date").set_index("week_start_date")

    return data[["year", "weekofyear", "total_cases"]]


def make_weekly_series(data: pd.DataFrame) -> pd.Series:
    series = data["total_cases"].astype(float)

    if series.isna().any():
        series = series.interpolate(method="time").ffill().bfill()

    return series


def decompose_cases(series: pd.Series, period: int = 52):
    if len(series) < period * 2:
        raise ValueError(
            f"Need at least {period * 2} observations for a stable decomposition, got {len(series)}."
        )

    return STL(series, period=period, robust=True).fit()


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


def plot_decomposition(city_id: str, result, output_dir: Path) -> Path:
    output_path = output_dir / f"{city_id}_stl_decomposition.png"

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

    fig.suptitle(f"STL decomposition for {city_id}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def plot_seasonal_weeks(city_id: str, profile: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / f"{city_id}_seasonal_week_profile.png"

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(profile["weekofyear"], profile["seasonal_mean"], color="#0f766e")
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title(f"Average seasonal effect by week for {city_id}")
    ax.set_xlabel("Week of year")
    ax.set_ylabel("Seasonal effect")
    ax.set_xlim(0.5, 53.5)
    ax.set_xticks(range(1, 54, 4))

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def print_season_summary(city_id: str, profile: pd.DataFrame, top_n: int) -> None:
    strongest_positive = profile.nlargest(top_n, "seasonal_mean")
    strongest_negative = profile.nsmallest(top_n, "seasonal_mean")

    print(f"\nCity: {city_id}")
    print("Weeks with strongest positive seasonal effect:")
    print(strongest_positive[["weekofyear", "seasonal_mean"]].to_string(index=False))
    print("\nWeeks with strongest negative seasonal effect:")
    print(strongest_negative[["weekofyear", "seasonal_mean"]].to_string(index=False))


def run_city(city_id: str, period: int, output_dir: Path, top_weeks: int) -> None:
    data = load_city_cases(city_id)
    series = make_weekly_series(data)
    result = decompose_cases(series, period=period)
    profile = seasonal_profile(result, data)

    output_dir.mkdir(parents=True, exist_ok=True)
    decomposition_path = plot_decomposition(city_id, result, output_dir)
    seasonal_path = plot_seasonal_weeks(city_id, profile, output_dir)
    profile_path = output_dir / f"{city_id}_seasonal_week_profile.csv"
    profile.to_csv(profile_path, index=False)

    print_season_summary(city_id, profile, top_weeks)
    print("\nSaved files:")
    print(decomposition_path)
    print(seasonal_path)
    print(profile_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STL decomposition on dengue weekly case counts."
    )
    parser.add_argument(
        "--city",
        choices=["sj", "iq", "all"],
        default="all",
        help="City to decompose. Use 'all' for San Juan and Iquitos.",
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
        run_city(city_id, args.period, args.output_dir, args.top_weeks)


if __name__ == "__main__":
    main()
