"""
Example:
    python3 matthias/sarimax_seasonal.py
    python3 matthias/sarimax_seasonal.py --order 1 0 1 --seasonal-order 1 1 1 52
    python3 matthias/sarimax_seasonal.py --order 2 0 2 --seasonal-order 1 1 0 52
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


script_name = Path(__file__).stem
default_output_dir = Path(__file__).resolve().parent / "outputs" / script_name
default_main_dir = Path("../oliver/outputs/main")


### CLI arguments
parser = argparse.ArgumentParser(
    description="Fit per-city SARIMAX models with native seasonal orders (no Fourier terms)."
)
parser.add_argument(
    "--train-csv",
    default=str(default_main_dir / "preprocessed_train.csv"),
    help="Preprocessed+engineered training CSV from main.py. "
    "Default: oliver/outputs/main/preprocessed_train.csv",
)
parser.add_argument(
    "--test-csv",
    default=str(default_main_dir / "preprocessed_test.csv"),
    help="Preprocessed+engineered test CSV from main.py. "
    "Default: oliver/outputs/main/preprocessed_test.csv",
)
parser.add_argument(
    "--submission-format-csv",
    default="../data/submission_format.csv",
    help="Path to the submission format CSV. Default: data/submission_format.csv",
)
parser.add_argument(
    "--output-dir",
    default=str(default_output_dir),
    help=f"Directory for generated outputs. Default: matthias/outputs/{script_name}",
)
parser.add_argument(
    "--submission-output", default="submission.csv",
    help="Filename for final submission predictions. Default: submission.csv",
)
parser.add_argument(
    "--validation-output", default="validation_scores.csv",
    help="Filename for expanding-year validation scores. Default: validation_scores.csv",
)
parser.add_argument(
    "--validation-predictions-output", default="validation_predictions.csv",
    help="Filename for row-level validation predictions. Default: validation_predictions.csv",
)
parser.add_argument(
    "--order", type=int, nargs=3, metavar=("P", "D", "Q"), default=[1, 0, 1],
    help="Non-seasonal ARIMA order (p d q). Default: 1 0 1",
)
parser.add_argument(
    "--seasonal-order", type=int, nargs=4, metavar=("P", "D", "Q", "S"),
    default=[1, 1, 1, 52],
    help="Seasonal ARIMA order (P D Q S). Default: 1 1 1 52",
)
parser.add_argument(
    "--maxiter", type=int, default=200,
    help="Maximum optimiser iterations per SARIMAX fit. Default: 200",
)
args = parser.parse_args()

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as error:
    raise ImportError(
        "statsmodels is required. Install with: python3 -m pip install statsmodels"
    ) from error


### Configuration
train_csv_path = Path(args.train_csv)
test_csv_path = Path(args.test_csv)
submission_format_path = Path(args.submission_format_csv)
output_dir = Path(args.output_dir)
submission_path = output_dir / args.submission_output
validation_scores_path = output_dir / args.validation_output
validation_predictions_path = output_dir / args.validation_predictions_output
merge_keys = ["city", "year", "weekofyear"]
sarimax_order = tuple(args.order)
sarimax_seasonal_order = tuple(args.seasonal_order)

if args.maxiter < 1:
    raise ValueError("--maxiter must be at least 1.")
if sarimax_seasonal_order[3] < 1:
    raise ValueError("Seasonal period S must be at least 1.")

# Exogenous climate feature subsets per city.
# Features are standardised before fitting; any column in the preprocessed CSV is valid.
city_climate_exog = {
    "sj": [
        # Temperature
        "station_avg_temp_c_lag_12",
        "station_avg_temp_c_lag_2",
        "station_avg_temp_c_lag_6",
        "station_avg_temp_c_rolling_12_mean",
        "station_max_temp_c_rolling_14_mean",
        "station_max_temp_c_rolling_5_mean",
        "station_max_temp_c_rolling_6_mean",
        "station_min_temp_c_rolling_14_mean",
        "reanalysis_avg_temp_k_lag_12",
        "reanalysis_max_air_temp_k_lag_12",
        "reanalysis_max_air_temp_k_rolling_12_mean",
        "reanalysis_min_air_temp_k_rolling_14_mean",
        "reanalysis_air_temp_k_rolling_8_mean",
        # Humidity / dew point
        "reanalysis_dew_point_temp_k_lag_10",
        "reanalysis_dew_point_temp_k_lag_3",
        "reanalysis_dew_point_temp_k_lag_5",
        "reanalysis_dew_point_temp_k_lag_8",
        "reanalysis_dew_point_temp_k_rolling_6_mean",
        "reanalysis_dew_point_temp_k_rolling_8_mean",
        "reanalysis_dew_point_temp_k_rolling_10_mean",
        "reanalysis_specific_humidity_g_per_kg_lag_3",
        "reanalysis_specific_humidity_g_per_kg_lag_12",
        "reanalysis_specific_humidity_g_per_kg_rolling_12_mean",
        "reanalysis_relative_humidity_percent",
    ],
    "iq": [
        # Temperature
        "reanalysis_min_air_temp_k_rolling_7_mean",
        "reanalysis_min_air_temp_k_rolling_5_mean",
        "reanalysis_min_air_temp_k_rolling_8_mean",
        "reanalysis_min_air_temp_k_lag_3",
        "reanalysis_min_air_temp_k_lag_2",
        "station_avg_temp_c_rolling_6_mean",
        "station_avg_temp_c_rolling_4_mean",
        "station_avg_temp_c_lag_2",
        "station_avg_temp_c_lag_4",
        "station_avg_temp_c_lag_6",
        "station_max_temp_c_rolling_6_mean",
        "station_max_temp_c_rolling_5_mean",
        "station_max_temp_c_lag_6",
        "station_min_temp_c_lag_6",
        "station_min_temp_c_lag_1",
        # NDVI / vegetation
        "ndvi_sw_lag_11",
        "ndvi_sw_lag_10",
        "ndvi_sw_lag_14",
        "ndvi_nw_lag_14",
        "ndvi_nw_lag_10",
        "ndvi_nw",
        # Precipitation
        "reanalysis_precip_amt_kg_per_m2",
        "precipitation_amt_mm_rolling_7_mean",
        "precipitation_amt_mm_lag_3",
        "precipitation_amt_mm_rolling_14_mean",
        # Humidity
        "reanalysis_specific_humidity_g_per_kg_lag_5",
        "reanalysis_specific_humidity_g_per_kg_lag_2",
        "reanalysis_specific_humidity_g_per_kg_rolling_8_mean",
    ],
}


### Data import
train_data = pd.read_csv(train_csv_path, parse_dates=["week_start_date"])
test_data  = pd.read_csv(test_csv_path,  parse_dates=["week_start_date"])
submission_format = pd.read_csv(submission_format_path)

for required in merge_keys + ["week_start_date"]:
    if required not in train_data.columns or required not in test_data.columns:
        raise ValueError(f"Column '{required}' missing from the preprocessed CSVs.")
if "total_cases" not in train_data.columns:
    raise ValueError("preprocessed train CSV must contain total_cases.")


### Helpers
def build_exog(df, climate_columns):
    """Return the exogenous matrix: just the climate feature subset."""
    return df[climate_columns].copy()


def standardise(train_exog, *other_exog):
    """Z-score using train statistics; fill residual NaNs with 0 (the train mean)."""
    mean = train_exog.mean(axis=0)
    std  = train_exog.std(axis=0).replace(0.0, 1.0)
    scaled = [((train_exog - mean) / std).fillna(0.0)]
    for exog in other_exog:
        scaled.append(((exog - mean) / std).fillna(0.0))
    return scaled


def seasonal_naive(train_df, target_weeks):
    """Fallback: predict the training week-of-year mean for each target week."""
    print("  Fallback to seasonal-naive!")
    woy_mean    = train_df.groupby("weekofyear")["total_cases"].mean()
    global_mean = train_df["total_cases"].mean()
    return target_weeks.map(woy_mean).fillna(global_mean).to_numpy()


def finalize(raw_values):
    """Clip at 0 and round to integer."""
    return [int(round(max(0.0, float(v)))) for v in raw_values]


def fit_forecast(y_log_train, exog_train, exog_future, n_steps):
    """Fit SARIMAX on log1p(cases) and return expm1 forecasts; None on failure."""
    # Use a constant trend only when there is no non-seasonal differencing,
    # to avoid over-parameterising an already-differenced series.
    trend = "c" if sarimax_order[1] == 0 and sarimax_seasonal_order[1] == 0 else "n"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                y_log_train.to_numpy(),
                exog=exog_train.to_numpy(),
                order=sarimax_order,
                seasonal_order=sarimax_seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=args.maxiter, method="lbfgs")
            forecast_log = result.get_forecast(
                steps=n_steps, exog=exog_future.to_numpy()
            ).predicted_mean
            return np.expm1(np.asarray(forecast_log, dtype=float))
        except Exception as exc:
            print(f"  SARIMAX fit/forecast failed: {exc}")
            return None


def validation_years_for(city_df):
    """Expanding full-year validation schedule, identical to main.py."""
    year_counts = city_df.groupby("year").size()
    full_years  = [y for y, n in year_counts.items() if n == 52]
    if not full_years:
        return []
    first_year           = int(city_df["year"].min())
    first_full_after_start = min(y for y in full_years if y > first_year)
    return [y for y in full_years if y > first_full_after_start]


def resolve_columns(df, requested, city):
    """Keep only requested columns that exist; warn about any that don't."""
    present = [c for c in requested if c in df.columns]
    missing = [c for c in requested if c not in df.columns]
    if missing:
        print(f"  [{city}] warning: skipping {len(missing)} missing exog column(s): "
              f"{', '.join(missing)}")
    if not present:
        raise ValueError(f"No requested exog columns are available for city '{city}'.")
    return present


### Expanding full-year validation
validation_score_records      = []
validation_prediction_records = []
all_actuals, all_predictions  = [], []

for city, city_train in train_data.groupby("city", sort=False):
    city_train = city_train.sort_values("week_start_date").reset_index(drop=True)
    climate_columns = resolve_columns(city_train, city_climate_exog[city], city)
    years = validation_years_for(city_train)
    print(f"[{city}] validation years: {years if years else 'none'}")

    for validation_year in years:
        fold_train = city_train[city_train["year"] < validation_year]
        fold_val   = city_train[city_train["year"] == validation_year]

        exog_train_raw = build_exog(fold_train, climate_columns)
        exog_val_raw   = build_exog(fold_val,   climate_columns)
        exog_train_s, exog_val_s = standardise(exog_train_raw, exog_val_raw)

        y_log    = np.log1p(fold_train["total_cases"].astype(float))
        forecast = fit_forecast(y_log, exog_train_s, exog_val_s, len(fold_val))
        if forecast is None:
            forecast = seasonal_naive(fold_train, fold_val["weekofyear"])
            used = "seasonal_naive_fallback"
        else:
            used = f"sarimax{sarimax_order}_s{sarimax_seasonal_order}"

        predictions = finalize(forecast)
        for row, prediction, raw in zip(
            fold_val.itertuples(index=False), predictions, forecast
        ):
            validation_prediction_records.append({
                "city": row.city, "year": row.year, "weekofyear": row.weekofyear,
                "week_start_date": row.week_start_date,
                "validation_year": validation_year,
                "actual_total_cases": row.total_cases,
                "predicted_total_cases": prediction,
                "predicted_total_cases_raw": max(0.0, float(raw)),
                "model": used,
            })

        fold_mae = mean_absolute_error(fold_val["total_cases"], predictions)
        all_actuals.extend(fold_val["total_cases"].tolist())
        all_predictions.extend(predictions)
        validation_score_records.append({
            "city": city, "validation_year": validation_year,
            "train_rows": len(fold_train), "validation_rows": len(fold_val),
            "model": used, "mae": fold_mae,
        })
        print(f"  [{city}] {validation_year}: MAE={fold_mae:.3f} ({used})")

if all_actuals:
    overall = mean_absolute_error(all_actuals, all_predictions)
    validation_score_records.append({
        "city": "all", "validation_year": "all", "train_rows": "",
        "validation_rows": len(all_actuals), "model": "", "mae": overall,
    })

validation_scores      = pd.DataFrame(validation_score_records)
validation_predictions = pd.DataFrame(validation_prediction_records)


### Final per-city models and submission
submission_predictions = {}

for city, city_train in train_data.groupby("city", sort=False):
    city_train = city_train.sort_values("week_start_date").reset_index(drop=True)
    city_test  = (
        test_data[test_data["city"] == city]
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )
    climate_columns = resolve_columns(city_train, city_climate_exog[city], city)

    exog_train_raw = build_exog(city_train, climate_columns)
    exog_test_raw  = build_exog(city_test,  climate_columns)
    exog_train_s, exog_test_s = standardise(exog_train_raw, exog_test_raw)

    y_log    = np.log1p(city_train["total_cases"].astype(float))
    forecast = fit_forecast(y_log, exog_train_s, exog_test_s, len(city_test))
    if forecast is None:
        print(f"[{city}] final model: SARIMAX failed, using seasonal-naive fallback.")
        forecast = seasonal_naive(city_train, city_test["weekofyear"])
    predictions = finalize(forecast)

    for row, prediction in zip(city_test.itertuples(index=False), predictions):
        submission_predictions[(row.city, row.year, row.weekofyear)] = prediction


### Output export
output_dir.mkdir(parents=True, exist_ok=True)

submission = submission_format.copy()
submission["total_cases"] = [
    submission_predictions[(row.city, row.year, row.weekofyear)]
    for row in submission.itertuples(index=False)
]

if list(submission.columns) != ["city", "year", "weekofyear", "total_cases"]:
    raise ValueError("Submission columns must be city, year, weekofyear, total_cases.")
if not submission[merge_keys].equals(submission_format[merge_keys]):
    raise ValueError("Submission row order must match the submission format.")
if submission["total_cases"].isna().any():
    raise ValueError("Submission contains missing predictions.")
if (submission["total_cases"] < 0).any():
    raise ValueError("Submission contains negative predictions.")
submission["total_cases"] = submission["total_cases"].astype(int)

validation_scores.to_csv(validation_scores_path, index=False)
validation_predictions.to_csv(validation_predictions_path, index=False)
submission.to_csv(submission_path, index=False)


### Summary
print("\nSARIMAX seasonal pipeline complete.")
print(f"Non-seasonal order (p,d,q):  {sarimax_order}")
print(f"Seasonal order (P,D,Q,S):    {sarimax_seasonal_order}")
print(f"Validation folds: {len(validation_scores[validation_scores['city'] != 'all'])}")
overall_row = validation_scores.loc[validation_scores["city"] == "all", "mae"]
if not overall_row.empty:
    print(f"Overall validation MAE: {overall_row.iloc[0]:.4f}")
print(f"Saved validation scores:      {validation_scores_path}")
print(f"Saved validation predictions: {validation_predictions_path}")
print(f"Saved submission:             {submission_path}")


### Visualisation
# Figure 1: per-city validation — actuals vs predictions, fold bands + MAE labels.
# Figure 2: full training history stitched to test-period forecast per city.
# Both are saved as high-resolution PNGs next to the CSV outputs.

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("\n[visualisation] matplotlib not found – skipping plots. "
          "Install with: pip install matplotlib")
else:
    CITY_LABELS     = {"sj": "San José", "iq": "Iquitos"}
    COLOR_ACTUAL    = "#1a1a2e"
    COLOR_PREDICTED = "#e84545"
    COLOR_FORECAST  = "#2d6a4f"
    BAND_ALPHA      = 0.06

    def _fold_bands(ax, fold_years, date_series):
        for idx, year in enumerate(fold_years):
            mask = date_series.dt.year == year
            if not mask.any():
                continue
            ax.axvspan(
                date_series[mask].iloc[0], date_series[mask].iloc[-1],
                color="#888888" if idx % 2 == 0 else "#444444",
                alpha=BAND_ALPHA, linewidth=0,
            )

    def _style(ax, title):
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylabel("Total cases", fontsize=9)
        ax.set_xlabel("Week start date", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    cities = (
        sorted(validation_predictions["city"].unique())
        if not validation_predictions.empty
        else sorted(train_data["city"].unique())
    )

    # ── Figure 1: Validation actuals vs predictions ────────────────────────
    if not validation_predictions.empty:
        fig1, axes1 = plt.subplots(
            len(cities), 1, figsize=(16, 5 * len(cities)), squeeze=False
        )
        fig1.suptitle(
            "SARIMAX (seasonal) – Validation: Actual vs Predicted dengue cases",
            fontsize=14, fontweight="bold", y=1.01,
        )
        for ax, city in zip(axes1[:, 0], cities):
            vp = (
                validation_predictions[validation_predictions["city"] == city]
                .sort_values("week_start_date")
                .reset_index(drop=True)
            )
            if vp.empty:
                ax.set_visible(False)
                continue

            fold_years = sorted(vp["validation_year"].unique())
            _fold_bands(ax, fold_years, vp["week_start_date"])

            ax.plot(vp["week_start_date"], vp["actual_total_cases"],
                    color=COLOR_ACTUAL, linewidth=1.5, label="Actual", zorder=3)
            ax.plot(vp["week_start_date"], vp["predicted_total_cases"],
                    color=COLOR_PREDICTED, linewidth=1.5, linestyle="--",
                    alpha=0.88, label="Predicted", zorder=4)

            y_ann = vp["actual_total_cases"].max() * 0.93
            for year in fold_years:
                fold_df = vp[vp["validation_year"] == year]
                if fold_df.empty:
                    continue
                mae_val = mean_absolute_error(
                    fold_df["actual_total_cases"], fold_df["predicted_total_cases"]
                )
                x_mid = fold_df["week_start_date"].iloc[len(fold_df) // 2]
                ax.annotate(
                    f"{year}\nMAE {mae_val:.1f}",
                    xy=(x_mid, y_ann),
                    ha="center", va="top",
                    fontsize=7.5, color="#555555",
                    annotation_clip=True,
                )

            city_scores = validation_scores[
                (validation_scores["city"] == city)
                & (validation_scores["validation_year"] != "all")
            ]
            avg_mae = (
                city_scores["mae"].astype(float).mean()
                if not city_scores.empty else float("nan")
            )
            _style(ax, f"{CITY_LABELS.get(city, city.upper())}  –  "
                       f"mean validation MAE: {avg_mae:.2f}")

        fig1.tight_layout()
        p1 = output_dir / "validation_actual_vs_predicted.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"Saved validation plot:        {p1}")

    # ── Figure 2: Training history + test forecast ─────────────────────────
    fig2, axes2 = plt.subplots(
        len(cities), 1, figsize=(16, 5 * len(cities)), squeeze=False
    )
    fig2.suptitle(
        "SARIMAX (seasonal) – Training history & test-period forecast",
        fontsize=14, fontweight="bold", y=1.01,
    )
    for ax, city in zip(axes2[:, 0], cities):
        tr = (
            train_data[train_data["city"] == city]
            .sort_values("week_start_date")
            .reset_index(drop=True)
        )
        city_test_rows = test_data[test_data["city"] == city].sort_values("week_start_date")
        fc = (
            submission[submission["city"] == city]
            .merge(
                city_test_rows[["city", "year", "weekofyear", "week_start_date"]],
                on=["city", "year", "weekofyear"],
                how="left",
            )
            .sort_values("week_start_date")
            .reset_index(drop=True)
        )

        ax.plot(tr["week_start_date"], tr["total_cases"],
                color=COLOR_ACTUAL, linewidth=1.2, label="Training actuals", zorder=3)

        if not fc.empty:
            bridge_x = [tr["week_start_date"].iloc[-1], fc["week_start_date"].iloc[0]]
            bridge_y = [tr["total_cases"].iloc[-1],      fc["total_cases"].iloc[0]]
            ax.plot(bridge_x, bridge_y,
                    color=COLOR_FORECAST, linewidth=1.5, linestyle="--", zorder=4)
            ax.plot(fc["week_start_date"], fc["total_cases"],
                    color=COLOR_FORECAST, linewidth=1.8, linestyle="--",
                    label="Test forecast", zorder=4)
            ax.axvspan(fc["week_start_date"].iloc[0], fc["week_start_date"].iloc[-1],
                       color=COLOR_FORECAST, alpha=0.07, linewidth=0)
            ax.axvline(fc["week_start_date"].iloc[0],
                       color=COLOR_FORECAST, linewidth=0.8, linestyle=":")

        _style(ax, f"{CITY_LABELS.get(city, city.upper())}  –  "
                   f"training history & forecast")

    fig2.tight_layout()
    p2 = output_dir / "forecast_vs_history.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved forecast plot:          {p2}")