"""SARIMAX baseline for DengAI, reusing the engineered features from main.py.

This reads the preprocessed train/test CSVs written by `oliver/main.py`
(`preprocessed_train.csv`, `preprocessed_test.csv`) and fits one SARIMAX model
per city on a compact subset of the engineered exogenous features, then runs the
same expanding full-year validation and writes a submission in the required
format. Run `oliver/main.py` first so the preprocessed CSVs exist.

Design notes
------------
* Seasonality: the 52-week annual cycle is modelled with Fourier (sine/cosine)
  terms passed as exogenous regressors, not a seasonal AR/MA order with period
  52 (which is numerically intractable for weekly data). This is the standard
  "ARIMA errors + Fourier terms" / dynamic-regression approach for long periods.
  --fourier-harmonics 1 reproduces main.py's single weekofyear sin/cos pair.
* Counts: total_cases is overdispersed and non-negative, so we model
  log1p(total_cases) with a Gaussian SARIMAX and invert with expm1. Because
  expm1 is monotonic, the back-transformed point forecast is the predictive
  *median*, which is the estimator that minimises MAE - the competition metric.
* Exogenous regressors are standardised using training-fold statistics only
  (no leakage). Residual NaNs from the head-of-series lag/rolling features are
  filled with 0 after standardisation (i.e. the training mean).
* If a SARIMAX fit or forecast fails to converge, the fold falls back to a
  seasonal-naive prediction (training week-of-year mean) so the pipeline always
  produces a complete, valid submission.

Example:
    python3 matthias/sarimax.py
    python3 matthias/sarimax.py --order 1 1 1 --fourier-harmonics 2
"""

import argparse
import math
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
    description="Fit per-city SARIMAX models on main.py's engineered features."
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
    help=f"Directory for generated outputs. Default: oliver/outputs/{script_name}",
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
    "--order", type=int, nargs=3, metavar=("P", "D", "Q"), default=[2, 0, 1],
    help="Non-seasonal SARIMAX order (p d q). Default: 2 0 1",
)
parser.add_argument(
    "--fourier-harmonics", type=int, default=3,
    help="Number of Fourier harmonic pairs for the 52-week season. "
    "Use 1 to match main.py's single sin/cos pair. Default: 3",
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
        "statsmodels is required for the SARIMAX pipeline. Install it, e.g.: "
        "python3 -m pip install statsmodels"
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
SEASONAL_PERIOD = 52.0
sarimax_order = tuple(args.order)

if args.fourier_harmonics < 1:
    raise ValueError("--fourier-harmonics must be at least 1.")
if args.maxiter < 1:
    raise ValueError("--maxiter must be at least 1.")

# Compact, biologically motivated exogenous subsets, drawn from the engineered
# columns main.py produces. San Juan leans on longer (6-12 week) temperature and
# humidity signals; Iquitos on shorter-lag humidity / dew point / min temperature.
# Edit these freely - any column present in the preprocessed CSV may be used.
city_climate_exog = {
    "sj": [
        "reanalysis_specific_humidity_g_per_kg_lag_8",
        "reanalysis_dew_point_temp_k_lag_10",
        "reanalysis_min_air_temp_k_lag_6",
        "station_avg_temp_c_rolling_8_mean",
        "reanalysis_specific_humidity_g_per_kg_rolling_12_mean",
    ],
    "iq": [
        "reanalysis_specific_humidity_g_per_kg_lag_2",
        "reanalysis_dew_point_temp_k_lag_2",
        "reanalysis_min_air_temp_k_lag_3",
        "station_min_temp_c_lag_1",
        "reanalysis_specific_humidity_g_per_kg_rolling_4_mean",
    ],
}


### Data import
train_data = pd.read_csv(train_csv_path, parse_dates=["week_start_date"])
test_data = pd.read_csv(test_csv_path, parse_dates=["week_start_date"])
submission_format = pd.read_csv(submission_format_path)

for required in merge_keys + ["week_start_date"]:
    if required not in train_data.columns or required not in test_data.columns:
        raise ValueError(f"Column '{required}' missing from the preprocessed CSVs.")
if "total_cases" not in train_data.columns:
    raise ValueError("preprocessed train CSV must contain total_cases.")


### Helpers
def fourier_terms(weekofyear, harmonics):
    """Sine/cosine pairs for the annual cycle: columns fourier_sin_k / fourier_cos_k."""
    angle = 2.0 * math.pi * np.asarray(weekofyear, dtype=float) / SEASONAL_PERIOD
    out = {}
    for k in range(1, harmonics + 1):
        out[f"fourier_sin_{k}"] = np.sin(k * angle)
        out[f"fourier_cos_{k}"] = np.cos(k * angle)
    return pd.DataFrame(out)


def build_exog(df, climate_columns, harmonics):
    """Assemble the exogenous matrix (Fourier season + climate subset) for df."""
    fourier = fourier_terms(df["weekofyear"].to_numpy(), harmonics)
    fourier.index = df.index
    climate = df[climate_columns].copy()
    return pd.concat([fourier, climate], axis=1)


def standardise(train_exog, *other_exog):
    """Z-score using train statistics; fill residual NaNs with 0 (the train mean)."""
    mean = train_exog.mean(axis=0)
    std = train_exog.std(axis=0).replace(0.0, 1.0)
    scaled = [((train_exog - mean) / std).fillna(0.0)]
    for exog in other_exog:
        scaled.append(((exog - mean) / std).fillna(0.0))
    return scaled


def seasonal_naive(train_df, target_weeks):
    """Fallback: predict the training week-of-year mean for each target week."""
    woy_mean = train_df.groupby("weekofyear")["total_cases"].mean()
    global_mean = train_df["total_cases"].mean()
    return target_weeks.map(woy_mean).fillna(global_mean).to_numpy()


def finalize(raw_values):
    """Clip at 0, round to integer - matching main.py's submission convention."""
    return [int(round(max(0.0, float(value)))) for value in raw_values]


def fit_forecast(y_log_train, exog_train, exog_future, n_steps):
    """Fit SARIMAX on log1p target and return expm1 forecasts; None on failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                y_log_train.to_numpy(),
                exog=exog_train.to_numpy(),
                order=sarimax_order,
                seasonal_order=(0, 0, 0, 0),
                trend="c" if sarimax_order[1] == 0 else "n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=args.maxiter, method="lbfgs")
            forecast_log = result.get_forecast(
                steps=n_steps, exog=exog_future.to_numpy()
            ).predicted_mean
            return np.expm1(np.asarray(forecast_log, dtype=float))
        except Exception:
            return None


def validation_years_for(city_df):
    """Expanding full-year validation schedule, identical to main.py."""
    year_counts = city_df.groupby("year").size()
    full_years = [year for year, rows in year_counts.items() if rows == 52]
    if not full_years:
        return []
    first_year = int(city_df["year"].min())
    first_full_after_start = min(year for year in full_years if year > first_year)
    return [year for year in full_years if year > first_full_after_start]


def resolve_columns(df, requested, city):
    """Keep only requested exog columns that exist; warn about any that don't."""
    present = [column for column in requested if column in df.columns]
    missing = [column for column in requested if column not in df.columns]
    if missing:
        print(f"  [{city}] warning: skipping {len(missing)} missing exog column(s): "
              f"{', '.join(missing)}")
    if not present:
        raise ValueError(f"No requested exog columns are available for city '{city}'.")
    return present


### Expanding full-year validation
validation_score_records = []
validation_prediction_records = []
all_actuals, all_predictions = [], []

for city, city_train in train_data.groupby("city", sort=False):
    city_train = city_train.sort_values("week_start_date").reset_index(drop=True)
    climate_columns = resolve_columns(city_train, city_climate_exog[city], city)
    years = validation_years_for(city_train)
    print(f"[{city}] validation years: {years if years else 'none'}")

    for validation_year in years:
        fold_train = city_train[city_train["year"] < validation_year]
        fold_val = city_train[city_train["year"] == validation_year]

        exog_train = build_exog(fold_train, climate_columns, args.fourier_harmonics)
        exog_val = build_exog(fold_val, climate_columns, args.fourier_harmonics)
        exog_train_s, exog_val_s = standardise(exog_train, exog_val)

        y_log = np.log1p(fold_train["total_cases"].astype(float))
        forecast = fit_forecast(y_log, exog_train_s, exog_val_s, len(fold_val))
        if forecast is None:
            forecast = seasonal_naive(fold_train, fold_val["weekofyear"])
            used = "seasonal_naive_fallback"
        else:
            used = f"sarimax{sarimax_order}"

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

validation_scores = pd.DataFrame(validation_score_records)
validation_predictions = pd.DataFrame(validation_prediction_records)


### Final per-city models and submission
submission_predictions = {}

for city, city_train in train_data.groupby("city", sort=False):
    city_train = city_train.sort_values("week_start_date").reset_index(drop=True)
    city_test = (
        test_data[test_data["city"] == city]
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )
    climate_columns = resolve_columns(city_train, city_climate_exog[city], city)

    exog_train = build_exog(city_train, climate_columns, args.fourier_harmonics)
    exog_test = build_exog(city_test, climate_columns, args.fourier_harmonics)
    exog_train_s, exog_test_s = standardise(exog_train, exog_test)

    y_log = np.log1p(city_train["total_cases"].astype(float))
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
print("\nSARIMAX pipeline complete.")
print(f"Order (p,d,q): {sarimax_order} | Fourier harmonics: {args.fourier_harmonics}")
print(f"Validation folds: {len(validation_scores[validation_scores['city'] != 'all'])}")
overall_row = validation_scores.loc[validation_scores["city"] == "all", "mae"]
if not overall_row.empty:
    print(f"Overall validation MAE: {overall_row.iloc[0]:.4f}")
print(f"Saved validation scores: {validation_scores_path}")
print(f"Saved validation predictions: {validation_predictions_path}")
print(f"Saved submission: {submission_path}")