"""SARIMAX + Optuna hyperparameter search for DengAI.

Searches over ARIMA order (p, 0, q) and seasonal order (P, 0, Q, 52) using
Optuna, evaluated by expanding-window cross-validated MAE. Fourier features
are removed; seasonality is handled entirely by the (P,D,Q,52) seasonal order.

Run oliver/main.py first to produce the preprocessed CSVs.

Example:
    python3 matthias/sarimax_optuna.py
    python3 matthias/sarimax_optuna.py --n-trials 100 --timeout 3600
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as error:
    raise ImportError(
        "optuna is required. Install with: pip install optuna"
    ) from error

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as error:
    raise ImportError(
        "statsmodels is required. Install with: pip install statsmodels"
    ) from error


script_name = Path(__file__).stem
default_output_dir = Path(__file__).resolve().parent / "outputs" / script_name
default_main_dir = Path("../oliver/outputs/main")


### CLI arguments
parser = argparse.ArgumentParser(
    description="SARIMAX with Optuna hyperparameter search for DengAI."
)
parser.add_argument(
    "--train-csv",
    default=str(default_main_dir / "preprocessed_train.csv"),
)
parser.add_argument(
    "--test-csv",
    default=str(default_main_dir / "preprocessed_test.csv"),
)
parser.add_argument(
    "--submission-format-csv",
    default="../data/submission_format.csv",
)
parser.add_argument(
    "--output-dir",
    default=str(default_output_dir),
)
parser.add_argument("--submission-output",              default="submission.csv")
parser.add_argument("--validation-output",              default="validation_scores.csv")
parser.add_argument("--validation-predictions-output",  default="validation_predictions.csv")
parser.add_argument("--best-params-output",             default="best_params.csv")
parser.add_argument(
    "--n-trials", type=int, default=50,
    help="Number of Optuna trials per city. Default: 50",
)
parser.add_argument(
    "--timeout", type=float, default=None,
    help="Time limit in seconds for the Optuna search per city. Default: none",
)
parser.add_argument(
    "--maxiter", type=int, default=200,
    help="Maximum optimiser iterations per SARIMAX fit. Default: 200",
)
parser.add_argument(
    "--sampler", default="tpe", choices=["tpe", "random"],
    help="Optuna sampler. tpe is Bayesian; random is a useful baseline. Default: tpe",
)
args = parser.parse_args()


### Fixed configuration
SEASONAL_PERIOD = 52
D_ORDER  = 0   # non-seasonal differencing: fixed (log transform handles stationarity)
SD_ORDER = 0   # seasonal differencing: fixed (avoids losing 52 rows per fold)
merge_keys = ["city", "year", "weekofyear"]

# Search bounds.
# p, q non-seasonal: 0-4, but never both zero simultaneously.
# P, Q seasonal: capped at 2. Seasonal terms at lag 52 are expensive and
# the dataset has ~18 annual cycles — P=3 or Q=3 would be overparameterised.
P_RANGE = (0, 2)
Q_RANGE = (0, 4)
SP_RANGE = (0, 2)
SQ_RANGE = (0, 2)

output_dir = Path(args.output_dir)
train_csv_path = Path(args.train_csv)
test_csv_path = Path(args.test_csv)
submission_format_path = Path(args.submission_format_csv)
submission_path = output_dir / args.submission_output
validation_scores_path = output_dir / args.validation_output
validation_predictions_path = output_dir / args.validation_predictions_output
best_params_path = output_dir / args.best_params_output

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
test_data  = pd.read_csv(test_csv_path,  parse_dates=["week_start_date"])
submission_format = pd.read_csv(submission_format_path)

for required in merge_keys + ["week_start_date"]:
    if required not in train_data.columns or required not in test_data.columns:
        raise ValueError(f"Column '{required}' missing from the preprocessed CSVs.")
if "total_cases" not in train_data.columns:
    raise ValueError("preprocessed train CSV must contain total_cases.")


### Helpers
def build_exog(df, climate_columns):
    """Return the climate-only exogenous matrix."""
    return df[climate_columns].copy()


def standardise(train_exog, *other_exog):
    """Z-score using train statistics only; fill residual NaNs with 0."""
    mean = train_exog.mean(axis=0)
    std  = train_exog.std(axis=0).replace(0.0, 1.0)
    scaled = [((train_exog - mean) / std).fillna(0.0)]
    for exog in other_exog:
        scaled.append(((exog - mean) / std).fillna(0.0))
    return scaled


def seasonal_naive(train_df, target_weeks):
    """Fallback: training week-of-year mean."""
    woy_mean    = train_df.groupby("weekofyear")["total_cases"].mean()
    global_mean = train_df["total_cases"].mean()
    return target_weeks.map(woy_mean).fillna(global_mean).to_numpy()


def finalize(raw_values):
    """Clip at 0 and round to integer."""
    return [int(round(max(0.0, float(v)))) for v in raw_values]


def fit_forecast(y_log_train, exog_train, exog_future, n_steps, order, s_order):
    """Fit SARIMAX and return expm1 forecasts; None on any failure.

    trend="c" is included when neither d nor D is non-zero, anchoring the
    model to the long-run mean of the log-transformed series. If any
    differencing is active the constant would become a linear trend after
    un-differencing, so it is suppressed with "n". Since D_ORDER and
    D_ORDER are both fixed at 0 here, trend is always "c", but the logic
    is kept explicit for clarity.
    """
    p, d, q     = order
    P, D, Q, s  = s_order
    trend = "c" if d == 0 and D == 0 else "n"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                y_log_train.to_numpy(),
                exog=exog_train.to_numpy(),
                order=order,
                seasonal_order=s_order,
                trend=trend,
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
    """Expanding full-year validation folds.

    Skips two lead years so the seasonal AR(1) at lag 52 always has at
    least one complete annual cycle of history in the earliest fold.
    """
    year_counts = city_df.groupby("year").size()
    full_years  = [yr for yr, rows in year_counts.items() if rows == 52]
    if not full_years:
        return []
    first_year             = int(city_df["year"].min())
    first_full_after_start = min(yr for yr in full_years if yr > first_year)
    candidates             = [yr for yr in full_years if yr > first_full_after_start]
    if not candidates:
        return []
    second_full = min(candidates)
    return [yr for yr in full_years if yr > second_full]


def resolve_columns(df, requested, city):
    """Return only the requested columns that exist in df."""
    present = [col for col in requested if col in df.columns]
    missing = [col for col in requested if col not in df.columns]
    if missing:
        print(f"  [{city}] warning: skipping {len(missing)} missing exog column(s): "
              f"{', '.join(missing)}")
    if not present:
        raise ValueError(f"No requested exog columns available for city '{city}'.")
    return present


def cross_val_mae(city_df, climate_columns, order, s_order):
    """Expanding-window cross-validated MAE for one city and one (order, s_order) pair.

    Returns the mean MAE across all validation folds. If every fold falls
    back to seasonal-naive (i.e. SARIMAX failed on all of them), returns
    a large penalty value so Optuna avoids that region of the search space.
    """
    years = validation_years_for(city_df)
    if not years:
        return float("inf")

    fold_maes      = []
    sarimax_folds  = 0

    for val_year in years:
        fold_train = city_df[city_df["year"] < val_year]
        fold_val   = city_df[city_df["year"] == val_year]

        exog_tr = build_exog(fold_train, climate_columns)
        exog_vl = build_exog(fold_val,   climate_columns)
        exog_tr_s, exog_vl_s = standardise(exog_tr, exog_vl)

        y_log    = np.log1p(fold_train["total_cases"].astype(float))
        forecast = fit_forecast(y_log, exog_tr_s, exog_vl_s, len(fold_val), order, s_order)

        if forecast is None:
            preds = finalize(seasonal_naive(fold_train, fold_val["weekofyear"]))
        else:
            preds = finalize(forecast)
            sarimax_folds += 1

        fold_maes.append(mean_absolute_error(fold_val["total_cases"], preds))

    # Penalise parameter combinations that never converge — they are not
    # genuinely better than the fallback, and we want Optuna to move away.
    if sarimax_folds == 0:
        return float("inf")

    return float(np.mean(fold_maes))


### Optuna search — one study per city
if args.sampler == "tpe":
    sampler = optuna.samplers.TPESampler(seed=42)
else:
    sampler = optuna.samplers.RandomSampler(seed=42)

best_orders = {}   # city -> {"order": (p,d,q), "seasonal_order": (P,D,Q,s)}
best_params_records = []

for city, city_train in train_data.groupby("city", sort=False):
    city_train     = city_train.sort_values("week_start_date").reset_index(drop=True)
    climate_columns = resolve_columns(city_train, city_climate_exog[city], city)

    print(f"\n[{city}] Starting Optuna search ({args.n_trials} trials)...")

    def objective(trial):
        p  = trial.suggest_int("p",  *P_RANGE)
        q  = trial.suggest_int("q",  *Q_RANGE)
        sp = trial.suggest_int("sp", *SP_RANGE)
        sq = trial.suggest_int("sq", *SQ_RANGE)

        # Skip degenerate case: no non-seasonal dynamics at all.
        # A model with p=0, q=0 has no short-term structure — the residuals
        # would be pure white noise at non-seasonal lags, which is almost
        # never true for weekly disease counts.
        if p == 0 and q == 0:
            raise optuna.exceptions.TrialPruned()

        # Skip fully empty model: no dynamics at any timescale.
        if p == 0 and q == 0 and sp == 0 and sq == 0:
            raise optuna.exceptions.TrialPruned()

        order   = (p,  D_ORDER,  q)
        s_order = (sp, SD_ORDER, sq, SEASONAL_PERIOD)

        return cross_val_mae(city_train, climate_columns, order, s_order)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"sarimax_{city}",
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best   = study.best_trial
    best_p  = best.params["p"]
    best_q  = best.params["q"]
    best_sp = best.params["sp"]
    best_sq = best.params["sq"]

    best_orders[city] = {
        "order":          (best_p,  D_ORDER,  best_q),
        "seasonal_order": (best_sp, SD_ORDER, best_sq, SEASONAL_PERIOD),
    }
    best_params_records.append({
        "city": city,
        "p":  best_p,  "d":  D_ORDER,  "q":  best_q,
        "P":  best_sp, "D":  SD_ORDER, "Q":  best_sq, "s": SEASONAL_PERIOD,
        "cv_mae": best.value,
        "n_trials_completed": len(study.trials),
    })

    print(f"[{city}] Best order: ({best_p},{D_ORDER},{best_q})"
          f"({best_sp},{SD_ORDER},{best_sq},{SEASONAL_PERIOD})"
          f"  CV MAE={best.value:.4f}")


### Expanding validation with best parameters (for output CSVs)
validation_score_records      = []
validation_prediction_records = []
all_actuals, all_predictions  = [], []


for city, city_train in train_data.groupby("city", sort=False):
    city_train      = city_train.sort_values("week_start_date").reset_index(drop=True)
    climate_columns = resolve_columns(city_train, city_climate_exog[city], city)
    order           = best_orders[city]["order"]
    s_order         = best_orders[city]["seasonal_order"]
    years           = validation_years_for(city_train)

    print(f"\n[{city}] Re-running validation with best params {order}x{s_order}...")

    for val_year in years:
        fold_train = city_train[city_train["year"] < val_year]
        fold_val   = city_train[city_train["year"] == val_year]

        exog_tr = build_exog(fold_train, climate_columns)
        exog_vl = build_exog(fold_val,   climate_columns)
        exog_tr_s, exog_vl_s = standardise(exog_tr, exog_vl)

        y_log    = np.log1p(fold_train["total_cases"].astype(float))
        forecast = fit_forecast(y_log, exog_tr_s, exog_vl_s, len(fold_val), order, s_order)

        if forecast is None:
            forecast = seasonal_naive(fold_train, fold_val["weekofyear"])
            used = "seasonal_naive_fallback"
        else:
            used = f"sarimax{order}x{s_order}"

        predictions = finalize(forecast)
        for row, pred, raw in zip(fold_val.itertuples(index=False), predictions, forecast):
            validation_prediction_records.append({
                "city":                      row.city,
                "year":                      row.year,
                "weekofyear":                row.weekofyear,
                "week_start_date":           row.week_start_date,
                "validation_year":           val_year,
                "actual_total_cases":        row.total_cases,
                "predicted_total_cases":     pred,
                "predicted_total_cases_raw": max(0.0, float(raw)),
                "model":                     used,
            })

        fold_mae = mean_absolute_error(fold_val["total_cases"], predictions)
        all_actuals.extend(fold_val["total_cases"].tolist())
        all_predictions.extend(predictions)
        validation_score_records.append({
            "city":            city,
            "validation_year": val_year,
            "train_rows":      len(fold_train),
            "validation_rows": len(fold_val),
            "model":           used,
            "mae":             fold_mae,
        })
        print(f"  [{city}] {val_year}: MAE={fold_mae:.3f} ({used})")

if all_actuals:
    overall = mean_absolute_error(all_actuals, all_predictions)
    validation_score_records.append({
        "city": "all", "validation_year": "all", "train_rows": "",
        "validation_rows": len(all_actuals), "model": "", "mae": overall,
    })

validation_scores      = pd.DataFrame(validation_score_records)
validation_predictions = pd.DataFrame(validation_prediction_records)

### Validation plot export
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    for city, city_train in train_data.groupby("city", sort=False):
        city_train  = city_train.sort_values("week_start_date").reset_index(drop=True)
        city_preds  = validation_predictions[validation_predictions["city"] == city].copy()
        city_preds  = city_preds.sort_values("week_start_date").reset_index(drop=True)

        if city_preds.empty:
            print(f"[{city}] No validation predictions to plot.")
            continue

        fig, ax = plt.subplots(figsize=(16, 5))

        # --- Actual cases across the full training timeline ---
        ax.plot(
            city_train["week_start_date"],
            city_train["total_cases"],
            color="#444444",
            linewidth=1.0,
            label="actual cases",
            zorder=2,
        )

        # --- Predicted cases per validation fold, each in its own colour ---
        # Each fold is a distinct year; colouring them separately makes it
        # easy to see whether early or late folds are harder to predict.
        val_years  = sorted(city_preds["validation_year"].unique())
        cmap       = plt.cm.get_cmap("tab10", len(val_years))
        year_color = {yr: cmap(i) for i, yr in enumerate(val_years)}

        for val_year in val_years:
            fold = city_preds[city_preds["validation_year"] == val_year]
            ax.plot(
                fold["week_start_date"],
                fold["predicted_total_cases"],
                color=year_color[val_year],
                linewidth=1.5,
                linestyle="--",
                zorder=3,
                label=f"predicted {val_year}",
            )

        # --- Vertical lines at fold boundaries ---
        # Drawn at the first week of each validation year so the expanding
        # train/validation split is visually unambiguous.
        for val_year in val_years:
            fold_start = city_preds.loc[
                city_preds["validation_year"] == val_year, "week_start_date"
            ].min()
            ax.axvline(
                fold_start,
                color=year_color[val_year],
                linewidth=0.8,
                linestyle=":",
                alpha=0.6,
                zorder=1,
            )

        # --- Per-fold MAE annotations ---
        # Placed at the top of each fold's time window so you can immediately
        # see which years were hardest to predict without consulting the CSV.
        y_max = city_train["total_cases"].max()
        for val_year in val_years:
            fold = city_preds[city_preds["validation_year"] == val_year]
            fold_mae = mean_absolute_error(
                fold["actual_total_cases"], fold["predicted_total_cases"]
            )
            mid_date = fold["week_start_date"].iloc[len(fold) // 2]
            ax.text(
                mid_date,
                y_max * 1.02,
                f"MAE\n{fold_mae:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=year_color[val_year],
            )

        # --- Overall MAE in the title ---
        overall_mae = mean_absolute_error(
            city_preds["actual_total_cases"], city_preds["predicted_total_cases"]
        )
        best = best_orders[city]
        order_str = (
            f"SARIMAX{best['order']}x{best['seasonal_order']} "
            f"| overall CV MAE: {overall_mae:.2f}"
        )

        ax.set_title(f"{city.upper()} — actual vs predicted  ({order_str})", fontsize=11)
        ax.set_xlabel("date")
        ax.set_ylabel("total cases")
        ax.set_xlim(city_train["week_start_date"].min(), city_train["week_start_date"].max())
        ax.margins(y=0.12)   # headroom for MAE annotations
        ax.legend(fontsize=8, ncol=min(len(val_years) + 1, 6), loc="upper left")
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

        fig.tight_layout()
        plot_path = output_dir / f"validation_plot_{city}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{city}] Saved validation plot: {plot_path}")

except ImportError:
    print("matplotlib not found — skipping plots. Install with: pip install matplotlib")

### Final per-city models and submission
submission_predictions = {}

for city, city_train in train_data.groupby("city", sort=False):
    city_train      = city_train.sort_values("week_start_date").reset_index(drop=True)
    city_test       = (test_data[test_data["city"] == city]
                       .sort_values("week_start_date").reset_index(drop=True))
    climate_columns = resolve_columns(city_train, city_climate_exog[city], city)
    order           = best_orders[city]["order"]
    s_order         = best_orders[city]["seasonal_order"]

    exog_tr = build_exog(city_train, climate_columns)
    exog_te = build_exog(city_test,  climate_columns)
    exog_tr_s, exog_te_s = standardise(exog_tr, exog_te)

    y_log    = np.log1p(city_train["total_cases"].astype(float))
    forecast = fit_forecast(y_log, exog_tr_s, exog_te_s, len(city_test), order, s_order)

    if forecast is None:
        print(f"[{city}] final model failed; using seasonal-naive fallback.")
        forecast = seasonal_naive(city_train, city_test["weekofyear"])

    predictions = finalize(forecast)
    for row, pred in zip(city_test.itertuples(index=False), predictions):
        submission_predictions[(row.city, row.year, row.weekofyear)] = pred


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

best_params_df = pd.DataFrame(best_params_records)

validation_scores.to_csv(validation_scores_path,           index=False)
validation_predictions.to_csv(validation_predictions_path, index=False)
submission.to_csv(submission_path,                         index=False)
best_params_df.to_csv(best_params_path,                    index=False)


### Summary
print("\nSARIMAX + Optuna search complete.")
for rec in best_params_records:
    print(f"  [{rec['city']}] best: "
          f"({rec['p']},{rec['d']},{rec['q']})"
          f"({rec['P']},{rec['D']},{rec['Q']},{rec['s']})  "
          f"CV MAE={rec['cv_mae']:.4f}  "
          f"trials={rec['n_trials_completed']}")
overall_row = validation_scores.loc[validation_scores["city"] == "all", "mae"]
if not overall_row.empty:
    print(f"Overall validation MAE: {overall_row.iloc[0]:.4f}")
print(f"Saved best params:            {best_params_path}")
print(f"Saved validation scores:      {validation_scores_path}")
print(f"Saved validation predictions: {validation_predictions_path}")
print(f"Saved submission:             {submission_path}")