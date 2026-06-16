"""
preprocess.py — loading, imputation, feature engineering, year-based CV splits.
"""
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ── Constants ────────────────────────────────────────────────────────────────
# Number of validation years per fold.  Each fold holds out this many
# consecutive years; the training fold is everything before them.
# San Juan spans ~19 years, Iquitos ~10 years in training data.
# 2 years per fold means ~5 usable folds for SJ, ~3 for IQ — enough to
# cover multiple full seasonal cycles per validation window.
VAL_YEARS = 2

# Minimum number of training *years* before the first fold is used.
# ARIMA needs enough history; fewer than 4 years risks poor parameter estimates.
MIN_TRAIN_YEARS = 4

EXOG_DRIVERS = [
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "reanalysis_min_air_temp_k",
    "station_avg_temp_c",
    "reanalysis_precip_amt_kg_per_m2",
    "reanalysis_relative_humidity_percent",
]

# Autoregressive lags on target — for Random Forest only
CASE_LAGS = [1, 2, 3, 4, 6, 8, 12, 26, 52]

# Lagged climate drivers — peak lags from EDA cross-correlation analysis
DRIVER_LAGS = {
    "reanalysis_specific_humidity_g_per_kg": [4, 6, 8],
    "reanalysis_dew_point_temp_k":           [4, 6, 8],
    "reanalysis_min_air_temp_k":             [3, 5],
    "station_avg_temp_c":                    [3, 5],
    "reanalysis_precip_amt_kg_per_m2":       [4, 6],
    "reanalysis_relative_humidity_percent":  [4, 6],
}

ROLLING_WINDOWS = [4, 8]


# ── Stationarity ──────────────────────────────────────────────────────────────
def adf_d(series: pd.Series, alpha: float = 0.05, max_d: int = 2) -> int:
    """
    Run ADF on log1p(series) and its successive differences.
    Returns the minimum d for stationarity at significance level alpha.
    Log1p is applied first — variance stabilisation must precede the test.
    """
    y = np.log1p(series.dropna().values.astype(float))
    for d in range(max_d + 1):
        y_d  = np.diff(y, n=d) if d > 0 else y
        pval = adfuller(y_d, autolag="AIC")[1]
        status = "stationary ✓" if pval < alpha else "non-stationary"
        print(f"    d={d}  ADF p={pval:.4f}  {status}")
        if pval < alpha:
            return d
    print(f"    Warning: not stationary after {max_d} diffs; using d={max_d}")
    return max_d


# ── Loading ───────────────────────────────────────────────────────────────────
def load_data(data_dir: Path):
    """Load the four DrivenData DengAI CSVs."""
    idx = ["city", "year", "weekofyear"]
    feat_tr = pd.read_csv(data_dir / "dengue_features_train.csv",
                          index_col=idx, parse_dates=["week_start_date"])
    labels  = pd.read_csv(data_dir / "dengue_labels_train.csv", index_col=idx)
    train   = feat_tr.join(labels)

    sj = train.loc["sj"].sort_values("week_start_date").copy()
    iq = train.loc["iq"].sort_values("week_start_date").copy()

    feat_te = sub = None
    if (data_dir / "dengue_features_test.csv").exists():
        feat_te = pd.read_csv(data_dir / "dengue_features_test.csv",
                              index_col=idx, parse_dates=["week_start_date"])
    if (data_dir / "submission_format.csv").exists():
        sub = pd.read_csv(data_dir / "submission_format.csv", index_col=idx)

    print(f"Loaded — SJ: {len(sj)} rows  IQ: {len(iq)} rows")
    return sj, iq, feat_te, sub


# ── Imputation ────────────────────────────────────────────────────────────────
def impute(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Linear interpolation along the time axis, then median fallback.
    Always called on training rows only so imputation never sees future data.
    """
    out = df.copy()
    out[cols] = out[cols].interpolate(method="linear", limit_direction="both")
    out[cols] = out[cols].fillna(out[cols].median())
    return out


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add autoregressive lags, rolling statistics, lagged climate drivers,
    and sin/cos seasonal encoding.

    Called separately on each fold's training and validation data after
    imputation, so lag features at the start of the validation window
    correctly reference the tail of the training fold (the val df is
    passed in with training rows prepended — see _split_with_context).

    NaN rows from lags are kept; RF training drops them via .dropna().
    ARIMA/ARIMAX operate on total_cases directly and ignore these columns.
    """
    d = df.copy()

    for k in CASE_LAGS:
        d[f"cases_lag{k}"] = d["total_cases"].shift(k)

    for w in ROLLING_WINDOWS:
        base = d["total_cases"].shift(1)
        d[f"cases_roll{w}_mean"] = base.rolling(w).mean()
        d[f"cases_roll{w}_std"]  = base.rolling(w).std()

    for col, lags in DRIVER_LAGS.items():
        if col not in d.columns:
            continue
        for k in lags:
            d[f"{col}_lag{k}"] = d[col].shift(k)

    woy = (d.index.get_level_values("weekofyear")
           if "weekofyear" in d.index.names else d["weekofyear"])
    d["sin_week"] = np.sin(2 * np.pi * woy / 52)
    d["cos_week"] = np.cos(2 * np.pi * woy / 52)

    yr = (d.index.get_level_values("year")
          if "year" in d.index.names else d["year"])
    d["year_centered"] = yr - yr.min()

    return d


# ── Full preprocessing (used for final submission fit) ────────────────────────
def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Impute then engineer features on a full city dataset."""
    feat_cols = [c for c in raw.columns if c not in {"week_start_date", "total_cases"}]
    d = impute(raw, [c for c in feat_cols if c in raw.columns])
    return engineer_features(d)


def preprocess_test(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepend training data before imputing so edge NaNs in test are filled
    using training context.  Returns only the test rows.
    """
    combined  = pd.concat([train_raw, test_raw]).sort_values("week_start_date")
    feat_cols = [c for c in combined.columns
                 if c not in {"week_start_date", "total_cases"}]
    combined  = impute(combined, [c for c in feat_cols if c in combined.columns])
    combined  = engineer_features(combined)
    return combined.iloc[len(train_raw):]


# ── Year-based CV splits ──────────────────────────────────────────────────────
def _get_years(df: pd.DataFrame) -> list[int]:
    """Extract sorted unique years from index or column."""
    if "year" in df.index.names:
        return sorted(df.index.get_level_values("year").unique())
    return sorted(df["year"].unique())


def _split_with_context(raw: pd.DataFrame, tr_years: list, val_years: list):
    """
    Split raw (pre-feature-engineering) data into training and validation folds.

    Imputation is fitted on training rows only, with training medians used
    to fill any remaining NaNs in validation rows.

    For lag feature computation on the validation fold: we concatenate the
    training tail (max lag = 52 rows) with the raw validation rows, engineer
    features on the combined frame, then return only the validation portion.
    This ensures cases_lag52 at the first validation row correctly references
    a true training observation, not NaN.
    """
    feat_cols = [c for c in raw.columns if c not in {"week_start_date", "total_cases"}]

    yr_col = (raw.index.get_level_values("year")
              if "year" in raw.index.names else raw["year"])

    tr_raw  = raw[yr_col.isin(tr_years)].copy()
    val_raw = raw[yr_col.isin(val_years)].copy()

    # Impute training fold
    tr_imp  = impute(tr_raw, [c for c in feat_cols if c in tr_raw.columns])
    tr_meds = tr_imp[[c for c in feat_cols if c in tr_imp.columns]].median()

    # Impute validation fold using training medians for fallback
    val_imp = val_raw.copy()
    val_imp[list(tr_meds.index)] = (
        val_imp[list(tr_meds.index)]
        .interpolate(method="linear", limit_direction="both")
        .fillna(tr_meds)
    )

    # Engineer training features
    tr_feat = engineer_features(tr_imp)

    # Engineer validation features with training tail as context so that
    # lag features at the fold boundary are non-NaN
    max_lag     = max(CASE_LAGS)
    tail        = tr_imp.iloc[-max_lag:]
    val_context = pd.concat([tail, val_imp])
    val_context_feat = engineer_features(val_context)
    val_feat = val_context_feat.iloc[len(tail):]   # drop the tail rows

    return tr_feat, val_feat


def year_based_splits(
    raw: pd.DataFrame,
    val_years: int = VAL_YEARS,
    min_train_years: int = MIN_TRAIN_YEARS,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """
    Yield (fold_idx, train_df, val_df) with fold boundaries at year boundaries.

    Design choices
    ──────────────
    Year-aligned boundaries
        Dengue is strongly annual.  Splitting mid-year would put part of an
        outbreak in training and part in validation, making the validation
        task easier than the real test scenario (whole future years).
        The test set itself spans whole years per city, so year-aligned
        folds are the closest match.

    Expanding window
        Each fold's training set grows by val_years.  More history is always
        useful for a non-stationary epidemic series; a sliding window would
        discard early outbreak patterns that inform later ones.

    val_years = 2
        Two years per validation fold covers at least two full seasonal cycles,
        giving a robust MAE estimate.  One year risks a fold that happens to
        be a low-outbreak year dominating the score.

    min_train_years = 4
        ARIMA needs sufficient history.  Fewer than 4 years risks poor
        parameter estimates, especially for the seasonal lag-52 RF features.

    Lag feature context
        Training tail rows are prepended to the validation data before
        engineer_features() is called, so lag features at the fold boundary
        correctly reference true training observations rather than NaN.
    """
    years = _get_years(raw)
    n     = len(years)

    fold_idx = 0
    # Slide the val window one val_years step at a time
    for val_start in range(min_train_years, n - val_years + 1, val_years):
        tr_years  = years[:val_start]
        val_yrs   = years[val_start: val_start + val_years]

        tr_feat, val_feat = _split_with_context(raw, tr_years, val_yrs)

        print(f"  Fold {fold_idx}: "
              f"train {tr_years[0]}–{tr_years[-1]} ({len(tr_feat)} rows)  "
              f"val {val_yrs[0]}–{val_yrs[-1]} ({len(val_feat)} rows)")

        yield fold_idx, tr_feat, val_feat
        fold_idx += 1