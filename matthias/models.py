import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
from numpy.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from preprocess import (CASE_LAGS, DRIVER_LAGS, EXOG_DRIVERS,
                        MIN_TRAIN_YEARS, ROLLING_WINDOWS, VAL_YEARS,
                        adf_d, year_based_splits)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
_PQ = {"sj": (3, 2), "iq": (2, 1)}

RF_PARAM_GRID = [
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 4},
    {"n_estimators": 300, "max_depth": 10,   "min_samples_leaf": 4},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 8},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 4},
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _log(y):  return np.log1p(np.asarray(y, dtype=float))
def _exp(y):  return np.clip(np.expm1(np.asarray(y, dtype=float)), 0, None)

def _exog_cols(df):  return [c for c in EXOG_DRIVERS if c in df.columns]
def _exog_matrix(df): return df[_exog_cols(df)].values.astype(float)

def _rf_feature_cols(df):
    return [c for c in df.columns if c not in {"total_cases", "week_start_date"}]

def _diff_cols(X: np.ndarray, d: int) -> np.ndarray:
    for _ in range(d):
        X = np.diff(X, axis=0)
    return X


def _arima_fit(model):
    """
    Fit an ARIMA/ARIMAX model with a stable method, falling back gracefully.

    - innovations_mle is faster and more stable for low-order models but
      raises LinAlgError when the initial state covariance is numerically
      singular (short series, near-unit-root after differencing).
    - disp is NOT a valid kwarg for statsmodels.tsa.arima.model.ARIMA.fit()
      (it was only valid for the legacy statsmodels.tsa.arima_model.ARIMA).
    - Fallback to lbfgs (the default optimizer) which is more robust.
    """
    try:
        return model.fit(method="innovations_mle", low_memory=True)
    except (LinAlgError, ValueError):
        return model.fit(method="lbfgs", low_memory=True)


# ── Baseline ──────────────────────────────────────────────────────────────────
def seasonal_naive(tr, val) -> float:
    tr_r  = tr.reset_index()  if "weekofyear" in tr.index.names  else tr
    val_r = val.reset_index() if "weekofyear" in val.index.names else val
    woy_mean = tr_r.groupby("weekofyear")["total_cases"].mean()
    preds = val_r["weekofyear"].map(woy_mean).fillna(tr_r["total_cases"].mean())
    return mean_absolute_error(val_r["total_cases"], preds)


# ── ARIMA — recursive multi-step ──────────────────────────────────────────────
def _arima_forecast(tr: pd.DataFrame, val: pd.DataFrame, order: tuple):
    """
    Fit ARIMA on the full training fold, forecast the entire val horizon at once.
    No true observations from val are used — purely recursive.
    """
    y_tr  = _log(tr["total_cases"].values)
    y_val = val["total_cases"].values.astype(float)
    H     = len(y_val)

    fitted = _arima_fit(ARIMA(y_tr, order=order))
    # forecast(H) returns H steps ahead on the log1p scale (statsmodels
    # undoes its internal differencing automatically)
    fc_log = fitted.forecast(steps=H)
    preds  = _exp(fc_log)
    return preds, mean_absolute_error(y_val, preds)

def fit_arima(folds: list, city: str = "sj"):
    """Evaluate ARIMA across CV folds using recursive H-step forecasting."""
    p, q     = _PQ.get(city, (2, 1))
    first_tr = folds[0][1]
    print(f"\n[ARIMA] {city.upper()} — ADF on log1p(total_cases), fold 0:")
    d        = adf_d(first_tr["total_cases"])
    order    = (p, d, q)
    print(f"[ARIMA] {city.upper()} — order={order}, {len(folds)} folds")

    fold_maes, last_preds = [], None
    for fold_idx, tr, val in folds:
        preds, mae = _arima_forecast(tr, val, order)
        fold_maes.append(mae)
        last_preds = preds
        print(f"  fold {fold_idx}  train={len(tr)}  val={len(val)}  H={len(val)}  MAE={mae:.2f}")

    return fold_maes, last_preds


# ── ARIMAX — recursive multi-step ─────────────────────────────────────────────
def _arimax_forecast(tr: pd.DataFrame, val: pd.DataFrame, p: int, q: int, d: int):
    """
    Fit ARIMAX on the full training fold, forecast the entire val horizon.

    Climate exog for the future horizon is taken directly from the val rows
    (available in the test CSV).  Only total_cases is unknown — the exog is
    NOT recursively estimated; it is observed future covariate data.

    When d > 0:
      - Manually difference training target and training exog d times.
      - Stack the last d training exog rows with all val exog rows, then
        difference d times to get the future exog on the differenced scale.
      - Fit ARIMA with order=(p,0,q) (we own the differencing).
      - Forecast H steps; invert differencing, then invert log1p.
    """
    y_tr   = _log(tr["total_cases"].values)
    y_val  = val["total_cases"].values.astype(float)
    H      = len(y_val)

    X_tr  = _exog_matrix(tr)
    X_val = _exog_matrix(val)

    # Scale on training levels (before differencing)
    scaler  = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)

    if d > 0:
        y_model = np.diff(y_tr, n=d)
        X_model = _diff_cols(X_tr_s.copy(), d)
        # Need the last d training rows as context to difference the future exog
        context = np.vstack([X_tr_s[-d:], X_val_s])   # (d + H, F)
        X_fc    = _diff_cols(context, d)                # (H, F)
    else:
        y_model = y_tr
        X_model = X_tr_s
        X_fc    = X_val_s

    fitted   = _arima_fit(ARIMA(y_model, exog=X_model, order=(p, 0, q)))
    fc_diff  = fitted.forecast(steps=H, exog=X_fc)

    # Invert manual differencing back to log1p level
    if d == 1:
        # cumsum of differences starting from the last training level
        fc_log = y_tr[-1] + np.cumsum(fc_diff)
    elif d == 2:
        # two cumulative sums; first reconstruct the first-difference level
        last_diff = y_tr[-1] - y_tr[-2]
        first_diffs = last_diff + np.cumsum(fc_diff)
        fc_log = y_tr[-1] + np.cumsum(first_diffs)
    else:
        fc_log = fc_diff

    preds = _exp(fc_log)
    return preds, mean_absolute_error(y_val, preds)


def fit_arimax(folds: list, city: str = "sj"):
    """Evaluate ARIMAX across CV folds using recursive H-step forecasting."""
    p, q     = _PQ.get(city, (2, 1))
    first_tr = folds[0][1]
    print(f"\n[ARIMAX] {city.upper()} — ADF on log1p(total_cases), fold 0:")
    d        = adf_d(first_tr["total_cases"])
    print(f"[ARIMAX] {city.upper()} — d={d}, {len(folds)} folds")

    fold_maes, last_preds = [], None
    for fold_idx, tr, val in folds:
        preds, mae = _arimax_forecast(tr, val, p, q, d)
        fold_maes.append(mae)
        last_preds = preds
        print(f"  fold {fold_idx}  train={len(tr)}  val={len(val)}  H={len(val)}  MAE={mae:.2f}")

    return fold_maes, last_preds


# ── Random Forest ────────────────────────────────────────────────────────────
def _rf_fit_and_forecast(tr, val, params):
    """
    Fit RF on training fold, predict validation fold directly.

    Lag and rolling features for the validation rows were computed by
    preprocess._split_with_context, which prepended the training tail as
    context before calling engineer_features — so cases_lag1…52 in the
    first val row already reference true training observations, not NaN.
    No recursion is needed: the features are pre-computed and fixed.

    Target is log1p-transformed; predictions are clipped to >= 0 then expm1'd.
    """
    feat_cols = _rf_feature_cols(tr)
    tr_c  = tr[feat_cols + ["total_cases"]].dropna()
    val_c = val[feat_cols + ["total_cases"]].dropna()

    X_tr  = tr_c[feat_cols].values
    y_tr  = _log(tr_c["total_cases"].values)
    X_val = val_c[feat_cols].values
    y_val = val_c["total_cases"].values.astype(float)

    scaler = StandardScaler().fit(X_tr)
    rf = RandomForestRegressor(
        n_jobs=-1, random_state=RANDOM_STATE, **params
    ).fit(scaler.transform(X_tr), y_tr)

    preds = _exp(rf.predict(scaler.transform(X_val)))
    mae   = mean_absolute_error(y_val, preds)
    return preds, mae, (rf, scaler, feat_cols)


def _rf_tune(tr_raw: pd.DataFrame) -> dict:
    """
    Tune RF hyperparameters via inner year-based splits on the training fold.

    tr_raw is the *raw* (pre-engineered) training fold, passed through so
    that year_based_splits can re-impute and re-engineer each inner fold
    correctly — the same pipeline as the outer loop.

    Each inner fold uses recursive forecasting so the tuning objective
    matches the actual prediction task (multi-step horizon, no true obs).
    """
    best_params, best_mae = RF_PARAM_GRID[0], float("inf")

    inner_folds = list(year_based_splits(
        tr_raw,
        val_years=VAL_YEARS,
        min_train_years=MIN_TRAIN_YEARS,
    ))

    if not inner_folds:
        # Training fold too short for inner CV — return default params
        print("    RF tuning: inner fold too small, using default params")
        return RF_PARAM_GRID[0]

    for params in RF_PARAM_GRID:
        fold_maes = []
        for _, inner_tr, inner_val in inner_folds:
            if len(inner_tr) < 50:
                continue
            _, mae, _ = _rf_fit_and_forecast(inner_tr, inner_val, params)
            fold_maes.append(mae)

        if fold_maes:
            mean_mae = np.mean(fold_maes)
            if mean_mae < best_mae:
                best_mae, best_params = mean_mae, params

    print(f"    RF tuning → {best_params}  inner-CV MAE={best_mae:.2f}")
    return best_params


def fit_random_forest(folds: list, raw: pd.DataFrame):
    """
    Evaluate RF across CV folds:
      1. Tune hyperparameters via inner year_based_splits on the raw training
         fold (raw is passed so inner splits can re-impute/re-engineer correctly).
      2. Refit with best params on the full (engineered) training fold.
      3. Recursive multi-step forecast over the entire validation horizon.
    """
    fold_maes, last_preds, last_model = [], None, None

    # Build a map from fold_idx to the corresponding raw training years,
    # so _rf_tune receives the right raw slice for each outer fold.
    from preprocess import _get_years, VAL_YEARS, MIN_TRAIN_YEARS
    all_years = _get_years(raw)
    yr_col    = (raw.index.get_level_values("year")
                 if "year" in raw.index.names else raw["year"])

    raw_tr_slices = {}
    fold_counter  = 0
    for val_start in range(MIN_TRAIN_YEARS, len(all_years) - VAL_YEARS + 1, VAL_YEARS):
        tr_years = all_years[:val_start]
        raw_tr_slices[fold_counter] = raw[yr_col.isin(tr_years)]
        fold_counter += 1

    for fold_idx, tr, val in folds:
        print(f"\n  [RF] fold {fold_idx}  train={len(tr)}  val={len(val)}  H={len(val)}")
        tr_raw_fold = raw_tr_slices.get(fold_idx, raw)
        best_params             = _rf_tune(tr_raw_fold)
        preds, mae, model       = _rf_fit_and_forecast(tr, val, best_params)
        fold_maes.append(mae)
        last_preds, last_model  = preds, model
        print(f"    outer MAE={mae:.2f}")

    return fold_maes, last_preds, last_model