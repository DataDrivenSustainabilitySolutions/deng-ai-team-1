"""Tiny Transformer for DengAI total_cases forecasting.

Architecture
------------
Encoder-only causal transformer (decoder-style, with a causal mask so each
position can only attend to itself and earlier positions). This lets us use the
same model for both teacher-forced training and autoregressive rollout at
inference time without any structural change.

Each timestep token is the concatenation of:
  • climate features (standardised)       — dim = n_features
  • log1p(total_cases) lagged by 1 step   — dim = 1  (0-filled at t=0)
  • sin/cos of week-of-year               — dim = 2  (fixed seasonal signal)

That vector is projected to d_model via a linear input embedding, then
positional encodings are added and the sequence is fed through
num_layers transformer encoder layers with a causal mask.
The final token's representation is projected to a single scalar:
the predicted log1p(total_cases) for the *next* week.

At training time the model sees a sliding window of context_len weeks and
predicts the immediate next value — a standard teacher-forced 1-step-ahead
regression on the full sequence. Loss is smooth-L1 on log1p scale.

At forecast time (validation and test) the model starts from the last
context_len weeks of observed data and rolls out autoregressively, one step at
a time, feeding each predicted log1p value back as the lagged target feature
for the next step.

Model size (defaults):
  d_model=32, nhead=4, num_layers=2, dim_feedforward=128
  → ~35 k parameters — trains in a few seconds on CPU.

Example:
    python3 matthias/transformer_dengai.py
    python3 matthias/transformer_dengai.py --d-model 64 --num-layers 3 --epochs 300
    python3 matthias/transformer_dengai.py --context-len 26 --lr 3e-4
"""

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required. Install with: pip install torch"
    ) from exc


script_name = Path(__file__).stem
default_output_dir = Path(__file__).resolve().parent / "outputs" / script_name
default_main_dir   = Path("../oliver/outputs/main")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Tiny causal transformer for DengAI total_cases forecasting."
)
parser.add_argument("--train-csv",
    default=str(default_main_dir / "preprocessed_train.csv"))
parser.add_argument("--test-csv",
    default=str(default_main_dir / "preprocessed_test.csv"))
parser.add_argument("--submission-format-csv",
    default="../data/submission_format.csv")
parser.add_argument("--output-dir",
    default=str(default_output_dir))
parser.add_argument("--submission-output",      default="submission.csv")
parser.add_argument("--validation-output",      default="validation_scores.csv")
parser.add_argument("--validation-predictions-output",
    default="validation_predictions.csv")
# Model hyperparameters
parser.add_argument("--d-model",        type=int,   default=32,
    help="Transformer embedding dimension. Default: 32")
parser.add_argument("--nhead",          type=int,   default=4,
    help="Number of attention heads. Default: 4")
parser.add_argument("--num-layers",     type=int,   default=2,
    help="Number of transformer encoder layers. Default: 2")
parser.add_argument("--dim-feedforward",type=int,   default=128,
    help="Feed-forward hidden size inside each layer. Default: 128")
parser.add_argument("--dropout",        type=float, default=0.1,
    help="Dropout probability. Default: 0.1")
parser.add_argument("--context-len",    type=int,   default=52,
    help="Number of past weeks fed as context. Default: 52 (1 year)")
# Training hyperparameters
parser.add_argument("--epochs",         type=int,   default=200,
    help="Training epochs per city/fold. Default: 200")
parser.add_argument("--batch-size",     type=int,   default=32,
    help="Mini-batch size. Default: 32")
parser.add_argument("--lr",             type=float, default=1e-3,
    help="Adam learning rate. Default: 1e-3")
parser.add_argument("--weight-decay",   type=float, default=1e-4,
    help="Adam weight decay. Default: 1e-4")
parser.add_argument("--patience",       type=int,   default=30,
    help="Early-stopping patience (epochs without improvement). Default: 30")
parser.add_argument("--seed",           type=int,   default=42,
    help="Random seed. Default: 42")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
train_csv_path       = Path(args.train_csv)
test_csv_path        = Path(args.test_csv)
submission_format_path = Path(args.submission_format_csv)
output_dir           = Path(args.output_dir)
submission_path      = output_dir / args.submission_output
validation_scores_path       = output_dir / args.validation_output
validation_predictions_path  = output_dir / args.validation_predictions_output
merge_keys = ["city", "year", "weekofyear"]

torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Climate feature subsets — same as sarimax_seasonal.py
city_climate_exog = {
    "sj": [
        "station_avg_temp_c_lag_12", "station_avg_temp_c_lag_2",
        "station_avg_temp_c_lag_6", "station_avg_temp_c_rolling_12_mean",
        "station_max_temp_c_rolling_14_mean", "station_max_temp_c_rolling_5_mean",
        "station_max_temp_c_rolling_6_mean", "station_min_temp_c_rolling_14_mean",
        "reanalysis_avg_temp_k_lag_12", "reanalysis_max_air_temp_k_lag_12",
        "reanalysis_max_air_temp_k_rolling_12_mean",
        "reanalysis_min_air_temp_k_rolling_14_mean",
        "reanalysis_air_temp_k_rolling_8_mean",
        "reanalysis_dew_point_temp_k_lag_10", "reanalysis_dew_point_temp_k_lag_3",
        "reanalysis_dew_point_temp_k_lag_5", "reanalysis_dew_point_temp_k_lag_8",
        "reanalysis_dew_point_temp_k_rolling_6_mean",
        "reanalysis_dew_point_temp_k_rolling_8_mean",
        "reanalysis_dew_point_temp_k_rolling_10_mean",
        "reanalysis_specific_humidity_g_per_kg_lag_3",
        "reanalysis_specific_humidity_g_per_kg_lag_12",
        "reanalysis_specific_humidity_g_per_kg_rolling_12_mean",
        "reanalysis_relative_humidity_percent",
    ],
    "iq": [
        "reanalysis_min_air_temp_k_rolling_7_mean",
        "reanalysis_min_air_temp_k_rolling_5_mean",
        "reanalysis_min_air_temp_k_rolling_8_mean",
        "reanalysis_min_air_temp_k_lag_3", "reanalysis_min_air_temp_k_lag_2",
        "station_avg_temp_c_rolling_6_mean", "station_avg_temp_c_rolling_4_mean",
        "station_avg_temp_c_lag_2", "station_avg_temp_c_lag_4",
        "station_avg_temp_c_lag_6", "station_max_temp_c_rolling_6_mean",
        "station_max_temp_c_rolling_5_mean", "station_max_temp_c_lag_6",
        "station_min_temp_c_lag_6", "station_min_temp_c_lag_1",
        "ndvi_sw_lag_11", "ndvi_sw_lag_10", "ndvi_sw_lag_14",
        "ndvi_nw_lag_14", "ndvi_nw_lag_10", "ndvi_nw",
        "reanalysis_precip_amt_kg_per_m2",
        "precipitation_amt_mm_rolling_7_mean", "precipitation_amt_mm_lag_3",
        "precipitation_amt_mm_rolling_14_mean",
        "reanalysis_specific_humidity_g_per_kg_lag_5",
        "reanalysis_specific_humidity_g_per_kg_lag_2",
        "reanalysis_specific_humidity_g_per_kg_rolling_8_mean",
    ],
}


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------
class DengaiTransformer(nn.Module):
    """Causal encoder-only transformer for 1-step-ahead log1p(cases) regression.

    Input token dimension = n_climate_features + 1 (lagged log-cases) + 2 (sin/cos woy).
    Each token is linearly projected to d_model, positional encodings are added,
    then passed through `num_layers` self-attention layers with a causal mask.
    The last token's output is projected to a single scalar prediction.
    """

    def __init__(self, n_features: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        token_dim = n_features + 1 + 2      # climate + lagged_y + sin/cos(woy)
        self.input_proj = nn.Linear(token_dim, d_model)

        # Sinusoidal positional encoding (fixed, not learned — keeps the model tiny)
        self.register_buffer("pos_enc", self._make_pos_enc(512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,            # Pre-LN — more stable for small models
        )
        # norm_first=True disables the nested-tensor fast path; suppress the warning.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head     = nn.Linear(d_model, 1)

        # Weight init: small uniform for the projection and zero bias on head
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _make_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe  = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe.unsqueeze(0)          # (1, max_len, d_model)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position t only attends to 0..t."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, token_dim)  — the full token sequence
        Returns:
            (batch, seq_len)  — log1p predictions for each position's *next* step
        """
        seq_len = x.size(1)
        h = self.input_proj(x) + self.pos_enc[:, :seq_len, :]
        mask = self._causal_mask(seq_len, x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(h).squeeze(-1)     # (batch, seq_len)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def resolve_columns(df: pd.DataFrame, requested: list, city: str) -> list:
    present = [c for c in requested if c in df.columns]
    missing = [c for c in requested if c not in df.columns]
    if missing:
        print(f"  [{city}] skipping {len(missing)} missing columns: {', '.join(missing)}")
    if not present:
        raise ValueError(f"No exog columns available for city '{city}'.")
    return present


def standardise_fit(df: pd.DataFrame, cols: list):
    """Compute (mean, std) on df[cols]; return filled array + stats."""
    sub  = df[cols].copy().fillna(0.0)
    mean = sub.mean().values.astype(np.float32)
    std  = sub.std().values.astype(np.float32)
    std[std == 0] = 1.0
    return (sub.values.astype(np.float32) - mean) / std, mean, std


def standardise_apply(df: pd.DataFrame, cols: list,
                      mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    sub = df[cols].copy().fillna(0.0)
    return ((sub.values.astype(np.float32) - mean) / std)


def woy_sincos(weekofyear: np.ndarray) -> np.ndarray:
    """Return (N, 2) array of sin/cos encoding for week-of-year."""
    angle = 2.0 * math.pi * weekofyear.astype(float) / 52.0
    return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)


def build_token_matrix(climate: np.ndarray, log_cases: np.ndarray,
                       weekofyear: np.ndarray) -> np.ndarray:
    """Assemble (T, token_dim) input matrix.

    Token at position t = [climate[t], log_cases[t-1], sin(woy[t]), cos(woy[t])].
    log_cases[-1] (the lag at t=0) is 0.
    """
    T = len(climate)
    lagged = np.zeros((T, 1), dtype=np.float32)
    lagged[1:, 0] = log_cases[:-1]
    sc = woy_sincos(weekofyear)
    return np.concatenate([climate, lagged, sc], axis=1)   # (T, F+3)


def make_windows(tokens: np.ndarray, targets: np.ndarray,
                 context_len: int):
    """Sliding-window dataset: X (N, context_len, F), y (N,)."""
    N = len(tokens) - context_len
    if N <= 0:
        return None, None
    X = np.stack([tokens[i : i + context_len] for i in range(N)])
    y = targets[context_len:]           # target = value at position context_len+i
    return X.astype(np.float32), y.astype(np.float32)


def validation_years_for(city_df: pd.DataFrame) -> list:
    year_counts = city_df.groupby("year").size()
    full_years  = [y for y, n in year_counts.items() if n == 52]
    if not full_years:
        return []
    first_year = int(city_df["year"].min())
    first_full_after_start = min(y for y in full_years if y > first_year)
    return [y for y in full_years if y > first_full_after_start]


def finalize(raw: np.ndarray) -> list:
    return [int(round(max(0.0, float(v)))) for v in raw]


def seasonal_naive(train_df: pd.DataFrame, target_woy: pd.Series) -> np.ndarray:
    woy_mean    = train_df.groupby("weekofyear")["total_cases"].mean()
    global_mean = train_df["total_cases"].mean()
    return target_woy.map(woy_mean).fillna(global_mean).to_numpy()


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------
def train_model(model: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
    """Train model on (X, y) with Adam + early stopping. Returns best model."""
    model.to(DEVICE)
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optim   = torch.optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    loss_fn = nn.SmoothL1Loss()

    best_loss   = float("inf")
    best_state  = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve  = 0

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            # Forward: predict from each position, take the last token's pred
            # For the sliding-window formulation each window of length context_len
            # yields exactly one target (the next step), so we use the last position.
            pred = model(xb)[:, -1]     # (batch,)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        epoch_loss /= len(dataset)

        if epoch_loss < best_loss - 1e-6:
            best_loss  = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def autoregressive_forecast(model: nn.Module, context_tokens: np.ndarray,
                             future_climate: np.ndarray,
                             future_woy: np.ndarray) -> np.ndarray:
    """Roll out the model one step at a time over future_climate.

    Args:
        context_tokens: (context_len, token_dim)  — the seed window
        future_climate: (n_steps, n_climate_feat) — standardised climate features
        future_woy:     (n_steps,)                — week-of-year integers
    Returns:
        (n_steps,) array of expm1-transformed predictions (raw case counts)
    """
    model.eval()
    context = context_tokens.copy()     # (context_len, token_dim)
    n_steps = len(future_climate)
    preds_log = []
    sc_future = woy_sincos(future_woy)

    last_log_case = context[-1, -3]     # position of lagged_y in token: index -(1+2) = -3
    # token layout: [climate(F) | lagged_y(1) | sin(1) | cos(1)]
    # lagged_y is at index n_climate = token_dim - 3
    lag_idx = context.shape[1] - 3

    for t in range(n_steps):
        # Build the next token: use the most recently predicted value as the lag
        next_token = np.concatenate([
            future_climate[t],                              # climate features
            np.array([last_log_case], dtype=np.float32),   # log1p(cases) from previous step
            sc_future[t],                                   # sin/cos(woy)
        ]).astype(np.float32)
        # Extend context window (drop oldest, append new)
        context = np.concatenate([context[1:], next_token[np.newaxis, :]], axis=0)

        x = torch.from_numpy(context[np.newaxis]).to(DEVICE)   # (1, ctx, F)
        log_pred = model(x)[0, -1].item()
        last_log_case = log_pred
        preds_log.append(log_pred)

    return np.expm1(np.array(preds_log, dtype=float))


# ---------------------------------------------------------------------------
# Data import
# ---------------------------------------------------------------------------
train_data        = pd.read_csv(train_csv_path, parse_dates=["week_start_date"])
test_data         = pd.read_csv(test_csv_path,  parse_dates=["week_start_date"])
submission_format = pd.read_csv(submission_format_path)

for col in merge_keys + ["week_start_date"]:
    if col not in train_data.columns or col not in test_data.columns:
        raise ValueError(f"Column '{col}' missing from the preprocessed CSVs.")
if "total_cases" not in train_data.columns:
    raise ValueError("preprocessed train CSV must contain total_cases.")


# ---------------------------------------------------------------------------
# Expanding full-year validation
# ---------------------------------------------------------------------------
validation_score_records      = []
validation_prediction_records = []
all_actuals, all_predictions  = [], []

for city, city_train in train_data.groupby("city", sort=False):
    city_train = city_train.sort_values("week_start_date").reset_index(drop=True)
    cols       = resolve_columns(city_train, city_climate_exog[city], city)
    years      = validation_years_for(city_train)
    print(f"\n[{city}] validation years: {years if years else 'none'}")
    n_feat = len(cols)
    token_dim = n_feat + 3              # climate + lagged_y + sin + cos

    for val_year in years:
        fold_train = city_train[city_train["year"] < val_year].reset_index(drop=True)
        fold_val   = city_train[city_train["year"] == val_year].reset_index(drop=True)

        # Standardise climate features using train fold statistics
        climate_tr, feat_mean, feat_std = standardise_fit(fold_train, cols)
        climate_vl = standardise_apply(fold_val, cols, feat_mean, feat_std)

        log_cases_tr = np.log1p(fold_train["total_cases"].values.astype(np.float32))
        log_cases_vl = np.log1p(fold_val["total_cases"].values.astype(np.float32))
        woy_tr = fold_train["weekofyear"].values
        woy_vl = fold_val["weekofyear"].values

        # Build token matrices
        tokens_tr = build_token_matrix(climate_tr, log_cases_tr, woy_tr)
        tokens_vl = build_token_matrix(climate_vl, log_cases_vl, woy_vl)

        # Sliding-window training dataset (teacher-forced)
        X_tr, y_tr = make_windows(tokens_tr, log_cases_tr, args.context_len)
        if X_tr is None or len(X_tr) < 4:
            print(f"  [{city}] {val_year}: not enough data to train, using seasonal-naive.")
            raw = seasonal_naive(fold_train, fold_val["weekofyear"])
            preds = finalize(raw)
            used = "seasonal_naive_fallback"
        else:
            model = DengaiTransformer(
                n_features=n_feat, d_model=args.d_model, nhead=args.nhead,
                num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
            )
            model = train_model(model, X_tr, y_tr)

            # Seed context: last context_len rows of training tokens
            seed_ctx = tokens_tr[-args.context_len:]
            raw = autoregressive_forecast(
                model, seed_ctx, climate_vl, woy_vl
            )
            preds = finalize(raw)
            used = "transformer"

        fold_mae = mean_absolute_error(fold_val["total_cases"], preds)
        all_actuals.extend(fold_val["total_cases"].tolist())
        all_predictions.extend(preds)

        for row, pred, r in zip(fold_val.itertuples(index=False), preds, raw):
            validation_prediction_records.append({
                "city": row.city, "year": row.year, "weekofyear": row.weekofyear,
                "week_start_date": row.week_start_date,
                "validation_year": val_year,
                "actual_total_cases": row.total_cases,
                "predicted_total_cases": pred,
                "predicted_total_cases_raw": max(0.0, float(r)),
                "model": used,
            })
        validation_score_records.append({
            "city": city, "validation_year": val_year,
            "train_rows": len(fold_train), "validation_rows": len(fold_val),
            "model": used, "mae": fold_mae,
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


# ---------------------------------------------------------------------------
# Final models on full training data → submission
# ---------------------------------------------------------------------------
submission_predictions = {}

for city, city_train in train_data.groupby("city", sort=False):
    city_train = city_train.sort_values("week_start_date").reset_index(drop=True)
    city_test  = (
        test_data[test_data["city"] == city]
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )
    cols   = resolve_columns(city_train, city_climate_exog[city], city)
    n_feat = len(cols)

    climate_tr, feat_mean, feat_std = standardise_fit(city_train, cols)
    climate_te = standardise_apply(city_test, cols, feat_mean, feat_std)
    log_cases_tr = np.log1p(city_train["total_cases"].values.astype(np.float32))
    woy_tr = city_train["weekofyear"].values
    woy_te = city_test["weekofyear"].values

    tokens_tr = build_token_matrix(climate_tr, log_cases_tr, woy_tr)
    X_tr, y_tr = make_windows(tokens_tr, log_cases_tr, args.context_len)

    print(f"\n[{city}] training final model on {len(city_train)} rows …")
    model = DengaiTransformer(
        n_features=n_feat, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = train_model(model, X_tr, y_tr)

    seed_ctx = tokens_tr[-args.context_len:]
    raw  = autoregressive_forecast(model, seed_ctx, climate_te, woy_te)
    preds = finalize(raw)

    for row, pred in zip(city_test.itertuples(index=False), preds):
        submission_predictions[(row.city, row.year, row.weekofyear)] = pred


# ---------------------------------------------------------------------------
# Output export
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nTransformer pipeline complete.")
n_params = sum(
    sum(p.numel() for p in DengaiTransformer(
        n_features=len(resolve_columns(
            train_data[train_data["city"] == city], city_climate_exog[city], city
        )),
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=0.0,
    ).parameters())
    for city in train_data["city"].unique()
)
print(f"Model: d_model={args.d_model}, nhead={args.nhead}, "
      f"layers={args.num_layers}, ff={args.dim_feedforward}")
print(f"Approx total parameters across cities: {n_params:,}")
print(f"Context length: {args.context_len} weeks | Epochs: {args.epochs}")
folds_df = validation_scores[validation_scores["city"] != "all"]
print(f"Validation folds: {len(folds_df)}")
overall_row = validation_scores.loc[validation_scores["city"] == "all", "mae"]
if not overall_row.empty:
    print(f"Overall validation MAE: {overall_row.iloc[0]:.4f}")
print(f"Saved validation scores:      {validation_scores_path}")
print(f"Saved validation predictions: {validation_predictions_path}")
print(f"Saved submission:             {submission_path}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("\n[visualisation] matplotlib not found – skipping plots.")
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
            ax.axvspan(date_series[mask].iloc[0], date_series[mask].iloc[-1],
                       color="#888888" if idx % 2 == 0 else "#444444",
                       alpha=BAND_ALPHA, linewidth=0)

    def _style(ax, title):
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylabel("Total cases", fontsize=9)
        ax.set_xlabel("Week start date", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    cities = (sorted(validation_predictions["city"].unique())
              if not validation_predictions.empty
              else sorted(train_data["city"].unique()))

    # Figure 1: Validation actuals vs predictions
    if not validation_predictions.empty:
        fig1, axes1 = plt.subplots(len(cities), 1,
                                   figsize=(16, 5 * len(cities)), squeeze=False)
        fig1.suptitle("Transformer – Validation: Actual vs Predicted dengue cases",
                      fontsize=14, fontweight="bold", y=1.01)
        for ax, city in zip(axes1[:, 0], cities):
            vp = (validation_predictions[validation_predictions["city"] == city]
                  .sort_values("week_start_date").reset_index(drop=True))
            if vp.empty:
                ax.set_visible(False); continue
            fold_years = sorted(vp["validation_year"].unique())
            _fold_bands(ax, fold_years, vp["week_start_date"])
            ax.plot(vp["week_start_date"], vp["actual_total_cases"],
                    color=COLOR_ACTUAL, linewidth=1.5, label="Actual", zorder=3)
            ax.plot(vp["week_start_date"], vp["predicted_total_cases"],
                    color=COLOR_PREDICTED, linewidth=1.5, linestyle="--",
                    alpha=0.88, label="Predicted", zorder=4)
            y_ann = vp["actual_total_cases"].max() * 0.93
            for year in fold_years:
                fd = vp[vp["validation_year"] == year]
                if fd.empty: continue
                mae_v = mean_absolute_error(fd["actual_total_cases"],
                                            fd["predicted_total_cases"])
                x_mid = fd["week_start_date"].iloc[len(fd) // 2]
                ax.annotate(f"{year}\nMAE {mae_v:.1f}", xy=(x_mid, y_ann),
                            ha="center", va="top", fontsize=7.5, color="#555555",
                            annotation_clip=True)
            cs = validation_scores[(validation_scores["city"] == city)
                                   & (validation_scores["validation_year"] != "all")]
            avg_mae = cs["mae"].astype(float).mean() if not cs.empty else float("nan")
            _style(ax, f"{CITY_LABELS.get(city, city.upper())}  –  "
                       f"mean validation MAE: {avg_mae:.2f}")
        fig1.tight_layout()
        p1 = output_dir / "validation_actual_vs_predicted.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"Saved validation plot:        {p1}")

    # Figure 2: Training history + test forecast
    fig2, axes2 = plt.subplots(len(cities), 1,
                               figsize=(16, 5 * len(cities)), squeeze=False)
    fig2.suptitle("Transformer – Training history & test-period forecast",
                  fontsize=14, fontweight="bold", y=1.01)
    for ax, city in zip(axes2[:, 0], cities):
        tr = (train_data[train_data["city"] == city]
              .sort_values("week_start_date").reset_index(drop=True))
        city_test_rows = test_data[test_data["city"] == city].sort_values("week_start_date")
        fc = (submission[submission["city"] == city]
              .merge(city_test_rows[["city", "year", "weekofyear", "week_start_date"]],
                     on=["city", "year", "weekofyear"], how="left")
              .sort_values("week_start_date").reset_index(drop=True))
        ax.plot(tr["week_start_date"], tr["total_cases"],
                color=COLOR_ACTUAL, linewidth=1.2, label="Training actuals", zorder=3)
        if not fc.empty:
            bridge_x = [tr["week_start_date"].iloc[-1], fc["week_start_date"].iloc[0]]
            bridge_y = [tr["total_cases"].iloc[-1],      fc["total_cases"].iloc[0]]
            ax.plot(bridge_x, bridge_y, color=COLOR_FORECAST,
                    linewidth=1.5, linestyle="--", zorder=4)
            ax.plot(fc["week_start_date"], fc["total_cases"],
                    color=COLOR_FORECAST, linewidth=1.8, linestyle="--",
                    label="Test forecast", zorder=4)
            ax.axvspan(fc["week_start_date"].iloc[0], fc["week_start_date"].iloc[-1],
                       color=COLOR_FORECAST, alpha=0.07, linewidth=0)
            ax.axvline(fc["week_start_date"].iloc[0],
                       color=COLOR_FORECAST, linewidth=0.8, linestyle=":")
        _style(ax, f"{CITY_LABELS.get(city, city.upper())}  –  training history & forecast")
    fig2.tight_layout()
    p2 = output_dir / "forecast_vs_history.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved forecast plot:          {p2}")