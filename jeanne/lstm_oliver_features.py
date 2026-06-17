"""Train a PyTorch LSTM with Oliver's feature roots, without lag or rolling features.

Example:
    python3 jeanne/lstm_oliver_features.py --city all --epochs 80
"""

import argparse
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "lstm_outputs"
MERGE_KEYS = ["city", "year", "weekofyear"]
WEEKS_PER_YEAR = 52
IDENTIFIER_COLUMNS = {
    "city",
    "year",
    "weekofyear",
    "week_start_date",
    "split",
    "total_cases",
}
SEASONALITY_FEATURE_COLUMNS = ["weekofyear_sin", "weekofyear_cos"]


SJ_WEATHER_ROOTS = [
    "reanalysis_air_temp_k",
    "reanalysis_avg_temp_k",
    "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k",
    "reanalysis_min_air_temp_k",
    "reanalysis_relative_humidity_percent",
    "reanalysis_specific_humidity_g_per_kg",
    "station_avg_temp_c",
    "station_max_temp_c",
    "station_min_temp_c",
]
IQ_WEATHER_ROOTS = [
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "reanalysis_min_air_temp_k",
    "station_min_temp_c",
    "station_avg_temp_c",
    "station_max_temp_c",
    "ndvi_ne",
    "ndvi_nw",
    "ndvi_se",
    "ndvi_sw",
    "precipitation_amt_mm",
    "reanalysis_sat_precip_amt_mm",
    "reanalysis_precip_amt_kg_per_m2",
]

CITY_LABELS = {"sj": "San Juan", "iq": "Iquitos"}
COLOR_ACTUAL = "#1a1a2e"
COLOR_PREDICTED = "#e84545"
COLOR_FORECAST = "#2d6a4f"
BAND_ALPHA = 0.06


@dataclass
class FeaturePreprocessor:
    medians: pd.Series
    means: pd.Series
    stds: pd.Series

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        filled = features.fillna(self.medians).fillna(0.0)
        scaled = (filled - self.means) / self.stds
        return scaled.to_numpy(dtype=np.float32)


@dataclass
class TrainingResult:
    model: nn.Module
    best_epoch: int
    best_val_mae: float | None
    history: list[dict[str, float | int | str]]


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(16, hidden_size // 2), 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        return self.head(hidden[-1])


def select_oliver_feature_columns(data_columns: pd.Index) -> dict[str, list[str]]:
    city_feature_columns = {}

    for city in ["sj", "iq"]:
        weather_feature_columns = []

        if city == "sj":
            for feature in SJ_WEATHER_ROOTS:
                if feature in data_columns:
                    weather_feature_columns.append(feature)

        if city == "iq":
            for feature in IQ_WEATHER_ROOTS:
                if feature in data_columns:
                    weather_feature_columns.append(feature)

        selected_columns = SEASONALITY_FEATURE_COLUMNS + weather_feature_columns
        city_feature_columns[city] = [
            column
            for column in dict.fromkeys(selected_columns)
            if column in data_columns
        ]

    return city_feature_columns


@lru_cache(maxsize=1)
def build_oliver_feature_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    train_features = pd.read_csv(DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    test_features = pd.read_csv(DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    train_labels = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")

    if list(train_features.columns) != list(test_features.columns):
        raise ValueError("Train and test feature CSVs must have the same columns.")

    train_data = train_features.merge(train_labels, on=MERGE_KEYS, how="left", validate="one_to_one")
    if train_data["total_cases"].isna().any():
        raise ValueError("Merged training data contains missing total_cases values.")

    train_data["split"] = "train"
    test_data = test_features.copy()
    test_data["split"] = "test"
    train_data["_row_order"] = range(len(train_data))
    test_data["_row_order"] = range(len(test_data))

    data = pd.concat([train_data, test_data], ignore_index=True, sort=False)
    numeric_feature_columns = [
        column
        for column in train_features.columns
        if column not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(train_features[column])
    ]

    interpolated_city_data = []
    for _, city_data in data.groupby("city", sort=False):
        city_data = city_data.sort_values("week_start_date").copy()
        city_data[numeric_feature_columns] = city_data[numeric_feature_columns].interpolate(
            method="linear",
            limit_direction="both",
        )
        interpolated_city_data.append(city_data)

    data = pd.concat(interpolated_city_data, ignore_index=True)
    engineered_feature_data = {}

    week_angle = 2 * math.pi * data["weekofyear"] / 52.0
    engineered_feature_data["weekofyear_sin"] = week_angle.map(math.sin)
    engineered_feature_data["weekofyear_cos"] = week_angle.map(math.cos)

    data = pd.concat([data, pd.DataFrame(engineered_feature_data, index=data.index)], axis=1)
    city_feature_columns = select_oliver_feature_columns(data.columns)

    return data, city_feature_columns


def load_engineered_city_data(city_id: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data, city_feature_columns = build_oliver_feature_data()
    if city_id not in city_feature_columns:
        raise ValueError(f"Unknown city_id: {city_id}")

    feature_columns = city_feature_columns[city_id]
    city_train = (
        data[(data["split"] == "train") & (data["city"] == city_id)]
        .sort_values("week_start_date")
        .copy()
    )
    city_test = (
        data[(data["split"] == "test") & (data["city"] == city_id)]
        .sort_values("week_start_date")
        .copy()
    )

    X_train = city_train.set_index(["year", "weekofyear"])[feature_columns]
    y_train = city_train.set_index(["year", "weekofyear"])["total_cases"]
    X_test = city_test.set_index(["year", "weekofyear"])[feature_columns]
    test_dates = pd.to_datetime(city_test.set_index(["year", "weekofyear"])["week_start_date"])
    test_dates.name = "week_start_date"

    return X_train, y_train, X_test, test_dates


def load_data(city_id: str) -> tuple[pd.DataFrame, pd.Series]:
    train_features, train_labels, _, _ = load_engineered_city_data(city_id)
    return train_features.copy(), train_labels.copy()


def load_test_data(city_id: str) -> tuple[pd.DataFrame, pd.Series]:
    _, _, test_features, dates = load_engineered_city_data(city_id)
    return test_features.copy(), dates.copy()


def load_week_start_dates(city_id: str, source: str = "train") -> pd.Series:
    path = DATA_DIR / "dengue_features_train.csv" if source == "train" else DATA_DIR / "dengue_features_test.csv"
    features = pd.read_csv(path, parse_dates=["week_start_date"], index_col=[0, 1, 2])
    dates = pd.to_datetime(features.loc[city_id].sort_index()["week_start_date"])
    dates.name = "week_start_date"
    return dates


def fit_feature_preprocessor(features: pd.DataFrame) -> FeaturePreprocessor:
    medians = features.median(numeric_only=True).fillna(0.0)
    filled = features.fillna(medians).fillna(0.0)
    means = filled.mean()
    stds = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return FeaturePreprocessor(medians=medians, means=means, stds=stds)


def build_sequences(
    feature_values: np.ndarray,
    index: pd.MultiIndex,
    sequence_length: int,
    target_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, pd.MultiIndex]:
    if sequence_length < 1:
        raise ValueError("--sequence-length must be at least 1.")
    if len(feature_values) < sequence_length:
        raise ValueError("Not enough rows to build LSTM sequences.")

    sequences = [
        feature_values[position - sequence_length + 1 : position + 1]
        for position in range(sequence_length - 1, len(feature_values))
    ]
    sequence_index = index[sequence_length - 1 :]

    targets = None
    if target_values is not None:
        targets = np.asarray(target_values[sequence_length - 1 :], dtype=np.float32)

    return np.stack(sequences).astype(np.float32), targets, sequence_index


def get_last_full_year(index: pd.MultiIndex) -> int:
    years = index.get_level_values("year")
    year_counts = pd.Series(1, index=years).groupby(level=0).sum()
    full_years = [
        int(year)
        for year, row_count in year_counts.items()
        if row_count == WEEKS_PER_YEAR
    ]
    if not full_years:
        raise ValueError("Need at least one complete 52-week year.")
    return max(full_years)


def transform_target(values: np.ndarray, target_transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if target_transform == "log1p":
        return np.log1p(values)
    if target_transform == "none":
        return values
    raise ValueError(f"Unsupported target transform: {target_transform}")


def inverse_transform_prediction(values: np.ndarray, target_transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if target_transform == "log1p":
        return np.expm1(values)
    if target_transform == "none":
        return values
    raise ValueError(f"Unsupported target transform: {target_transform}")


def make_case_count_predictions(
    raw_predictions: np.ndarray,
    index: pd.Index | None,
    target_transform: str,
) -> pd.Series:
    predictions = pd.Series(
        inverse_transform_prediction(raw_predictions, target_transform),
        index=index,
        name="total_cases",
    )
    return predictions.clip(lower=0).round().astype(int)


def set_random_seed(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def predict_raw(
    model: nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(sequences))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device)
            batch_predictions = model(batch_features).detach().cpu().numpy().reshape(-1)
            predictions.append(batch_predictions)

    return np.concatenate(predictions).astype(np.float32)


def train_lstm_model(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    val_sequences: np.ndarray | None,
    val_targets: np.ndarray | None,
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    target_transform: str,
    random_state: int,
    device: torch.device,
    city: str,
    phase: str,
) -> TrainingResult:
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()
    y_train = transform_target(train_targets, target_transform).reshape(-1, 1)
    train_dataset = TensorDataset(
        torch.from_numpy(train_sequences),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )

    best_state = clone_model_state(model)
    best_epoch = 0
    best_val_mae = math.inf if val_sequences is not None and val_targets is not None else None
    epochs_without_improvement = 0
    history: list[dict[str, float | int | str]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            loss = criterion(prediction, batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(batch_losses))
        history_record: dict[str, float | int | str] = {
            "city": city,
            "phase": phase,
            "epoch": epoch,
            "train_loss": train_loss,
        }

        if val_sequences is not None and val_targets is not None:
            raw_val_predictions = predict_raw(model, val_sequences, device, batch_size)
            val_predictions = make_case_count_predictions(
                raw_val_predictions,
                index=None,
                target_transform=target_transform,
            )
            val_mae = float(mean_absolute_error(val_targets, val_predictions.to_numpy()))
            history_record["val_mae"] = val_mae

            if val_mae < float(best_val_mae):
                best_val_mae = val_mae
                best_state = clone_model_state(model)
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                history.append(history_record)
                break
        else:
            best_state = clone_model_state(model)
            best_epoch = epoch

        history.append(history_record)

    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_val_mae=None if best_val_mae is None else float(best_val_mae),
        history=history,
    )


def fold_bands(ax, fold_years, date_series: pd.Series) -> None:
    dates = pd.Series(pd.to_datetime(date_series)).reset_index(drop=True)
    for idx, year in enumerate(fold_years):
        mask = dates.dt.year == int(year)
        if not mask.any():
            continue
        ax.axvspan(
            dates[mask].iloc[0],
            dates[mask].iloc[-1],
            color="#888888" if idx % 2 == 0 else "#444444",
            alpha=BAND_ALPHA,
            linewidth=0,
        )


def style_time_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylabel("Total cases", fontsize=9)
    ax.set_xlabel("Week start date", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_holdout_prediction(
    city_id: str,
    y: pd.Series,
    prediction: pd.Series,
    holdout_year: int,
    final_mae: float,
    output_dir: Path,
) -> Path:
    dates = load_week_start_dates(city_id).reindex(y.index)
    holdout_data = pd.DataFrame(
        {
            "date": dates.reindex(prediction.index),
            "total_cases": y.reindex(prediction.index),
            "prediction": prediction,
        }
    ).sort_values("date")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{city_id}_lstm_holdout_prediction.png"

    fig, ax = plt.subplots(figsize=(14, 6))
    fold_bands(ax, [holdout_year], holdout_data["date"])
    ax.plot(
        holdout_data["date"],
        holdout_data["total_cases"],
        color=COLOR_ACTUAL,
        linewidth=1.5,
        label="Actual",
        zorder=3,
    )
    ax.plot(
        holdout_data["date"],
        holdout_data["prediction"],
        color=COLOR_PREDICTED,
        linewidth=1.5,
        linestyle="--",
        alpha=0.88,
        label="Predicted",
        zorder=4,
    )
    y_ann = max(float(holdout_data["total_cases"].max()), float(holdout_data["prediction"].max()), 1.0) * 0.93
    x_mid = holdout_data["date"].iloc[len(holdout_data) // 2]
    ax.annotate(
        f"{holdout_year}\nMAE {final_mae:.1f}",
        xy=(x_mid, y_ann),
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
        annotation_clip=True,
    )
    style_time_axis(
        ax,
        f"{CITY_LABELS.get(city_id, city_id.upper())} - LSTM holdout MAE: {final_mae:.2f}",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_submission_prediction(
    city_id: str,
    y_train: pd.Series,
    prediction: pd.Series,
    test_dates: pd.Series,
    prediction_weeks: int,
    output_dir: Path,
) -> Path:
    train_dates = load_week_start_dates(city_id, source="train")
    first_test_date = test_dates.min()
    train_mask = train_dates < first_test_date
    train_plot = pd.DataFrame(
        {
            "date": train_dates[train_mask],
            "total_cases": y_train.loc[train_mask],
        }
    ).sort_values("date")
    test_plot = pd.DataFrame(
        {
            "date": test_dates.reindex(prediction.index.droplevel("city")),
            "prediction": prediction.to_numpy(),
        }
    ).sort_values("date")

    output_dir.mkdir(parents=True, exist_ok=True)
    horizon_label = "full test set" if prediction_weeks == 0 else f"next {prediction_weeks} weeks"
    plot_suffix = "full_test" if prediction_weeks == 0 else f"{prediction_weeks}_weeks"
    output_path = output_dir / f"{city_id}_lstm_{plot_suffix}_prediction.png"

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        train_plot["date"],
        train_plot["total_cases"],
        color=COLOR_ACTUAL,
        linewidth=1.2,
        label="Training actuals",
        zorder=3,
    )
    if not test_plot.empty:
        bridge_x = [train_plot["date"].iloc[-1], test_plot["date"].iloc[0]]
        bridge_y = [train_plot["total_cases"].iloc[-1], test_plot["prediction"].iloc[0]]
        ax.plot(bridge_x, bridge_y, color=COLOR_FORECAST, linewidth=1.5, linestyle="--", zorder=4)
        ax.plot(
            test_plot["date"],
            test_plot["prediction"],
            color=COLOR_FORECAST,
            linewidth=1.8,
            linestyle="--",
            label=f"LSTM {horizon_label} prediction",
            zorder=4,
        )
        ax.axvspan(
            test_plot["date"].iloc[0],
            test_plot["date"].iloc[-1],
            color=COLOR_FORECAST,
            alpha=0.07,
            linewidth=0,
        )
        ax.axvline(test_plot["date"].iloc[0], color=COLOR_FORECAST, linewidth=0.8, linestyle=":")
    style_time_axis(
        ax,
        f"{CITY_LABELS.get(city_id, city_id.upper())} - LSTM training history & {horizon_label} forecast",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def run_holdout_for_city(
    city_id: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, int, Path]:
    X, y = load_data(city_id)
    years = X.index.get_level_values("year")
    holdout_year = get_last_full_year(X.index)
    row_train_mask = years < holdout_year

    if not row_train_mask.any():
        raise ValueError(f"{city_id}: Need at least two years for a holdout split.")

    preprocessor = fit_feature_preprocessor(X[row_train_mask])
    scaled_features = preprocessor.transform(X)
    sequences, sequence_targets, sequence_index = build_sequences(
        scaled_features,
        X.index,
        args.sequence_length,
        y.to_numpy(),
    )
    sequence_years = sequence_index.get_level_values("year")
    sequence_train_mask = sequence_years < holdout_year
    sequence_val_mask = sequence_years == holdout_year

    if not sequence_train_mask.any() or not sequence_val_mask.any():
        raise ValueError(f"{city_id}: Sequence split produced empty train or validation data.")

    result = train_lstm_model(
        sequences[sequence_train_mask],
        sequence_targets[sequence_train_mask],
        sequences[sequence_val_mask],
        sequence_targets[sequence_val_mask],
        input_size=sequences.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        target_transform=args.target_transform,
        random_state=args.random_state,
        device=device,
        city=city_id,
        phase="holdout",
    )

    val_index = sequence_index[sequence_val_mask]
    raw_predictions = predict_raw(result.model, sequences[sequence_val_mask], device, args.batch_size)
    predictions = make_case_count_predictions(raw_predictions, val_index, args.target_transform)
    actual = pd.Series(sequence_targets[sequence_val_mask], index=val_index, name="actual_total_cases")
    final_mae = float(mean_absolute_error(actual, predictions))
    clipped_raw_predictions = np.maximum(
        0.0,
        inverse_transform_prediction(raw_predictions, args.target_transform),
    )
    dates = load_week_start_dates(city_id).reindex(val_index)
    validation_predictions = pd.DataFrame(
        {
            "city": city_id,
            "year": val_index.get_level_values("year"),
            "weekofyear": val_index.get_level_values("weekofyear"),
            "week_start_date": dates.to_numpy(),
            "validation_year": holdout_year,
            "actual_total_cases": actual.to_numpy(dtype=np.float32),
            "predicted_total_cases": predictions.to_numpy(dtype=int),
            "predicted_total_cases_raw": clipped_raw_predictions,
        }
    )
    score_record = {
        "city": city_id,
        "validation_year": holdout_year,
        "train_sequences": int(sequence_train_mask.sum()),
        "validation_sequences": int(sequence_val_mask.sum()),
        "feature_count": int(X.shape[1]),
        "sequence_length": args.sequence_length,
        "best_epoch": result.best_epoch,
        "mae": final_mae,
    }
    history = pd.DataFrame(result.history)
    plot_path = plot_holdout_prediction(
        city_id,
        y,
        predictions,
        holdout_year,
        final_mae,
        args.output_dir / "plots",
    )

    return score_record, validation_predictions, history, result.best_epoch, plot_path


def run_submission_for_city(
    city_id: str,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
) -> tuple[pd.Series, pd.DataFrame, Path]:
    X_train, y_train = load_data(city_id)
    X_test, test_dates = load_test_data(city_id)

    preprocessor = fit_feature_preprocessor(X_train)
    scaled_train = preprocessor.transform(X_train)
    train_sequences, train_targets, _ = build_sequences(
        scaled_train,
        X_train.index,
        args.sequence_length,
        y_train.to_numpy(),
    )
    result = train_lstm_model(
        train_sequences,
        train_targets,
        None,
        None,
        input_size=train_sequences.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        target_transform=args.target_transform,
        random_state=args.random_state,
        device=device,
        city=city_id,
        phase="prediction",
    )

    combined_features = pd.concat([X_train, X_test], axis=0)
    combined_sources = np.array(["train"] * len(X_train) + ["test"] * len(X_test))
    scaled_combined = preprocessor.transform(combined_features)
    combined_sequences, _, combined_sequence_index = build_sequences(
        scaled_combined,
        combined_features.index,
        args.sequence_length,
    )
    sequence_sources = combined_sources[args.sequence_length - 1 :]
    test_mask = sequence_sources == "test"
    test_positions = np.flatnonzero(test_mask)
    if args.prediction_weeks:
        test_positions = test_positions[: args.prediction_weeks]
    test_sequences = combined_sequences[test_positions]
    test_index = combined_sequence_index[test_positions]

    expected_test_rows = len(X_test) if args.prediction_weeks == 0 else min(args.prediction_weeks, len(X_test))
    if len(test_index) != expected_test_rows:
        raise ValueError(
            f"{city_id}: Expected {expected_test_rows} test sequences but built {len(test_index)}."
        )

    raw_predictions = predict_raw(result.model, test_sequences, device, args.batch_size)
    predictions = make_case_count_predictions(raw_predictions, test_index, args.target_transform)
    predictions.index = pd.MultiIndex.from_arrays(
        [
            [city_id] * len(predictions),
            predictions.index.get_level_values("year"),
            predictions.index.get_level_values("weekofyear"),
        ],
        names=["city", "year", "weekofyear"],
    )
    history = pd.DataFrame(result.history)
    plot_path = plot_submission_prediction(
        city_id,
        y_train,
        predictions,
        test_dates,
        args.prediction_weeks,
        args.output_dir / "plots",
    )

    return predictions, history, plot_path


def save_prediction_rows(predictions: list[pd.Series], output_path: Path) -> Path:
    prediction_frame = pd.concat(predictions).reset_index()[["city", "year", "weekofyear", "total_cases"]]
    prediction_frame["total_cases"] = prediction_frame["total_cases"].clip(lower=0).round().astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_path, index=False)
    return output_path


def save_submission_predictions(predictions: list[pd.Series], output_path: Path) -> Path:
    submission_format = pd.read_csv(DATA_DIR / "submission_format.csv")
    submission = submission_format.copy()
    prediction_frame = pd.concat(predictions).reset_index()[["city", "year", "weekofyear", "total_cases"]]

    submission = submission.merge(
        prediction_frame,
        on=MERGE_KEYS,
        how="left",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    submission["total_cases"] = submission["total_cases_prediction"]
    submission = submission.drop(columns=["total_cases_prediction"])

    if submission["total_cases"].isna().any():
        fallback_path = BASE_DIR / "submission_prediction.csv"
        if fallback_path.exists():
            fallback = pd.read_csv(fallback_path)
            expected_columns = ["city", "year", "weekofyear", "total_cases"]
            if list(fallback.columns) == expected_columns and fallback[MERGE_KEYS].equals(submission_format[MERGE_KEYS]):
                missing_mask = submission["total_cases"].isna()
                submission.loc[missing_mask, "total_cases"] = fallback.loc[missing_mask, "total_cases"]

    if submission["total_cases"].isna().any():
        missing_cities = sorted(submission.loc[submission["total_cases"].isna(), "city"].unique())
        raise ValueError(
            "Submission is incomplete. Train all cities or provide fallback predictions for: "
            + ", ".join(missing_cities)
        )

    submission = submission[["city", "year", "weekofyear", "total_cases"]]
    submission["total_cases"] = submission["total_cases"].clip(lower=0).round().astype(int)

    if list(submission.columns) != ["city", "year", "weekofyear", "total_cases"]:
        raise ValueError("Submission columns must be city, year, weekofyear, total_cases.")
    if len(submission) != len(submission_format):
        raise ValueError("Submission row count must match the submission format.")
    if not submission[MERGE_KEYS].equals(submission_format[MERGE_KEYS]):
        raise ValueError("Submission row order must match the submission format.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def build_forecast_plot_frame(city_id: str, prediction: pd.Series) -> pd.DataFrame:
    _, test_dates = load_test_data(city_id)
    prediction_index = prediction.index.droplevel("city")
    return pd.DataFrame(
        {
            "city": city_id,
            "year": prediction_index.get_level_values("year"),
            "weekofyear": prediction_index.get_level_values("weekofyear"),
            "week_start_date": test_dates.reindex(prediction_index).to_numpy(),
            "total_cases": prediction.to_numpy(),
        }
    ).sort_values("week_start_date")


def plot_sarimax_style_visualizations(
    validation_predictions: pd.DataFrame,
    validation_scores: pd.DataFrame,
    test_predictions: list[pd.Series],
    output_dir: Path,
    prediction_weeks: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    forecast_by_city = {
        prediction.index.get_level_values("city")[0]: prediction
        for prediction in test_predictions
        if len(prediction) > 0
    }
    cities = sorted(forecast_by_city)
    if not cities and not validation_predictions.empty:
        cities = sorted(validation_predictions["city"].unique())

    if not validation_predictions.empty:
        fig1, axes1 = plt.subplots(
            len(cities),
            1,
            figsize=(16, 5 * len(cities)),
            squeeze=False,
        )
        fig1.suptitle(
            "LSTM - Validation: Actual vs Predicted dengue cases",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        validation_predictions = validation_predictions.copy()
        validation_predictions["week_start_date"] = pd.to_datetime(validation_predictions["week_start_date"])

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
            fold_bands(ax, fold_years, vp["week_start_date"])
            ax.plot(
                vp["week_start_date"],
                vp["actual_total_cases"],
                color=COLOR_ACTUAL,
                linewidth=1.5,
                label="Actual",
                zorder=3,
            )
            ax.plot(
                vp["week_start_date"],
                vp["predicted_total_cases"],
                color=COLOR_PREDICTED,
                linewidth=1.5,
                linestyle="--",
                alpha=0.88,
                label="Predicted",
                zorder=4,
            )

            y_ann = max(float(vp["actual_total_cases"].max()), float(vp["predicted_total_cases"].max()), 1.0) * 0.93
            for year in fold_years:
                fold_df = vp[vp["validation_year"] == year]
                if fold_df.empty:
                    continue
                mae_val = mean_absolute_error(
                    fold_df["actual_total_cases"],
                    fold_df["predicted_total_cases"],
                )
                x_mid = fold_df["week_start_date"].iloc[len(fold_df) // 2]
                ax.annotate(
                    f"{year}\nMAE {mae_val:.1f}",
                    xy=(x_mid, y_ann),
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    color="#555555",
                    annotation_clip=True,
                )

            city_scores = validation_scores[validation_scores["city"] == city]
            avg_mae = city_scores["mae"].astype(float).mean() if not city_scores.empty else float("nan")
            style_time_axis(
                ax,
                f"{CITY_LABELS.get(city, city.upper())} - mean validation MAE: {avg_mae:.2f}",
            )

        fig1.tight_layout()
        validation_plot_path = output_dir / "validation_actual_vs_predicted.png"
        fig1.savefig(validation_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        plot_paths.append(validation_plot_path)

    if cities:
        fig2, axes2 = plt.subplots(
            len(cities),
            1,
            figsize=(16, 5 * len(cities)),
            squeeze=False,
        )
        horizon_label = "full test set" if prediction_weeks == 0 else f"next {prediction_weeks} weeks"
        fig2.suptitle(
            f"LSTM - Training history & {horizon_label} forecast",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        for ax, city in zip(axes2[:, 0], cities):
            _, y_train = load_data(city)
            train_dates = load_week_start_dates(city, source="train").reindex(y_train.index)
            train_plot = pd.DataFrame(
                {
                    "week_start_date": train_dates,
                    "total_cases": y_train,
                }
            ).sort_values("week_start_date")
            forecast_plot = build_forecast_plot_frame(city, forecast_by_city[city])

            ax.plot(
                train_plot["week_start_date"],
                train_plot["total_cases"],
                color=COLOR_ACTUAL,
                linewidth=1.2,
                label="Training actuals",
                zorder=3,
            )

            if not forecast_plot.empty:
                bridge_x = [
                    train_plot["week_start_date"].iloc[-1],
                    forecast_plot["week_start_date"].iloc[0],
                ]
                bridge_y = [
                    train_plot["total_cases"].iloc[-1],
                    forecast_plot["total_cases"].iloc[0],
                ]
                ax.plot(bridge_x, bridge_y, color=COLOR_FORECAST, linewidth=1.5, linestyle="--", zorder=4)
                ax.plot(
                    forecast_plot["week_start_date"],
                    forecast_plot["total_cases"],
                    color=COLOR_FORECAST,
                    linewidth=1.8,
                    linestyle="--",
                    label="Test forecast",
                    zorder=4,
                )
                ax.axvspan(
                    forecast_plot["week_start_date"].iloc[0],
                    forecast_plot["week_start_date"].iloc[-1],
                    color=COLOR_FORECAST,
                    alpha=0.07,
                    linewidth=0,
                )
                ax.axvline(
                    forecast_plot["week_start_date"].iloc[0],
                    color=COLOR_FORECAST,
                    linewidth=0.8,
                    linestyle=":",
                )

            style_time_axis(
                ax,
                f"{CITY_LABELS.get(city, city.upper())} - training history & forecast",
            )

        fig2.tight_layout()
        forecast_plot_path = output_dir / "forecast_vs_history.png"
        fig2.savefig(forecast_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        plot_paths.append(forecast_plot_path)

    return plot_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM on the same Oliver engineered features used by the Optuna template."
    )
    parser.add_argument("--city", choices=["all", "sj", "iq"], default="all")
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--final-epochs", type=int, default=0, help="Use 0 to reuse each holdout best epoch.")
    parser.add_argument("--prediction-weeks", type=int, default=WEEKS_PER_YEAR, help="Use 0 to predict all test rows.")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="log1p")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction-output", default="one_year_prediction_lstm.csv")
    parser.add_argument("--write-full-submission", action="store_true")
    parser.add_argument("--submission-output", default="submission_prediction_lstm.csv")

    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.final_epochs < 0:
        raise ValueError("--final-epochs must be non-negative.")
    if args.prediction_weeks < 0:
        raise ValueError("--prediction-weeks must be non-negative.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.hidden_size < 1:
        raise ValueError("--hidden-size must be at least 1.")
    if args.num_layers < 1:
        raise ValueError("--num-layers must be at least 1.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be at least 1.")

    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    set_random_seed(args.random_state)
    device = resolve_device(args.device)
    cities = ["sj", "iq"] if args.city == "all" else [args.city]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_scores = []
    validation_prediction_frames = []
    history_frames = []
    test_predictions = []
    holdout_plot_paths = []
    test_plot_paths = []

    for city_id in cities:
        score_record, validation_predictions, holdout_history, best_epoch, holdout_plot_path = run_holdout_for_city(
            city_id,
            args,
            device,
        )
        final_epochs = args.final_epochs or max(1, best_epoch)
        city_test_predictions, test_history, test_plot_path = run_submission_for_city(
            city_id,
            args,
            device,
            final_epochs,
        )

        score_record["final_epochs"] = final_epochs
        validation_scores.append(score_record)
        validation_prediction_frames.append(validation_predictions)
        history_frames.extend([holdout_history, test_history])
        test_predictions.append(city_test_predictions)
        holdout_plot_paths.append(holdout_plot_path)
        test_plot_paths.append(test_plot_path)

    validation_scores_path = args.output_dir / "validation_scores.csv"
    validation_predictions_path = args.output_dir / "validation_predictions.csv"
    training_history_path = args.output_dir / "training_history.csv"
    prediction_path = save_prediction_rows(
        test_predictions,
        args.output_dir / args.prediction_output,
    )
    submission_path = None
    if args.write_full_submission:
        submission_path = save_submission_predictions(
            test_predictions,
            args.output_dir / args.submission_output,
        )

    validation_scores_df = pd.DataFrame(validation_scores)
    validation_predictions_df = pd.concat(validation_prediction_frames, ignore_index=True)
    training_history_df = pd.concat(history_frames, ignore_index=True)
    combined_plot_paths = plot_sarimax_style_visualizations(
        validation_predictions_df,
        validation_scores_df,
        test_predictions,
        args.output_dir / "plots",
        args.prediction_weeks,
    )

    validation_scores_df.to_csv(validation_scores_path, index=False)
    validation_predictions_df.to_csv(validation_predictions_path, index=False)
    training_history_df.to_csv(training_history_path, index=False)

    print("LSTM Oliver-feature run")
    print("Cities:", ", ".join(cities))
    print("Device:", device)
    print("Target transform:", args.target_transform)
    print("Sequence length:", args.sequence_length)
    print("Prediction weeks:", "all" if args.prediction_weeks == 0 else args.prediction_weeks)
    print("Validation scores:", validation_scores_path)
    print("Validation predictions:", validation_predictions_path)
    print("Training history:", training_history_path)
    print("Prediction CSV:", prediction_path)
    if submission_path is not None:
        print("Full submission CSV:", submission_path)
    print("Holdout plots:", ", ".join(str(path) for path in holdout_plot_paths))
    print("Prediction plots:", ", ".join(str(path) for path in test_plot_paths))
    print("Combined plots:", ", ".join(str(path) for path in combined_plot_paths))


if __name__ == "__main__":
    main()
