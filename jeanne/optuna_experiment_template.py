from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from codecarbon import EmissionsTracker

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DB_PATH = BASE_DIR / "optuna_forecasting.db"
CODECARBON_DIR = BASE_DIR / "codecarbon_logs"
PLOT_DIR = BASE_DIR / "optuna_plots"
STUDY_NAME = "random_forest_forecasting_with_lags_and_rollings"
STORAGE = f"sqlite:///{DB_PATH}"
CITY_ID = "sj"
RANDOM_STATE = 7


def load_data(city_id: str, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    train_features = pd.read_csv(DATA_DIR / "dengue_features_train.csv", index_col=[0, 1, 2])
    train_labels = pd.read_csv(DATA_DIR / "dengue_labels_train.csv", index_col=[0, 1, 2])

    train_features = train_features.loc[city_id].sort_index()
    train_labels = train_labels.loc[city_id].sort_index()

    if feature_cols is not None:
        train_features = train_features[feature_cols]

    train_features = train_features.select_dtypes(include="number")
    train_features = train_features.interpolate(method="linear", limit_direction="both")
    train_labels = train_labels["total_cases"]

    base_features = train_features
    lagged_features = [base_features.shift(lag).add_suffix(f"_lag_{lag}") for lag in range(0, 14)]
    rolling_features = [
        base_features.shift(1).rolling(window=window, min_periods=1).mean().add_suffix(f"_roll_{window}_mean")
        for window in (3, 6)
    ]
    train_features = pd.concat([base_features, *lagged_features, *rolling_features], axis=1)
    train_features = train_features.bfill()

    return train_features, train_labels


def load_week_start_dates(city_id: str) -> pd.Series:
    train_features = pd.read_csv(
        DATA_DIR / "dengue_features_train.csv",
        index_col=[0, 1, 2],
        usecols=["city", "year", "weekofyear", "week_start_date"],
    )
    dates = pd.to_datetime(train_features.loc[city_id].sort_index()["week_start_date"])
    dates.name = "week_start_date"

    return dates


def build_model_from_params(params: dict[str, object]) -> RandomForestRegressor:
    return RandomForestRegressor(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_model(trial: optuna.Trial) -> RandomForestRegressor:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

    return build_model_from_params(params)


def yearly_time_series_splits(X: pd.DataFrame):
    years = X.index.get_level_values("year").unique().sort_values()

    for val_year in years[1:]:
        year = X.index.get_level_values("year")

        train_index = year < val_year
        val_index = year == val_year

        yield train_index, val_index


def final_year_holdout_split(X: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    years = X.index.get_level_values("year")
    final_year = years.max()
    train_index = years < final_year
    test_index = years == final_year

    if not train_index.any() or not test_index.any():
        raise ValueError("Need at least two years for a final year holdout split.")

    return train_index, test_index


def objective(trial: optuna.Trial) -> float:
    X, y = load_data(CITY_ID)
    cv_index, _ = final_year_holdout_split(X)
    X_cv = X.iloc[cv_index]
    y_cv = y.iloc[cv_index]

    CODECARBON_DIR.mkdir(exist_ok=True)
    tracker = EmissionsTracker(
        project_name=STUDY_NAME,
        output_dir=str(CODECARBON_DIR),
        output_file=f"trial_{trial.number}_emissions.csv",
        log_level="error",
        save_to_file=True,
    )
    tracker.start()

    try:
        fold_maes = []

        for fold, (train_index, test_index) in enumerate(yearly_time_series_splits(X_cv), start=1):
            model = build_model(trial)
            model.fit(X_cv.iloc[train_index], y_cv.iloc[train_index])

            pred = model.predict(X_cv.iloc[test_index])
            mae = float(mean_absolute_error(y_cv.iloc[test_index], pred))
            fold_maes.append(mae)
            trial.set_user_attr(f"fold_{fold}_mae", mae)

        return float(sum(fold_maes) / len(fold_maes))
    finally:
        emissions_kg = tracker.stop()
        trial.set_user_attr("emissions_kg_co2eq", emissions_kg)
        trial.set_user_attr("codecarbon_csv", str(CODECARBON_DIR / f"trial_{trial.number}_emissions.csv"))


def plot_best_model_prediction(
    city_id: str,
    best_params: dict[str, object],
    output_dir: Path = PLOT_DIR,
) -> tuple[Path, float]:
    X, y = load_data(city_id)
    train_index, test_index = final_year_holdout_split(X)

    model = build_model_from_params(best_params)
    model.fit(X.iloc[train_index], y.iloc[train_index])
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]
    print(y_test.value_counts())
    prediction = model.predict(X_test)
    final_mae = float(mean_absolute_error(y_test, prediction))
    holdout_year = X_test.index.get_level_values("year").max()

    dates = load_week_start_dates(city_id).reindex(X.index)
    train_data = pd.DataFrame(
        {
            "date": dates.iloc[train_index],
            "total_cases": y.iloc[train_index],
        }
    )
    test_data = pd.DataFrame(
        {
            "date": dates.iloc[test_index],
            "total_cases": y_test,
            "prediction": prediction,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{city_id}_best_model_prediction.png"

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train_data["date"], train_data["total_cases"], label="Train labels", color="#1f2937")
    ax.plot(test_data["date"], test_data["total_cases"], label="Test labels", color="#2563eb")
    ax.plot(test_data["date"], test_data["prediction"], label="Best model prediction", color="#dc2626")
    ax.axvline(test_data["date"].iloc[0], color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title(f"{city_id}: {holdout_year} holdout prediction (MAE: {final_mae:.3f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path, final_mae


def main() -> None:
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)
    plot_path, final_mae = plot_best_model_prediction(CITY_ID, study.best_params)

    print("Study:", STUDY_NAME)
    print("SQLite backup:", DB_PATH)
    print("CodeCarbon logs:", CODECARBON_DIR)
    print("Best MAE:", round(study.best_value, 3))
    print("Best params:", study.best_params)
    print("Final holdout MAE:", round(final_mae, 3))
    print("Final prediction plot:", plot_path)
    print("\nDashboard:")
    print(f"optuna-dashboard {STORAGE}")


if __name__ == "__main__":
    main()
