from pathlib import Path

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from codecarbon import EmissionsTracker

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DB_PATH = BASE_DIR / "optuna_forecasting.db"
CODECARBON_DIR = BASE_DIR / "codecarbon_logs"
STUDY_NAME = "random_forest_forecasting_template"
STORAGE = f"sqlite:///{DB_PATH}"


def load_data(city_id: str, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    train_features = pd.read_csv(DATA_DIR / "dengue_features_train.csv", index_col=[0, 1, 2])
    train_labels = pd.read_csv(DATA_DIR / "dengue_labels_train.csv", index_col=[0, 1, 2])

    train_features = train_features.loc[city_id].sort_index()
    train_labels = train_labels.loc[city_id].sort_index()

    if feature_cols is not None:
        train_features = train_features[feature_cols]

    train_features = train_features.select_dtypes(include="number")
    train_features = train_features.ffill().bfill()
    train_labels = train_labels["total_cases"]

    return train_features, train_labels


def build_model(trial: optuna.Trial) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 16),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        n_jobs=-1,
    )

def yearly_time_series_splits(X: pd.DataFrame):
    years = X.index.get_level_values("year").unique().sort_values()

    for val_year in years[1:]:
        year = X.index.get_level_values("year")

        train_index = year < val_year
        val_index = year == val_year

        yield train_index, val_index


def objective(trial: optuna.Trial) -> float:
    X, y = load_data("sj")
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

        for fold, (train_index, test_index) in enumerate(yearly_time_series_splits(X)):
            model = build_model(trial)
            model.fit(X.iloc[train_index], y.iloc[train_index])

            pred = model.predict(X.iloc[test_index])
            mae = float(mean_absolute_error(y.iloc[test_index], pred))
            fold_maes.append(mae)
            trial.set_user_attr(f"fold_{fold}_mae", mae)

        return float(sum(fold_maes) / len(fold_maes))
    finally:
        emissions_kg = tracker.stop()
        trial.set_user_attr("emissions_kg_co2eq", emissions_kg)
        trial.set_user_attr("codecarbon_csv", str(CODECARBON_DIR / f"trial_{trial.number}_emissions.csv"))


def main() -> None:
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=30)

    print("Study:", STUDY_NAME)
    print("SQLite backup:", DB_PATH)
    print("CodeCarbon logs:", CODECARBON_DIR)
    print("Best MAE:", round(study.best_value, 3))
    print("Best params:", study.best_params)
    print("\nDashboard:")
    print(f"optuna-dashboard {STORAGE}")


if __name__ == "__main__":
    main()
