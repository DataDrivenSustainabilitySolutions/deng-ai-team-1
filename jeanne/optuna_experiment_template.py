from pathlib import Path

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from codecarbon import EmissionsTracker

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "optuna_forecasting.db"
CODECARBON_DIR = BASE_DIR / "codecarbon_logs"
STUDY_NAME = "random_forest_forecasting_template"
STORAGE = f"sqlite:///{DB_PATH}"


def load_data(city_id: str, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = pd.read_csv(
        "../data/dengue_features_train.csv", index_col=[0, 1, 2]
    )

    train_labels = pd.read_csv(
        "../data/dengue_labels_train.csv", index_col=[0, 1, 2]
    )
    train_features = train_features.loc[city_id][feature_cols]
    train_labels = train_labels.loc[city_id][feature_cols]

    # TODO Cross Validation, Moving Window, etc.
    valid = None

    return train_features, train_labels, valid

def objective(trial: optuna.Trial) -> float:
    features, labels, valid = load_data("sj")
    tracker = EmissionsTracker(
        project_name=STUDY_NAME,
        output_dir=str(CODECARBON_DIR),
        output_file=f"trial_{trial.number}_emissions.csv",
        log_level="error",
        save_to_file=True,
    )
    tracker.start()

    try:
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 16),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            random_state=7,
            n_jobs=-1,
        )
        model.fit(features, labels)

        pred = model.predict(valid)
        mae = mean_absolute_error(labels, pred)
        return float(mae)
    finally:
        if tracker is not None:
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
