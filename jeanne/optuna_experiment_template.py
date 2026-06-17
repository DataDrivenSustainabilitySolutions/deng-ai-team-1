import math
import os
from functools import lru_cache
from pathlib import Path

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
SUBMISSION_OUTPUT_PATH = BASE_DIR / "submission_prediction.csv"
STORAGE = f"sqlite:///{DB_PATH}"
CITY_ID = "iq"
STUDY_NAME = "random_forest_forecasting_oliver_features_{}".format(CITY_ID)
RANDOM_STATE = 7
N_TRIALS = 50
MERGE_KEYS = ["city", "year", "weekofyear"]
IDENTIFIER_COLUMNS = {
    "city",
    "year",
    "weekofyear",
    "week_start_date",
    "split",
    "total_cases",
}
SEASONALITY_FEATURE_COLUMNS = ["weekofyear_sin", "weekofyear_cos"]


WEATHER_LAG_PLAN = {
    "reanalysis_specific_humidity_g_per_kg": [1, 2, 3, 5, 6, 8, 10, 12],
    "reanalysis_dew_point_temp_k": [1, 2, 3, 5, 6, 8, 10, 12],
    "reanalysis_min_air_temp_k": [1, 2, 3, 5, 6, 8, 10, 12],
    "reanalysis_air_temp_k": [6, 8, 10, 12],
    "reanalysis_avg_temp_k": [6, 8, 10, 12],
    "reanalysis_max_air_temp_k": [6, 8, 10, 12],
    "station_avg_temp_c": [2, 4, 5, 6, 8, 10, 12],
    "station_min_temp_c": [1, 2, 3, 4, 6, 8, 10, 12],
    "station_max_temp_c": [4, 6, 8, 10, 12],
    "ndvi_sw": [10, 11, 14],
    "ndvi_nw": [10, 11, 14],
    "precipitation_amt_mm": [2, 3],
    "reanalysis_sat_precip_amt_mm": [2, 3],
}


WEATHER_ROLLING_PLAN = {
    "reanalysis_specific_humidity_g_per_kg": [3, 4, 6, 8, 10, 12, 14],
    "reanalysis_dew_point_temp_k": [3, 4, 6, 8, 10, 12, 14],
    "reanalysis_min_air_temp_k": [3, 4, 5, 6, 7, 8, 10, 12, 14],
    "reanalysis_air_temp_k": [8, 10, 12, 14],
    "reanalysis_avg_temp_k": [8, 10, 12, 14],
    "reanalysis_max_air_temp_k": [8, 10, 12, 14],
    "station_avg_temp_c": [3, 4, 5, 6, 8, 10, 12, 14],
    "station_min_temp_c": [3, 4, 5, 6, 8, 10, 12, 14],
    "station_max_temp_c": [5, 6, 8, 10, 12, 14],
    "precipitation_amt_mm": [4, 7, 10, 14],
    "reanalysis_sat_precip_amt_mm": [4, 7, 10, 14],
}


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


def select_oliver_feature_columns(
    data_columns: pd.Index,
    engineered_feature_columns: list[str],
) -> dict[str, list[str]]:
    city_feature_columns = {}

    for city in ["sj", "iq"]:
        weather_feature_columns = []

        if city == "sj":
            for feature in SJ_WEATHER_ROOTS:
                if feature in data_columns:
                    weather_feature_columns.append(feature)
                for column in engineered_feature_columns:
                    if column.startswith(f"{feature}_lag_") or column.startswith(f"{feature}_rolling_"):
                        weather_feature_columns.append(column)

        if city == "iq":
            short_lag_suffixes = ["lag_1", "lag_2", "lag_3", "lag_5", "lag_0_3_mean", "lag_1_3_mean"]
            station_suffixes = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_6", "lag_1_3_mean"]
            ndvi_suffixes = ["lag_10", "lag_11", "lag_14"]
            precip_suffixes = ["lag_2", "lag_3", "lag_0_3_mean", "lag_1_3_mean", "lag_2_5_mean"]
            short_rolling_suffixes = [
                "rolling_3_mean",
                "rolling_4_mean",
                "rolling_5_mean",
                "rolling_6_mean",
                "rolling_7_mean",
                "rolling_8_mean",
            ]
            station_rolling_suffixes = [
                "rolling_3_mean",
                "rolling_4_mean",
                "rolling_5_mean",
                "rolling_6_mean",
            ]
            precip_rolling_suffixes = [
                "rolling_4_mean",
                "rolling_7_mean",
                "rolling_10_mean",
                "rolling_14_mean",
            ]

            for feature in IQ_WEATHER_ROOTS:
                if feature in data_columns:
                    weather_feature_columns.append(feature)

                for column in engineered_feature_columns:
                    if not column.startswith(f"{feature}_"):
                        continue

                    suffix = column.removeprefix(f"{feature}_")
                    if feature.startswith("ndvi_") and suffix in ndvi_suffixes:
                        weather_feature_columns.append(column)
                    elif "precip" in feature and suffix in precip_suffixes:
                        weather_feature_columns.append(column)
                    elif "precip" in feature and suffix in precip_rolling_suffixes:
                        weather_feature_columns.append(column)
                    elif feature.startswith("station_") and suffix in station_suffixes:
                        weather_feature_columns.append(column)
                    elif feature.startswith("station_") and suffix in station_rolling_suffixes:
                        weather_feature_columns.append(column)
                    elif feature.startswith("reanalysis_") and suffix in short_lag_suffixes:
                        weather_feature_columns.append(column)
                    elif feature.startswith("reanalysis_") and suffix in short_rolling_suffixes:
                        weather_feature_columns.append(column)

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
    engineered_feature_columns = []
    engineered_feature_data = {}

    week_angle = 2 * math.pi * data["weekofyear"] / 52.0
    engineered_feature_data["weekofyear_sin"] = week_angle.map(math.sin)
    engineered_feature_data["weekofyear_cos"] = week_angle.map(math.cos)
    engineered_feature_columns.extend(SEASONALITY_FEATURE_COLUMNS)

    for feature, lags in WEATHER_LAG_PLAN.items():
        if feature not in data.columns:
            continue
        for lag in lags:
            column = f"{feature}_lag_{lag}"
            engineered_feature_data[column] = data.groupby("city", sort=False)[feature].shift(lag)
            engineered_feature_columns.append(column)

    for feature, rolling_windows in WEATHER_ROLLING_PLAN.items():
        if feature not in data.columns:
            continue
        for window in rolling_windows:
            column = f"{feature}_rolling_{window}_mean"
            engineered_feature_data[column] = (
                data.groupby("city", sort=False)[feature]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            engineered_feature_columns.append(column)

    data = pd.concat([data, pd.DataFrame(engineered_feature_data, index=data.index)], axis=1)
    city_feature_columns = select_oliver_feature_columns(data.columns, engineered_feature_columns)

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


def load_data(city_id: str, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    train_features, train_labels, _, _ = load_engineered_city_data(city_id)

    if feature_cols is not None:
        train_features = train_features[feature_cols]

    return train_features.copy(), train_labels.copy()


def load_test_data(city_id: str, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    _, _, test_features, dates = load_engineered_city_data(city_id)

    if feature_cols is not None:
        test_features = test_features[feature_cols]

    return test_features.copy(), dates.copy()


def load_week_start_dates(city_id: str, source: str = "train") -> pd.Series:
    path = DATA_DIR / "dengue_features_train.csv" if source == "train" else DATA_DIR / "dengue_features_test.csv"
    features = pd.read_csv(path, parse_dates=["week_start_date"], index_col=[0, 1, 2])
    dates = pd.to_datetime(features.loc[city_id].sort_index()["week_start_date"])
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


def make_case_count_predictions(raw_predictions, index: pd.Index | None = None) -> pd.Series:
    predictions = pd.Series(raw_predictions, index=index, name="total_cases")

    return predictions.clip(lower=0).round().astype(int)


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
        output_file=f"{STUDY_NAME}_trial_{trial.number}_emissions.csv",
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
        trial.set_user_attr(
            "codecarbon_csv",
            str(CODECARBON_DIR / f"{STUDY_NAME}_trial_{trial.number}_emissions.csv"),
        )


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
    prediction = make_case_count_predictions(model.predict(X_test), index=X_test.index)
    final_mae = float(mean_absolute_error(y_test, prediction))
    holdout_year = X_test.index.get_level_values("year").max()

    dates = load_week_start_dates(city_id).reindex(X.index)
    train_data = pd.DataFrame(
        {
            "date": dates.iloc[train_index],
            "total_cases": y.iloc[train_index],
        }
    ).sort_values("date")
    test_data = pd.DataFrame(
        {
            "date": dates.iloc[test_index],
            "total_cases": y_test,
            "prediction": prediction,
        }
    ).sort_values("date")

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


def predict_submission_from_test(city_id: str, best_params: dict[str, object]) -> tuple[pd.Series, pd.Series]:
    X_train, y_train = load_data(city_id)
    X_test, test_dates = load_test_data(city_id)

    model = build_model_from_params(best_params)
    model.fit(X_train, y_train)
    predictions = make_case_count_predictions(model.predict(X_test), index=X_test.index)

    if "city" not in predictions.reset_index().columns:
        predictions.index = pd.MultiIndex.from_arrays(
            [
                [city_id] * len(predictions),
                predictions.index.get_level_values("year"),
                predictions.index.get_level_values("weekofyear"),
            ],
            names=["city", "year", "weekofyear"],
        )

    return predictions, test_dates


def save_submission_predictions(predictions: pd.Series, output_path: Path = SUBMISSION_OUTPUT_PATH) -> Path:
    submission_format = pd.read_csv(DATA_DIR / "submission_format.csv")
    submission = submission_format.copy()

    if output_path.exists():
        existing_submission = pd.read_csv(output_path)
        expected_columns = ["city", "year", "weekofyear", "total_cases"]
        if list(existing_submission.columns) == expected_columns and existing_submission[MERGE_KEYS].equals(
            submission_format[MERGE_KEYS]
        ):
            submission["total_cases"] = existing_submission["total_cases"]

    prediction_frame = predictions.reset_index()[["city", "year", "weekofyear", "total_cases"]]
    submission = submission.merge(
        prediction_frame,
        on=["city", "year", "weekofyear"],
        how="left",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    prediction_column = "total_cases_prediction"
    submission["total_cases"] = submission[prediction_column].combine_first(submission["total_cases"])
    submission = submission.drop(columns=[prediction_column])
    submission = submission[["city", "year", "weekofyear", "total_cases"]]
    submission["total_cases"] = make_case_count_predictions(submission["total_cases"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    return output_path


def plot_submission_prediction(
    city_id: str,
    best_params: dict[str, object],
    output_dir: Path = PLOT_DIR,
) -> Path:
    X_train, y_train = load_data(city_id)
    X_test, test_dates = load_test_data(city_id)

    model = build_model_from_params(best_params)
    model.fit(X_train, y_train)
    prediction = make_case_count_predictions(model.predict(X_test), index=X_test.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{city_id}_final_submission_prediction.png"

    train_dates = load_week_start_dates(city_id, source="train")
    first_test_date = test_dates.min()
    train_mask = train_dates < first_test_date
    train_dates = train_dates[train_mask]
    y_train_plot = y_train.loc[train_mask]

    train_plot = pd.DataFrame({"date": train_dates, "total_cases": y_train_plot}).sort_values("date")
    test_plot = pd.DataFrame({"date": test_dates, "prediction": prediction}).sort_values("date")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train_plot["date"], train_plot["total_cases"], label="Train labels", color="#1f2937")
    ax.plot(test_plot["date"], test_plot["prediction"], label="Final test prediction", color="#dc2626")
    ax.axvline(first_test_date, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title(f"{city_id}: Test set submission prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def main() -> None:
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    plot_path, final_mae = plot_best_model_prediction(CITY_ID, study.best_params)
    submission_predictions, _ = predict_submission_from_test(CITY_ID, study.best_params)
    submission_path = save_submission_predictions(submission_predictions)
    submission_plot_path = plot_submission_prediction(CITY_ID, study.best_params)

    print("Study:", STUDY_NAME)
    print("Trials requested:", N_TRIALS)
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    print("Completed trials:", len(completed_trials))
    print("SQLite backup:", DB_PATH)
    print("CodeCarbon logs:", CODECARBON_DIR)
    print("Best MAE:", round(study.best_value, 3))
    print("Best params:", study.best_params)
    print("Final holdout MAE:", round(final_mae, 3))
    print("Final holdout prediction plot:", plot_path)
    print("Submission CSV:", submission_path)
    print("Submission prediction plot:", submission_plot_path)
    print("\nDashboard:")
    print(f"optuna-dashboard {STORAGE}")


if __name__ == "__main__":
    main()
