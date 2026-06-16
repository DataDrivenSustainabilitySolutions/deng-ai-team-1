"""
DengAI — Modeling Framework
Usage:  python main.py [--data-dir ../data] [--submit]
"""
import argparse
from pathlib import Path

from preprocess import load_data, preprocess, preprocess_test, year_based_splits
from models import seasonal_naive, fit_arima, fit_arimax, fit_random_forest
from evaluate import (print_results, plot_forecasts,
                      plot_importance, generate_submission)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--submit", action="store_true",
                        help="Generate submission CSV (RF on full training data)")
    args    = parser.parse_args()
    data_dir = Path(args.data_dir)

    sj_raw, iq_raw, test_raw, submission_fmt = load_data(data_dir)

    cv_results = {}
    last_fold  = {}
    last_preds = {}
    rf_models  = {}

    for city, raw in (("sj", sj_raw), ("iq", iq_raw)):
        print(f"\n{'='*60}\n  City: {city.upper()}\n{'='*60}")

        # year_based_splits operates on raw data and re-imputes + re-engineers
        # each fold internally, so we pass raw here (not preprocess(raw)).
        folds = list(year_based_splits(raw))   # [(fold_idx, tr, val), ...]

        naive_maes               = [seasonal_naive(tr, val) for _, tr, val in folds]
        arima_maes,  arima_last  = fit_arima(folds,  city=city)
        arimax_maes, arimax_last = fit_arimax(folds, city=city)

        # fit_random_forest also receives raw so _rf_tune can run inner
        # year_based_splits on the raw training fold for each outer fold.
        rf_maes, rf_last, rf_model = fit_random_forest(folds, raw)

        cv_results[city] = {
            "Seasonal Naive": naive_maes,
            "ARIMA":          arima_maes,
            "ARIMAX":         arimax_maes,
            "Random Forest":  rf_maes,
        }
        last_fold[city]  = (folds[-1][1], folds[-1][2])
        last_preds[city] = {
            "ARIMA":         arima_last,
            "ARIMAX":        arimax_last,
            "Random Forest": rf_last,
        }
        rf_models[city] = rf_model

    print_results(cv_results)
    plot_forecasts(last_fold, last_preds)
    plot_importance(rf_models)

    if args.submit and test_raw is not None:
        generate_submission(sj_raw, iq_raw, test_raw,
                            rf_models, submission_fmt, data_dir)


if __name__ == "__main__":
    main()