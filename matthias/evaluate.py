"""
evaluate.py — results table, forecast plots, feature importance, submission.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from preprocess import preprocess_test

CITY_NAMES = {"sj": "San Juan", "iq": "Iquitos"}


# ── Console summary ───────────────────────────────────────────────────────────
def print_results(cv_results: dict):
    """
    Print mean ± std MAE across CV folds for each model and city.
    cv_results[city][model] = list of per-fold MAEs.
    """
    print("\n── Cross-validated MAE (mean ± std across folds, lower is better) ──")
    models = ["Seasonal Naive", "ARIMA", "ARIMAX", "Random Forest"]
    rows   = []
    for city in ("sj", "iq"):
        row = {"City": CITY_NAMES[city]}
        for model in models:
            maes = cv_results[city].get(model, [])
            if maes:
                row[model] = f"{np.mean(maes):.2f} ± {np.std(maes):.2f}"
            else:
                row[model] = "—"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("City")
    print(df.to_string())

    # Per-fold detail
    print("\n── Per-fold MAE detail ──")
    for city in ("sj", "iq"):
        print(f"\n  {CITY_NAMES[city]}:")
        for model in models:
            maes = cv_results[city].get(model, [])
            if maes:
                fold_str = "  ".join(f"fold{i}={m:.1f}" for i, m in enumerate(maes))
                print(f"    {model:<16} {fold_str}")


# ── Forecast visualisation (last fold) ───────────────────────────────────────
def plot_forecasts(last_fold: dict, last_preds: dict, out: str = "forecasts.png"):
    """
    Plot actual vs predicted for the last CV fold of each city.
    last_fold[city] = (tr, val)
    last_preds[city] = {model: preds_array}
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), tight_layout=True)
    colours = {"ARIMA": "steelblue", "ARIMAX": "darkorange", "Random Forest": "seagreen"}
    styles  = {"ARIMA": "--",        "ARIMAX": "-.",          "Random Forest": ":"}

    for ax, city in zip(axes, ("sj", "iq")):
        _, val   = last_fold[city]
        dates    = val["week_start_date"].values
        actuals  = val["total_cases"].values
        ax.plot(dates, actuals, color="black", lw=1.4, label="Actual")

        for model, preds in last_preds[city].items():
            n = min(len(dates), len(preds))
            ax.plot(dates[len(dates) - n:], preds[-n:],
                    color=colours[model], ls=styles[model], lw=1.2, label=model)

        ax.set_title(f"{CITY_NAMES[city]} — last CV fold forecasts")
        ax.set_ylabel("weekly cases")
        ax.legend(fontsize=9)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


# ── Feature importance (last fold model) ─────────────────────────────────────
def plot_importance(rf_models: dict, out: str = "importance.png"):
    """rf_models[city] = (rf, scaler, feat_cols) from the last CV fold."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), tight_layout=True)
    for ax, city in zip(axes, ("sj", "iq")):
        rf, _, feat_cols = rf_models[city]
        imp = pd.Series(rf.feature_importances_, index=feat_cols).nlargest(20)
        imp.sort_values().plot.barh(ax=ax, color="steelblue")
        ax.set_title(f"{CITY_NAMES[city]} — RF feature importance (top 20, last fold)")

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# ── Submission ────────────────────────────────────────────────────────────────
def generate_submission(sj_raw, iq_raw, test_raw, rf_models, submission_fmt, data_dir):
    """Refit RF on the full training data per city and predict on test."""
    out = submission_fmt.copy()
    for city, raw in (("sj", sj_raw), ("iq", iq_raw)):
        te_city = test_raw.loc[city].sort_values("week_start_date")
        te_proc = preprocess_test(raw, te_city)

        rf, scaler, feat_cols = rf_models[city]
        avail = [c for c in feat_cols if c in te_proc.columns]
        X     = scaler.transform(te_proc[avail].fillna(0).values)
        preds = np.clip(np.expm1(rf.predict(X)), 0, None).round().astype(int)
        out.loc[city, "total_cases"] = preds

    path = data_dir / "submission_rf.csv"
    out.to_csv(path)
    print(f"Submission saved → {path}")