import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from optuna_experiment_template import load_data, load_week_start_dates


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "pca_outputs"


def fit_pca(X: pd.DataFrame) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(X_scaled.shape)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    score_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    scores_df = pd.DataFrame(scores, index=X.index, columns=score_cols)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=X.columns, columns=score_cols)

    return pca, scores_df, loadings_df


def build_explained_variance_table(pca: PCA) -> pd.DataFrame:
    component = np.arange(1, len(pca.explained_variance_ratio_) + 1)

    return pd.DataFrame(
        {
            "component": component,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )


def plot_explained_variance(city_id: str, explained: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / f"{city_id}_pca_explained_variance.png"
    shown = explained.head(40)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        shown["component"],
        shown["explained_variance_ratio"],
        color="#2563eb",
        alpha=0.8,
        label="Explained variance",
    )
    ax.set_title(f"{city_id}: PCA explained variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xticks(shown["component"])

    ax_cum = ax.twinx()
    ax_cum.plot(
        shown["component"],
        shown["cumulative_explained_variance"],
        color="#dc2626",
        marker="o",
        linewidth=2,
        label="Cumulative variance",
    )
    ax_cum.set_ylabel("Cumulative explained variance")
    ax_cum.set_ylim(0, 1.02)

    lines, labels = ax.get_legend_handles_labels()
    cum_lines, cum_labels = ax_cum.get_legend_handles_labels()
    ax.legend(lines + cum_lines, labels + cum_labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def plot_pc_scatter(
    city_id: str,
    scores: pd.DataFrame,
    y: pd.Series,
    explained: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"{city_id}_pca_pc1_pc2_cases.png"

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        scores["PC1"],
        scores["PC2"],
        c=y.to_numpy(),
        cmap="viridis",
        s=38,
        alpha=0.82,
        edgecolors="none",
    )
    pc1_var = explained.loc[0, "explained_variance_ratio"] * 100
    pc2_var = explained.loc[1, "explained_variance_ratio"] * 100
    ax.set_title(f"{city_id}: PCA scores colored by total cases")
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Total cases")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def plot_pc_time_path(
    city_id: str,
    scores: pd.DataFrame,
    dates: pd.Series,
    explained: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"{city_id}_pca_pc1_pc2_time.png"

    date_numbers = matplotlib.dates.date2num(dates)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(scores["PC1"], scores["PC2"], color="#9ca3af", linewidth=1, alpha=0.6)
    scatter = ax.scatter(
        scores["PC1"],
        scores["PC2"],
        c=date_numbers,
        cmap="plasma",
        s=34,
        alpha=0.88,
        edgecolors="none",
    )
    pc1_var = explained.loc[0, "explained_variance_ratio"] * 100
    pc2_var = explained.loc[1, "explained_variance_ratio"] * 100
    ax.set_title(f"{city_id}: PCA trajectory over time")
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Week start date")
    colorbar.ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def top_loadings(loadings: pd.DataFrame, pc: str, top_n: int) -> pd.DataFrame:
    pc_loadings = loadings[pc].sort_values(key=lambda values: values.abs(), ascending=False)

    return (
        pc_loadings.head(top_n)
        .rename("loading")
        .reset_index()
        .rename(columns={"index": "feature"})
    )


def plot_top_loadings(
    city_id: str,
    loadings: pd.DataFrame,
    pc: str,
    top_n: int,
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"{city_id}_pca_{pc.lower()}_top_loadings.png"
    data = top_loadings(loadings, pc, top_n).iloc[::-1]
    colors = np.where(data["loading"] >= 0, "#0f766e", "#be123c")

    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.35)))
    ax.barh(data["feature"], data["loading"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title(f"{city_id}: top {top_n} feature loadings for {pc}")
    ax.set_xlabel("Loading")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def save_scores(
    city_id: str,
    scores: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    output_dir: Path,
    n_components: int,
) -> Path:
    output_path = output_dir / f"{city_id}_pca_scores.csv"
    scores_to_save = scores.iloc[:, :n_components].copy()
    scores_to_save.insert(0, "total_cases", y.to_numpy())
    scores_to_save.insert(0, "week_start_date", dates.to_numpy())
    scores_to_save.insert(0, "city", city_id)
    scores_to_save.reset_index().to_csv(output_path, index=False)

    return output_path


def run_city(city_id: str, output_dir: Path, top_n: int, score_components: int) -> list[Path]:
    X, y = load_data(city_id)
    dates = load_week_start_dates(city_id).reindex(X.index)

    pca, scores, loadings = fit_pca(X)
    explained = build_explained_variance_table(pca)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [
        output_dir / f"{city_id}_pca_explained_variance.csv",
        output_dir / f"{city_id}_pca_loadings.csv",
        output_dir / f"{city_id}_pca_top_loadings.csv",
    ]

    explained.to_csv(saved_paths[0], index=False)
    loadings.to_csv(saved_paths[1], index_label="feature")

    top_loading_rows = []
    for pc in ["PC1", "PC2", "PC3"]:
        pc_top = top_loadings(loadings, pc, top_n)
        pc_top.insert(0, "component", pc)
        top_loading_rows.append(pc_top)
    pd.concat(top_loading_rows, ignore_index=True).to_csv(saved_paths[2], index=False)

    saved_paths.extend(
        [
            save_scores(
                city_id=city_id,
                scores=scores,
                y=y,
                dates=dates,
                output_dir=output_dir,
                n_components=min(score_components, scores.shape[1]),
            ),
            plot_explained_variance(city_id, explained, output_dir),
            plot_pc_scatter(city_id, scores, y, explained, output_dir),
            plot_pc_time_path(city_id, scores, dates, explained, output_dir),
            plot_top_loadings(city_id, loadings, "PC1", top_n, output_dir),
            plot_top_loadings(city_id, loadings, "PC2", top_n, output_dir),
            plot_top_loadings(city_id, loadings, "PC3", top_n, output_dir),
        ]
    )

    print(f"\nCity: {city_id}")
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")
    print(
        "Components for 90% / 95% variance:",
        int((explained["cumulative_explained_variance"] >= 0.90).idxmax() + 1),
        "/",
        int((explained["cumulative_explained_variance"] >= 0.95).idxmax() + 1),
    )
    print("Saved files:")
    for path in saved_paths:
        print(path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PCA on the Optuna feature matrix, including original features, "
            "lags, and rolling-window means."
        )
    )
    parser.add_argument(
        "--city",
        choices=["sj", "iq", "all"],
        default="all",
        help="City to analyze. Use 'all' for San Juan and Iquitos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for PCA plots and CSV outputs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of strongest absolute loadings to plot per component.",
    )
    parser.add_argument(
        "--score-components",
        type=int,
        default=10,
        help="Number of PCA score columns to save in the score CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city_ids = ["sj", "iq"] if args.city == "all" else [args.city]

    for city_id in city_ids:
        run_city(city_id, args.output_dir, args.top_n, args.score_components)


if __name__ == "__main__":
    main()
