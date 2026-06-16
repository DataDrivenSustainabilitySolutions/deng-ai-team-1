"""Plot DengAI features and available target values in city-specific grids.

Example:
    python3 oliver/visualization.py
"""

import argparse
import math
import os
from pathlib import Path

import pandas as pd


script_name = Path(__file__).stem
default_output_dir = Path(__file__).resolve().parent / "outputs" / script_name
default_output_dir_label = f"oliver/outputs/{script_name}"


### CLI arguments
parser = argparse.ArgumentParser(
    description="Plot train/test feature time series and the labeled train target."
)
parser.add_argument(
    "--features-csv",
    "--train-features-csv",
    dest="train_features_csv",
    default="data/dengue_features_train.csv",
    help="Path to the training feature CSV. Default: data/dengue_features_train.csv",
)
parser.add_argument(
    "--test-features-csv",
    default="data/dengue_features_test.csv",
    help="Path to the test feature CSV. Default: data/dengue_features_test.csv",
)
parser.add_argument(
    "--labels-csv",
    default="data/dengue_labels_train.csv",
    help="Path to the training label CSV. Default: data/dengue_labels_train.csv",
)
parser.add_argument(
    "--output-dir",
    default=default_output_dir,
    help=f"Directory for generated plots. Default: {default_output_dir_label}",
)
parser.add_argument(
    "--output-file",
    default="feature_target_grid.png",
    help="Base filename for generated city plot grids. Default: feature_target_grid.png",
)
parser.add_argument(
    "--max-weeks-per-city",
    type=int,
    default=0,
    help="Earliest weeks to plot per city and split. Use 0 to plot all rows. Default: 0",
)
parser.add_argument(
    "--columns",
    type=int,
    default=3,
    help="Number of subplot columns in the grid. Default: 3",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=150,
    help="Resolution for the saved figure. Default: 150",
)
args = parser.parse_args()


### Configuration
train_features_path = Path(args.train_features_csv)
test_features_path = Path(args.test_features_csv)
labels_path = Path(args.labels_csv)
output_dir = Path(args.output_dir)
base_output_path = output_dir / args.output_file
merge_keys = ["city", "year", "weekofyear"]
identifier_columns = {"city", "year", "weekofyear", "week_start_date", "split", "total_cases"}


### Data loading
train_features = pd.read_csv(train_features_path, parse_dates=["week_start_date"])
test_features = pd.read_csv(test_features_path, parse_dates=["week_start_date"])
labels = pd.read_csv(labels_path)

if list(train_features.columns) != list(test_features.columns):
    raise ValueError("Train and test feature CSVs must have the same columns.")

train_data = train_features.merge(labels, on=merge_keys, how="left", validate="one_to_one")
train_data["split"] = "train"

test_data = test_features.copy()
test_data["total_cases"] = float("nan")
test_data["split"] = "test"

data = pd.concat([train_data, test_data], ignore_index=True)


### Preprocessing
if train_data["total_cases"].isna().any():
    raise ValueError("Merged data contains missing total_cases values.")

data = data.sort_values(["city", "week_start_date", "split"])

feature_columns = [
    column
    for column in data.columns
    if column not in identifier_columns and pd.api.types.is_numeric_dtype(data[column])
]
plot_columns = feature_columns + ["total_cases"]

if args.max_weeks_per_city > 0:
    plot_data = (
        data.groupby(["city", "split"], group_keys=False).head(args.max_weeks_per_city).copy()
    )
else:
    plot_data = data.copy()


### Plot grids
os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

columns = max(1, args.columns)
rows = math.ceil(len(plot_columns) / columns)
figure_width = columns * 5
figure_height = rows * 3
output_suffix = base_output_path.suffix or ".png"
output_stem = base_output_path.stem if base_output_path.suffix else base_output_path.name
saved_paths = []

for city, city_plot_data in plot_data.groupby("city"):
    figure, axes = plt.subplots(rows, columns, figsize=(figure_width, figure_height), sharex=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for index, column in enumerate(plot_columns):
        axis = axes[index]

        for split, split_data in city_plot_data.groupby("split"):
            if column == "total_cases" and split == "test":
                continue

            missing_dates = split_data.loc[split_data[column].isna(), "week_start_date"]
            for missing_date in missing_dates:
                axis.axvline(
                    missing_date,
                    color="red",
                    linewidth=0.6,
                    alpha=0.35,
                    zorder=1,
                )

            observed_data = split_data.dropna(subset=[column])
            if observed_data.empty:
                continue

            axis.plot(
                observed_data["week_start_date"],
                observed_data[column],
                label=split,
                linewidth=0.9,
                alpha=0.9,
                linestyle="-" if split == "train" else "--",
                zorder=2,
            )

        axis.set_title(column.replace("_", " "), fontsize=9)
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="x", labelrotation=35, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)

    for axis in axes[len(plot_columns) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", title="split")
    figure.suptitle(f"DengAI {city} features and labeled target cases", fontsize=14)
    figure.tight_layout(rect=[0, 0, 0.98, 0.97])


    ### Output export
    output_dir.mkdir(parents=True, exist_ok=True)
    city_output_path = output_dir / f"{output_stem}_{city}{output_suffix}"
    figure.savefig(city_output_path, dpi=args.dpi)
    plt.close(figure)
    saved_paths.append(city_output_path)

print("Saved plot grids:")
for saved_path in saved_paths:
    print(f"- {saved_path}")
