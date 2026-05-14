import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["accuracy", "precision", "recall", "f1_score", "csi"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot UNet3D metrics over time for each region."
    )
    parser.add_argument(
        "--metrics-csv",
        required=True,
        help="Path to metrics_per_sample.csv produced by evaluate_unet3d_full.py",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/unet3d_full_evaluation_timeseries_full/figures",
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def plot_region_timeseries(region_df: pd.DataFrame, region_id: str, output_dir: str):
    fig, ax = plt.subplots(figsize=(11, 6))

    for metric in METRICS:
        ax.plot(region_df["target_year"], region_df[metric], marker="o", linewidth=1.8, label=metric)

    ax.set_title(f"UNet3D Metrics Over Time - {region_id}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    out_path = os.path.join(output_dir, f"metrics_timeseries_{region_id}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metric_across_regions(df: pd.DataFrame, metric: str, output_dir: str):
    fig, ax = plt.subplots(figsize=(11, 6))

    for region_id, region_df in df.groupby("region_id"):
        region_df = region_df.sort_values("target_year")
        ax.plot(region_df["target_year"], region_df[metric], marker="o", linewidth=1.8, label=region_id)

    ax.set_title(f"{metric} Over Time by Region")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    out_path = os.path.join(output_dir, f"timeseries_{metric}_all_regions.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()

    if not os.path.exists(args.metrics_csv):
        raise FileNotFoundError(f"Missing CSV: {args.metrics_csv}")

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)
    required_cols = {"region_id", "target_year", *METRICS}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metrics CSV: {sorted(missing)}")

    df = df.sort_values(["region_id", "target_year"]).reset_index(drop=True)

    for region_id, region_df in df.groupby("region_id"):
        plot_region_timeseries(region_df, region_id, args.output_dir)

    for metric in METRICS:
        plot_metric_across_regions(df, metric, args.output_dir)

    print(f"Figures written to: {os.path.abspath(args.output_dir)}")
    print(f"Region figures: {df['region_id'].nunique()}")
    print(f"Cross-region metric figures: {len(METRICS)}")


if __name__ == "__main__":
    main()
