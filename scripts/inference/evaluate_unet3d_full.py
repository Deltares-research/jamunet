import argparse
import json
import os

import numpy as np
import pandas as pd
import tifffile
import torch

from model.st_unet.st_unet import UNet3D_full


DEFAULT_CHECKPOINT = (
    "model/models_trained/"
    "UNet3D_full_bloss_spatial_month3_4dwns_8ihiddim_3ker_"
    "maxpool_0.05ilr_15step_0.75gamma_16batch_300epochs_0.5wthr.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate UNet3D_full on one or more neutral regions/years and report classification metrics."
    )
    parser.add_argument("--data-root", default="data/satellite", help="Dataset root folder.")
    parser.add_argument(
        "--collection",
        default="JRC_GSW1_4_MonthlyHistory",
        help="Collection name prefix used in folder names.",
    )
    parser.add_argument("--region", default="lat24p6515_lon88p0207", help="Single region id, used when --regions is empty.")
    parser.add_argument(
        "--regions",
        default="",
        help="Comma-separated region ids for batch evaluation, e.g. lat24p6515_lon88p0207,lat24p4349_lon88p4594.",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="Dataset month folder to evaluate.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=None,
        help="Single target year to evaluate. If omitted, all valid years are evaluated.",
    )
    parser.add_argument(
        "--target-years",
        default="",
        help="Comma-separated target years. Overrides --target-year when provided.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Water threshold for binarizing predictions.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to pretrained model weights.")
    parser.add_argument(
        "--output-dir",
        default="outputs/run_example_evaluation",
        help="Directory where metrics reports are written.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save probability and binary prediction TIFFs for each evaluated sample.",
    )
    return parser.parse_args()


def get_region_ids(region, regions):
    if regions.strip():
        return [region_id.strip() for region_id in regions.split(",") if region_id.strip()]
    return [region]


def get_dataset_paths(data_root, collection, region_id, month):
    dataset_name = f"{collection}_{region_id}"
    sample_dir = os.path.join(data_root, f"dataset_month{month}", dataset_name)
    averages_dir = os.path.join(data_root, "averages", f"average_{region_id}")

    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"Missing dataset folder: {sample_dir}")
    if not os.path.isdir(averages_dir):
        raise FileNotFoundError(f"Missing averages folder: {averages_dir}")

    return dataset_name, sample_dir, averages_dir


def get_available_years(sample_dir):
    all_tifs = sorted(file_name for file_name in os.listdir(sample_dir) if file_name.endswith(".tif"))
    if len(all_tifs) < 5:
        raise RuntimeError(f"Need at least 5 TIFF files in {sample_dir} to build a 4->1 sample.")

    tif_years = [int(file_name.split("_")[0]) for file_name in all_tifs]
    return all_tifs, tif_years


def get_target_years(args, tif_years):
    if args.target_years.strip():
        years = [int(year.strip()) for year in args.target_years.split(",") if year.strip()]
    elif args.target_year is not None:
        years = [args.target_year]
    else:
        years = tif_years[4:]

    missing_years = [year for year in years if year not in tif_years]
    if missing_years:
        raise ValueError(f"Target years not found in dataset: {missing_years}. Available: {tif_years}")

    invalid_years = [year for year in years if tif_years.index(year) < 4]
    if invalid_years:
        earliest_valid_year = tif_years[4]
        raise ValueError(
            f"These target years do not have four previous inputs available: {invalid_years}. "
            f"Earliest valid target year is {earliest_valid_year}."
        )

    return years


def load_image_with_average(sample_dir, averages_dir, tif_name, region_id):
    year = int(tif_name.split("_")[0])
    image = tifffile.imread(os.path.join(sample_dir, tif_name)).astype(np.int32)
    image[image == 0] = -1
    image[image == 1] = 0
    image[image == 2] = 1

    avg_path = os.path.join(averages_dir, f"average_{year}_{region_id}.csv")
    if not os.path.exists(avg_path):
        raise FileNotFoundError(f"Missing average file: {avg_path}")

    average_image = pd.read_csv(avg_path, header=None).to_numpy(dtype=np.float32)
    return np.where(image == -1, average_image, image).astype(np.float32)


def build_sample(sample_dir, averages_dir, region_id, target_year):
    all_tifs, tif_years = get_available_years(sample_dir)
    if target_year not in tif_years:
        raise ValueError(f"Target year {target_year} not found for {region_id}. Available: {tif_years}")

    target_idx = tif_years.index(target_year)
    if target_idx < 4:
        raise ValueError(
            f"Need at least 4 years before {target_year} for {region_id}. Earliest target is {tif_years[4]}."
        )

    selected = all_tifs[target_idx - 4 : target_idx + 1]
    images = [load_image_with_average(sample_dir, averages_dir, tif_name, region_id) for tif_name in selected]
    sample_input_np = np.stack(images[:4], axis=0)
    sample_target_np = images[4]
    return sample_input_np, sample_target_np, selected


def compute_confusion(binary_prediction, target):
    tp = int(((binary_prediction == 1) & (target == 1)).sum().item())
    fp = int(((binary_prediction == 1) & (target == 0)).sum().item())
    tn = int(((binary_prediction == 0) & (target == 0)).sum().item())
    fn = int(((binary_prediction == 0) & (target == 1)).sum().item())
    return tp, fp, tn, fn


def compute_binary_metrics(prediction, target, threshold):
    binary_prediction = (prediction >= threshold).float()
    tp, fp, tn, fn = compute_confusion(binary_prediction, target)
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return binary_prediction, accuracy, precision, recall, f1_score, csi, tp, fp, tn, fn


def summarize_global_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "csi": csi,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_region_year(model, device, args, region_id, target_year):
    _, sample_dir, averages_dir = get_dataset_paths(args.data_root, args.collection, region_id, args.month)
    sample_input_np, sample_target_np, selected = build_sample(sample_dir, averages_dir, region_id, target_year)

    sample_input = torch.tensor(sample_input_np, dtype=torch.float32, device=device).unsqueeze(0)
    sample_target = torch.tensor(sample_target_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        prediction = model(sample_input).squeeze(0)

    binary_prediction, accuracy, precision, recall, f1_score, csi, tp, fp, tn, fn = compute_binary_metrics(
        prediction,
        sample_target,
        args.threshold,
    )

    result = {
        "region_id": region_id,
        "target_year": target_year,
        "input_years": ",".join(file_name.split("_")[0] for file_name in selected[:4]),
        "target_file": selected[4],
        "threshold": args.threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "csi": csi,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pred_water_pixels": int(binary_prediction.sum().item()),
        "target_water_pixels": int(sample_target.sum().item()),
        "prediction_min": float(prediction.min().item()),
        "prediction_max": float(prediction.max().item()),
    }

    if args.save_predictions:
        region_output_dir = os.path.join(args.output_dir, region_id)
        os.makedirs(region_output_dir, exist_ok=True)
        probability_path = os.path.join(
            region_output_dir,
            f"eval_probabilities_{region_id}_{target_year}.tif",
        )
        binary_path = os.path.join(
            region_output_dir,
            f"eval_binary_{region_id}_{target_year}.tif",
        )
        tifffile.imwrite(probability_path, prediction.cpu().numpy().astype(np.float32))
        tifffile.imwrite(binary_path, binary_prediction.cpu().numpy().astype(np.uint8))
        result["probability_path"] = probability_path
        result["binary_path"] = binary_path

    return result


def load_model(checkpoint, device):
    model = UNet3D_full(in_channels=1, out_channels=1, init_features=8, temporal=3, seed=42).to(device)
    state_dict = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    region_ids = get_region_ids(args.region, args.regions)
    device = "cpu"
    model = load_model(args.checkpoint, device)

    results = []
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    print(f"Evaluating regions: {region_ids}")
    for region_id in region_ids:
        _, sample_dir, _ = get_dataset_paths(args.data_root, args.collection, region_id, args.month)
        _, tif_years = get_available_years(sample_dir)
        target_years = get_target_years(args, tif_years)

        print(f"[{region_id}] Target years: {target_years}")
        for target_year in target_years:
            result = evaluate_region_year(model, device, args, region_id, target_year)
            results.append(result)
            total_tp += result["tp"]
            total_fp += result["fp"]
            total_tn += result["tn"]
            total_fn += result["fn"]

            print(
                f"[{region_id}] {target_year}: "
                f"accuracy={result['accuracy']:.4f}, precision={result['precision']:.4f}, "
                f"recall={result['recall']:.4f}, f1={result['f1_score']:.4f}, csi={result['csi']:.4f}"
            )

    results_df = pd.DataFrame(results)
    per_sample_csv = os.path.join(args.output_dir, "metrics_per_sample.csv")
    results_df.to_csv(per_sample_csv, index=False)

    macro_summary = {
        "accuracy": float(results_df["accuracy"].mean()),
        "precision": float(results_df["precision"].mean()),
        "recall": float(results_df["recall"].mean()),
        "f1_score": float(results_df["f1_score"].mean()),
        "csi": float(results_df["csi"].mean()),
    }
    global_summary = summarize_global_metrics(total_tp, total_fp, total_tn, total_fn)
    summary = {
        "num_samples": len(results),
        "regions": region_ids,
        "month": args.month,
        "threshold": args.threshold,
        "checkpoint": args.checkpoint,
        "macro_average": macro_summary,
        "global_pixel_metrics": global_summary,
        "per_sample_csv": per_sample_csv,
    }

    summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Evaluation completed.")
    print(f"Per-sample metrics: {per_sample_csv}")
    print(f"Summary metrics: {summary_path}")
    print(
        "Macro average -> "
        f"accuracy={macro_summary['accuracy']:.4f}, "
        f"precision={macro_summary['precision']:.4f}, "
        f"recall={macro_summary['recall']:.4f}, "
        f"f1={macro_summary['f1_score']:.4f}, "
        f"csi={macro_summary['csi']:.4f}"
    )
    print(
        "Global pixel metrics -> "
        f"accuracy={global_summary['accuracy']:.4f}, "
        f"precision={global_summary['precision']:.4f}, "
        f"recall={global_summary['recall']:.4f}, "
        f"f1={global_summary['f1_score']:.4f}, "
        f"csi={global_summary['csi']:.4f}"
    )


if __name__ == "__main__":
    main()