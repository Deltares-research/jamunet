import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import tifffile
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare satellite_01 data for UNet3D_full inference.")
    parser.add_argument("--data-root", default="data/satellite_01", help="Root folder for neutral dataset.")
    parser.add_argument("--collection", default="JRC_GSW1_4_MonthlyHistory", help="Collection tag in folder names.")
    parser.add_argument("--target-height", type=int, default=1000, help="Output image height.")
    parser.add_argument("--target-width", type=int, default=500, help="Output image width.")
    parser.add_argument(
        "--align-flow-to-south",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate each reach so flow direction is top-to-bottom before crop/pad.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed/dataset files.")
    return parser.parse_args()


def list_region_dirs(base_dir, collection):
    region_dirs = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Missing folder: {base_dir}")
    for folder in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, folder)
        if os.path.isdir(full) and folder.startswith(f"{collection}_eval_r"):
            region_dirs.append(folder)
    return region_dirs


def center_crop_or_pad(image, target_h, target_w):
    src_h, src_w = image.shape[:2]

    if src_h >= target_h:
        top = (src_h - target_h) // 2
        cropped_h = image[top : top + target_h, :]
    else:
        pad_top = (target_h - src_h) // 2
        pad_bottom = target_h - src_h - pad_top
        cropped_h = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0)

    cur_h, cur_w = cropped_h.shape
    if cur_w >= target_w:
        left = (cur_w - target_w) // 2
        out = cropped_h[:, left : left + target_w]
    else:
        pad_left = (target_w - cur_w) // 2
        pad_right = target_w - cur_w - pad_left
        out = np.pad(cropped_h, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)

    return out


def load_heading_map(data_root):
    metadata_path = os.path.join(data_root, "regions", "eval_reaches.json")
    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {item["region_id"]: float(item.get("flow_heading_deg", 180.0)) for item in data}


def rotate_to_south(image, flow_heading_deg):
    # flow_heading_deg is clockwise from north; southward is 180 deg.
    angle_ccw = flow_heading_deg - 180.0
    pil_image = Image.fromarray(image)
    rotated = pil_image.rotate(angle=angle_ccw, resample=Image.NEAREST, expand=True, fillcolor=0)
    return np.array(rotated)


def preprocess_images(args, region_dirs):
    original_root = os.path.join(args.data_root, "original")
    preprocessed_root = os.path.join(args.data_root, "preprocessed")
    os.makedirs(preprocessed_root, exist_ok=True)
    heading_map = load_heading_map(args.data_root)

    total = 0
    for region_folder in region_dirs:
        region_id = region_folder.replace(f"{args.collection}_", "")
        src_dir = os.path.join(original_root, region_folder)
        dst_dir = os.path.join(preprocessed_root, region_folder)
        os.makedirs(dst_dir, exist_ok=True)

        for file_name in sorted(os.listdir(src_dir)):
            if not file_name.endswith(".tif"):
                continue

            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            if os.path.exists(dst_path) and not args.overwrite:
                continue

            image = tifffile.imread(src_path)
            if args.align_flow_to_south and region_id in heading_map:
                image = rotate_to_south(image, heading_map[region_id])

            processed = center_crop_or_pad(image, args.target_height, args.target_width).astype(np.uint8)
            tifffile.imwrite(dst_path, processed)
            total += 1

    print(f"Preprocessed TIFFs written: {total}")


def build_dataset_month_folders(args, region_dirs):
    preprocessed_root = os.path.join(args.data_root, "preprocessed")
    total = 0

    for month in [1, 2, 3, 4]:
        month_dir = os.path.join(args.data_root, f"dataset_month{month}")
        os.makedirs(month_dir, exist_ok=True)

        for region_folder in region_dirs:
            src_dir = os.path.join(preprocessed_root, region_folder)
            dst_dir = os.path.join(month_dir, region_folder)
            os.makedirs(dst_dir, exist_ok=True)

            for file_name in sorted(os.listdir(src_dir)):
                if not file_name.endswith(".tif"):
                    continue

                parts = file_name.split("_")
                file_month = int(parts[1])
                if file_month != month:
                    continue

                src_path = os.path.join(src_dir, file_name)
                dst_path = os.path.join(dst_dir, file_name)
                if os.path.exists(dst_path) and not args.overwrite:
                    continue

                shutil.copy2(src_path, dst_path)
                total += 1

    print(f"Dataset-month TIFFs copied: {total}")


def average_for_year(data_root, region_folder, year):
    monthly_images = []
    for month in [1, 2, 3, 4]:
        tif_path = os.path.join(
            data_root,
            f"dataset_month{month}",
            region_folder,
            f"{year}_{month:02d}_01_{region_folder.split('_')[-2]}_{region_folder.split('_')[-1]}.tif",
        )
        # region id is eval_rXX, reconstructed from folder suffix
        if not os.path.exists(tif_path):
            alt_name = f"{year}_{month:02d}_01_{region_folder.replace('JRC_GSW1_4_MonthlyHistory_', '')}.tif"
            tif_path = os.path.join(data_root, f"dataset_month{month}", region_folder, alt_name)

        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"Missing month image for average: {tif_path}")

        image = tifffile.imread(tif_path).astype(np.float32)
        image = np.where(image == 0, np.nan, image)
        image = np.where(image == 1, 0, image)
        image = np.where(image == 2, 1, image)
        monthly_images.append(image)

    avg = np.nanmean(monthly_images, axis=0)
    avg = np.where(np.isnan(avg), 0, avg)
    return (avg > 0.5).astype(np.float32)


def build_averages(args, region_dirs):
    averages_root = os.path.join(args.data_root, "averages")
    os.makedirs(averages_root, exist_ok=True)

    total = 0
    for region_folder in region_dirs:
        region_id = region_folder.replace(f"{args.collection}_", "")
        output_dir = os.path.join(averages_root, f"average_{region_id}")
        os.makedirs(output_dir, exist_ok=True)

        month3_dir = os.path.join(args.data_root, "dataset_month3", region_folder)
        years = sorted(
            int(file_name.split("_")[0])
            for file_name in os.listdir(month3_dir)
            if file_name.endswith(".tif")
        )

        for year in years:
            out_csv = os.path.join(output_dir, f"average_{year}_{region_id}.csv")
            if os.path.exists(out_csv) and not args.overwrite:
                continue

            avg = average_for_year(args.data_root, region_folder, year)
            pd.DataFrame(avg).to_csv(out_csv, index=False, header=False)
            total += 1

    print(f"Average CSVs written: {total}")


def main():
    args = parse_args()
    region_dirs = list_region_dirs(os.path.join(args.data_root, "original"), args.collection)
    if not region_dirs:
        raise RuntimeError("No eval_rXX region folders found under original/")

    preprocess_images(args, region_dirs)
    build_dataset_month_folders(args, region_dirs)
    build_averages(args, region_dirs)
    print("satellite_01 preparation completed.")


if __name__ == "__main__":
    main()