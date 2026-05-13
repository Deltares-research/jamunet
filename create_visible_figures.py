import argparse
import os

import numpy as np
import tifffile
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create original_visible and preprocesses_visible PNG figures scaled to 0-255."
    )
    parser.add_argument("--data-root", default="data/satellite_01", help="Dataset root.")
    parser.add_argument(
        "--collection",
        default="JRC_GSW1_4_MonthlyHistory",
        help="Collection prefix used in region folder names.",
    )
    parser.add_argument("--year", type=int, default=2021, help="Year to export.")
    parser.add_argument("--month", type=int, default=3, help="Month to export.")
    parser.add_argument("--day", type=int, default=1, help="Day to export.")
    parser.add_argument("--start-year", type=int, default=None, help="Start year for bulk export.")
    parser.add_argument("--end-year", type=int, default=None, help="End year for bulk export.")
    parser.add_argument(
        "--months",
        default="1,2,3,4",
        help="Comma-separated months for bulk export, e.g. 1,2,3,4.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/visible_figures",
        help="Root output folder where figure sets are written.",
    )
    return parser.parse_args()


def scale_to_0_255(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    min_val = float(arr[finite].min())
    max_val = float(arr[finite].max())
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr - min_val) / (max_val - min_val)
    scaled = np.clip(scaled * 255.0, 0.0, 255.0)
    return scaled.astype(np.uint8)


def list_region_folders(base_dir: str, collection: str):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Missing folder: {base_dir}")

    folders = []
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith(f"{collection}_eval_r"):
            folders.append(name)
    return folders


def export_set(source_root: str, region_folders, file_name: str, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    written = 0

    for folder in region_folders:
        region_id = folder.replace("JRC_GSW1_4_MonthlyHistory_", "")
        tif_path = os.path.join(source_root, folder, file_name)
        if not os.path.exists(tif_path):
            print(f"Skip missing file: {tif_path}")
            continue

        img = tifffile.imread(tif_path)
        img_u8 = scale_to_0_255(img)

        out_name = f"{prefix}_{region_id}_{file_name.replace('.tif', '.png')}"
        out_path = os.path.join(out_dir, out_name)
        Image.fromarray(img_u8).save(out_path)
        written += 1

    return written


def main():
    args = parse_args()

    original_root = os.path.join(args.data_root, "original")
    preprocessed_root = os.path.join(args.data_root, "preprocessed")
    region_folders = list_region_folders(original_root, args.collection)

    original_visible_dir = os.path.join(args.output_root, "original_visible")
    preprocesses_visible_dir = os.path.join(args.output_root, "preprocesses_visible")

    original_written = 0
    preprocesses_written = 0

    if args.start_year is not None or args.end_year is not None:
        if args.start_year is None or args.end_year is None:
            raise ValueError("Both --start-year and --end-year are required for bulk export.")
        if args.end_year < args.start_year:
            raise ValueError("--end-year must be >= --start-year.")

        months = [int(m.strip()) for m in args.months.split(",") if m.strip()]
        for m in months:
            if m < 1 or m > 12:
                raise ValueError("Months in --months must be between 1 and 12.")

        dates = [(y, m) for y in range(args.start_year, args.end_year + 1) for m in months]
    else:
        dates = [(args.year, args.month)]

    for year, month in dates:
        date_file = f"{year}_{month:02d}_{args.day:02d}"
        for folder in region_folders:
            region_id = folder.replace(f"{args.collection}_", "")
            file_name = f"{date_file}_{region_id}.tif"

            original_written += export_set(
                source_root=original_root,
                region_folders=[folder],
                file_name=file_name,
                out_dir=original_visible_dir,
                prefix="original_visible",
            )
            preprocesses_written += export_set(
                source_root=preprocessed_root,
                region_folders=[folder],
                file_name=file_name,
                out_dir=preprocesses_visible_dir,
                prefix="preprocesses_visible",
            )

    print(f"original_visible files written: {original_written}")
    print(f"preprocesses_visible files written: {preprocesses_written}")
    print(f"Output root: {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()
