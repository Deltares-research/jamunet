"""Run UNet3D_full inference for coordinate-based regions.

UNet3D_full is the best model available in this repository at the moment.
If you want to use a different model, you can change the checkpoint or swap
the model implementation while keeping the rest of the workflow unchanged.
"""

import os
import sys
import argparse
import subprocess
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.st_unet.st_unet import UNet3D_full


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "model"
    / "models_trained"
    / "UNet3D_full_bloss_spatial_month3_4dwns_8ihiddim_3ker_maxpool_0.05ilr_15step_0.75gamma_16batch_300epochs_0.5wthr.pth"
)


def _center_crop_or_pad(image: np.ndarray, target_shape, fill_value=0):
    target_h, target_w = target_shape
    src_h, src_w = image.shape

    if src_h >= target_h:
        top = (src_h - target_h) // 2
        image_h = image[top : top + target_h, :]
    else:
        pad_top = (target_h - src_h) // 2
        pad_bottom = target_h - src_h - pad_top
        image_h = np.pad(
            image,
            ((pad_top, pad_bottom), (0, 0)),
            mode="constant",
            constant_values=fill_value,
        )

    cur_h, cur_w = image_h.shape
    if cur_w >= target_w:
        left = (cur_w - target_w) // 2
        out = image_h[:, left : left + target_w]
    else:
        pad_left = (target_w - cur_w) // 2
        pad_right = target_w - cur_w - pad_left
        out = np.pad(
            image_h,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=fill_value,
        )

    return out


def _reverse_center_crop_or_pad(image: np.ndarray, target_shape, fill_value=0):
    target_h, target_w = target_shape
    src_h, src_w = image.shape

    if target_h >= src_h:
        canvas = np.full((target_h, src_w), fill_value=fill_value, dtype=image.dtype)
        top = (target_h - src_h) // 2
        canvas[top : top + src_h, :] = image
        image_h = canvas
    else:
        top = (src_h - target_h) // 2
        image_h = image[top : top + target_h, :]

    cur_h, cur_w = image_h.shape
    if target_w >= cur_w:
        canvas = np.full((cur_h, target_w), fill_value=fill_value, dtype=image_h.dtype)
        left = (target_w - cur_w) // 2
        canvas[:, left : left + cur_w] = image_h
        out = canvas
    else:
        left = (cur_w - target_w) // 2
        out = image_h[:, left : left + target_w]

    return out


def _load_region_flow_heading(data_root: str, region_id: str) -> float:
    metadata_path = os.path.join(data_root, "regions", "region_catalog_model_ready.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            "Missing required model-ready catalog for strict georeferencing: "
            f"{metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        reaches = json.load(file)

    if not isinstance(reaches, list):
        raise ValueError(f"Catalog payload must be a JSON array: {metadata_path}")
    if len(reaches) == 0:
        raise ValueError(
            "Model-ready catalog is empty; strict georeferencing requires populated metadata. "
            f"Regenerate: {metadata_path}"
        )

    sample_region_ids = []
    for reach in reaches:
        if len(sample_region_ids) < 5 and "region_id" in reach:
            sample_region_ids.append(str(reach.get("region_id")))
        if reach.get("region_id") != region_id:
            continue
        if "flow_heading_deg" not in reach:
            raise KeyError(
                "Missing flow_heading_deg in model-ready catalog record for region "
                f"{region_id} ({metadata_path})"
            )
        heading = float(reach["flow_heading_deg"])
        if not (0.0 <= heading < 360.0):
            raise ValueError(
                f"Invalid flow_heading_deg={heading} for region {region_id}; expected [0, 360)."
            )
        return heading

    raise KeyError(
        f"Region {region_id} not found in model-ready catalog: {metadata_path}. "
        f"Catalog records={len(reaches)} sample_region_ids={sample_region_ids}"
    )


def _restore_to_reference_space(image: np.ndarray, ref_shape, flow_heading_deg, binary=False):
    angle_ccw = flow_heading_deg - 180.0

    dummy = Image.fromarray(np.zeros(ref_shape, dtype=np.uint8))
    rot_dummy = dummy.rotate(angle=angle_ccw, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
    rot_h, rot_w = np.array(rot_dummy).shape

    on_rot_canvas = _reverse_center_crop_or_pad(image, (rot_h, rot_w), fill_value=0)
    resample_mode = Image.Resampling.NEAREST if binary else Image.Resampling.BILINEAR
    back_rotated = Image.fromarray(on_rot_canvas).rotate(
        angle=-angle_ccw,
        resample=resample_mode,
        expand=True,
        fillcolor=0,
    )
    back_arr = np.array(back_rotated)
    return _center_crop_or_pad(back_arr, ref_shape, fill_value=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UNet3D_full inference on one or multiple regions identified by coordinate slugs."
    )
    parser.add_argument("--data-root", default="data/satellite", help="Dataset root folder.")
    parser.add_argument("--collection", default="JRC_GSW1_4_MonthlyHistory", help="Collection name prefix used in folder names.")
    parser.add_argument("--region", default="lat24p6515_lon88p0207", help="Single region id (e.g., lat24p6515_lon88p0207).")
    parser.add_argument("--regions", default="", help="Comma-separated region ids for batch inference (e.g., lat24p6515_lon88p0207,lat24p4349_lon88p4594).")
    parser.add_argument("--month", type=int, default=3, choices=[1, 2, 3, 4], help="Dataset month folder to use.")
    parser.add_argument("--target-year", type=int, default=2021, help="Year to predict (uses previous 4 years as input).")
    parser.add_argument("--output-dir", default="outputs/run_example", help="Base output directory.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to pretrained model weights.",
    )
    parser.add_argument(
        "--braided-python",
        default=sys.executable,
        help="Python executable used for georeferencing helper (must have rasterio or GDAL).",
    )
    parser.add_argument("--skip-georef", action="store_true", help="Skip georeferenced output generation.")
    return parser.parse_args()


def _resolve_python_command(command_or_path: str):
    """Resolve either an absolute python path or a command available on PATH."""
    if os.path.isabs(command_or_path):
        return command_or_path if os.path.exists(command_or_path) else None
    return shutil.which(command_or_path)


def get_region_ids(region, regions):
    if regions.strip():
        return [r.strip() for r in regions.split(",") if r.strip()]
    return [region]


def run_single_region(model, device, args, region_id):
    dataset_name = f"{args.collection}_{region_id}"
    sample_dir = os.path.join(args.data_root, f"dataset_month{args.month}", dataset_name)
    original_dir = os.path.join(args.data_root, "original", dataset_name)
    averages_dir = os.path.join(args.data_root, "averages", f"average_{region_id}")

    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"Missing dataset folder: {sample_dir}")
    if not os.path.isdir(averages_dir):
        raise FileNotFoundError(f"Missing averages folder: {averages_dir}")

    all_tifs = sorted([f for f in os.listdir(sample_dir) if f.endswith(".tif")])
    if len(all_tifs) < 5:
        raise RuntimeError(f"Need at least 5 TIFF files in {sample_dir} to build a 4->1 sample.")

    tif_years = [int(f.split("_")[0]) for f in all_tifs]
    if args.target_year not in tif_years:
        raise ValueError(f"TARGET_YEAR {args.target_year} not found for {region_id}. Available: {tif_years}")

    target_idx = tif_years.index(args.target_year)
    if target_idx < 4:
        raise ValueError(
            f"Need at least 4 years before {args.target_year} for {region_id}. "
            f"Earliest target is {tif_years[4]}."
        )

    selected = all_tifs[target_idx - 4 : target_idx + 1]
    print(f"[{region_id}] Predicting year {args.target_year} using inputs: {[f.split('_')[0] for f in selected[:4]]}")

    good_images = []
    for tif_name in selected:
        year = int(tif_name.split("_")[0])
        img = tifffile.imread(os.path.join(sample_dir, tif_name)).astype(np.float32)

        img = img.astype(np.int32)
        img[img == 0] = -1
        img[img == 1] = 0
        img[img == 2] = 1

        avg_path = os.path.join(averages_dir, f"average_{year}_{region_id}.csv")
        if not os.path.exists(avg_path):
            raise FileNotFoundError(f"Missing average file: {avg_path}")
        avg_img = pd.read_csv(avg_path, header=None).to_numpy(dtype=np.float32)

        img = np.where(img == -1, avg_img, img).astype(np.float32)
        good_images.append(img)

    sample_input_np = np.stack(good_images[:4], axis=0)
    sample_target_np = good_images[4]

    sample_input = torch.tensor(sample_input_np, dtype=torch.float32, device=device).unsqueeze(0)
    sample_target = torch.tensor(sample_target_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        prediction = model(sample_input)

    binary_prediction = (prediction >= 0.5).float()

    region_output_dir = os.path.join(args.output_dir, region_id)
    os.makedirs(region_output_dir, exist_ok=True)

    prediction_path = os.path.join(
        region_output_dir,
        f"prediction_probabilities_{region_id}_{args.target_year}.tif",
    )
    binary_path = os.path.join(
        region_output_dir,
        f"prediction_binary_{region_id}_{args.target_year}.tif",
    )
    binary_vis_tif_path = os.path.join(
        region_output_dir,
        f"prediction_binary_vis_{region_id}_{args.target_year}.tif",
    )

    tifffile.imwrite(prediction_path, prediction.squeeze(0).cpu().numpy().astype(np.float32))
    tifffile.imwrite(binary_path, binary_prediction.squeeze(0).cpu().numpy().astype(np.uint8))
    binary_vis = binary_prediction.squeeze(0).cpu().numpy().astype(np.uint8) * 255
    tifffile.imwrite(binary_vis_tif_path, binary_vis)

    georef_prob_path = os.path.join(
        region_output_dir,
        f"prediction_probabilities_georef_{region_id}_{args.target_year}.tif",
    )
    georef_vis_path = os.path.join(
        region_output_dir,
        f"prediction_binary_vis_georef_{region_id}_{args.target_year}.tif",
    )

    georef_script = str(REPO_ROOT / "scripts" / "postprocessing" / "georeference_output.py")

    if not args.skip_georef:
        georef_success_count = 0
        if not os.path.isdir(original_dir):
            raise FileNotFoundError(
                f"Missing original folder required for strict georeferencing: {original_dir}"
            )

        target_tif_name = selected[4]
        original_tifs = sorted(f for f in os.listdir(original_dir) if f.endswith(".tif"))
        if target_tif_name not in original_tifs:
            raise FileNotFoundError(
                "Strict georeferencing requires exact reference TIFF name match. "
                f"Target={target_tif_name}, original_dir={original_dir}"
            )
        ref_tif_name = target_tif_name
        ref_tif = os.path.join(original_dir, ref_tif_name)

        flow_heading_deg = _load_region_flow_heading(args.data_root, region_id)
        georef_python = _resolve_python_command(args.braided_python)

        if not georef_python:
            raise RuntimeError(
                f"Missing Python executable for georeferencing helper: {args.braided_python}"
            )
        if not os.path.exists(georef_script):
            raise FileNotFoundError(f"Missing georeferencing helper script: {georef_script}")
        if not os.path.exists(ref_tif):
            raise FileNotFoundError(f"Missing reference TIFF: {ref_tif}")

        print(f"[{region_id}] Georeferencing reference TIFF: {ref_tif_name}")
        print(f"[{region_id}] Using flow_heading_deg={flow_heading_deg:.6f}")
        ref_shape = tifffile.imread(ref_tif).shape
        if len(ref_shape) < 2:
            raise ValueError(f"Unexpected reference TIFF shape for {ref_tif}: {ref_shape}")
        ref_shape_2d = (int(ref_shape[0]), int(ref_shape[1]))

        for src, dst, binary_out in [
            (prediction_path, georef_prob_path, False),
            (binary_vis_tif_path, georef_vis_path, True),
        ]:
            src_arr = tifffile.imread(src).squeeze()
            src_arr = _restore_to_reference_space(
                src_arr,
                ref_shape=ref_shape_2d,
                flow_heading_deg=flow_heading_deg,
                binary=binary_out,
            )

            tmp_src = os.path.join(region_output_dir, f"__tmp_mapspace_{os.path.basename(src)}")
            try:
                if binary_out:
                    tifffile.imwrite(tmp_src, src_arr.astype(np.uint8))
                else:
                    tifffile.imwrite(tmp_src, src_arr.astype(np.float32))

                result = subprocess.run(
                    [georef_python, georef_script, tmp_src, ref_tif, dst],
                    capture_output=True,
                    text=True,
                )
            finally:
                if os.path.exists(tmp_src):
                    os.remove(tmp_src)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Georeferencing failed for {src} -> {dst}. "
                    f"stderr={result.stderr.strip()} stdout={result.stdout.strip()}"
                )

            georef_success_count += 1
            if result.stdout.strip():
                print(result.stdout.strip())

    target_binary = sample_target
    pred_water_pixels = int(binary_prediction.sum().item())
    target_water_pixels = int(target_binary.sum().item())
    intersection = int(((binary_prediction == 1) & (target_binary == 1)).sum().item())
    union = int(((binary_prediction == 1) | (target_binary == 1)).sum().item())
    iou = (intersection / union) if union > 0 else 0.0

    print(f"[{region_id}] UNet3D_full inference completed.")
    print(f"[{region_id}] Input tensor shape: {tuple(sample_input.shape)}")
    print(f"[{region_id}] Raw prediction shape: {tuple(prediction.shape)}")
    print(f"[{region_id}] Prediction range: min={prediction.min().item():.6f}, max={prediction.max().item():.6f}")
    print(f"[{region_id}] Predicted water pixels (>=0.5): {pred_water_pixels}")
    print(f"[{region_id}] Target water pixels: {target_water_pixels}")
    print(f"[{region_id}] IoU (single sample, threshold 0.5): {iou:.6f}")
    print(f"[{region_id}] Saved probability map: {prediction_path}")
    print(f"[{region_id}] Saved binary mask: {binary_path}")
    print(f"[{region_id}] Saved binary visualization TIFF (0/255): {binary_vis_tif_path}")
    if not args.skip_georef:
        if georef_success_count != 2 or not os.path.exists(georef_prob_path) or not os.path.exists(georef_vis_path):
            raise RuntimeError(
                f"Strict georeferencing expected 2 outputs but got {georef_success_count} for {region_id}."
            )
        print(f"[{region_id}] Saved georeferenced probability map: {georef_prob_path}")
        print(f"[{region_id}] Saved georeferenced binary visualization: {georef_vis_path}")


def main():
    args = parse_args()
    region_ids = get_region_ids(args.region, args.regions)
    device = "cpu"

    checkpoint = args.checkpoint
    if not os.path.isabs(checkpoint):
        checkpoint = str(REPO_ROOT / checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = UNet3D_full(in_channels=1, out_channels=1, init_features=8, temporal=3, seed=42).to(device)
    state_dict = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Running inference for regions: {region_ids}")
    for region_id in region_ids:
        run_single_region(model, device, args, region_id)


if __name__ == "__main__":
    main()

