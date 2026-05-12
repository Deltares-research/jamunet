import os
import subprocess
import sys

import numpy as np
import pandas as pd
import tifffile
import torch

from model.st_unet.st_unet import UNet3D_full


def main():
    device = "cpu"

    checkpoint = (
        "model/models_trained/"
        "UNet3D_full_bloss_spatial_month3_4dwns_8ihiddim_3ker_"
        "maxpool_0.05ilr_15step_0.75gamma_16batch_300epochs_0.5wthr.pth"
    )
    # ── Change TARGET_YEAR (1992–2021) and REGION to control the prediction. ──
    TARGET_YEAR = 2021
    # Available regions (from north to south along the Brahmaputra):
    #   testing_r1      lat 23.82–24.36  (current test region)
    #   validation_r1   lat 24.36–24.90  (immediately upstream/north)
    #   training_r1     lat 24.90–25.44  (further upstream)
    #   training_r2     lat 24.97–25.51
    #   training_r3     lat 25.09–25.62
    #   training_r4     lat 25.21–25.75
    #   training_r5/r6  lat 25.52–26.20  (wider tiles, further upstream)
    REGION = "validation_r1"
    # ──────────────────────────────────────────────────────────────────────────

    output_dir = "outputs/unet3d_full_example"
    dataset_name = f"JRC_GSW1_4_MonthlyHistory_{REGION}"

    sample_dir = f"data/satellite/dataset_month3/{dataset_name}"
    original_dir = f"data/satellite/original/{dataset_name}"
    braided_python = (
        r"C:\Users\chavarri\AppData\Local\miniforge3\envs\braided\python.exe"
    )
    all_tifs = sorted([f for f in os.listdir(sample_dir) if f.endswith(".tif")])
    if len(all_tifs) < 5:
        raise RuntimeError("Need at least 5 TIFF files to build a 4->1 sample.")

    # Find the index of TARGET_YEAR and select it plus the 4 preceding images.
    tif_years = [int(f.split("_")[0]) for f in all_tifs]
    if TARGET_YEAR not in tif_years:
        raise ValueError(f"TARGET_YEAR {TARGET_YEAR} not found. Available: {tif_years}")
    target_idx = tif_years.index(TARGET_YEAR)
    if target_idx < 4:
        raise ValueError(f"Need at least 4 years before {TARGET_YEAR}. Earliest target is {tif_years[4]}.")
    selected = all_tifs[target_idx - 4 : target_idx + 1]
    print(f"Predicting year {TARGET_YEAR} using inputs: {[f.split('_')[0] for f in selected[:4]]}")
    good_images = []
    for tif_name in selected:
        year = int(tif_name.split("_")[0])
        img = tifffile.imread(os.path.join(sample_dir, tif_name)).astype(np.float32)

        # Match scaled class convention used in training code.
        img = img.astype(np.int32)
        img[img == 0] = -1
        img[img == 1] = 0
        img[img == 2] = 1

        avg_path = f"data/satellite/averages/average_{REGION}/average_{year}_{REGION}.csv"
        avg_img = pd.read_csv(avg_path, header=None).to_numpy(dtype=np.float32)

        # Replace no-data pixels (-1) with binary average values.
        img = np.where(img == -1, avg_img, img).astype(np.float32)
        good_images.append(img)

    sample_input_np = np.stack(good_images[:4], axis=0)
    sample_target_np = good_images[4]

    sample_input = torch.tensor(sample_input_np, dtype=torch.float32, device=device).unsqueeze(0)
    sample_target = torch.tensor(sample_target_np, dtype=torch.float32, device=device)

    model = UNet3D_full(in_channels=1, out_channels=1, init_features=8, temporal=3, seed=42).to(device)
    state_dict = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        prediction = model(sample_input)

    binary_prediction = (prediction >= 0.5).float()
    target_binary = sample_target

    os.makedirs(output_dir, exist_ok=True)
    prediction_path = os.path.join(output_dir, f"unet3d_full_prediction_probabilities_{REGION}_{TARGET_YEAR}.tif")
    binary_path = os.path.join(output_dir, f"unet3d_full_prediction_binary_{REGION}_{TARGET_YEAR}.tif")
    binary_vis_tif_path = os.path.join(output_dir, f"unet3d_full_prediction_binary_vis_{REGION}_{TARGET_YEAR}.tif")
    tifffile.imwrite(prediction_path, prediction.squeeze(0).cpu().numpy().astype(np.float32))
    tifffile.imwrite(binary_path, binary_prediction.squeeze(0).cpu().numpy().astype(np.uint8))
    binary_vis = (binary_prediction.squeeze(0).cpu().numpy().astype(np.uint8) * 255)
    tifffile.imwrite(binary_vis_tif_path, binary_vis)

    # Use the target year's original TIF as the georeferencing reference.
    ref_tif_name = next(
        (f for f in os.listdir(original_dir) if f.startswith(str(TARGET_YEAR))), None
    )
    ref_tif = os.path.join(original_dir, ref_tif_name) if ref_tif_name else None
    georef_prob_path = os.path.join(output_dir, f"unet3d_full_prediction_probabilities_georef_{REGION}_{TARGET_YEAR}.tif")
    georef_vis_path = os.path.join(output_dir, f"unet3d_full_prediction_binary_vis_georef_{REGION}_{TARGET_YEAR}.tif")
    if os.path.exists(braided_python) and ref_tif and os.path.exists(ref_tif):
        for src, dst in [(prediction_path, georef_prob_path), (binary_vis_tif_path, georef_vis_path)]:
            result = subprocess.run(
                [braided_python, "georeference_output.py", src, ref_tif, dst],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"Warning: georeferencing failed for {src}: {result.stderr.strip()}")
    else:
        print("Skipped georeferencing: braided env or reference TIF not found.")

    pred_water_pixels = int(binary_prediction.sum().item())
    target_water_pixels = int(target_binary.sum().item())

    intersection = int(((binary_prediction == 1) & (target_binary == 1)).sum().item())
    union = int(((binary_prediction == 1) | (target_binary == 1)).sum().item())
    iou = (intersection / union) if union > 0 else 0.0

    print("UNet3D_full example prediction completed.")
    print(f"Input tensor shape: {tuple(sample_input.shape)}")
    print(f"Raw prediction shape: {tuple(prediction.shape)}")
    print(f"Prediction range: min={prediction.min().item():.6f}, max={prediction.max().item():.6f}")
    print(f"Predicted water pixels (>=0.5): {pred_water_pixels}")
    print(f"Target water pixels: {target_water_pixels}")
    print(f"IoU (single sample, threshold 0.5): {iou:.6f}")
    print(f"Saved probability map: {prediction_path}")
    print(f"Saved binary mask: {binary_path}")
    print(f"Saved binary visualization TIFF (0/255): {binary_vis_tif_path}")
    print(f"Saved georeferenced probability map: {georef_prob_path}")
    print(f"Saved georeferenced binary visualization: {georef_vis_path}")


if __name__ == "__main__":
    main()
