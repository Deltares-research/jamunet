import argparse
import json
import math
import os
import re
import shutil

import numpy as np
import pandas as pd
import tifffile
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare satellite data for UNet3D_full inference.")
    parser.add_argument("--data-root", default="data/satellite", help="Root folder for dataset.")
    parser.add_argument("--collection", default="JRC_GSW1_4_MonthlyHistory", help="Collection tag in folder names.")
    parser.add_argument("--target-height", type=int, default=1000, help="Output image height.")
    parser.add_argument("--target-width", type=int, default=500, help="Output image width.")
    parser.add_argument(
        "--pixel-size-m",
        type=float,
        default=60.0,
        help="Pixel size in meters used to derive model-ready physical footprint.",
    )
    parser.add_argument(
        "--align-flow-to-south",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate each reach so flow direction is top-to-bottom before crop/pad.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed/dataset files.")
    parser.add_argument(
        "--model-ready-catalog-only",
        action="store_true",
        help="Only write model-ready region catalogs and skip preprocessing/dataset/averages generation.",
    )
    return parser.parse_args()


def list_region_dirs(base_dir, collection):
    region_dirs = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Missing folder: {base_dir}")
    for folder in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, folder)
        if os.path.isdir(full) and folder.startswith(f"{collection}_"):
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
    candidates = [
        os.path.join(data_root, "regions", "region_catalog.json"),
        os.path.join(data_root, "regions", "eval_reaches.json"),
    ]
    metadata_path = next((path for path in candidates if os.path.exists(path)), None)
    if metadata_path is None:
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


def clamp(x, x_min, x_max):
    return max(x_min, min(x, x_max))


def km_offsets_to_latlon(center_lat, center_lon, east_km, north_km):
    dlat = north_km / 111.32
    cos_lat = max(math.cos(math.radians(center_lat)), 1e-6)
    dlon = east_km / (111.32 * cos_lat)
    lat = clamp(center_lat + dlat, -89.9, 89.9)
    lon = clamp(center_lon + dlon, -179.9, 179.9)
    return lon, lat


def build_rotated_polygon(center_lat, center_lon, u_east, u_north, tile_length_km, tile_width_km):
    # v is the local cross-section unit vector (left side looking downstream).
    v_east = -u_north
    v_north = u_east

    half_len = tile_length_km / 2.0
    half_wid = tile_width_km / 2.0

    corners = [
        (+half_len, +half_wid),
        (+half_len, -half_wid),
        (-half_len, -half_wid),
        (-half_len, +half_wid),
    ]

    polygon = []
    for along_km, across_km in corners:
        east_km = along_km * u_east + across_km * v_east
        north_km = along_km * u_north + across_km * v_north
        lon, lat = km_offsets_to_latlon(center_lat, center_lon, east_km, north_km)
        polygon.append([lon, lat])

    polygon.append(polygon[0])
    return polygon


def write_geojson(records, geojson_path):
    features = []
    for item in records:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "region_id": item["region_id"],
                    "source_region_id": item["source_region_id"],
                    "flow_heading_deg": item["flow_heading_deg"],
                    "model_ready_height_px": item["model_ready_height_px"],
                    "model_ready_width_px": item["model_ready_width_px"],
                    "pixel_size_m": item["pixel_size_m"],
                    "model_ready_length_km": item["model_ready_length_km"],
                    "model_ready_width_km": item["model_ready_width_km"],
                    "center_lat": item["center_lat"],
                    "center_lon": item["center_lon"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [item["polygon_lonlat"]],
                },
            }
        )

    payload = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(geojson_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_model_ready_region_catalog(args):
    regions_dir = os.path.join(args.data_root, "regions")
    os.makedirs(regions_dir, exist_ok=True)

    source_catalog_path = os.path.join(regions_dir, "region_catalog.json")
    source_catalog = []

    model_ready_json = os.path.join(regions_dir, "region_catalog_model_ready.json")
    model_ready_geojson = os.path.join(regions_dir, "region_catalog_model_ready.geojson")

    if os.path.exists(source_catalog_path):
        with open(source_catalog_path, "r", encoding="utf-8") as file:
            source_catalog = json.load(file)
    else:
        print(f"Source catalog not found, using region-id fallback: {source_catalog_path}")
        source_catalog = _build_source_catalog_from_region_ids(args)
        if not source_catalog:
            with open(model_ready_json, "w", encoding="utf-8") as file:
                json.dump([], file, indent=2)
            write_geojson([], model_ready_geojson)
            print("No region information available to build model-ready catalog.")
            print(f"Empty model-ready region catalog written: {model_ready_json}")
            print(f"Empty model-ready region polygons written: {model_ready_geojson}")
            return

    model_ready_length_km = (args.target_height * args.pixel_size_m) / 1000.0
    model_ready_width_km = (args.target_width * args.pixel_size_m) / 1000.0

    records = []
    for item in source_catalog:
        source_region_id = item.get("region_id", "")
        region_id = source_region_id.replace("eval_", "") if source_region_id.startswith("eval_") else source_region_id

        center_lat = float(item["center_lat"])
        center_lon = float(item["center_lon"])
        flow_heading_deg = float(item.get("flow_heading_deg", 180.0))

        heading_rad = math.radians(flow_heading_deg)
        u_east = math.sin(heading_rad)
        u_north = math.cos(heading_rad)

        polygon = build_rotated_polygon(
            center_lat=center_lat,
            center_lon=center_lon,
            u_east=u_east,
            u_north=u_north,
            tile_length_km=model_ready_length_km,
            tile_width_km=model_ready_width_km,
        )

        lon_values = [p[0] for p in polygon[:-1]]
        lat_values = [p[1] for p in polygon[:-1]]
        lon_min, lon_max = min(lon_values), max(lon_values)
        lat_min, lat_max = min(lat_values), max(lat_values)

        records.append(
            {
                "region_id": region_id,
                "source_region_id": source_region_id,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "flow_heading_deg": flow_heading_deg,
                "model_ready_height_px": int(args.target_height),
                "model_ready_width_px": int(args.target_width),
                "pixel_size_m": float(args.pixel_size_m),
                "model_ready_length_km": float(model_ready_length_km),
                "model_ready_width_km": float(model_ready_width_km),
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "polygon_lonlat": polygon,
            }
        )

    with open(model_ready_json, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    write_geojson(records, model_ready_geojson)

    print(f"Model-ready region catalog written: {model_ready_json}")
    print(f"Model-ready region polygons written: {model_ready_geojson}")


def _parse_region_id_center(region_id):
    # Expected formats: lat24p6515_lon88p0207, latm24p6515_lonm88p0207
    pattern = re.compile(r"^(latm|lat)(\d+p\d+)_(lonm|lon)(\d+p\d+)$")
    match = pattern.match(region_id)
    if not match:
        return None

    lat_prefix, lat_token, lon_prefix, lon_token = match.groups()
    lat = float(lat_token.replace("p", "."))
    lon = float(lon_token.replace("p", "."))

    if lat_prefix == "latm":
        lat *= -1.0
    if lon_prefix == "lonm":
        lon *= -1.0

    return lat, lon


def _collect_region_ids_from_folders(args):
    region_ids = set()
    candidate_roots = [
        os.path.join(args.data_root, "original"),
        os.path.join(args.data_root, "preprocessed"),
        os.path.join(args.data_root, "dataset_month1"),
    ]

    for root in candidate_roots:
        if not os.path.isdir(root):
            continue
        for folder in os.listdir(root):
            full = os.path.join(root, folder)
            if not os.path.isdir(full):
                continue
            if folder.startswith(f"{args.collection}_"):
                region_ids.add(folder.replace(f"{args.collection}_", ""))

    return sorted(region_ids)


def _build_source_catalog_from_region_ids(args):
    source_catalog = []
    for region_id in _collect_region_ids_from_folders(args):
        parsed = _parse_region_id_center(region_id)
        if parsed is None:
            continue
        center_lat, center_lon = parsed
        source_catalog.append(
            {
                "region_id": region_id,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "flow_heading_deg": 180.0,
            }
        )

    return source_catalog


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
    region_id = region_folder.replace("JRC_GSW1_4_MonthlyHistory_", "")
    monthly_images = []
    for month in [1, 2, 3, 4]:
        tif_path = os.path.join(
            data_root,
            f"dataset_month{month}",
            region_folder,
            f"{year}_{month:02d}_01_{region_id}.tif",
        )
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

    if args.model_ready_catalog_only:
        build_model_ready_region_catalog(args)
        print("Model-ready catalog generation completed.")
        return

    region_dirs = list_region_dirs(os.path.join(args.data_root, "original"), args.collection)
    if not region_dirs:
        raise RuntimeError(f"No {args.collection}_* region folders found under original/")

    preprocess_images(args, region_dirs)
    build_model_ready_region_catalog(args)
    build_dataset_month_folders(args, region_dirs)
    build_averages(args, region_dirs)
    print("satellite preparation completed.")


if __name__ == "__main__":
    main()
