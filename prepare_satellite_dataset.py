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


def index_available_regions(data_root, collection):
    """Index regions that actually have available TIFF images on disk."""
    original_root = os.path.join(data_root, "original")
    available = {}
    if not os.path.isdir(original_root):
        return available

    for folder_name in os.listdir(original_root):
        folder_path = os.path.join(original_root, folder_name)
        if not (os.path.isdir(folder_path) and folder_name.startswith(f"{collection}_")):
            continue

        region_id = extract_region_id_from_folder(folder_name, collection)
        tif_count = len([name for name in os.listdir(folder_path) if name.endswith(".tif")])
        if tif_count > 0:
            available[region_id] = {
                "folder": folder_name,
                "image_count": tif_count,
            }

    return available


def _is_valid_polygon(coords):
    return isinstance(coords, list) and len(coords) >= 4


def _bbox_from_polygon(coords):
    lon_values = [float(p[0]) for p in coords[:-1]]
    lat_values = [float(p[1]) for p in coords[:-1]]
    return min(lon_values), max(lon_values), min(lat_values), max(lat_values)


def _build_bbox_polygon(lon_min, lon_max, lat_min, lat_max):
    return [
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
        [lon_max, lat_min],
    ]


def infer_rotation_angle_from_shape(image_shape):
    """Infer ccw rotation angle needed to align flow top-to-bottom from source image shape."""
    ratio_h_w = round(float(image_shape[0]) / float(image_shape[1]), 2)
    if ratio_h_w == 2:
        return 0.0
    if ratio_h_w == 1:
        return 45.0
    if ratio_h_w == 0.5:
        return 90.0
    return None


def infer_flow_heading_from_image_shape(image_shape):
    """Convert inferred ccw rotation angle to downstream heading (clockwise from north)."""
    angle_ccw = infer_rotation_angle_from_shape(image_shape)
    if angle_ccw is None:
        return None
    return (180.0 + angle_ccw) % 360.0


def extract_region_id_from_folder(region_folder, collection):
    prefix = f"{collection}_"
    if region_folder.startswith(prefix):
        return region_folder.replace(prefix, "")
    return region_folder


def get_region_image_shape(data_root, collection, region_key):
    original_root = os.path.join(data_root, "original")
    if not os.path.isdir(original_root):
        return None

    folder_candidates = []
    if region_key.startswith(f"{collection}_"):
        folder_candidates.append(region_key)
    folder_candidates.append(f"{collection}_{region_key}")

    for folder_name in folder_candidates:
        folder_path = os.path.join(original_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        tif_names = sorted(name for name in os.listdir(folder_path) if name.endswith(".tif"))
        if not tif_names:
            continue
        sample_path = os.path.join(folder_path, tif_names[0])
        with tifffile.TiffFile(sample_path) as tif:
            return tif.pages[0].shape

    return None


def infer_heading_for_region_key(data_root, collection, region_key):
    shape = get_region_image_shape(data_root, collection, region_key)
    if shape is None:
        return None
    return infer_flow_heading_from_image_shape(shape)


def load_heading_overrides(data_root):
    """Load optional per-region heading overrides from JSON."""
    overrides_path = os.path.join(data_root, "regions", "model_ready_heading_overrides.json")
    if not os.path.exists(overrides_path):
        return {}

    with open(overrides_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Heading override file must be a JSON object mapping region IDs to heading degrees: {overrides_path}"
        )

    return {str(key): float(value) for key, value in payload.items()}


def heading_from_vector(east_km, north_km):
    if abs(east_km) < 1e-9 and abs(north_km) < 1e-9:
        return None
    return float((math.degrees(math.atan2(east_km, north_km)) + 360.0) % 360.0)


def infer_headings_from_centerline(metadata):
    """Infer per-region heading from local tangent of nearby region centers."""
    points = []
    for item in metadata:
        region_id = item.get("region_id")
        if not region_id:
            continue
        if "center_lat" not in item or "center_lon" not in item:
            continue
        points.append(
            {
                "region_id": str(region_id),
                "center_lat": float(item["center_lat"]),
                "center_lon": float(item["center_lon"]),
            }
        )

    if len(points) < 2:
        return {}

    mean_lat = float(sum(p["center_lat"] for p in points) / len(points))
    cos_mean_lat = max(math.cos(math.radians(mean_lat)), 1e-6)

    coords = np.array(
        [
            [p["center_lon"] * 111.32 * cos_mean_lat, p["center_lat"] * 111.32]
            for p in points
        ],
        dtype=np.float64,
    )

    headings = {}
    n_points = len(points)
    k = min(5, n_points)

    for i, point in enumerate(points):
        delta = coords - coords[i]
        dist2 = np.sum(delta * delta, axis=1)
        neighbor_idx = np.argsort(dist2)[:k]
        neighborhood = coords[neighbor_idx]
        if neighborhood.shape[0] < 2:
            continue

        neighborhood_centered = neighborhood - neighborhood.mean(axis=0)
        cov = np.cov(neighborhood_centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        principal_axis = eig_vecs[:, int(np.argmax(eig_vals))]

        heading = heading_from_vector(float(principal_axis[0]), float(principal_axis[1]))
        if heading is not None:
            headings[point["region_id"]] = heading

    return headings


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


def load_heading_map(data_root, collection):
    candidates = [
        os.path.join(data_root, "regions", "region_catalog.json"),
        os.path.join(data_root, "regions", "eval_reaches.json"),
    ]
    metadata_path = next((path for path in candidates if os.path.exists(path)), None)
    if metadata_path is None:
        metadata = []
    else:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    heading_map = {}

    centerline_heading_map = infer_headings_from_centerline(metadata)

    for item in metadata:
        region_id = item.get("region_id")
        source_region_id = item.get("source_region_id")
        source_folder = item.get("source_folder")

        heading = item.get("flow_heading_deg")
        if heading is None and region_id in centerline_heading_map:
            heading = centerline_heading_map[region_id]
        if heading is None:
            for key in [source_region_id, region_id, source_folder]:
                if key:
                    inferred = infer_heading_for_region_key(data_root, collection, key)
                    if inferred is not None:
                        heading = inferred
                        break
        if heading is None:
            heading = 180.0

        heading = float(heading)

        for key in [region_id, source_region_id]:
            if key:
                heading_map[key] = heading

        if source_folder and source_folder.startswith(f"{collection}_"):
            source_folder_region_id = extract_region_id_from_folder(source_folder, collection)
            heading_map[source_folder_region_id] = heading

    original_root = os.path.join(data_root, "original")
    if os.path.isdir(original_root):
        for folder_name in os.listdir(original_root):
            folder_path = os.path.join(original_root, folder_name)
            if not (os.path.isdir(folder_path) and folder_name.startswith(f"{collection}_")):
                continue

            region_id = extract_region_id_from_folder(folder_name, collection)
            if region_id in heading_map:
                continue

            inferred = infer_heading_for_region_key(data_root, collection, region_id)
            if inferred is not None:
                heading_map[region_id] = float(inferred)

    for key, heading in load_heading_overrides(data_root).items():
        heading_map[key] = float(heading)

    return heading_map


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
                    "source_folder": item.get("source_folder"),
                    "available_image_count": item.get("available_image_count"),
                    "footprint_source": item.get("footprint_source"),
                    "flow_heading_deg": item["flow_heading_deg"],
                    "model_ready_height_px": item["model_ready_height_px"],
                    "model_ready_width_px": item["model_ready_width_px"],
                    "pixel_size_m": item["pixel_size_m"],
                    "model_ready_length_km": item["model_ready_length_km"],
                    "model_ready_width_km": item["model_ready_width_km"],
                    "model_ready_area_km2": item["model_ready_area_km2"],
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
    model_ready_area_km2 = model_ready_length_km * model_ready_width_km
    heading_map = load_heading_map(args.data_root, args.collection)
    available_regions = index_available_regions(args.data_root, args.collection)

    records = []
    for item in source_catalog:
        source_region_id = item.get("region_id", "")
        region_id = source_region_id.replace("eval_", "") if source_region_id.startswith("eval_") else source_region_id

        # Keep only regions that actually have available source TIFFs.
        if region_id not in available_regions:
            continue

        polygon = item.get("polygon_lonlat")
        if not _is_valid_polygon(polygon):
            lon_min = item.get("lon_min")
            lon_max = item.get("lon_max")
            lat_min = item.get("lat_min")
            lat_max = item.get("lat_max")
            if None not in [lon_min, lon_max, lat_min, lat_max]:
                polygon = _build_bbox_polygon(float(lon_min), float(lon_max), float(lat_min), float(lat_max))
            else:
                # Skip records with no trustworthy footprint geometry.
                continue

        lon_min, lon_max, lat_min, lat_max = _bbox_from_polygon(polygon)

        center_lat = float(item["center_lat"])
        center_lon = float(item["center_lon"])
        flow_heading_deg = item.get("flow_heading_deg")
        if flow_heading_deg is None:
            flow_heading_deg = heading_map.get(source_region_id)
        if flow_heading_deg is None:
            flow_heading_deg = heading_map.get(region_id)
        if flow_heading_deg is None:
            flow_heading_deg = 180.0
        flow_heading_deg = float(flow_heading_deg)

        records.append(
            {
                "region_id": region_id,
                "source_region_id": source_region_id,
                "source_folder": available_regions[region_id]["folder"],
                "available_image_count": int(available_regions[region_id]["image_count"]),
                "center_lat": center_lat,
                "center_lon": center_lon,
                "flow_heading_deg": flow_heading_deg,
                "model_ready_height_px": int(args.target_height),
                "model_ready_width_px": int(args.target_width),
                "pixel_size_m": float(args.pixel_size_m),
                "model_ready_length_km": float(model_ready_length_km),
                "model_ready_width_km": float(model_ready_width_km),
                "model_ready_area_km2": float(model_ready_area_km2),
                "footprint_source": "region_catalog_available_images",
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
    heading_map = load_heading_map(args.data_root, args.collection)

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
