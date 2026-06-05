import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

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
    """Build heading map using only available original TIFF image shapes."""
    heading_map = {}
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

    return heading_map


def _center_crop_or_pad_constant(image, target_h, target_w, fill_value):
    src_h, src_w = image.shape[:2]

    if src_h >= target_h:
        top = (src_h - target_h) // 2
        cropped_h = image[top : top + target_h, :]
    else:
        pad_top = (target_h - src_h) // 2
        pad_bottom = target_h - src_h - pad_top
        cropped_h = np.pad(
            image,
            ((pad_top, pad_bottom), (0, 0)),
            mode="constant",
            constant_values=fill_value,
        )

    cur_h, cur_w = cropped_h.shape
    if cur_w >= target_w:
        left = (cur_w - target_w) // 2
        out = cropped_h[:, left : left + target_w]
    else:
        pad_left = (target_w - cur_w) // 2
        pad_right = target_w - cur_w - pad_left
        out = np.pad(
            cropped_h,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=fill_value,
        )

    return out


def _list_tif_files(folder_path):
    if not os.path.isdir(folder_path):
        return []
    return sorted(name for name in os.listdir(folder_path) if name.endswith(".tif"))


def _read_geotiff_model(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        height, width = page.shape
        tags = page.tags

        if 33550 in tags and 33922 in tags:
            scale = tags[33550].value
            tiepoint = tags[33922].value
            return {
                "kind": "scale_tiepoint",
                "width": int(width),
                "height": int(height),
                "sx": float(scale[0]),
                "sy": float(scale[1]),
                "x0": float(tiepoint[3]),
                "y0": float(tiepoint[4]),
            }

        if 34264 in tags:
            transform = [float(value) for value in tags[34264].value]
            if len(transform) != 16:
                raise ValueError(f"Unexpected ModelTransformationTag length in {tif_path}: {len(transform)}")
            return {
                "kind": "transform",
                "width": int(width),
                "height": int(height),
                "transform": transform,
            }

    raise KeyError(f"Missing georeferencing tags (33550/33922 or 34264) in {tif_path}")


def _pixel_to_lonlat(model, x_px, y_px):
    if model["kind"] == "scale_tiepoint":
        lon = model["x0"] + model["sx"] * float(x_px)
        lat = model["y0"] - model["sy"] * float(y_px)
        return lon, lat

    matrix = model["transform"]
    lon = matrix[0] * float(x_px) + matrix[1] * float(y_px) + matrix[3]
    lat = matrix[4] * float(x_px) + matrix[5] * float(y_px) + matrix[7]
    return lon, lat


def _polygon_from_model(model):
    width = model["width"]
    height = model["height"]
    corners_px = [
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    ]
    polygon = [[_pixel_to_lonlat(model, x_px, y_px)[0], _pixel_to_lonlat(model, x_px, y_px)[1]] for x_px, y_px in corners_px]
    polygon.append(polygon[0])
    return polygon


def _center_from_polygon(polygon):
    lon_min, lon_max, lat_min, lat_max = _bbox_from_polygon(polygon)
    center_lon = (lon_min + lon_max) / 2.0
    center_lat = (lat_min + lat_max) / 2.0
    return center_lat, center_lon


def _build_region_index_from_root(root_path, collection):
    index = {}
    if not os.path.isdir(root_path):
        return index

    for folder_name in sorted(os.listdir(root_path)):
        folder_path = os.path.join(root_path, folder_name)
        if not (os.path.isdir(folder_path) and folder_name.startswith(f"{collection}_")):
            continue

        tif_names = _list_tif_files(folder_path)
        if not tif_names:
            continue

        region_id = extract_region_id_from_folder(folder_name, collection)
        index[region_id] = {
            "folder": folder_name,
            "tif_names": tif_names,
            "sample_tif_path": os.path.join(folder_path, tif_names[0]),
        }

    return index


def _model_ready_data_roots(data_root):
    """Return model-ready roots, preferring the full dataset when present."""
    roots = []
    full_dataset_root = os.path.join(data_root, "dataset")
    if os.path.isdir(full_dataset_root):
        roots.append(full_dataset_root)

    for month in [1, 2, 3, 4]:
        month_root = os.path.join(data_root, f"dataset_month{month}")
        if os.path.isdir(month_root):
            roots.append(month_root)

    return roots


def _build_model_ready_region_index(data_root, collection):
    region_index = {}
    for root_path in _model_ready_data_roots(data_root):
        month_index = _build_region_index_from_root(root_path, collection)
        for region_id, item in month_index.items():
            if region_id not in region_index:
                region_index[region_id] = {
                    "folder": item["folder"],
                    "tif_names": set(),
                    "sample_tif_path": item["sample_tif_path"],
                }
            region_index[region_id]["tif_names"].update(item["tif_names"])

    for item in region_index.values():
        item["tif_names"] = sorted(item["tif_names"])

    return region_index


def _build_model_ready_pairs(data_root, collection):
    """Pair each model-ready TIFF with the same-name original TIFF (TIFF-only)."""
    pairs_by_region = {}
    missing_pairs = []
    seen_pairs = set()

    for root_path in _model_ready_data_roots(data_root):
        month_index = _build_region_index_from_root(root_path, collection)
        month = None
        if os.path.basename(root_path).startswith("dataset_month"):
            month = int(os.path.basename(root_path).replace("dataset_month", ""))

        for region_id, item in month_index.items():
            original_dir = os.path.join(data_root, "original", item["folder"])
            if not os.path.isdir(original_dir):
                continue

            pairs_by_region.setdefault(region_id, [])
            for tif_name in item["tif_names"]:
                pair_key = (region_id, tif_name)
                if pair_key in seen_pairs:
                    continue

                original_tif_path = os.path.join(original_dir, tif_name)
                model_ready_tif_path = os.path.join(root_path, item["folder"], tif_name)

                if not os.path.exists(original_tif_path):
                    missing_pairs.append((region_id, tif_name))
                    continue

                seen_pairs.add(pair_key)
                pairs_by_region[region_id].append(
                    {
                        "region_id": region_id,
                        "folder": item["folder"],
                        "tif_name": tif_name,
                        "month": month,
                        "original_tif_path": original_tif_path,
                        "model_ready_tif_path": model_ready_tif_path,
                    }
                )

    for region_id in pairs_by_region:
        pairs_by_region[region_id] = sorted(pairs_by_region[region_id], key=lambda pair: pair["tif_name"])

    if missing_pairs:
        print(f"Warning: skipped {len(missing_pairs)} model-ready TIFFs without matching original TIFF.")

    return pairs_by_region


def _recover_rotation_from_pair(original_image, model_ready_image, target_height, target_width):
    """Recover angle_ccw used during preprocessing by direct image matching."""
    if original_image.ndim != 2:
        original_image = np.squeeze(original_image)
    if model_ready_image.ndim != 2:
        model_ready_image = np.squeeze(model_ready_image)

    original_image = original_image.astype(np.uint8)
    model_ready_image = model_ready_image.astype(np.uint8)

    # Keep only expected classes to avoid artifacts from unexpected values.
    original_image = np.where(np.isin(original_image, [0, 1, 2]), original_image, 0).astype(np.uint8)
    model_ready_image = np.where(np.isin(model_ready_image, [0, 1, 2]), model_ready_image, 0).astype(np.uint8)

    candidate_angles = [0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0]

    best_angle = 0.0
    best_score = -1.0
    for angle_ccw in candidate_angles:
        rotated = np.array(
            Image.fromarray(original_image.astype(np.uint8)).rotate(
                angle=angle_ccw,
                resample=Image.NEAREST,
                expand=True,
                fillcolor=0,
            )
        )
        candidate = center_crop_or_pad(rotated, target_height, target_width).astype(np.uint8)
        if candidate.shape != model_ready_image.shape:
            continue

        valid = (candidate != 0) | (model_ready_image != 0)
        if np.any(valid):
            score = float(np.mean(candidate[valid] == model_ready_image[valid]))
        else:
            score = float(np.mean(candidate == model_ready_image))
        if score > best_score:
            best_score = score
            best_angle = angle_ccw

    return best_angle, best_score


def _recover_region_transform_from_pairs(pairs, target_height, target_width):
    """Recover transform from all available TIFF pairs in region, then choose representative pair."""
    recovered = []
    for pair in pairs:
        try:
            original_image = tifffile.imread(pair["original_tif_path"])
            model_ready_image = tifffile.imread(pair["model_ready_tif_path"])
        except Exception:
            continue

        angle_ccw, match_score = _recover_rotation_from_pair(
            original_image=original_image,
            model_ready_image=model_ready_image,
            target_height=target_height,
            target_width=target_width,
        )
        recovered.append(
            {
                "pair": pair,
                "angle_ccw": float(angle_ccw),
                "match_score": float(match_score),
            }
        )

    if not recovered:
        return None

    # Choose most frequent recovered angle; break ties by average score.
    grouped = {}
    for item in recovered:
        key = round(item["angle_ccw"], 3)
        grouped.setdefault(key, []).append(item)

    best_angle_key = None
    best_count = -1
    best_group_score = -1.0
    for angle_key, group in grouped.items():
        count = len(group)
        mean_score = float(np.mean([entry["match_score"] for entry in group]))
        if count > best_count or (count == best_count and mean_score > best_group_score):
            best_count = count
            best_group_score = mean_score
            best_angle_key = angle_key

    best_group = grouped[best_angle_key]
    representative = sorted(
        best_group,
        key=lambda entry: (entry["pair"]["tif_name"], entry["match_score"]),
        reverse=True,
    )[0]

    return {
        "angle_ccw": float(best_angle_key),
        "match_score": float(representative["match_score"]),
        "representative_pair": representative["pair"],
    }


def _model_ready_polygon_from_inverse_transform(original_tif_path, angle_ccw, target_height, target_width):
    model = _read_geotiff_model(original_tif_path)
    original_h = model["height"]
    original_w = model["width"]

    x_map = np.tile(np.arange(original_w, dtype=np.float32), (original_h, 1))
    y_map = np.tile(np.arange(original_h, dtype=np.float32).reshape(-1, 1), (1, original_w))

    angle_ccw = float(angle_ccw)
    x_rot = np.array(
        Image.fromarray(x_map).rotate(
            angle=angle_ccw,
            resample=Image.NEAREST,
            expand=True,
            fillcolor=-1.0,
        )
    )
    y_rot = np.array(
        Image.fromarray(y_map).rotate(
            angle=angle_ccw,
            resample=Image.NEAREST,
            expand=True,
            fillcolor=-1.0,
        )
    )

    x_model = _center_crop_or_pad_constant(x_rot, target_height, target_width, -1.0)
    y_model = _center_crop_or_pad_constant(y_rot, target_height, target_width, -1.0)

    valid = (x_model >= 0.0) & (y_model >= 0.0)
    ys, xs = np.where(valid)
    if len(xs) == 0 or len(ys) == 0:
        return None

    valid_points = np.column_stack((ys, xs))
    target_corners = np.array(
        [
            [0, 0],
            [0, target_width - 1],
            [target_height - 1, target_width - 1],
            [target_height - 1, 0],
        ],
        dtype=np.int32,
    )

    corner_rc = []
    for corner in target_corners:
        d2 = np.sum((valid_points - corner) ** 2, axis=1)
        nearest_idx = int(np.argmin(d2))
        row, col = valid_points[nearest_idx]
        corner_rc.append((int(row), int(col)))

    polygon = []
    for row, col in corner_rc:
        orig_x = float(x_model[row, col])
        orig_y = float(y_model[row, col])
        lon, lat = _pixel_to_lonlat(model, orig_x, orig_y)
        polygon.append([lon, lat])

    polygon.append(polygon[0])
    return polygon


def _build_catalog_record_from_polygon(
    region_id,
    source_region_id,
    source_folder,
    available_image_count,
    flow_heading_deg,
    polygon,
    target_height,
    target_width,
    pixel_size_m,
    footprint_source,
):
    lon_min, lon_max, lat_min, lat_max = _bbox_from_polygon(polygon)
    center_lat, center_lon = _center_from_polygon(polygon)
    model_ready_length_km = (target_height * pixel_size_m) / 1000.0
    model_ready_width_km = (target_width * pixel_size_m) / 1000.0
    model_ready_area_km2 = model_ready_length_km * model_ready_width_km

    return {
        "region_id": region_id,
        "source_region_id": source_region_id,
        "source_folder": source_folder,
        "available_image_count": int(available_image_count),
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "flow_heading_deg": float(flow_heading_deg),
        "model_ready_height_px": int(target_height),
        "model_ready_width_px": int(target_width),
        "pixel_size_m": float(pixel_size_m),
        "model_ready_length_km": float(model_ready_length_km),
        "model_ready_width_km": float(model_ready_width_km),
        "model_ready_area_km2": float(model_ready_area_km2),
        "footprint_source": footprint_source,
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "lon_min": float(lon_min),
        "lon_max": float(lon_max),
        "polygon_lonlat": polygon,
    }


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
                    "source_region_id": item.get("source_region_id"),
                    "source_folder": item.get("source_folder"),
                    "available_image_count": item.get("available_image_count"),
                    "footprint_source": item.get("footprint_source"),
                    "flow_heading_deg": item.get("flow_heading_deg"),
                    "model_ready_height_px": item.get("model_ready_height_px"),
                    "model_ready_width_px": item.get("model_ready_width_px"),
                    "pixel_size_m": item.get("pixel_size_m"),
                    "model_ready_length_km": item.get("model_ready_length_km"),
                    "model_ready_width_km": item.get("model_ready_width_km"),
                    "model_ready_area_km2": item.get("model_ready_area_km2"),
                    "center_lat": item.get("center_lat"),
                    "center_lon": item.get("center_lon"),
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

    model_ready_json = os.path.join(regions_dir, "region_catalog_model_ready.json")
    model_ready_geojson = os.path.join(regions_dir, "region_catalog_model_ready.geojson")

    original_regions = _build_region_index_from_root(
        os.path.join(args.data_root, "original"),
        args.collection,
    )
    model_ready_regions = _build_model_ready_region_index(args.data_root, args.collection)
    pairs_by_region = _build_model_ready_pairs(args.data_root, args.collection)

    records = []
    errors = []
    for region_id in sorted(model_ready_regions):
        if region_id not in original_regions:
            errors.append(
                f"[{region_id}] Missing original region folder required for strict model-ready catalog."
            )
            continue

        model_ready_info = model_ready_regions[region_id]
        pairs = pairs_by_region.get(region_id, [])
        recovered = None
        if pairs:
            recovered = _recover_region_transform_from_pairs(
                pairs=pairs,
                target_height=args.target_height,
                target_width=args.target_width,
            )

        if recovered is not None:
            angle_ccw = recovered["angle_ccw"]
            match_score = recovered["match_score"]
            representative_pair = recovered["representative_pair"]
            original_tif_path = representative_pair["original_tif_path"]
            flow_heading_deg = (180.0 + angle_ccw) % 360.0
            footprint_source = "recovered_per_image_transform_to_original_geotiff"
        else:
            errors.append(
                f"[{region_id}] Could not recover transform from model-ready/original TIFF pairs; strict mode forbids fallback heading defaults."
            )
            continue

        polygon = _model_ready_polygon_from_inverse_transform(
            original_tif_path=original_tif_path,
            angle_ccw=angle_ccw,
            target_height=args.target_height,
            target_width=args.target_width,
        )
        if not _is_valid_polygon(polygon):
            errors.append(
                f"[{region_id}] Invalid polygon reconstructed from inverse transform ({original_tif_path})."
            )
            continue

        record = _build_catalog_record_from_polygon(
            region_id=region_id,
            source_region_id=region_id,
            source_folder=model_ready_info["folder"],
            available_image_count=len(model_ready_info["tif_names"]),
            flow_heading_deg=flow_heading_deg,
            polygon=polygon,
            target_height=args.target_height,
            target_width=args.target_width,
            pixel_size_m=args.pixel_size_m,
            footprint_source=footprint_source,
        )
        record["recovered_angle_ccw"] = float(angle_ccw)
        if match_score is not None:
            record["recovered_match_score"] = float(match_score)
        if representative_pair is not None:
            record["representative_tif_name"] = representative_pair["tif_name"]
        else:
            record["representative_tif_name"] = os.path.basename(original_tif_path)
        records.append(record)

    if errors:
        preview = "\n".join(errors[:20])
        extra = ""
        if len(errors) > 20:
            extra = f"\n... and {len(errors) - 20} more errors."
        raise RuntimeError(
            "Strict model-ready catalog generation failed due to missing required data:\n"
            f"{preview}{extra}"
        )

    if not records:
        raise RuntimeError("Strict model-ready catalog generation produced zero valid records.")

    with open(model_ready_json, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    write_geojson(records, model_ready_geojson)

    print(f"Model-ready region catalog written: {model_ready_json}")
    print(f"Model-ready region polygons written: {model_ready_geojson}")


def build_raw_region_catalog(args):
    regions_dir = os.path.join(args.data_root, "regions")
    os.makedirs(regions_dir, exist_ok=True)

    raw_json = os.path.join(regions_dir, "region_catalog_raw_available.json")
    raw_geojson = os.path.join(regions_dir, "region_catalog_raw_available.geojson")

    original_regions = _build_region_index_from_root(
        os.path.join(args.data_root, "original"),
        args.collection,
    )
    heading_map = load_heading_map(args.data_root, args.collection)

    records = []
    for region_id, info in sorted(original_regions.items()):
        try:
            model = _read_geotiff_model(info["sample_tif_path"])
        except (KeyError, ValueError):
            continue

        polygon = _polygon_from_model(model)
        if not _is_valid_polygon(polygon):
            continue

        records.append(
            _build_catalog_record_from_polygon(
                region_id=region_id,
                source_region_id=region_id,
                source_folder=info["folder"],
                available_image_count=len(info["tif_names"]),
                flow_heading_deg=heading_map.get(region_id, 180.0),
                polygon=polygon,
                target_height=args.target_height,
                target_width=args.target_width,
                pixel_size_m=args.pixel_size_m,
                footprint_source="raw_original_geotiff",
            )
        )

    with open(raw_json, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    write_geojson(records, raw_geojson)

    print(f"Raw region catalog written: {raw_json}")
    print(f"Raw region polygons written: {raw_geojson}")


def generate_osm_catalog_figures(args):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    plot_script = os.path.join(repo_root, "scripts", "utils", "plot_model_ready_footprints.py")
    python_exec = sys.executable

    outputs = [
        (
            os.path.join(args.data_root, "regions", "region_catalog_model_ready.geojson"),
            os.path.join(args.data_root, "regions", "model_ready_footprints_osm.png"),
        ),
        (
            os.path.join(args.data_root, "regions", "region_catalog_raw_available.geojson"),
            os.path.join(args.data_root, "regions", "raw_available_footprints_osm.png"),
        ),
    ]

    for geojson_path, output_path in outputs:
        if not os.path.exists(geojson_path):
            print(f"Warning: GeoJSON missing, skipping OSM figure: {geojson_path}")
            continue

        command = [
            python_exec,
            plot_script,
            "--geojson",
            geojson_path,
            "--output",
            output_path,
            "--with-basemap",
            "--label-regions",
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"Warning: OSM figure generation failed for {geojson_path}: {result.stderr.strip()}")
        except Exception as exc:
            print(f"Warning: OSM figure generation raised exception for {geojson_path}: {exc}")


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
        build_raw_region_catalog(args)
        build_model_ready_region_catalog(args)
        generate_osm_catalog_figures(args)
        print("Raw and model-ready catalog generation completed.")
        return

    region_dirs = list_region_dirs(os.path.join(args.data_root, "original"), args.collection)
    if not region_dirs:
        raise RuntimeError(f"No {args.collection}_* region folders found under original/")

    preprocess_images(args, region_dirs)
    build_raw_region_catalog(args)
    build_model_ready_region_catalog(args)
    generate_osm_catalog_figures(args)
    build_dataset_month_folders(args, region_dirs)
    build_averages(args, region_dirs)
    print("satellite preparation completed.")


if __name__ == "__main__":
    main()
