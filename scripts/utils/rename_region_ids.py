"""Rename satellite region folders and files to coordinate-based slugs.

This script migrates the current `eval_rXX` / `training_rXX` / `validation_rX`
layout to coordinate-based region identifiers of the form
`lat24p6515_lon88p0207`.

It supports `data/satellite`, where the center coordinates are derived from the
GeoTIFF georeferencing tags of the first TIFF in each region folder.

Usage:
    python rename_region_ids.py --dry-run
    python rename_region_ids.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import tifffile


COLLECTION_TAG = "JRC_GSW1_4_MonthlyHistory"


def slugify_coord(value: float, prefix: str) -> str:
    magnitude = f"{abs(value):.4f}".replace(".", "p")
    sign_prefix = f"{prefix}m" if value < 0 else prefix
    return f"{sign_prefix}{magnitude}"


def region_slug(lat: float, lon: float) -> str:
    return f"{slugify_coord(lat, 'lat')}_{slugify_coord(lon, 'lon')}"


def read_geotiff_center(tif_path: Path) -> tuple[float, float, float, float, float, float]:
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        height, width = page.shape

        tags = page.tags
        if 33550 in tags and 33922 in tags:
            scale = tags[33550].value
            tiepoint = tags[33922].value

            pixel_size_x = float(scale[0])
            pixel_size_y = float(scale[1])
            x0 = float(tiepoint[3])
            y0 = float(tiepoint[4])

            center_lon = x0 + pixel_size_x * (width - 1) / 2.0
            center_lat = y0 - pixel_size_y * (height - 1) / 2.0

            lon_min = x0
            lon_max = x0 + pixel_size_x * (width - 1)
            lat_max = y0
            lat_min = y0 - pixel_size_y * (height - 1)
        elif 34264 in tags:
            # ModelTransformationTag is a 4x4 matrix in row-major order.
            # Map pixel coords (i, j, 0, 1) -> model (x, y, z, 1).
            transform = [float(value) for value in tags[34264].value]
            if len(transform) != 16:
                raise ValueError(f"Unexpected ModelTransformationTag length in {tif_path}: {len(transform)}")

            def to_model(i: float, j: float) -> tuple[float, float]:
                x = transform[0] * i + transform[1] * j + transform[3]
                y = transform[4] * i + transform[5] * j + transform[7]
                return x, y

            corners = [
                to_model(0.0, 0.0),
                to_model(float(width - 1), 0.0),
                to_model(0.0, float(height - 1)),
                to_model(float(width - 1), float(height - 1)),
            ]
            lon_values = [corner[0] for corner in corners]
            lat_values = [corner[1] for corner in corners]

            lon_min = min(lon_values)
            lon_max = max(lon_values)
            lat_min = min(lat_values)
            lat_max = max(lat_values)
            center_lon = (lon_min + lon_max) / 2.0
            center_lat = (lat_min + lat_max) / 2.0
        else:
            raise KeyError(f"Missing georeferencing tags (33550/33922 or 34264) in {tif_path}")

    return center_lat, center_lon, lat_min, lat_max, lon_min, lon_max


def rename_path(src: Path, dst: Path, apply: bool) -> None:
    if src == dst:
        return
    if not src.exists():
        return
    if dst.exists():
        raise FileExistsError(f"Target already exists: {dst}")
    print(f"MOVE {src} -> {dst}")
    if apply:
        src.rename(dst)


def rename_children(dir_path: Path, old: str, new: str, apply: bool) -> None:
    if not dir_path.exists():
        return

    for child_name in sorted(os.listdir(dir_path)):
        child = dir_path / child_name
        if old not in child.name:
            continue
        new_name = child.name.replace(old, new)
        rename_path(child, child.with_name(new_name), apply)


def write_json(path: Path, payload: object, apply: bool) -> None:
    print(f"WRITE {path}")
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_geojson(path: Path, records: list[dict], apply: bool) -> None:
    features = []
    for record in records:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "region_id": record["region_id"],
                    "center_lat": record["center_lat"],
                    "center_lon": record["center_lon"],
                    "lat_min": record.get("lat_min"),
                    "lat_max": record.get("lat_max"),
                    "lon_min": record.get("lon_min"),
                    "lon_max": record.get("lon_max"),
                    "source_region_id": record.get("source_region_id"),
                    "source_folder": record.get("source_folder"),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [record["polygon_lonlat"]],
                },
            }
        )

    payload = {"type": "FeatureCollection", "features": features}
    print(f"WRITE {path}")
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_satellite_01_records(root: Path) -> list[dict]:
    catalog_path = root / "regions" / "eval_reaches.json"
    records = json.loads(catalog_path.read_text(encoding="utf-8"))
    for record in records:
        record["source_region_id"] = record["region_id"]
        record["region_id"] = region_slug(float(record["center_lat"]), float(record["center_lon"]))
        record["source_folder"] = f"{COLLECTION_TAG}_{record['source_region_id']}"
        record["polygon_lonlat"] = record.get("polygon_lonlat", [])
    return records


def load_satellite_records(root: Path) -> list[dict]:
    original_root = root / "original"
    records: list[dict] = []
    for folder_name in sorted(os.listdir(original_root)):
        folder = original_root / folder_name
        if not folder.is_dir() or not folder.name.startswith(f"{COLLECTION_TAG}_"):
            continue
        tif_files = sorted(folder.glob("*.tif"))
        if not tif_files:
            continue
        center_lat, center_lon, lat_min, lat_max, lon_min, lon_max = read_geotiff_center(tif_files[0])
        old_region_id = folder.name.replace(f"{COLLECTION_TAG}_", "")
        region_id = region_slug(center_lat, center_lon)
        records.append(
            {
                "source_region_id": old_region_id,
                "region_id": region_id,
                "source_folder": folder.name,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "polygon_lonlat": [
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                ],
            }
        )
    return records


def rename_tree(root: Path, records: list[dict], apply: bool, rename_folders: bool) -> None:
    month_dirs = [root / f"dataset_month{i}" for i in range(1, 5)]
    for record in records:
        old_region_id = record["source_region_id"]
        new_region_id = record["region_id"]

        parent_dirs = [
            root / "original",
            root / "preprocessed",
            root / "dataset",
            *month_dirs,
        ]

        for parent in parent_dirs:
            old_folder = parent / f"{COLLECTION_TAG}_{old_region_id}"
            if old_folder.exists():
                rename_children(old_folder, old_region_id, new_region_id, apply)
                if rename_folders:
                    rename_path(old_folder, parent / f"{COLLECTION_TAG}_{new_region_id}", apply)

        averages_old = root / "averages" / f"average_{old_region_id}"
        if averages_old.exists():
            rename_children(averages_old, old_region_id, new_region_id, apply)
            if rename_folders:
                rename_path(averages_old, root / "averages" / f"average_{new_region_id}", apply)


def migrate_satellite_01(root: Path, apply: bool, rename_folders: bool) -> None:
    records = load_satellite_01_records(root)
    rename_tree(root, records, apply, rename_folders)

    catalog_payload = [
        {
            "region_id": record["region_id"],
            "source_region_id": record["source_region_id"],
            "center_lat": record["center_lat"],
            "center_lon": record["center_lon"],
            "lat_min": record["lat_min"],
            "lat_max": record["lat_max"],
            "lon_min": record["lon_min"],
            "lon_max": record["lon_max"],
            "flow_heading_deg": record.get("flow_heading_deg"),
            "polygon_lonlat": record.get("polygon_lonlat", []),
            "source_folder": record.get("source_folder"),
        }
        for record in records
    ]

    legacy_json = root / "regions" / "eval_reaches.json"
    legacy_geojson = root / "regions" / "eval_reaches.geojson"
    new_json = root / "regions" / "region_catalog.json"
    new_geojson = root / "regions" / "region_catalog.geojson"

    if apply:
        write_json(new_json, catalog_payload, True)
        write_geojson(new_geojson, catalog_payload, True)
        if legacy_json.exists():
            legacy_json.unlink()
        if legacy_geojson.exists():
            legacy_geojson.unlink()
    else:
        write_json(new_json, catalog_payload, False)
        write_geojson(new_geojson, catalog_payload, False)


def migrate_satellite(root: Path, apply: bool, rename_folders: bool) -> None:
    records = load_satellite_records(root)
    rename_tree(root, records, apply, rename_folders)

    catalog_payload = [
        {
            "region_id": record["region_id"],
            "source_region_id": record["source_region_id"],
            "center_lat": record["center_lat"],
            "center_lon": record["center_lon"],
            "lat_min": record["lat_min"],
            "lat_max": record["lat_max"],
            "lon_min": record["lon_min"],
            "lon_max": record["lon_max"],
            "polygon_lonlat": record["polygon_lonlat"],
            "source_folder": record["source_folder"],
        }
        for record in records
    ]

    write_json(root / "regions" / "region_catalog.json", catalog_payload, apply)
    write_geojson(root / "regions" / "region_catalog.geojson", catalog_payload, apply)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename satellite region folders and files to coordinate-based slugs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Apply the renames instead of doing a dry run.")
    mode.add_argument("--dry-run", action="store_true", help="Show the renames without changing anything.")
    parser.add_argument("--skip-folders", action="store_true", help="Rename files and metadata, but leave directories in place.")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["data/satellite"],
        help="Dataset roots to migrate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = not args.apply
    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            print(f"Skip missing root: {root}")
            continue

        print(f"\n== Migrating {root} ==")
        if (root / "regions" / "eval_reaches.json").exists():
            migrate_satellite_01(root, not dry_run, not args.skip_folders)
        else:
            migrate_satellite(root, not dry_run, not args.skip_folders)


if __name__ == "__main__":
    main()
