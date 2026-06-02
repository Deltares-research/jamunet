"""Plot model-ready region footprints for quick visual QA.

Default input:
    data/satellite/regions/region_catalog_model_ready.geojson

Example:
    python scripts/utils/plot_model_ready_footprints.py --label-regions
"""

from __future__ import annotations

import argparse
import json
import math
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model-ready region footprints and save a QA figure."
    )
    parser.add_argument(
        "--geojson",
        default="data/satellite/regions/region_catalog_model_ready.geojson",
        help="Path to model-ready GeoJSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/satellite/regions/model_ready_footprints.png",
        help="Path to save output figure.",
    )
    parser.add_argument(
        "--label-regions",
        action="store_true",
        help="Annotate region IDs at polygon centroids.",
    )
    parser.add_argument(
        "--with-basemap",
        action="store_true",
        help="Draw OpenStreetMap tiles in the background (requires internet).",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=8,
        help="OSM tile zoom level used when --with-basemap is set (default: 8).",
    )
    return parser.parse_args()


def load_features(geojson_path: Path) -> list[dict]:
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not features:
        raise ValueError(f"No features found in {geojson_path}")
    return features


def centroid(coords: list[list[float]]) -> tuple[float, float]:
    # Ignore duplicated closing point if present.
    ring = coords[:-1] if len(coords) > 1 and coords[0] == coords[-1] else coords
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_lonlat(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = xtile / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
    return lon, lat


def fetch_osm_basemap(min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int) -> tuple[Image.Image, list[float]]:
    x_min, y_max = lonlat_to_tile(min_lon, min_lat, zoom)
    x_max, y_min = lonlat_to_tile(max_lon, max_lat, zoom)

    tile_count_x = x_max - x_min + 1
    tile_count_y = y_max - y_min + 1
    canvas = Image.new("RGB", (tile_count_x * 256, tile_count_y * 256))

    headers = {"User-Agent": "jamunet-footprint-plotter/1.0"}
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            request = Request(url, headers=headers)
            with urlopen(request, timeout=20) as response:
                tile = Image.open(BytesIO(response.read())).convert("RGB")
            canvas.paste(tile, ((x - x_min) * 256, (y - y_min) * 256))

    left_lon, top_lat = tile_to_lonlat(x_min, y_min, zoom)
    right_lon, bottom_lat = tile_to_lonlat(x_max + 1, y_max + 1, zoom)
    extent = [left_lon, right_lon, bottom_lat, top_lat]
    return canvas, extent


def plot_plain(features: list[dict], output_path: Path, label_regions: bool, with_basemap: bool, zoom: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)

    min_lon = float("inf")
    max_lon = float("-inf")
    min_lat = float("inf")
    max_lat = float("-inf")

    for feature in features:
        props = feature.get("properties", {})
        region_id = props.get("region_id", "unknown")
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon":
            continue

        ring = geometry.get("coordinates", [[]])[0]
        if len(ring) < 3:
            continue

        patch = MplPolygon(
            ring,
            closed=True,
            facecolor="#4e79a7",
            alpha=0.22,
            edgecolor="#1f3a5f",
            linewidth=1.0,
        )
        ax.add_patch(patch)

        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        min_lon = min(min_lon, min(xs))
        max_lon = max(max_lon, max(xs))
        min_lat = min(min_lat, min(ys))
        max_lat = max(max_lat, max(ys))

        if label_regions:
            cx, cy = centroid(ring)
            ax.text(cx, cy, region_id, fontsize=6, ha="center", va="center", color="#0d1b2a")

    if min_lon == float("inf"):
        raise ValueError("No valid polygon coordinates found in GeoJSON.")

    if with_basemap:
        try:
            basemap_img, basemap_extent = fetch_osm_basemap(min_lon, min_lat, max_lon, max_lat, zoom)
            ax.imshow(basemap_img, extent=basemap_extent, origin="upper", alpha=0.9)
        except Exception as exc:
            print(f"Could not fetch OSM basemap ({exc}). Falling back to plain background.")

    pad_lon = (max_lon - min_lon) * 0.05
    pad_lat = (max_lat - min_lat) * 0.05
    ax.set_xlim(min_lon - pad_lon, max_lon + pad_lon)
    ax.set_ylim(min_lat - pad_lat, max_lat + pad_lat)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Model-ready image footprints")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    geojson_path = Path(args.geojson)
    output_path = Path(args.output)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    features = load_features(geojson_path)
    plot_plain(features, output_path, args.label_regions, args.with_basemap, args.zoom)

    print(f"Footprint figure written: {output_path}")


if __name__ == "__main__":
    main()
