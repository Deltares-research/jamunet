import argparse
import json
import math
import os
from datetime import date

import ee
import requests


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def month_start_next(year, month):
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)


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

    # Polygon ring (clockwise), closed by repeating first point.
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


def build_reaches(
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    num_reaches,
    tile_length_km,
    tile_width_km,
    overlap,
    southwest_shift_reaches=None,
    shift_west_km=0.0,
    shift_south_km=0.0,
):
    avg_lat = (start_lat + end_lat) / 2.0
    dx_east_km = (end_lon - start_lon) * 111.32 * max(math.cos(math.radians(avg_lat)), 1e-6)
    dy_north_km = (end_lat - start_lat) * 111.32
    flow_norm = math.hypot(dx_east_km, dy_north_km)
    if flow_norm < 1e-9:
        # Fallback southward direction if start/end are equal.
        u_east, u_north = 0.0, -1.0
    else:
        u_east = dx_east_km / flow_norm
        u_north = dy_north_km / flow_norm

    if num_reaches <= 0:
        total_km = haversine_km(start_lat, start_lon, end_lat, end_lon)
        step_km = max(tile_length_km * (1.0 - overlap), 1.0)
        num_reaches = max(1, math.ceil(total_km / step_km))

    reaches = []
    southwest_shift_reaches = southwest_shift_reaches or set()
    for i in range(num_reaches):
        if num_reaches == 1:
            t = 0.5
        else:
            t = i / (num_reaches - 1)

        center_lat = start_lat + t * (end_lat - start_lat)
        center_lon = start_lon + t * (end_lon - start_lon)
        region_id = f"eval_r{i + 1:02d}"

        if region_id in southwest_shift_reaches and (shift_west_km > 0.0 or shift_south_km > 0.0):
            center_lon, center_lat = km_offsets_to_latlon(
                center_lat,
                center_lon,
                east_km=-shift_west_km,
                north_km=-shift_south_km,
            )

        polygon = build_rotated_polygon(
            center_lat=center_lat,
            center_lon=center_lon,
            u_east=u_east,
            u_north=u_north,
            tile_length_km=tile_length_km,
            tile_width_km=tile_width_km,
        )

        lon_values = [p[0] for p in polygon[:-1]]
        lat_values = [p[1] for p in polygon[:-1]]
        lon_min, lon_max = min(lon_values), max(lon_values)
        lat_min, lat_max = min(lat_values), max(lat_values)

        # Angle of flow with respect to north, clockwise (degrees).
        heading_deg = (math.degrees(math.atan2(u_east, u_north)) + 360.0) % 360.0

        reaches.append(
            {
                "region_id": f"eval_r{i + 1:02d}",
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "flow_heading_deg": heading_deg,
                "polygon_lonlat": polygon,
                "shifted_southwest": region_id in southwest_shift_reaches,
                "shift_west_km": shift_west_km if region_id in southwest_shift_reaches else 0.0,
                "shift_south_km": shift_south_km if region_id in southwest_shift_reaches else 0.0,
            }
        )

    return reaches


def initialize_ee(project=None):
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def download_one_image(image, region_coords, out_path, scale, crs):
    url = image.getDownloadURL(
        {
            "scale": scale,
            "crs": crs,
            "region": region_coords,
            "format": "GEO_TIFF",
        }
    )

    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(out_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)


def parse_args():
    parser = argparse.ArgumentParser(description="Download JRC monthly history images for neutral eval reaches.")
    parser.add_argument("--start-lat", type=float, default=24.6515)
    parser.add_argument("--start-lon", type=float, default=88.020697)
    parser.add_argument("--end-lat", type=float, default=23.785158)
    parser.add_argument("--end-lon", type=float, default=89.775544)

    parser.add_argument("--data-root", default="data/satellite_01")
    parser.add_argument("--collection", default="JRC/GSW1_4/MonthlyHistory")
    parser.add_argument("--collection-tag", default="JRC_GSW1_4_MonthlyHistory")

    parser.add_argument("--num-reaches", type=int, default=5)
    parser.add_argument("--tile-length-km", type=float, default=60.0)
    parser.add_argument("--tile-width-km", type=float, default=30.0)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--southwest-shift-reaches", default="eval_r03,eval_r04")
    parser.add_argument("--shift-west-km", type=float, default=6.0)
    parser.add_argument("--shift-south-km", type=float, default=6.0)

    parser.add_argument("--start-year", type=int, default=1988)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--months", default="1,2,3,4")

    parser.add_argument("--scale", type=int, default=60)
    parser.add_argument("--crs", default="EPSG:4326")
    parser.add_argument("--ee-project", default="")
    parser.add_argument(
        "--geometry-mode",
        choices=["polygon", "bbox"],
        default="polygon",
        help="Use rotated polygon clip or axis-aligned bounding box for download region.",
    )

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_reaches_geojson(reaches, geojson_path):
    features = []
    for reach in reaches:
        feature = {
            "type": "Feature",
            "properties": {
                "region_id": reach["region_id"],
                "flow_heading_deg": reach.get("flow_heading_deg"),
                "center_lat": reach["center_lat"],
                "center_lon": reach["center_lon"],
                "lat_min": reach["lat_min"],
                "lat_max": reach["lat_max"],
                "lon_min": reach["lon_min"],
                "lon_max": reach["lon_max"],
                "shifted_southwest": reach.get("shifted_southwest", False),
                "shift_west_km": reach.get("shift_west_km", 0.0),
                "shift_south_km": reach.get("shift_south_km", 0.0),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [reach["polygon_lonlat"]],
            },
        }
        features.append(feature)

    collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(geojson_path, "w", encoding="utf-8") as file:
        json.dump(collection, file, indent=2)


def main():
    args = parse_args()

    overlap = clamp(args.overlap, 0.0, 0.95)
    months = [int(m.strip()) for m in args.months.split(",") if m.strip()]
    for month in months:
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month in --months: {month}")

    southwest_shift_reaches = {
        reach.strip()
        for reach in args.southwest_shift_reaches.split(",")
        if reach.strip()
    }

    reaches = build_reaches(
        args.start_lat,
        args.start_lon,
        args.end_lat,
        args.end_lon,
        args.num_reaches,
        args.tile_length_km,
        args.tile_width_km,
        overlap,
        southwest_shift_reaches=southwest_shift_reaches,
        shift_west_km=max(0.0, args.shift_west_km),
        shift_south_km=max(0.0, args.shift_south_km),
    )

    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(os.path.join(args.data_root, "original"), exist_ok=True)
    os.makedirs(os.path.join(args.data_root, "regions"), exist_ok=True)

    reaches_metadata_path = os.path.join(args.data_root, "regions", "eval_reaches.json")
    with open(reaches_metadata_path, "w", encoding="utf-8") as f:
        json.dump(reaches, f, indent=2)
    print(f"Saved reach metadata: {reaches_metadata_path}")

    reaches_geojson_path = os.path.join(args.data_root, "regions", "eval_reaches.geojson")
    write_reaches_geojson(reaches, reaches_geojson_path)
    print(f"Saved reach polygons: {reaches_geojson_path}")

    if args.dry_run:
        print("Dry run enabled. No Earth Engine calls will be executed.")
        return

    initialize_ee(project=args.ee_project if args.ee_project else None)

    for reach in reaches:
        region_id = reach["region_id"]
        if args.geometry_mode == "bbox":
            region_geom = ee.Geometry.Rectangle(
                [reach["lon_min"], reach["lat_min"], reach["lon_max"], reach["lat_max"]],
                geodesic=False,
            )
            region_coords = region_geom.getInfo()["coordinates"]
        else:
            region_coords = [reach["polygon_lonlat"]]
            region_geom = ee.Geometry.Polygon(region_coords, proj=None, geodesic=False)

        out_dir = os.path.join(args.data_root, "original", f"{args.collection_tag}_{region_id}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing {region_id} -> {out_dir}")

        for year in range(args.start_year, args.end_year + 1):
            for month in months:
                start_dt = date(year, month, 1)
                end_dt = month_start_next(year, month)

                out_name = f"{year}_{month:02d}_01_{region_id}.tif"
                out_path = os.path.join(out_dir, out_name)

                if args.skip_existing and os.path.exists(out_path):
                    print(f"  Skip existing: {out_name}")
                    continue

                collection = (
                    ee.ImageCollection(args.collection)
                    .filterDate(start_dt.isoformat(), end_dt.isoformat())
                    .filterBounds(region_geom)
                )

                count = collection.size().getInfo()
                if count == 0:
                    print(f"  No image for {year}-{month:02d} in {region_id}")
                    continue

                image = ee.Image(collection.first()).clip(region_geom).toUint8()

                try:
                    download_one_image(image, region_coords, out_path, args.scale, args.crs)
                    print(f"  Downloaded: {out_name}")
                except Exception as exc:
                    print(f"  Failed: {out_name} -> {exc}")


if __name__ == "__main__":
    main()
