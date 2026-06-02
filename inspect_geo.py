"""Audit TIFF shape consistency across satellite data stages.

Example:
    python inspect_geo.py --data-root data/satellite --expected-height 1000 --expected-width 500
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import tifffile


Shape = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect TIFF dimensions in original, preprocessed, and dataset_month* folders "
            "to verify shape normalization."
        )
    )
    parser.add_argument("--data-root", default="data/satellite", help="Satellite data root folder.")
    parser.add_argument("--collection", default="JRC_GSW1_4_MonthlyHistory", help="Collection prefix in region folder names.")
    parser.add_argument("--expected-height", type=int, default=1000, help="Expected normalized height.")
    parser.add_argument("--expected-width", type=int, default=500, help="Expected normalized width.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Max sample file paths shown for each stage.",
    )
    return parser.parse_args()


def list_region_dirs(base_dir: Path, collection: str) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted(
        child
        for child in base_dir.iterdir()
        if child.is_dir() and child.name.startswith(f"{collection}_")
    )


def iter_tiffs(region_dirs: Iterable[Path]) -> Iterable[Path]:
    for region_dir in region_dirs:
        yield from sorted(region_dir.glob("*.tif"))


def get_shape(path: Path) -> Shape:
    image = tifffile.imread(path)
    if image.ndim < 2:
        raise ValueError(f"Unexpected TIFF dimensionality ({image.ndim}) for {path}")
    return int(image.shape[0]), int(image.shape[1])


def summarize_stage(stage_name: str, region_dirs: List[Path], max_examples: int) -> Dict[str, object]:
    counter: Counter[Shape] = Counter()
    examples: Dict[Shape, List[str]] = {}
    total = 0
    failed: List[str] = []

    for tif_path in iter_tiffs(region_dirs):
        try:
            shape = get_shape(tif_path)
            counter[shape] += 1
            if len(examples.get(shape, [])) < max_examples:
                examples.setdefault(shape, []).append(str(tif_path))
            total += 1
        except Exception as exc:
            failed.append(f"{tif_path} ({exc})")

    return {
        "stage": stage_name,
        "region_count": len(region_dirs),
        "total_tifs": total,
        "shape_counts": counter,
        "examples": examples,
        "failed": failed,
    }


def print_summary(summary: Dict[str, object], expected_shape: Shape | None = None) -> None:
    stage = summary["stage"]
    region_count = summary["region_count"]
    total_tifs = summary["total_tifs"]
    shape_counts: Counter[Shape] = summary["shape_counts"]
    examples: Dict[Shape, List[str]] = summary["examples"]
    failed: List[str] = summary["failed"]

    print(f"\n=== {stage} ===")
    print(f"Regions: {region_count}")
    print(f"TIFF files scanned: {total_tifs}")

    if not shape_counts:
        print("No TIFF files found.")
    else:
        print("Shape distribution (height x width):")
        for shape, count in shape_counts.most_common():
            h, w = shape
            print(f"  - {h} x {w}: {count}")
            for sample in examples.get(shape, []):
                print(f"      sample: {sample}")

    if expected_shape is not None and total_tifs > 0:
        expected_count = shape_counts.get(expected_shape, 0)
        all_match = expected_count == total_tifs
        status = "PASS" if all_match else "FAIL"
        eh, ew = expected_shape
        print(f"Expected shape check ({eh} x {ew}): {status} ({expected_count}/{total_tifs})")

    if failed:
        print(f"Failed reads: {len(failed)}")
        for item in failed[:10]:
            print(f"  - {item}")


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    expected_shape = (args.expected_height, args.expected_width)

    original_dirs = list_region_dirs(data_root / "original", args.collection)
    preprocessed_dirs = list_region_dirs(data_root / "preprocessed", args.collection)

    dataset_month_dirs = []
    for month in (1, 2, 3, 4):
        month_root = data_root / f"dataset_month{month}"
        month_region_dirs = list_region_dirs(month_root, args.collection)
        dataset_month_dirs.append((f"dataset_month{month}", month_region_dirs))

    print("TIFF shape audit")
    print(f"Data root: {data_root}")
    print(f"Collection: {args.collection}")

    print_summary(summarize_stage("original", original_dirs, args.max_examples))
    print_summary(
        summarize_stage("preprocessed", preprocessed_dirs, args.max_examples),
        expected_shape=expected_shape,
    )

    for stage_name, region_dirs in dataset_month_dirs:
        print_summary(
            summarize_stage(stage_name, region_dirs, args.max_examples),
            expected_shape=expected_shape,
        )


if __name__ == "__main__":
    main()
