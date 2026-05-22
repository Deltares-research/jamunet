# Preprocessing

This folder contains reusable Python modules used to prepare satellite and river data before model training, validation, testing, and inference.

## Scripts in this folder

### dataset_generation.py
Builds model-ready input/target datasets from preprocessed satellite imagery.

Main responsibilities:
- load single GeoTIFF images into NumPy arrays (`load_image_array`)
- build ordered image lists for a given reach (`create_list_images`)
- create sliding-window input/target pairs (4 input years -> 1 target year by default) (`create_datasets`)
- filter bad samples based on non-water pixel threshold (`combine_datasets`)
- stack samples across reaches and return PyTorch datasets (`create_full_dataset` and related helpers)

Typical output objects:
- Python lists of input/target arrays
- `torch.utils.data.TensorDataset` objects ready for training loops

Notes:
- many functions in this module still use legacy split naming (`training`, `validation`, `testing`) in their arguments
- the newer coordinate-based folder naming is represented in the data folders and region catalogs

### satellite_analysis_pre.py
Preprocessing utilities for satellite images.

Main responsibilities:
- construct image paths (`get_path_images`, `list_paths`)
- rename downloaded files to the internal naming format (`rename_images`)
- rotate images to align flow direction top-to-bottom (`get_angle_rotation`, `rotate_images`)
- crop/pad images to model shape (default `1000x500`) (`reshape_images`)
- compute and support pixel-level operations used by dataset generation (`count_pixels`, averaging helpers used by `dataset_generation.py`)

Typical output:
- transformed NumPy arrays
- preprocessed TIFF files (when used in preprocessing pipelines outside this module)

### images_analysis.py
Image loading and exploratory analysis utilities.

Main responsibilities:
- load and visualize TIFF/PNG-like imagery (`load_image`, `show_image_array`)
- load per-image CSV pixel-count summaries from preprocessed folders (`load_all_csv`)
- merge multiple reaches into long-format analysis dataframes (`create_long_df`)
- compute data availability statistics based on no-data percentage (`get_info_images`)

Typical output:
- plotted figures for QA
- pandas DataFrames for exploratory analysis

### river_analysis_pre.py
Utilities for hydrological tabular data preprocessing (discharge, water level, velocity).

Main responsibilities:
- load flow/water-level Excel datasets into standardized dataframes
- reshape wide time tables into long daily/monthly series
- fill missing values with day- or month-based averages

Typical output:
- cleaned and aligned pandas DataFrames for downstream analysis

## Expected input data structure

The preprocessing code expects data under `data/` in this repository.

Primary satellite tree:
- `data/satellite/original/`
- `data/satellite/preprocessed/`
- `data/satellite/dataset_month1/` ... `data/satellite/dataset_month4/`
- `data/satellite/dataset/`
- `data/satellite/averages/`
- `data/satellite/regions/region_catalog.json`
- `data/satellite/regions/region_catalog.geojson`

Region folder naming:
- coordinate-based slug format is used (example: `JRC_GSW1_4_MonthlyHistory_lat24p6515_lon88p0207`)
- region metadata (including legacy source labels) is stored in `region_catalog.json`

Image value convention (JRC MonthlyHistory, single channel):
- original classes: `0=no-data`, `1=non-water`, `2=water`
- many preprocessing/model functions remap to: `-1=no-data`, `0=non-water`, `1=water`

Hydrology input tree used by `river_analysis_pre.py`:
- `data/flow/original/` (Excel files with discharge/water-level records)

## Output data structure

Outputs produced by preprocessing functions are a mix of in-memory objects and files:

In-memory outputs:
- NumPy arrays (single images or image stacks)
- pandas DataFrames (pixel summaries, merged long tables, cleaned hydrology time series)
- PyTorch `TensorDataset` objects for model training/evaluation pipelines

On-disk outputs (when preprocessing pipelines write files):
- preprocessed GeoTIFF images in `data/satellite/preprocessed/...`
- filtered/organized datasets in `data/satellite/dataset*...`
- seasonal average CSVs in `data/satellite/averages/...`

## How these modules are used

Common flow in the project:
1. Prepare and standardize satellite imagery (`satellite_analysis_pre.py`)
2. Inspect quality and pixel distributions (`images_analysis.py`)
3. Build model training samples (`dataset_generation.py`)
4. Optionally preprocess hydrology records (`river_analysis_pre.py`)

These modules are mostly utility libraries imported by notebooks and higher-level scripts rather than standalone CLI programs.