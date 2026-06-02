# JamUNet: Predicting Morphological Changes of Braided Sand-Bed Rivers with Deep Learning

JamUNet is a deep-learning project for predicting river morphology changes in braided sand-bed rivers using satellite imagery.

## Project Origin and Repository Evolution

This work originates from the MSc thesis research of Antonio Magherini at TU Delft and Deltares.

The codebase and workflows were later extended and modified by students in the CEGM2003 course (Data Science and Artificial Intelligence for Engineers).

This repository contains a refactored and separated version of that evolving work, with reorganized scripts, documentation, and dataset conventions to support reproducible training and inference.

For the original thesis manuscript, see the TU Delft repository:
https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348

## Repository Structure

Main folders:

- `data`: Dataset folders and region metadata for training and evaluation.
- `Images`: Poster and visual project assets.
- `model`: Deep-learning modules, notebooks, and trained model checkpoints.
- `postprocessing`: Utilities for metrics, plotting, and result export.
- `preprocessing`: Utilities for dataset generation and preprocessing workflows.
- `benchmarks`: Baseline and no-change scenario experiments.

Supporting scripts are available in the repository root for data preparation, evaluation, georeferencing, and figure generation.

## Data Naming Convention

Satellite region folders use coordinate-based IDs, for example:
`lat24p6515_lon88p0207`

Region metadata is stored in:
`data/satellite/regions/region_catalog.json`

Model-ready region metadata and polygons are stored in:
`data/satellite/regions/region_catalog_model_ready.json`
`data/satellite/regions/region_catalog_model_ready.geojson`

Catalog usage notes:

- `region_catalog.json` describes the original download footprints (geographic regions).
- `region_catalog_model_ready.json` and `region_catalog_model_ready.geojson` describe model-ready footprints derived from the fixed preprocessing image size (default 1000x500 pixels at 60 m resolution).
- Use the model-ready catalog files when you want polygons that match the standardized model input dimensions.

Generate or refresh the model-ready catalog files with:

```powershell
cd <path-to-repo>
python prepare_satellite_dataset.py --model-ready-catalog-only
```

This command writes:
- `data/satellite/regions/region_catalog_model_ready.json`
- `data/satellite/regions/region_catalog_model_ready.geojson`

For detailed preprocessing inputs and outputs, see:
`preprocessing/README.md`

## Installation

Recommended full environment (scripts + notebooks):

```powershell
cd <path-to-repo>
conda env create -f braided.yml
conda activate braided
```

If the environment already exists:

```powershell
conda env update -f braided.yml --prune
conda activate braided
```

Minimal packages for running only the example inference script:

```powershell
pip install numpy pandas tifffile pillow torch
```

Optional dependency for georeferenced output generation:

```powershell
conda install -c conda-forge gdal
```

Notes:

- GDAL is not required when using `--skip-georef`.
- The default model checkpoint path is inside `model/models_trained`.
- The `--region` value must match a coordinate-based region ID from the region catalog.
- Running `prepare_satellite_dataset.py` also writes model-ready polygon catalogs under `data/satellite/regions/`.

## Quick Inference Example (Target Year 2021)

From the repository root:

```powershell
cd <path-to-repo>
python run_example.py --region lat24p6515_lon88p0207 --target-year 2021 --skip-georef
```

Notes:

- Replace `lat24p6515_lon88p0207` with your desired region ID.
- Remove `--skip-georef` to also generate georeferenced outputs.

## Citation

Please cite the original thesis as:

```bibtex
@mastersthesis{magherini2024,
  author       = {Magherini, A.},
  title        = {{JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning}},
  school       = {{Delft University of Technology}},
  year         = {2024},
  month        = {10},
  howpublished = {\url{https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348}}
}
```
