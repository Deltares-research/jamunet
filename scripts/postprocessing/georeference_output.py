"""
Add georeferencing to a prediction TIFF by copying CRS and geotransform
from a reference original satellite image.

Usage:
    python georeference_output.py <input_tif> <reference_tif> <output_tif>
"""

import os
import sys
import warnings

import numpy as np

try:
    import rasterio
except Exception:  # pragma: no cover - optional dependency
    rasterio = None

try:
    from osgeo import gdal
except Exception:  # pragma: no cover - optional dependency
    gdal = None

warnings.filterwarnings("ignore", category=FutureWarning)
if gdal is not None:
    gdal.UseExceptions()


def _georeference_with_rasterio(input_tif: str, reference_tif: str, output_tif: str) -> None:
    with rasterio.open(reference_tif) as ref_ds:
        transform = ref_ds.transform
        crs = ref_ds.crs

    with rasterio.open(input_tif) as src_ds:
        data = src_ds.read(1)
        profile = src_ds.profile.copy()

    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "transform": transform,
            "crs": crs,
            "compress": "lzw",
            "height": int(data.shape[0]),
            "width": int(data.shape[1]),
        }
    )
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)

    if os.path.exists(output_tif):
        os.remove(output_tif)
    with rasterio.open(output_tif, "w", **profile) as out_ds:
        out_ds.write(data.astype(profile["dtype"]) if profile.get("dtype") else np.asarray(data), 1)

    print(f"Georeferenced TIF written: {output_tif}")
    print(f"  CRS: {crs}")
    print(f"  Transform: {transform}")


def _georeference_with_gdal(input_tif: str, reference_tif: str, output_tif: str) -> None:
    if gdal is None:
        raise RuntimeError("GDAL backend unavailable")

    ref_ds = gdal.Open(reference_tif)
    if ref_ds is None:
        raise FileNotFoundError(f"Cannot open reference TIF: {reference_tif}")
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjectionRef()
    ref_ds = None  # close

    src_ds = gdal.Open(input_tif)
    if src_ds is None:
        raise FileNotFoundError(f"Cannot open input TIF: {input_tif}")
    band = src_ds.GetRasterBand(1)
    data = band.ReadAsArray()
    dtype = band.DataType
    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize
    src_ds = None  # close

    driver = gdal.GetDriverByName("GTiff")
    if os.path.exists(output_tif):
        os.remove(output_tif)
    out_ds = driver.Create(
        output_tif,
        xsize,
        ysize,
        1,
        dtype,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_ds.GetRasterBand(1).WriteArray(data)
    out_ds.FlushCache()
    out_ds = None

    print(f"Georeferenced TIF written: {output_tif}")
    print(f"  CRS: copied from reference")
    print(f"  Geotransform: {geotransform}")


def georeference(input_tif: str, reference_tif: str, output_tif: str) -> None:
    if rasterio is not None:
        _georeference_with_rasterio(input_tif, reference_tif, output_tif)
        return

    _georeference_with_gdal(input_tif, reference_tif, output_tif)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python georeference_output.py <input_tif> <reference_tif> <output_tif>")
        sys.exit(1)
    georeference(sys.argv[1], sys.argv[2], sys.argv[3])
