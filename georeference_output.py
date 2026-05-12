"""
Add georeferencing to a prediction TIFF by copying CRS and geotransform
from a reference original satellite image.

Usage:
    python georeference_output.py <input_tif> <reference_tif> <output_tif>
"""

import os
import sys
import warnings

from osgeo import gdal, osr

warnings.filterwarnings("ignore", category=FutureWarning)
gdal.UseExceptions()


def georeference(input_tif: str, reference_tif: str, output_tif: str) -> None:
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
    print(f"  CRS: EPSG:4326 (WGS 84)")
    print(f"  Geotransform: {geotransform}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python georeference_output.py <input_tif> <reference_tif> <output_tif>")
        sys.exit(1)
    georeference(sys.argv[1], sys.argv[2], sys.argv[3])
