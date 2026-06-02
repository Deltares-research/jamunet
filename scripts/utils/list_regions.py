from osgeo import gdal
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

base = "data/satellite/original"
regions = sorted(os.listdir(base))
for r in regions:
    folder = os.path.join(base, r)
    tifs = sorted([f for f in os.listdir(folder) if f.endswith(".tif")])
    if not tifs:
        continue
    ds = gdal.Open(os.path.join(folder, tifs[0]))
    if ds is None:
        continue
    gt = ds.GetGeoTransform()
    x0 = gt[0]
    y0 = gt[3]
    x1 = x0 + gt[1] * ds.RasterXSize
    y1 = y0 + gt[5] * ds.RasterYSize
    print(f"{r:55s}  lon=[{x0:.3f}, {x1:.3f}]  lat=[{y1:.3f}, {y0:.3f}]")
