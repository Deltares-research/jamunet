from osgeo import gdal

paths = [
    "data/satellite/original/JRC_GSW1_4_MonthlyHistory_testing_r1/1992_03_01_testing_r1.tif",
    "data/satellite/dataset_month3/JRC_GSW1_4_MonthlyHistory_testing_r1/1992_03_01_testing_r1.tif",
]

for p in paths:
    d = gdal.Open(p)
    if d is None:
        print(f"FILE {p}: CANNOT OPEN")
        continue
    print(f"FILE {p}")
    print(f"  SIZE {d.RasterXSize} x {d.RasterYSize}")
    gt = d.GetGeoTransform()
    print(f"  GT {gt}")
    pr = d.GetProjectionRef()
    print(f"  HAS_CRS {bool(pr)}")
    print(f"  PROJ_PREFIX {pr[:120].replace(chr(10), ' ')}")
    print("---")
