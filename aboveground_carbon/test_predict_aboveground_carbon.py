import numpy as np
import pytest
import rasterio
from affine import Affine

import predict_aboveground_carbon as script
from alphaearth_invest.invest_raster import NODATA


def write(path, data, transform, crs, nodata):
    profile = {"driver": "GTiff", "count": 1, "dtype": data.dtype.name, "crs": crs, "transform": transform,
               "width": data.shape[1], "height": data.shape[0], "nodata": nodata}
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data, 1)
    return path


def test_c_above_by_lulc_averages_per_class_and_skips_nodata(tmp_path):
    transform = Affine(10, 0, 500000, 0, -10, 4000040)
    carbon = np.array([[1, 3, 10, 20], [NODATA, 5, 30, 40], [2, 2, 2, 2], [2, 2, 2, 2]], dtype=np.float32)
    prediction = write(tmp_path / "carbon.tif", carbon, transform, "EPSG:32610", NODATA)
    # Land cover covers only the top two rows (outside its extent is ignored) and has nodata 0.
    codes = np.array([[7, 7, 9, 9], [7, 7, 0, 9]], dtype=np.uint8)
    lulc = write(tmp_path / "lulc.tif", codes, transform, "EPSG:32610", 0)

    rows = {row["lucode"]: row for row in script.c_above_by_lulc(prediction, lulc)}

    assert set(rows) == {7, 9}
    assert rows[7]["c_above"] == pytest.approx(3.0) and rows[7]["pixel_count"] == 3  # 1, 3, 5
    assert rows[9]["c_above"] == pytest.approx(70 / 3) and rows[9]["pixel_count"] == 3  # 10, 20, 40
