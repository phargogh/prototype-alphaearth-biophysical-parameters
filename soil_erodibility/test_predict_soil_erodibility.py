import pytest
import shapely

import erodibility
from alphaearth_invest.aoi import aoi_from_bbox, aoi_in_crs
from alphaearth_invest import soilgrids

CRS = "EPSG:32610"


@pytest.mark.network
def test_soilgrids_reference_samples_cover_the_aoi_with_plausible_k(network, tmp_path):
    aoi = aoi_from_bbox(-122.26, 37.39, -122.24, 37.41)
    samples = soilgrids.reference_samples(
        erodibility.epic_k, erodibility.SOILGRIDS_PROPERTIES, 30, aoi, CRS, tmp_path, source="test"
    )
    projected = aoi_in_crs(aoi, CRS)
    inside = shapely.contains_xy(projected, samples.x, samples.y)
    # The AOI holds about 63 SoilGrids pixels (250 m); expect nearly all of them.
    assert inside.sum() >= 0.8 * projected.area / 250**2
    assert ((samples.value > 0.009) & (samples.value < 0.092)).all()  # InVEST guide's typical range
    assert (tmp_path / "soilgrids" / "sand_0-5cm_mean.vrt").exists()
