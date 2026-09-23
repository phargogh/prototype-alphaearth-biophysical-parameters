import numpy as np
import pytest
import rasterio
import shapely
from sklearn.dummy import DummyRegressor

from alphaearth_invest.embeddings import RAW_NODATA
from alphaearth_invest.invest_raster import NODATA
from alphaearth_invest import pipeline
from alphaearth_invest.grid import OutputGrid

CRS = "EPSG:32610"


def write_stack(path, grid, raw):
    profile = {
        "driver": "GTiff", "count": 64, "dtype": "int8", "nodata": RAW_NODATA, "crs": grid.crs,
        "transform": grid.transform, "width": grid.width, "height": grid.height,
    }
    with rasterio.open(path, "w", **profile) as stack:
        stack.write(raw)
    return path


@pytest.fixture
def grid():
    return OutputGrid.covering(shapely.box(500000, 4000000, 500200, 4000100), CRS)  # 20 x 10 pixels


def test_reference_samples_validate_their_contract():
    with pytest.raises(ValueError, match="equal lengths"):
        pipeline.ReferenceSamples(np.zeros(2), np.zeros(3), np.zeros(2), "test")
    with pytest.raises(ValueError, match="finite"):
        pipeline.ReferenceSamples(np.zeros(1), np.zeros(1), np.array([np.nan]), "test")


def test_spatial_blocks_group_points_by_square():
    x = np.array([10, 1990, 2010, 10])
    y = np.array([10, 10, 10, 2010])
    groups = pipeline.spatial_blocks(x, y, 2000)
    assert groups[0] == groups[1]
    assert len({groups[0], groups[2], groups[3]}) == 3


def test_regression_metrics_for_known_errors():
    metrics = pipeline.regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["bias"] == pytest.approx(0)
    assert metrics["r2"] == pytest.approx(0)


def test_cross_validation_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 64)).astype(np.float32)
    groups = np.repeat(np.arange(20), 20)
    config = pipeline.TrainingConfig(number_of_trees=50)
    signal = pipeline.cross_validate(features, 3 * features[:, 0] + rng.normal(scale=0.1, size=400), groups, config)
    noise = pipeline.cross_validate(features, rng.normal(size=400), groups, config)
    assert signal["n_blocks"] == 20 and signal["n_samples"] == 400
    assert signal["r2"] > 0.4
    assert noise["r2"] < 0.1


def test_cross_validation_needs_enough_blocks():
    features = np.zeros((200, 64))
    with pytest.raises(pipeline.InsufficientSamplesError, match="blocks"):
        pipeline.cross_validate(features, np.arange(200.0), np.zeros(200, dtype=int), pipeline.TrainingConfig())


def test_training_samples_drop_points_outside_aoi_or_without_embedding(tmp_path, grid):
    raw = np.full((64, grid.height, grid.width), 10, dtype=np.int8)
    raw[:, :, 0] = RAW_NODATA  # first column has no embedding
    stack = write_stack(tmp_path / "stack.tif", grid, raw)
    aoi = shapely.box(500000, 4000000, 500150, 4000100)
    reference = pipeline.ReferenceSamples(
        x=np.array([500005.0, 500055.0, 500175.0]),  # no embedding, kept, outside AOI
        y=np.array([4000055.0, 4000055.0, 4000055.0]),
        value=np.array([1.0, 2.0, 3.0]),
        source="test",
    )
    features, samples = pipeline.training_samples(reference, aoi, stack, pipeline.TrainingConfig())
    np.testing.assert_array_equal(samples.value, [2.0])
    assert features.shape == (1, 64)
    assert features[0, 0] == pytest.approx((10 / 127.5) ** 2)


def test_prediction_fills_aoi_pixels_with_embeddings_and_nodata_elsewhere(tmp_path, grid):
    raw = np.full((64, grid.height, grid.width), 10, dtype=np.int8)
    raw[:, 0, :] = RAW_NODATA  # top row has no embedding
    stack = write_stack(tmp_path / "stack.tif", grid, raw)
    model = DummyRegressor(strategy="constant", constant=5.0).fit(np.zeros((1, 64)), [5.0])
    left_half = shapely.box(500000, 4000000, 500100, 4000100)
    output = tmp_path / "out" / "prediction.tif"

    pipeline.predict_to_geotiff(model, stack, left_half, grid, output, "test_parameter")

    with rasterio.open(output) as result:
        values = result.read(1)
        assert result.nodata == NODATA and result.dtypes[0] == "float32" and result.crs.to_string() == CRS
        assert result.descriptions == ("test_parameter",)
    assert (values[1:, :10] == 5.0).all()
    assert (values[0, :] == NODATA).all()
    assert (values[:, 10:] == NODATA).all()
    assert sorted(p.name for p in output.parent.iterdir()) == ["prediction.tif"]


def test_prediction_refuses_a_stack_on_another_grid(tmp_path, grid):
    other = OutputGrid.covering(shapely.box(500000, 4000000, 500100, 4000100), CRS)
    stack = write_stack(tmp_path / "stack.tif", other, np.zeros((64, other.height, other.width), dtype=np.int8))
    model = DummyRegressor().fit(np.zeros((1, 64)), [0.0])
    with pytest.raises(RuntimeError, match="not on the output grid"):
        pipeline.predict_to_geotiff(model, stack, shapely.box(500000, 4000000, 500200, 4000100), grid, tmp_path / "p.tif", "x")
    assert not (tmp_path / "p.tif").exists()
