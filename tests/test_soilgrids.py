import pytest

from alphaearth_invest import soilgrids


def test_depth_weights_follow_interval_thickness():
    assert soilgrids.depth_weights(30) == [((0, 5), 5 / 30), ((5, 15), 10 / 30), ((15, 30), 15 / 30)]
    for depth in soilgrids.INTERVAL_BOTTOMS_CM:
        assert sum(weight for _, weight in soilgrids.depth_weights(depth)) == pytest.approx(1)


def test_depth_must_be_an_interval_boundary():
    with pytest.raises(ValueError, match="depth_cm"):
        soilgrids.depth_weights(20)


def test_layer_names_match_soilgrids():
    assert soilgrids.layer_name("sand", (0, 5)) == "sand_0-5cm_mean"
    with pytest.raises(ValueError):
        soilgrids.layer_name("nitrogen", (0, 5))
