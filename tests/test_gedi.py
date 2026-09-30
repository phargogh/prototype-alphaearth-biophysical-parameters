import h5py
import numpy as np
import pytest

from alphaearth_invest import gedi


def write_granule(path, beams):
    with h5py.File(path, "w") as granule:
        granule.create_group("METADATA")  # non-beam groups are ignored
        for name, (lon, lat, agbd) in beams.items():
            beam = granule.create_group(name)
            beam[gedi.LONGITUDE], beam[gedi.LATITUDE], beam["agbd"] = lon, lat, agbd
    return path


def test_read_shots_keeps_shots_inside_bounds_across_beams(tmp_path):
    path = write_granule(
        tmp_path / "g.h5",
        {
            "BEAM0000": (np.array([-122.25, -121.0]), np.array([37.40, 37.40]), np.array([10.0, 20.0])),
            "BEAM0101": (np.array([-122.21]), np.array([37.42]), np.array([30.0])),
        },
    )
    shots = gedi.read_shots(path, (-122.30, 37.37, -122.20, 37.43), ("agbd",))
    np.testing.assert_array_equal(shots["agbd"], [10.0, 30.0])
    np.testing.assert_array_equal(shots[gedi.LONGITUDE], [-122.25, -122.21])


def test_read_shots_returns_empty_arrays_when_nothing_is_inside(tmp_path):
    path = write_granule(tmp_path / "g.h5", {"BEAM0000": (np.array([0.0]), np.array([0.0]), np.array([1.0]))})
    shots = gedi.read_shots(path, (10, 10, 11, 11), ("agbd",))
    assert all(len(values) == 0 for values in shots.values())


class FakeGranule:
    def __init__(self, name):
        self.name = name

    def data_links(self, access=None):
        return [f"https://example.test/data/{self.name}", f"https://example.test/data/{self.name}.sha256"]


def test_cached_granules_need_no_login(tmp_path, monkeypatch):
    (tmp_path / "a.h5").write_bytes(b"cached")
    monkeypatch.setattr(gedi, "login", lambda: pytest.fail("login should not be needed"))
    assert gedi.granule_files([FakeGranule("a.h5")], tmp_path) == [tmp_path / "a.h5"]


def test_missing_granules_are_downloaded_then_moved_into_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(gedi, "login", lambda: None)

    def fake_download(granules, local_path, show_progress):
        for granule in granules:
            (local_path / granule.name).write_bytes(b"downloaded")

    monkeypatch.setattr(gedi.earthaccess, "download", fake_download)
    paths = gedi.granule_files([FakeGranule("a.h5"), FakeGranule("b.h5")], tmp_path)
    assert [p.read_bytes() for p in paths] == [b"downloaded", b"downloaded"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.h5", "b.h5"]  # staging directory removed


def test_failed_downloads_raise_and_leave_no_partial_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gedi, "login", lambda: None)
    monkeypatch.setattr(gedi.earthaccess, "download", lambda granules, local_path, show_progress: [])
    with pytest.raises(RuntimeError, match="failed to download"):
        gedi.granule_files([FakeGranule("a.h5")], tmp_path)
    assert list(tmp_path.iterdir()) == []
