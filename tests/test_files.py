import pytest

from alphaearth_invest.files import atomic_output


def test_atomic_output_renames_on_success(tmp_path):
    target = tmp_path / "out" / "result.tif"
    with atomic_output(target) as temporary:
        temporary.write_text("complete")
    assert target.read_text() == "complete"
    assert list(target.parent.iterdir()) == [target]


def test_atomic_output_leaves_nothing_on_failure(tmp_path):
    target = tmp_path / "result.tif"
    with pytest.raises(RuntimeError), atomic_output(target) as temporary:
        temporary.write_text("partial")
        raise RuntimeError("interrupted")
    assert list(tmp_path.iterdir()) == []
