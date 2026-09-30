import shapely

from alphaearth_invest.grid import OutputGrid


def test_grid_is_aligned_to_pixel_multiples_and_covers_the_aoi():
    aoi = shapely.box(561934.2, 4136141.7, 570837.9, 4142871.3)
    grid = OutputGrid.covering(aoi, "EPSG:32610")
    west, south, east, north = grid.bounds
    assert (west, south, east, north) == (561930, 4136140, 570840, 4142880)
    assert (grid.width, grid.height) == (891, 674)


def test_windows_tile_the_grid_exactly():
    grid = OutputGrid.covering(shapely.box(0, 0, 12345, 6789), "EPSG:32631")
    windows = list(grid.windows(size=512))
    assert sum(w.width * w.height for w in windows) == grid.width * grid.height
    assert max(w.col_off + w.width for w in windows) == grid.width
    assert max(w.row_off + w.height for w in windows) == grid.height


def test_key_identifies_the_grid():
    a = OutputGrid.covering(shapely.box(0, 0, 1000, 1000), "EPSG:32610")
    b = OutputGrid.covering(shapely.box(0, 0, 1000, 1010), "EPSG:32610")
    assert a.key == OutputGrid.covering(shapely.box(0, 0, 1000, 1000), "EPSG:32610").key
    assert a.key != b.key


def test_inside_uses_pixel_centres():
    grid = OutputGrid.covering(shapely.box(0, 0, 100, 100), "EPSG:32610")
    left_half = shapely.box(0, 0, 50, 100)
    (window,) = list(grid.windows())
    inside = grid.inside(left_half, window)
    assert inside.shape == (10, 10)
    assert inside[:, :5].all() and not inside[:, 5:].any()
