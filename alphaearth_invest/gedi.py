"""GEDI L4A version 2.1 footprints from NASA's ORNL DAAC, via earthaccess.

Searching NASA's catalog needs no account. Downloading granules needs a free
NASA Earthdata login (https://urs.earthdata.nasa.gov), supplied through the
EARTHDATA_TOKEN or EARTHDATA_USERNAME/EARTHDATA_PASSWORD environment variables
or a urs.earthdata.nasa.gov entry in ~/.netrc.

Each granule is one HDF5 file per orbit segment (roughly 150-250 MB) with one
group per laser beam. Granules are cached, so a later run over the same area and
dates downloads nothing and needs no login.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import earthaccess
import h5py
import numpy as np
from earthaccess.exceptions import LoginStrategyUnavailable

SHORT_NAME = "GEDI_L4A_AGB_Density_V2_1_2056"
PRODUCT = "ORNL DAAC GEDI_L4A_AGB_Density_V2_1_2056, doi:10.3334/ORNLDAAC/2056"
LONGITUDE = "lon_lowestmode"
LATITUDE = "lat_lowestmode"

Bounds = tuple[float, float, float, float]  # west, south, east, north in degrees


def search(bounds: Bounds, start: str, end: str) -> list[earthaccess.DataGranule]:
    """Granules whose footprint intersects ``bounds`` between ``start`` and ``end``."""
    return earthaccess.search_data(short_name=SHORT_NAME, bounding_box=bounds, temporal=(start, end))


def _file_name(granule: earthaccess.DataGranule) -> str:
    links = [link for link in granule.data_links(access="external") if link.endswith(".h5")]
    if len(links) != 1:
        raise RuntimeError(f"Expected one HDF5 link for granule, found {links}")
    return links[0].rsplit("/", 1)[1]


def login() -> None:
    """Log in to NASA Earthdata from the environment or ~/.netrc; never prompts."""
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
        except LoginStrategyUnavailable:
            continue
        if auth.authenticated:
            return
    raise RuntimeError(
        "Downloading GEDI needs a NASA Earthdata login. Register free at https://urs.earthdata.nasa.gov, "
        "then set EARTHDATA_TOKEN (or EARTHDATA_USERNAME and EARTHDATA_PASSWORD) or add "
        "'machine urs.earthdata.nasa.gov login USER password PASS' to ~/.netrc."
    )


def granule_files(granules: list[earthaccess.DataGranule], directory: Path) -> list[Path]:
    """Local paths of ``granules`` in ``directory``, downloading any that are missing.

    Downloads land in a temporary subdirectory and move into ``directory`` only
    when complete, so a cached file is always whole.
    """
    by_name = {_file_name(granule): granule for granule in granules}
    missing = [granule for name, granule in by_name.items() if not (directory / name).exists()]
    if missing:
        login()
        directory.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".download-", dir=directory))
        try:
            earthaccess.download(missing, local_path=staging, show_progress=False)
            for granule in missing:
                name = _file_name(granule)
                if (staging / name).exists():
                    (staging / name).replace(directory / name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        still_missing = [name for name in by_name if not (directory / name).exists()]
        if still_missing:
            raise RuntimeError(f"{len(still_missing)} GEDI granules failed to download, e.g. {still_missing[0]}")
    return [directory / name for name in by_name]


def read_shots(path: Path, bounds: Bounds, datasets: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Longitude, latitude, and ``datasets`` for every shot inside ``bounds``, across all beams."""
    west, south, east, north = bounds
    names = (LONGITUDE, LATITUDE, *datasets)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    with h5py.File(path, "r") as granule:
        for beam_name in sorted(name for name in granule if name.startswith("BEAM")):
            beam = granule[beam_name]
            longitude, latitude = beam[LONGITUDE][()], beam[LATITUDE][()]
            inside = (longitude >= west) & (longitude <= east) & (latitude >= south) & (latitude <= north)
            if not inside.any():
                continue
            parts[LONGITUDE].append(longitude[inside])
            parts[LATITUDE].append(latitude[inside])
            for name in datasets:
                parts[name].append(beam[name][()][inside])
    return {name: np.concatenate(values) if values else np.empty(0) for name, values in parts.items()}
