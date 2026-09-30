"""Atomic file output and cached downloads.

Outputs and cache entries are written under a temporary name in the target
directory and renamed only when complete. An existing file at the target path is
therefore always whole, and a failed or interrupted run leaves nothing behind.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import requests


@contextlib.contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Yield a temporary path beside ``path``; rename it to ``path`` if the block succeeds, delete it otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_bytes(url: str) -> bytes:
    with requests.get(url, timeout=(30, 300)) as response:
        response.raise_for_status()
        return response.content


def download_once(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` unless it is already there."""
    if not path.exists():
        with atomic_output(path) as temporary, requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    file.write(chunk)
    return path
