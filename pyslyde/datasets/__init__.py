"""
Example data for PySlyde tutorials, documentation and integration tests.

PySlyde uses one small, openly licensed whole-slide image so that every
documented workflow can be reproduced from a fresh installation:

* **Slide** - ``CMU-1-Small-Region.svs`` (Aperio SVS, 2220 x 2967 px, ~1.9 MB)
  from the OpenSlide test-data collection, released under CC0-1.0
  (https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/).
  It is downloaded on first use, verified against a pinned SHA-256 checksum
  and cached locally, so it is only downloaded once per machine.
* **Annotations** - synthetic, computer-generated ``tissue`` and
  ``epithelium`` polygons (NOT pathologist annotations), shipped with the
  package in all six supported formats (GeoJSON, CSV, QuPath, ASAP, ImageJ and
  the custom PySlyde JSON). They are regenerated with
  ``scripts/make_example_annotations.py``.

Example
-------
>>> from pyslyde.datasets import example_data_dir
>>> root = example_data_dir()           # doctest: +SKIP
>>> sorted(p.name for p in (root / "annotations").iterdir())  # doctest: +SKIP
['example.csv', 'example.geojson', 'example_asap.xml', ...]

Environment variables
---------------------
``PYSLYDE_DATA_DIR``
    Use an existing directory (laid out as ``wsi/`` + ``annotations/``)
    instead of downloading anything.
``PYSLYDE_CACHE_DIR``
    Where downloaded files are cached (default: ``~/.cache/pyslyde``, or
    ``$XDG_CACHE_HOME/pyslyde``).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional
from urllib.request import Request, urlopen

__all__ = [
    "EXAMPLE_SLIDE",
    "ANNOTATION_FILES",
    "cache_dir",
    "download_verified",
    "fetch_example_slide",
    "example_annotations_dir",
    "example_data_dir",
]

#: Pinned description of the example slide (URL, checksum, licence).
EXAMPLE_SLIDE: Dict[str, str] = {
    "filename": "CMU-1-Small-Region.svs",
    "url": (
        "https://openslide.cs.cmu.edu/download/openslide-testdata/"
        "Aperio/CMU-1-Small-Region.svs"
    ),
    "sha256": "ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7",
    "license": "CC0-1.0",
    "source": "OpenSlide test data (https://openslide.org/)",
}

#: Annotation source name -> file shipped in ``pyslyde/datasets/annotations``.
ANNOTATION_FILES: Dict[str, str] = {
    "geojson": "example.geojson",
    "csv": "example.csv",
    "qupath": "example_qupath.json",
    "asap": "example_asap.xml",
    "imagej": "example_imagej.xml",
    "json": "example_custom.json",
}

_ANNOTATIONS_DIR = Path(__file__).resolve().parent / "annotations"


def cache_dir() -> Path:
    """Return (and create) the local cache directory for downloaded data."""
    env = os.environ.get("PYSLYDE_CACHE_DIR", "").strip()
    if env:
        root = Path(env).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
        root = (Path(xdg) if xdg else Path.home() / ".cache") / "pyslyde"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_verified(url: str, sha256: str, dest: Path, timeout: float = 60.0) -> Path:
    """
    Download ``url`` to ``dest`` unless a file with the expected checksum is
    already there.

    The download goes to a temporary file first and is only moved into place
    once its SHA-256 matches ``sha256``, so a cached file is always complete
    and verified.

    Args:
        url: Source URL.
        sha256: Expected SHA-256 hex digest (mandatory).
        dest: Destination file path.
        timeout: Socket timeout in seconds.

    Returns:
        Path to the verified file.

    Raises:
        ValueError: If ``sha256`` is empty.
        RuntimeError: If the downloaded file does not match ``sha256``.
    """
    if not sha256:
        raise ValueError("A SHA-256 checksum is required for every download.")
    dest = Path(dest)
    if dest.exists() and _sha256(dest) == sha256:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "pyslyde-datasets"})
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urlopen(req, timeout=timeout) as resp:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        got = _sha256(tmp)
        if got != sha256:
            raise RuntimeError(
                f"Checksum mismatch for {url}: expected {sha256}, got {got}."
            )
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def fetch_example_slide() -> Path:
    """
    Return the path to the example slide, downloading it on first use.

    Returns:
        Path to ``CMU-1-Small-Region.svs`` in the local cache.
    """
    dest = cache_dir() / EXAMPLE_SLIDE["filename"]
    return download_verified(EXAMPLE_SLIDE["url"], EXAMPLE_SLIDE["sha256"], dest)


def example_annotations_dir() -> Path:
    """Return the directory holding the synthetic example annotations."""
    return _ANNOTATIONS_DIR


def example_data_dir(root: Optional[os.PathLike] = None) -> Path:
    """
    Return a directory laid out as::

        <root>/
        |-- wsi/CMU-1-Small-Region.svs
        `-- annotations/example.geojson, example.csv, example_qupath.json,
                        example_asap.xml, example_imagej.xml, example_custom.json

    If ``PYSLYDE_DATA_DIR`` is set, that directory is returned unchanged.

    Args:
        root: Where to assemble the layout (default: ``<cache_dir>/example``).

    Returns:
        Path to the assembled example-data directory.
    """
    env = os.environ.get("PYSLYDE_DATA_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"PYSLYDE_DATA_DIR does not exist: {p}")
        return p

    root = Path(root) if root is not None else cache_dir() / "example"
    wsi_dir = root / "wsi"
    ann_dir = root / "annotations"
    wsi_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    slide = fetch_example_slide()
    target = wsi_dir / slide.name
    if not target.exists() or target.stat().st_size != slide.stat().st_size:
        shutil.copy2(slide, target)

    for name in ANNOTATION_FILES.values():
        shutil.copy2(_ANNOTATIONS_DIR / name, ann_dir / name)
    return root.resolve()
