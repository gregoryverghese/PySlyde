# conftest.py
"""
Shared pytest configuration and fixtures.

Integration data
----------------
Tests marked ``integration`` run on the PySlyde example dataset
(see :mod:`pyslyde.datasets`):

* the CC0-licensed OpenSlide slide ``CMU-1-Small-Region.svs``, downloaded once,
  verified against a pinned SHA-256 and cached (``~/.cache/pyslyde`` or
  ``$PYSLYDE_CACHE_DIR``);
* synthetic annotations in all six supported formats, shipped in
  ``pyslyde/datasets/annotations``.

Sources, in order of precedence:

1. ``PYSLYDE_IT_DATA_DIR`` - an existing directory (``wsi/`` + ``annotations/``)
   or a local ``.zip`` / ``.tar(.gz)`` archive with that layout.
2. ``PYSLYDE_IT_DATA_URL`` + ``PYSLYDE_IT_DATA_SHA256`` - a remote archive with
   that layout. The checksum is mandatory.
3. The bundled example dataset (default).

If the data cannot be obtained (for example, no network on first run) the
integration tests are skipped with the reason, unless ``PYSLYDE_IT_STRICT=1``
is set, in which case they fail. CI runs in strict mode.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

import pytest


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _strict() -> bool:
    return _env("PYSLYDE_IT_STRICT") in ("1", "true", "True", "yes")


def _unavailable(msg: str):
    """Skip, or fail in strict mode, when integration data is unavailable."""
    if _strict():
        pytest.fail(f"[PYSLYDE_IT_STRICT] {msg}", pytrace=False)
    pytest.skip(msg)


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a .zip or .tar(.gz) archive, refusing paths that escape ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()

    def _check(name: str) -> None:
        target = (dest / name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe path in archive {archive.name}: {name!r}")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for n in zf.namelist():
                _check(n)
            zf.extractall(dest)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            for m in tf.getmembers():
                _check(m.name)
            tf.extractall(dest)
        return
    raise ValueError(
        f"Unsupported archive format: {archive.name!r} "
        "(expected .zip or .tar/.tar.gz/.tgz)"
    )


def _resolve_root(extract_dir: Path) -> Path:
    """Use ``extract_dir/data`` if present, otherwise ``extract_dir``."""
    data_dir = extract_dir / "data"
    return (data_dir if data_dir.exists() else extract_dir).resolve()


@pytest.fixture(scope="session")
def integration_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Root directory of the integration data (``wsi/`` + ``annotations/``).

    See the module docstring for the supported sources.
    """
    local = _env("PYSLYDE_IT_DATA_DIR")
    if local:
        p = Path(local).expanduser().resolve()
        if not p.exists():
            _unavailable(f"PYSLYDE_IT_DATA_DIR does not exist: {p}")
        if p.is_dir():
            return p
        extract_dir = tmp_path_factory.mktemp("pyslyde_it_local")
        _safe_extract(p, extract_dir)
        return _resolve_root(extract_dir)

    from pyslyde.datasets import cache_dir, download_verified, example_data_dir

    url = _env("PYSLYDE_IT_DATA_URL")
    if url:
        sha = _env("PYSLYDE_IT_DATA_SHA256")
        if not sha:
            pytest.fail(
                "PYSLYDE_IT_DATA_URL is set but PYSLYDE_IT_DATA_SHA256 is empty; "
                "a checksum is required for remote integration data.",
                pytrace=False,
            )
        try:
            archive = download_verified(url, sha, cache_dir() / f"it-{sha[:16]}.archive")
        except Exception as e:  # network error or checksum mismatch
            _unavailable(f"Could not fetch integration data from {url}: {e}")
        extract_dir = tmp_path_factory.mktemp("pyslyde_it_url")
        _safe_extract(archive, extract_dir)
        return _resolve_root(extract_dir)

    try:
        return example_data_dir()
    except Exception as e:
        _unavailable(f"Could not prepare the PySlyde example dataset: {e}")
