# conftest.py
from __future__ import annotations

import hashlib
import os
import re
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, build_opener

import gdown
import pytest

DEFAULT_DATA_URL = (
    "https://drive.google.com/uc?export=download&id=1a5RvYeoqAmAOTaoRAxYINxlwp0lOIMHY"
)

DEFAULT_DATA_SHA256 = ""


def _get_env_with_default(var_name: str, default: str = "") -> str:
    """
    Get an environment variable with a fallback default value.

    Args:
        var_name: Name of the environment variable
        default: Default value to use if environment variable is not set

    Returns:
        The environment variable value if set and non-empty, otherwise the default
    """
    value = os.environ.get(var_name, "").strip()
    return value if value else default


def _sha256(path: Path) -> str:
    """Compute the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_google_drive_direct(url: str) -> str:
    """
    Convert common Google Drive share URLs into a direct-download URL.

    Supports:
      - https://drive.google.com/file/d/<ID>/view?...

    Rewrites to:
      - https://drive.google.com/uc?export=download&id=<ID>
    """
    m = re.search(r"drive\.google\.com/file/d/([^/]+)/", url)
    if not m:
        return url
    file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _download_url_to_file(url: str, dest: Path) -> None:
    """
    Download a URL to a local file.

    Uses streaming download and follows redirects.

    Args:
        url: Source URL.
        dest: Destination path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = Request(
        url, headers={"User-Agent": "Mozilla/5.0 (pytest-fixtures-downloader)"}
    )
    with build_opener().open(req) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _extract_file_id_from_url(url: str) -> str | None:
    """Extract Google Drive file ID from various URL formats."""
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        return m.group(1)

    m = re.search(r"/d/([^/]+)", url)
    if m:
        return m.group(1)

    return None


def _download_google_drive(url: str, dest: Path) -> None:
    """
    Download a Google Drive file using the gdown library.

    gdown handles Google Drive's complex download flow including:
    - Confirmation pages
    - Virus scan warnings
    - Large file confirmations
    - Cookies and redirects

    Args:
        url: A Drive URL (any format: share link, direct download, etc.)
        dest: Destination file path.

    Raises:
        RuntimeError: If gdown is not installed or download fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    file_id = _extract_file_id_from_url(url)
    if not file_id:
        file_id = url

    try:
        gdown.download(
            id=file_id, output=str(dest), quiet=False, fuzzy=True, resume=True
        )

        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError("Download failed: file is empty or does not exist")

    except Exception as e:
        raise RuntimeError(f"Google Drive download failed: {e}") from e


def _extract(archive: Path, dest: Path) -> None:
    """
    Extract a supported archive into a destination directory.

    Archive type is detected by inspecting file contents.

    Supported:
      - ZIP (zipfile.is_zipfile)
      - TAR (tarfile.is_tarfile; including gzip-compressed tar)
    """
    dest.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
        return

    if tarfile.is_tarfile(archive):
        try:
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(dest)
            return
        except tarfile.ReadError:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest)
            return

    raise ValueError(
        f"Unsupported archive format: {archive.name!r} "
        f"(expected a .zip or .tar(.gz/.tgz) archive)"
    )


def _resolve_extracted_root(extract_dir: Path) -> Path:
    """Use extract_dir/data if present; otherwise use extract_dir."""
    data_dir = extract_dir / "data"
    return data_dir.resolve() if data_dir.exists() else extract_dir.resolve()


@pytest.fixture(scope="session")
def integration_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """
    Resolve the integration fixture root directory (local dir/file or downloaded archive).

    The fixture source can be specified via environment variables.
    Default values can be set in the script (see DEFAULT_DATA_URL and DEFAULT_DATA_SHA256).

    Priority (highest to lowest):
    1. PYSLYDE_IT_DATA_DIR environment variable
    2. PYSLYDE_IT_DATA_URL environment variable
    3. DEFAULT_DATA_URL
    4. None (if no sources are configured)

    - PYSLYDE_IT_DATA_DIR: Path to a local directory or archive file
    - PYSLYDE_IT_DATA_URL: URL to download an archive from
    - PYSLYDE_IT_DATA_SHA256: Expected SHA-256 checksum (optional, for URL downloads)

    For Google Drive URLs, the gdown library is used for reliable downloads.

    Returns:
        Path to the extracted data directory, or None if no source is configured.
    """
    local = _get_env_with_default("PYSLYDE_IT_DATA_DIR")
    if local:
        p = Path(local).expanduser().resolve()
        if not p.exists():
            pytest.skip(f"PYSLYDE_IT_DATA_DIR does not exist: {p}")

        if p.is_dir():
            return p

        cache_root = tmp_path_factory.mktemp("pyslyde_it_data_local")
        extract_dir = cache_root / "extracted"
        _extract(p, extract_dir)
        return _resolve_extracted_root(extract_dir)

    url = _get_env_with_default("PYSLYDE_IT_DATA_URL", DEFAULT_DATA_URL)
    if not url:
        return None

    url = _maybe_google_drive_direct(url)

    cache_root = tmp_path_factory.mktemp("pyslyde_it_data_url")
    archive = cache_root / "fixtures.download"

    if "drive.google.com" in url:
        _download_google_drive(url, archive)
    else:
        _download_url_to_file(url, archive)

    expected = _get_env_with_default("PYSLYDE_IT_DATA_SHA256", DEFAULT_DATA_SHA256)
    if expected:
        got = _sha256(archive)
        if got != expected:
            raise RuntimeError(
                f"Integration fixture checksum mismatch: expected {expected}, got {got}"
            )

    extract_dir = cache_root / "extracted"
    _extract(archive, extract_dir)
    return _resolve_extracted_root(extract_dir)
