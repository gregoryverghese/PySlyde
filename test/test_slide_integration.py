"""
test_slide_integration.py

This module contains opt-in integration tests for PySlyde. It relies on
``conftest.py`` to resolve the integration fixture root directory via
environment variables and (optionally) download/extract an archive.

Fixture layout
--------------
The fixture root directory (referred to here as ``PARENT``) must contain:

PARENT
├── annotations
│   ├── asap.xml
│   ├── name.csv
│   ├── geojson.json or .geojson
│   ├── imagej.xml
│   └── qupath.json
└── wsi
    └── wsi.ndpi

Notes:
- The annotations directory may contain additional files; tests select the first
  match per supported pattern and will include the chosen path in skip messages.
- URL-based test data must point to a zipped archive (``.zip`` or
  ``.tar``, ``.tar.gz``, ``.tgz``).

Configuration and precedence
----------------------------
The fixture root is resolved using the following precedence:

1) ``PYSLYDE_IT_DATA_DIR`` (highest priority)
   - If it points to a directory: used as-is.
   - If it points to an archive file: extracted and used.

2) ``PYSLYDE_IT_DATA_URL``
   - Download an archive and extract it.
   - Optional integrity check via ``PYSLYDE_IT_DATA_SHA256``.

3) Fall back on ``PYSLYDE_IT_DATA_URL`` set in confest.py.

If neither is provided, tests will skip cleanly with an explanation.

Google Drive
------------
If ``PYSLYDE_IT_DATA_URL`` is a Google Drive link, ``conftest.py`` uses the
``gdown`` library for reliable downloads.

How to run
----------
Using defaults (only if ``conftest.py`` defines internal defaults and no env var
is provided):

    pytest path/to/test_slide_integration.py --basetemp=<optional_temp_dir>

Using a local fixture directory or archive:

    PYSLYDE_IT_DATA_DIR=/path/to/PARENT pytest path/to/test_slide_integration.py --basetemp=<optional_temp_dir>
    PYSLYDE_IT_DATA_DIR=/path/to/fixtures.zip pytest path/to/test_slide_integration.py --basetemp=<optional_temp_dir>

Using a remote archive URL:

    PYSLYDE_IT_DATA_URL=<archive_url> pytest path/to/test_slide_integration.py --basetemp=<optional_temp_dir>
    PYSLYDE_IT_DATA_URL=<archive_url> PYSLYDE_IT_DATA_SHA256=<sha256> pytest path/to/test_slide_integration.py --basetemp=<optional_temp_dir>
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from pyslyde.slide import Slide

pytestmark = pytest.mark.integration

_ANN_SOURCES = ("geojson", "csv", "qupath", "asap", "imagej", "json")

_ANN_PATTERNS = {
    "geojson": ("*.geojson", "*geojson*.json"),
    "csv": ("*.csv",),
    "qupath": ("*qupath*.json",),
    "asap": ("*asap*.xml",),
    "imagej": ("*imagej*.xml",),
    "json": ("*custom*.json",),
}

_WSI_PATTERNS = (
    "*.svs",
    "*.ome.tif",
    "*.ome.tiff",
    "*.tif",
    "*.tiff",
    "*.ndpi",
)


def _fmt_selected(*, wsi: Path | None = None, ann: Path | None = None) -> str:
    """
    Format selected fixture paths for inclusion in skip messages.

    Args:
        wsi:
            The selected whole-slide image path, if one was resolved.
        ann:
            The selected annotation fixture path, if one was resolved.

    Returns:
        str
            A formatted, multi-line string  when at least one path is provided;
            otherwise an empty string.
    """
    parts: list[str] = []
    if wsi is not None:
        parts.append(f"wsi={wsi}")
    if ann is not None:
        parts.append(f"ann={ann}")
    return "\nFound/selected:\n" + "\n".join(parts) if parts else ""


def _skip_missing(
    reason: str, *, missing: list[str], wsi: Path | None = None, ann: Path | None = None
) -> None:
    """
    Skip the current test with a consistent, informative message.

    Standardises skip output across the integration suite.

    Args:
        reason:
            High-level explanation of why the test is being skipped.
        missing:
            A list of required resources or compatibility checks that are not satisfied.
            These are presented as a comma-separated list.
        wsi:
            The selected whole-slide image path, if one was resolved.
        ann:
            The selected annotation fixture path, if one was resolved.
    """
    msg = f"{reason}. Missing: " + ", ".join(missing) + _fmt_selected(wsi=wsi, ann=ann)
    pytest.skip(msg)


def _csv_has_pyslyde_schema(path: Path) -> tuple[bool, str]:
    """
    Check whether a CSV annotation fixture minimally matches the expected CSV schema.

    The loader expects the following columns to be present:

    - ``label``
    - ``polygon_id``
    - ``vertex_id``
    - ``x``
    - ``y``

    Args
        path:
            Path to the CSV file to validate.

    Returns:
        (bool, str)
            ``(ok, detail)`` where:

            - ``ok`` is ``True`` if all required columns are present.
            - ``detail`` is a short message suitable for including
              in a skip reason when ``ok`` is ``False``.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception as e:
        return (
            False,
            f"CSV fixture could not be read ({type(e).__name__}: {e}).",
        )

    required = {"label", "polygon_id", "vertex_id", "x", "y"}
    cols = set(header)
    missing = required - cols
    if missing:
        return (
            False,
            "CSV fixture is not in required CSV schema; expected columns "
            f"{sorted(required)} but missing {sorted(missing)}. Found: {header}",
        )
    return True, ""


def _geojson_root_is_supported(path: Path) -> tuple[bool, str]:
    """
    Check whether a GeoJSON fixture matches the container types expected by PySlyde.
    Otherwise skip with a clear message.

    PySlyde's GeoJSON loader expects the JSON root to be a mapping (dict) with:

    - ``type == "FeatureCollection"`` (and a ``features`` list), or
    - ``type == "Feature"``

    Some tools export GeoJSON as a bare list of Feature objects at the root,
    similar to QuPath's GeoJson style; that is valid JSON but is not supported by
    the current loader.

    Args:
        path:
            Path to the GeoJSON file to validate.

    Returns:
        (bool, str)
            ``(ok, detail)`` where:

            - ``ok`` is ``True`` if the root container type is supported.
            - ``detail`` is a short message suitable for a skip reason when ``ok`` is
            ``False``.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return (
            False,
            f"GeoJSON fixture could not be parsed as JSON ({type(e).__name__}: {e}).",
        )

    if isinstance(data, list):
        return (
            False,
            "GeoJSON fixture appears to be a bare list of Feature objects (often QuPath-style). "
            "_geojson expects a FeatureCollection or Feature object at the JSON root.",
        )

    if not isinstance(data, dict):
        return False, "GeoJSON fixture root is not a JSON object/dict."

    t = str(data.get("type", ""))
    if t not in {"FeatureCollection", "Feature"}:
        return (
            False,
            f"GeoJSON fixture root type is {t!r}; expected 'FeatureCollection' or 'Feature'.",
        )

    return True, ""


def _first_match(root: Path, patterns: Iterable[str]) -> tuple[Path | None, list[Path]]:
    """
    Find candidate fixture files under ``root`` matching one or more glob patterns.

    All matches from all patterns are collected, de-duplicated, resolved to absolute
    paths, and sorted. The first entry in the sorted list is returned as the selected
    fixture.

    Args
        root:
            Directory to search.
        patterns:
            One or more glob patterns to apply under ``root``.

    Returns:
        (Path | None, list[Path])
            ``(selected, matches)`` where:

            - ``selected`` is the first match in sorted order, or ``None`` if no matches exist.
            - ``matches`` is the full sorted list of all matches (possibly empty).
    """
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(root.glob(pat))
    hits = sorted({p.resolve() for p in hits})
    return (hits[0] if hits else None), hits


def _pick_wsi(data_root: Path) -> tuple[Path | None, list[Path]]:
    """
    Locate a whole-slide image (WSI) fixture under ``data_root / "wsi"``.

    Args:
        data_root:
            Root directory containing the integration fixtures.

    Returns:
        (Path | None, list[Path])
            ``(selected, matches)`` where:
            - ``selected`` is the chosen WSI path (first match in sorted order), or ``None``.
            - ``matches`` is the full sorted list of discovered WSI candidates.
    """
    return _first_match(data_root / "wsi", _WSI_PATTERNS)


def _ann_path(data_root: Path, source: str) -> tuple[Path | None, list[Path]]:
    """
    Locate an annotation fixture under ``data_root / "annotations"`` for the given source.

    Args:
        data_root:
            Root directory containing the integration fixtures.
        source:
            Annotation source identifier. Must be a key in ``_ANN_PATTERNS``.

    Returns:
        (Path | None, list[Path])
            ``(selected, matches)`` where:

            - ``selected`` is the chosen annotation file (first match in sorted order), or ``None``.
            - ``matches`` is the full sorted list of discovered annotation candidates.
    """
    return _first_match(data_root / "annotations", _ANN_PATTERNS[source])


def _require_integration_root(integration_data_dir: Path | None) -> Path:
    """
    Resolve the integration fixture root directory or skip if it is not configured.

    Args:
        integration_data_dir:
            The path resolved by the integration-data fixture (commonly from
            ``PYSLYDE_IT_DATA_DIR``), or ``None`` when not configured.

    Returns:
        Path
            The resolved integration data root.
    """
    if integration_data_dir is None:
        pytest.skip(
            "Integration data not configured. Provide either "
            "PYSLYDE_IT_DATA_DIR=<path> or PYSLYDE_IT_DATA_URL=<archive_url>."
        )
    return integration_data_dir


def _require_source_resources(data_root: Path, source: str) -> tuple[Path, Path]:
    """
    Resolve the WSI and annotation fixture paths for a given annotation source.

    In addition to basic file presence checks, this applies lightweight
    compatibility checks for fixtures that are frequently encountered in multiple
    schemas:

    - CSV: validates required header columns.
    - GeoJSON: validates supported root container type.

    Args:
        data_root:
            Root directory containing integration fixtures, including ``wsi/`` and
            ``annotations/`` subdirectories.
        source:
            Annotation source identifier, e.g. ``"geojson"``, ``"csv"``, ``"qupath"``,
            ``"asap"``, ``"imagej"``, ``"json"``.

    Returns:
        (Path, Path)
            ``(wsi_path, ann_path)``: the selected WSI path and selected annotation path.
    """
    missing: list[str] = []

    wsi_sel, _ = _pick_wsi(data_root)
    if wsi_sel is None:
        missing.append(f"wsi/{' or '.join(_WSI_PATTERNS)}")

    ann_sel, _ = _ann_path(data_root, source)
    if ann_sel is None:
        pats = " or ".join(_ANN_PATTERNS[source])
        missing.append(f"annotations/{pats}")

    if missing:
        _skip_missing(
            f"Integration resources unavailable for source={source!r}",
            missing=missing,
            wsi=wsi_sel,
            ann=ann_sel,
        )

    if source == "csv":
        ok, detail = _csv_has_pyslyde_schema(ann_sel)
        if not ok:
            _skip_missing(
                f"Integration annotation fixture incompatible for source={source!r}",
                missing=[detail],
                wsi=wsi_sel,
                ann=ann_sel,
            )

    if source == "geojson":
        ok, detail = _geojson_root_is_supported(ann_sel)
        if not ok:
            _skip_missing(
                f"Integration annotation fixture incompatible for source={source!r}",
                missing=[detail],
                wsi=wsi_sel,
                ann=ann_sel,
            )

    return wsi_sel, ann_sel


def _assert_region_alignment_and_foreground(
    img: np.ndarray, roi_mask: np.ndarray
) -> None:
    """
    Assert basic invariants for the output of :meth:``pyslyde.slide.Slide.generate_region``.

    Args:
        img:
            RGB region image returned by :meth:``pyslyde.slide.Slide.generate_region``.
            Expected shape is ``(H, W, 3)``.
        roi_mask:
            ROI mask returned by :meth:``pyslyde.slide.Slide.generate_region``.
            Expected shape is ``(H, W)`` and must contain at least one foreground pixel.
    """
    assert isinstance(img, np.ndarray)
    assert isinstance(roi_mask, np.ndarray)

    assert img.ndim == 3 and img.shape[2] == 3
    assert roi_mask.ndim == 2
    assert img.shape[:2] == roi_mask.shape[:2]
    assert np.any(roi_mask > 0)


def _assert_label_filtering_produces_foreground(
    *,
    slide: Slide,
    level: int,
    roi_mask: np.ndarray,
) -> None:
    """
    Assert that label filtering works for both label names and encoded label IDs.

    This selects a class that is *actually present* in the provided ``roi_mask``
    (avoiding assumptions about label ordering), then verifies that
    :meth:``pyslyde.slide.Slide.generate_region`` returns non-empty masks
    when filtering by both the corresponding label string and numeric ID.

    Args:
        slide:
            A :class:``pyslyde.slide.Slide`` instance with annotations loaded.
        level:
            Pyramid level used when calling :meth:``pyslyde.slide.Slide.generate_region``.
        roi_mask:
            ROI mask generated without label filtering (used to discover which classes are
            present).
    """
    if slide.annotations is None:
        pytest.skip("No annotations available; cannot validate label filtering.")

    n_classes = len(slide.annotations.class_key)
    if n_classes < 1:
        pytest.skip(
            "Annotations exist but no classes are defined; cannot validate labels."
        )

    present = np.unique(roi_mask)
    present = present[present > 0]
    if present.size == 0:
        pytest.skip(
            "ROI mask contains no foreground classes; cannot validate label filtering."
        )

    chosen_id = int(present[0])
    chosen_label = slide.annotations.id_to_label[chosen_id]

    _, m_name = slide.generate_region(level=level, labels=[chosen_label])
    _, m_id = slide.generate_region(level=level, labels=[chosen_id])

    assert m_name.shape == roi_mask.shape
    assert m_id.shape == roi_mask.shape
    assert np.any(m_name > 0)
    assert np.any(m_id > 0)


@pytest.mark.parametrize("source", _ANN_SOURCES)
def test_generate_mask(integration_data_dir: Path | None, source: str) -> None:
    """
    Integration test for :meth:``pyslyde.slide.Slide.generate_mask``.

    Verifies end-to-end behaviour on real data:
    - OpenSlide can open the WSI and expose a pyramid.
    - Annotations are loaded via the requested loader.
    - A 2D mask is produced at the selected pyramid level.
    - The mask has the expected shape and contains foreground.

    Args:
        integration_data_dir:
            Root directory containing integration fixtures (WSI + annotations).
            If not configured, the test is skipped.
        source:
            Annotation source identifier used by :class:``pyslyde.slide.Annotations``.
    """
    data_root = _require_integration_root(integration_data_dir)
    wsi_path, ann_path = _require_source_resources(data_root, source)

    s = Slide(str(wsi_path), annotations_path=str(ann_path), source=source)
    level = s.level_count - 1

    mask = s.generate_mask(level=level)
    assert mask.ndim == 2

    w, h = s.level_dimensions[level]
    assert mask.shape == (h, w)
    assert int(mask.min()) == 0
    assert np.any(mask > 0)


@pytest.mark.parametrize("source", _ANN_SOURCES)
def test_generate_region_alignment(
    integration_data_dir: Path | None, source: str
) -> None:
    """
    Integration test for :meth:``pyslyde.slide.Slide.generate_region``.

    Validates:
    - returned region image has shape ``(H, W, 3)``,
    - ROI mask aligns spatially with the image,
    - ROI mask contains foreground,
    - ROI mask class IDs do not exceed the annotation class map.

    Args:
        integration_data_dir:
            Root directory containing integration fixtures.
            If not configured, the test is skipped.
        source:
            Annotation source identifier.
    """
    data_root = _require_integration_root(integration_data_dir)
    wsi_path, ann_path = _require_source_resources(data_root, source)

    s = Slide(str(wsi_path), annotations_path=str(ann_path), source=source)
    level = s.level_count - 1

    img, roi_mask = s.generate_region(level=level)
    _assert_region_alignment_and_foreground(img, roi_mask)

    n_classes = len(s.annotations.class_key) if s.annotations else 0
    assert int(roi_mask.max()) <= n_classes


@pytest.mark.parametrize("source", _ANN_SOURCES)
def test_generate_region_label_filtering(
    integration_data_dir: Path | None, source: str
) -> None:
    """
    Integration test for :meth:``pyslyde.slide.Slide.generate_region``.

    When annotations exist:
    - choose a class ID that is actually present in the ROI mask,
    - confirm filtering by the corresponding label name and numeric
      ID yields non-empty masks.

    Args:
        integration_data_dir:
            Root directory containing integration fixtures.
            If not configured, the test is skipped.
        source:
            Annotation source identifier.
    """
    data_root = _require_integration_root(integration_data_dir)
    wsi_path, ann_path = _require_source_resources(data_root, source)

    s = Slide(str(wsi_path), annotations_path=str(ann_path), source=source)
    level = s.level_count - 1

    _, roi_mask = s.generate_region(level=level)
    _assert_label_filtering_produces_foreground(slide=s, level=level, roi_mask=roi_mask)


@pytest.mark.parametrize("source", _ANN_SOURCES)
def test_save_artifacts(
    integration_data_dir: Path | None, source: str, tmp_path: Path
) -> None:
    """
    Integration test for :meth:``pyslyde.slide.Slide.save``.

    Validates that the method:
    - writes the expected set of outputs (mask, visualisation, metadata),
    - produces files that exist on disk,
    - writes metadata containing key fields required for downstream use.

    Args:
        integration_data_dir:
            Root directory containing integration fixtures.
            If not configured, the test is skipped.
        source:
            Annotation source identifier.
        tmp_path:
            Per-test temporary directory provided by pytest.
    """
    data_root = _require_integration_root(integration_data_dir)
    wsi_path, ann_path = _require_source_resources(data_root, source)

    s = Slide(str(wsi_path), annotations_path=str(ann_path), source=source)

    out = s.save(str(tmp_path), size=(256, 256), overwrite=True)
    assert set(out) == {"mask", "vis", "meta"}

    mask_p = Path(out["mask"])
    vis_p = Path(out["vis"])
    meta_p = Path(out["meta"])

    assert mask_p.exists()
    assert vis_p.exists()
    assert meta_p.exists()

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    assert meta["slide"] == s.name
    assert meta["mask_size"] == [256, 256]
    assert "created_utc" in meta
    assert "class_map" in meta


@pytest.mark.parametrize("num_component", [None, 1])
def test_detect_components_returns_valid_borders(
    integration_data_dir: Path | None,
    num_component: int | None,
) -> None:
    """
    Integration test for :meth:``pyslyde.slide.Slide.detect_components``.

    Ensures the image-processing pipeline runs end-to-end and returns valid
    bounding boxes (mapped into level-0 coordinates). The test is tolerant
    to “no contours found”.

    Args:
        integration_data_dir:
            Root directory containing integration fixtures.
            If not configured, the test is skipped.
        num_component:
            Optional cap on the number of detected components to retain.
            When ``None``, the production default behaviour is used.
    """
    data_root = _require_integration_root(integration_data_dir)

    wsi_path, _ = _pick_wsi(data_root)
    if wsi_path is None:
        _skip_missing(
            "Integration resources unavailable for detect_components",
            missing=[f"wsi/{' or '.join(_WSI_PATTERNS)}"],
            wsi=None,
            ann=None,
        )

    s = Slide(str(wsi_path))
    level = max(0, s.level_count - 1)

    components, borders = s.detect_components(level=level, num_component=num_component)

    assert isinstance(components, list)
    assert isinstance(borders, list)
    assert len(components) == len(borders)

    full_w, full_h = s.dims

    for b in borders:
        assert isinstance(b, list) and len(b) == 2
        (x1, x2), (y1, y2) = b

        assert isinstance(x1, int) and isinstance(x2, int)
        assert isinstance(y1, int) and isinstance(y2, int)

        assert 0 <= x1 <= x2 <= full_w
        assert 0 <= y1 <= y2 <= full_h

    for img in components:
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3 and img.shape[2] == 3
