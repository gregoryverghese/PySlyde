"""
Unit tests for ``pyslyde.slide`` (Annotations + Slide).

This module contains fast, deterministic tests that:
- validate annotation loaders using synthetic in-memory files (tmp_path)
- validate core Annotations behaviors (class_key stability, CSV round-trip, GeoJSON export)
- validate common error paths (schema/format validation)
- validate Slide utility behavior without requiring a real OpenSlide WSI, by using a
  lightweight "dummy" Slide object.

These tests are intended to run quickly in CI and to avoid reliance on large external
fixtures (e.g., real whole-slide images). Integration tests that exercise OpenSlide against
real WSI files is found in a separate integration test module (test_slide_integration.py).
"""

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pyslyde.slide import (
    Annotations,
    InvalidResizeBorderFactorError,
    InvalidResizeBorderOperatorError,
    InvalidRoundingPolicyError,
    Slide,
)


def _make_annotations_obj(ann_dict: dict, *, class_map=None) -> Annotations:
    """
    Construct an :class:``~pyslyde.slide.Annotations`` instance in-memory.

    This helper bypasses normal file loading by allocating an Annotations object
    via ``Annotations.__new__`` and populating the internal fields expected by the
    production code. It enables unit tests to focus on downstream behavior
    (e.g., class_key stability, rasterisation) without depending on loader I/O.

    Args
        ann_dict:
            Annotation mapping in canonical form::

                {
                "<label>": [
                    [[x1, y1], [x2, y2], ...],   # polygon 0
                    [[x1, y1], [x2, y2], ...],   # polygon 1
                ],
                ...
                }

            Labels must be strings; polygon coordinates should be integer-like.
        class_map:
            Optional mapping ``{label: id}`` to force deterministic IDs. If provided,
            it will be exposed via ``Annotations.class_map`` and used by the
            ``Annotations.class_key`` property.

    Returns
        Annotations
            An in-memory Annotations instance with ``_annotations`` set to ``ann_dict``.
    """
    a = Annotations.__new__(Annotations)
    a.paths = []
    a.source = "in_memory"
    a.labels = None
    a.encode = False
    a.class_map = class_map
    a._annotations = ann_dict
    return a


@pytest.fixture
def minimal_polys() -> dict:
    """
    Provide a minimal annotation set for unit tests.

    The returned structure contains two non-overlapping triangles in level-0
    coordinate space. This is used to test label selection, mask generation,
    and border calculations.

    Returns
        dict
            A canonical annotations dict with two labels ("tumour", "stroma"), each
            containing a single triangle polygon.
    """
    return {
        "tumour": [
            [[10, 10], [30, 10], [20, 30]],
        ],
        "stroma": [
            [[60, 60], [80, 60], [70, 80]],
        ],
    }


@pytest.fixture
def dummy_slide(minimal_polys):
    """
    Create a lightweight Slide-like object without invoking OpenSlide.

    We *do not* instantiate Slide/OpenSlide because OpenSlide exposes read-only
    properties (level_count, level_dimensions, etc.). Instead we build a plain
    Python object and bind Slide methods onto it.
    """
    s = SimpleNamespace()

    s.level = 0
    s.dims = (200, 100)  # (width, height)
    s.name = "dummy.svs"
    s._border = None

    s.level_count = 3
    s.level_dimensions = [
        (200, 100),
        (100, 50),
        (50, 25),
    ]
    s.level_downsamples = [1.0, 2.0, 4.0]
    s.dimensions = s.dims

    s.annotations = _make_annotations_obj(minimal_polys)

    def _read_region(location, level, size):
        """
        Stand-in for OpenSlide.read_region.

        Args
            location:
                (x, y) tuple in level-0 coordinates (ignored; deterministic image).
            level:
                Pyramid level (ignored; deterministic image).
            size:
                (width, height) in pixels; determines the returned image size.

        Returns
            PIL.Image.Image
                RGBA image of shape (height, width) filled with zeros.
        """
        w, h = size
        return Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8), mode="RGBA")

    s.read_region = _read_region

    s._validate_level = MethodType(Slide._validate_level, s)
    s._select_labels = MethodType(Slide._select_labels, s)
    s._validate_contour = MethodType(Slide._validate_contour, s)
    s.get_border = MethodType(Slide.get_border, s)

    s._infer_default_roi = MethodType(Slide._infer_default_roi, s)
    s._parse_axis = MethodType(Slide._parse_axis, s)
    s._normalise_roi_level0 = MethodType(Slide._normalise_roi_level0, s)
    s._rasterise_roi_mask = MethodType(Slide._rasterise_roi_mask, s)

    s.generate_mask = MethodType(Slide.generate_mask, s)
    s.generate_region = MethodType(Slide.generate_region, s)

    s.resize_border = Slide.resize_border

    return s


class TestAnnotationsLoaders:
    """
    Unit tests for Annotations loaders using synthetic on-disk fixtures.

    Each test creates the smallest viable file for a given loader format under
    ``tmp_path`` and asserts that:
    - the top-level structure is a dict
    - keys are label strings
    - polygons are lists of vertices
    - each polygon has >= 3 vertices
    - each vertex is ``[int, int]``
    """

    def test_load_imagej_returns_valid_structure(self, tmp_path: Path):
        """
        Validate the ImageJ XML loader produces the canonical in-memory format.

        The synthetic XML contains two annotations with one triangular polygon each.
        """
        xml_path = tmp_path / "ann.xml"
        xml_path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
            <Annotations>
              <Annotation Name="A">
                <Vertices>
                  <Vertex X="0" Y="0"/><Vertex X="10" Y="0"/><Vertex X="5" Y="10"/>
                </Vertices>
              </Annotation>
              <Annotation Name="B">
                <Vertices>
                  <Vertex X="20" Y="20"/><Vertex X="30" Y="20"/><Vertex X="25" Y="30"/>
                </Vertices>
              </Annotation>
            </Annotations>
            """,
            encoding="utf-8",
        )

        ann = Annotations(str(xml_path), source="imagej")
        d = ann.annotations
        assert isinstance(d, dict)
        assert all(isinstance(k, str) for k in d.keys())
        for polys in d.values():
            assert isinstance(polys, list)
            for poly in polys:
                assert len(poly) >= 3
                for pt in poly:
                    assert isinstance(pt, list) and len(pt) == 2
                    assert isinstance(pt[0], int) and isinstance(pt[1], int)

    def test_load_asap_returns_valid_structure(self, tmp_path: Path):
        """
        Validate the ASAP XML loader produces the canonical in-memory format.

        The synthetic XML contains one annotation group with a triangular polygon.
        """
        xml_path = tmp_path / "ann.xml"
        xml_path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
            <ASAP_Annotations>
              <Annotations>
                <Annotation PartOfGroup="G1">
                  <Coordinates>
                    <Coordinate X="0" Y="0"/><Coordinate X="10" Y="0"/><Coordinate X="5" Y="10"/>
                  </Coordinates>
                </Annotation>
              </Annotations>
            </ASAP_Annotations>
            """,
            encoding="utf-8",
        )

        ann = Annotations(str(xml_path), source="asap")
        d = ann.annotations
        assert isinstance(d, dict)
        assert all(isinstance(k, str) for k in d.keys())
        for polys in d.values():
            for poly in polys:
                assert len(poly) >= 3
                assert all(isinstance(v, int) for pt in poly for v in pt)

    def test_load_custom_json_returns_valid_structure(self, tmp_path: Path):
        """
        Validate the custom JSON loader produces the canonical in-memory format.

        The synthetic JSON follows the package-specific schema where each label maps
        to an object of polygon IDs, each of which is a list of {"x","y"} vertices.
        """
        p = tmp_path / "ann.json"
        p.write_text(
            json.dumps(
                {
                    "tumour": {
                        "0": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 5, "y": 10}]
                    },
                    "stroma": {
                        "0": [
                            {"x": 20, "y": 20},
                            {"x": 30, "y": 20},
                            {"x": 25, "y": 30},
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        ann = Annotations(str(p), source="json")
        d = ann.annotations
        assert isinstance(d, dict)
        assert set(d.keys()) == {"tumour", "stroma"}
        for poly in d["tumour"]:
            assert len(poly) >= 3
            assert all(isinstance(v, int) for pt in poly for v in pt)

    def test_load_qupath_returns_valid_structure(self, tmp_path: Path):
        """
        Validate the QuPath GeoJSON-like JSON loader produces the canonical format.

        QuPath exports can represent polygons in a list of features. This test uses a
        minimal single-feature polygon example.
        """
        p = tmp_path / "qupath.json"
        p.write_text(
            json.dumps(
                [
                    {
                        "properties": {"classification": {"name": "tumour"}},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [10, 0], [5, 10], [0, 0]]],
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )

        ann = Annotations(str(p), source="qupath")
        d = ann.annotations
        assert "tumour" in d
        assert len(d["tumour"]) == 1
        assert len(d["tumour"][0]) >= 3

    def test_load_geojson_returns_valid_structure(self, tmp_path: Path):
        """
        Validate the standard GeoJSON FeatureCollection loader produces canonical format.

        The synthetic GeoJSON contains a single Polygon feature with a "label" property.
        """
        p = tmp_path / "ann.geojson"
        p.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"label": "tumour"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[0, 0], [10, 0], [5, 10], [0, 0]]],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        ann = Annotations(str(p), source="geojson")
        d = ann.annotations
        assert "tumour" in d
        assert len(d["tumour"][0]) >= 3

    def test_load_csv_returns_valid_structure(self, tmp_path: Path):
        """
        Validate the CSV loader produces the canonical in-memory format.

        The synthetic CSV encodes two triangles as (label, polygon_id, vertex_id, x, y).
        """
        p = tmp_path / "ann.csv"
        p.write_text(
            "label,polygon_id,vertex_id,x,y\n"
            "tumour,0,0,0,0\n"
            "tumour,0,1,10,0\n"
            "tumour,0,2,5,10\n"
            "stroma,0,0,20,20\n"
            "stroma,0,1,30,20\n"
            "stroma,0,2,25,30\n",
            encoding="utf-8",
        )

        ann = Annotations(str(p), source="csv")
        d = ann.annotations
        assert set(d.keys()) == {"tumour", "stroma"}
        assert len(d["tumour"][0]) == 3


class TestAnnotationsCore:
    """Tests for core, format-agnostic behavior of the Annotations class."""

    def test_class_key_stable_sorted_labels(self, minimal_polys):
        """
        Ensure ``Annotations.class_key`` is deterministic without an explicit class_map.
        IDs are defined by sorting label names and assigning 1..K.

        This test asserts that:
        - keys are in sorted order
        - IDs correspond to {1, 2, ..., K}
        """
        ann = _make_annotations_obj(minimal_polys)
        ck = ann.class_key
        assert list(ck.keys()) == sorted(ck.keys())
        assert set(ck.values()) == {1, 2}

    def test_class_key_respects_class_map(self, minimal_polys):
        """
        Ensure a provided ``class_map`` is used verbatim by ``Annotations.class_key``.

        This guarantees that downstream masks and outputs can rely on stable,
        externally-defined label IDs.
        """
        ann = _make_annotations_obj(
            minimal_polys, class_map={"tumour": 10, "stroma": 20}
        )
        ck = ann.class_key
        assert ck["tumour"] == 10
        assert ck["stroma"] == 20

    def test_to_df_roundtrip_via_csv(self, tmp_path: Path, minimal_polys):
        """
        Validate that annotations can round-trip through CSV using ``to_df`` / ``save_csv``.

        The test writes the in-memory annotations to CSV and reloads them through the
        CSV loader, then compares the resulting DataFrames ignoring row order.
        """
        ann = _make_annotations_obj(minimal_polys)
        out = tmp_path / "roundtrip.csv"
        ann.save_csv(str(out), overwrite=True)

        ann2 = Annotations(str(out), source="csv")
        df1 = (
            ann.to_df()
            .sort_values(["label", "polygon_id", "vertex_id"])
            .reset_index(drop=True)
        )
        df2 = (
            ann2.to_df()
            .sort_values(["label", "polygon_id", "vertex_id"])
            .reset_index(drop=True)
        )
        assert df1.equals(df2)

    def test_to_geojson_rings_closed(self, minimal_polys):
        """
        Validate GeoJSON export closes polygon rings when ``close_rings=True``.

        GeoJSON polygon rings conventionally repeat the first coordinate as the last.
        This test ensures the exporter enforces that convention by default.
        """
        ann = _make_annotations_obj(minimal_polys)
        gj = ann.to_geojson(close_rings=True)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 2
        for feat in gj["features"]:
            coords = feat["geometry"]["coordinates"][0]
            assert coords[0] == coords[-1]


class TestAnnotationsErrors:
    """Negative tests ensuring loaders fail fast on invalid inputs."""

    def test_csv_missing_columns_raises(self, tmp_path: Path):
        """
        CSV loader should raise when required columns are missing.
        """
        p = tmp_path / "bad.csv"
        p.write_text("label,x,y\nfoo,0,0\n", encoding="utf-8")
        with pytest.raises(ValueError) as e:
            Annotations(str(p), source="csv")

        msg = str(e.value).lower()
        assert "csv" in msg or "columns" in msg
        assert "missing" in msg or "must contain" in msg

    def test_csv_duplicate_vertex_id_raises(self, tmp_path: Path):
        """
        CSV loader should reject duplicate vertex IDs within a single polygon.

        Duplicate vertex_id values indicate ambiguous vertex ordering and should be
        treated as invalid input.
        """
        p = tmp_path / "dup.csv"
        p.write_text(
            "label,polygon_id,vertex_id,x,y\n"
            "tumour,0,0,0,0\n"
            "tumour,0,0,10,0\n"
            "tumour,0,2,5,10\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as e:
            Annotations(str(p), source="csv")

        msg = str(e.value).lower()
        assert "vertex_id" in msg or "vertex id" in msg
        assert "duplicate" in msg

    def test_geojson_unsupported_geometry_type_raises(self, tmp_path: Path):
        """
        GeoJSON loader should raise for geometry types that are not currently
        supported by the loader .

        Supported: Polygon, MultiPolygon, LineString, MultiLineString.
        Ignored: Point, MultiPoint, GeometryCollection.
        """
        p = tmp_path / "bad.geojson"
        p.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"label": "tumour"},
                            "geometry": {
                                "type": "CircularString",
                                "coordinates": [[0, 0], [1, 1], [2, 0]],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError) as e:
            Annotations(str(p), source="geojson")

        msg = str(e.value).lower()
        assert "geometry" in msg
        assert "unsupported" in msg

    def test_geojson_point_and_geometrycollection_are_ignored(self, tmp_path: Path):
        """
        GeoJSON loader should explicitly ignore geometries currently not process.

        Point / MultiPoint / GeometryCollection are valid GeoJSON but are intentionally
        ignored to avoid surprising border expansion.
        """
        p = tmp_path / "ignored.geojson"
        p.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"label": "tumour"},
                            "geometry": {"type": "Point", "coordinates": [5, 5]},
                        },
                        {
                            "type": "Feature",
                            "properties": {"label": "tumour"},
                            "geometry": {
                                "type": "GeometryCollection",
                                "geometries": [
                                    {"type": "Point", "coordinates": [0, 0]},
                                ],
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        ann = Annotations(str(p), source="geojson")
        d = ann.annotations
        assert d == {} or (isinstance(d, dict) and len(d) == 0)

    def test_load_geojson_linestring_returns_valid_structure(self, tmp_path: Path):
        """
        Validate GeoJSON loader accepts LineString and returns canonical sequences.

        LineString is stored as a vertex sequence (list of [x, y]) under the label.
        """
        p = tmp_path / "ann_linestring.geojson"
        p.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"label": "tumour"},
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[0, 0], [10, 0], [5, 10]],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        ann = Annotations(str(p), source="geojson")
        d = ann.annotations

        assert isinstance(d, dict)
        assert "tumour" in d
        assert isinstance(d["tumour"], list)
        assert len(d["tumour"]) == 1

        seq = d["tumour"][0]
        assert isinstance(seq, list)
        assert len(seq) >= 3
        for pt in seq:
            assert isinstance(pt, list) and len(pt) == 2
            assert isinstance(pt[0], int) and isinstance(pt[1], int)

    def test_load_geojson_multipolygon_returns_valid_structure(self, tmp_path: Path):
        """
        Validate GeoJSON loader accepts MultiPolygon and returns canonical sequences.

        Only the exterior ring of each polygon is loaded.
        """
        p = tmp_path / "ann_multipolygon.geojson"
        p.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"label": "tumour"},
                            "geometry": {
                                "type": "MultiPolygon",
                                "coordinates": [
                                    [[[0, 0], [10, 0], [5, 10], [0, 0]]],
                                    [[[20, 20], [30, 20], [25, 30], [20, 20]]],
                                ],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        ann = Annotations(str(p), source="geojson")
        d = ann.annotations

        assert "tumour" in d
        assert len(d["tumour"]) == 2
        for seq in d["tumour"]:
            assert len(seq) >= 3
            assert all(isinstance(v, int) for pt in seq for v in pt)

    def test_custom_json_invalid_schema_raises(self, tmp_path: Path):
        """
        Custom JSON loader should reject invalid top-level schemas.

        The custom schema requires a dict at the top level; a list should fail.
        """
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        with pytest.raises(ValueError) as e:
            Annotations(str(p), source="json")

        msg = str(e.value).lower()
        assert "top-level" in msg or "top level" in msg
        assert "object" in msg or "dict" in msg


class TestSlideUnit:
    """
    Unit-ish tests for Slide behavior without relying on OpenSlide.

    These tests use the ``dummy_slide`` fixture to verify:
    - parameter validation and error handling
    - mask generation behavior (shape, label selection)
    - region generation alignment (image and mask dimensions)
    - rounding mode behavior for downsampled ROI sizing
    """

    def test_resize_border_validates_operator_and_factor(self):
        """
        ``Slide.resize_border`` should validate its inputs and enforce constraints.

        Ensures:
        - an unknown operator raises InvalidResizeBorderOperatorError
        - a non-positive factor raises InvalidResizeBorderFactorError
        - output is a multiple of factor and respects the >= threshold constraint
        """
        with pytest.raises(InvalidResizeBorderOperatorError) as e:
            Slide.resize_border(1000, factor=256, threshold=1000, operator="??")
        assert e.value.operator == "??"
        assert "??" not in e.value.allowed
        assert ">=" in e.value.allowed

        with pytest.raises(InvalidResizeBorderFactorError) as e:
            Slide.resize_border(1000, factor=0)
        assert e.value.factor == 0

        out = Slide.resize_border(1001, factor=256, threshold=1024, operator=">=")
        assert out % 256 == 0
        assert out >= 1024

    def test_generate_mask_size_vs_level_mutual_exclusion(self, dummy_slide):
        """
        ``Slide.generate_mask`` should reject mutually-exclusive ``size`` and ``level``.

        Providing both would create ambiguous scaling behavior; the API requires
        selecting exactly one output coordinate system.
        """
        with pytest.raises(ValueError) as e:
            dummy_slide.generate_mask(size=(50, 25), level=1)

        msg = str(e.value).lower()
        assert "size" in msg
        assert "level" in msg
        assert ("only one" in msg) or ("not both" in msg) or ("mutually" in msg)

    def test_generate_mask_full_res_disabled_by_default(self, dummy_slide):
        """
        ``Slide.generate_mask`` should not allow full-resolution output implicitly.

        Full-resolution masks can be extremely large; the implementation requires
        an explicit ``size`` or ``level`` unless ``full_res=True``.
        """
        with pytest.raises(ValueError) as e:
            dummy_slide.generate_mask()

        msg = str(e.value).lower()
        assert ("full-resolution" in msg) or ("full resolution" in msg)
        assert (
            ("disabled" in msg)
            or ("provide" in msg)
            or ("size" in msg)
            or ("level" in msg)
        )

    def test_generate_mask_preserve_aspect_rejects_wrong_aspect(self, dummy_slide):
        """
        ``preserve_aspect=True`` should reject sizes that distort slide geometry.

        With slide dims (200, 100) the aspect ratio is 2.0. A requested (100, 100)
        would distort polygons and should raise.
        """
        with pytest.raises(ValueError) as e:
            dummy_slide.generate_mask(size=(100, 100), preserve_aspect=True)

        msg = str(e.value).lower()
        assert "aspect" in msg
        assert ("ratio" in msg) or ("distort" in msg)

    def test_generate_mask_labels_subset_works(self, dummy_slide):
        """
        ``Slide.generate_mask`` should respect a subset of labels.

        The test generates:
        - a full mask using all labels
        - a mask using only "tumour"
        and asserts that the "stroma" class ID is not present in the subset mask.
        """
        mask_all = dummy_slide.generate_mask(size=(200, 100))
        u_all = set(np.unique(mask_all))
        assert 0 in u_all

        mask_t = dummy_slide.generate_mask(size=(200, 100), labels=["tumour"])
        u_t = set(np.unique(mask_t))
        stroma_id = dummy_slide.annotations.class_key["stroma"]
        assert stroma_id not in u_t

    def test_generate_region_default_roi_uses_border(self, dummy_slide):
        """
        ``Slide.generate_region`` should use annotation-derived border by default.

        When x/y are not specified, the implementation falls back to the padded
        annotation border (or full-slide if no annotations exist).

        This test verifies:
        - returned image is RGB (H, W, 3)
        - returned mask is (H, W)
        - shapes align pixel-for-pixel
        - mask values are stable class IDs
        """
        img, m = dummy_slide.generate_region(level=0)
        assert img.ndim == 3 and img.shape[2] == 3
        assert m.ndim == 2
        assert img.shape[:2] == m.shape[:2]
        assert set(np.unique(m)).issubset({0, 1, 2})

    def test_generate_region_rounding_direction(self, dummy_slide):
        """
        Rounding mode should affect ROI output dimensions in the expected direction.

        Using an ROI size not divisible by ds=2 at level 1:
        - floor should not exceed round
        - ceil should not be smaller than round

        This test compares image shapes to confirm monotonicity.
        """
        img_r, _ = dummy_slide.generate_region(
            level=1, x=0, y=0, x_size=101, y_size=51, rounding="round"
        )
        img_f, _ = dummy_slide.generate_region(
            level=1, x=0, y=0, x_size=101, y_size=51, rounding="floor"
        )
        img_c, _ = dummy_slide.generate_region(
            level=1, x=0, y=0, x_size=101, y_size=51, rounding="ceil"
        )

        assert img_f.shape[1] <= img_r.shape[1] <= img_c.shape[1]
        assert img_f.shape[0] <= img_r.shape[0] <= img_c.shape[0]

    def test_generate_region_invalid_rounding_raises(self, dummy_slide):
        """
        ``Slide.generate_region`` should reject invalid rounding modes.

        The API accepts {"round","floor","ceil"}. Any other value should raise
        a ValueError with a clear message.
        """
        with pytest.raises(InvalidRoundingPolicyError):
            dummy_slide.generate_region(
                level=1, x=0, y=0, x_size=100, y_size=50, rounding="nope"
            )

    def test_get_border_level_scaling(self, dummy_slide):
        """
        ``Slide.get_border(level=...)`` should scale border coordinates by downsample.

        When requesting a border at a downsampled level, coordinates are expected to be
        integer-divided by the level downsample factor.
        """
        b0 = dummy_slide.get_border(padding=0, level=None)
        b1 = dummy_slide.get_border(padding=0, level=1)
        ds = dummy_slide.level_downsamples[1]
        assert b1[0][0] == int(b0[0][0] / ds)
        assert b1[1][0] == int(b0[1][0] / ds)
