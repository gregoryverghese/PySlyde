"""
Pytest unit tests for pyslyde.slide_parser.

These tests focus on:
- WSIParser:
    - init/config in level and target_mpp modes
    - tiling and edge-case exclusion
    - mask validation/extraction/filtering
    - deterministic sampling
    - tile extraction with masking and normalization
- Stitching:
    - patch discovery and coordinate parsing
    - origin/coverage border inference
    - stride inference and completeness
    - overwrite/max stitching policies
    - resize behavior
    - validation errors

The tests mock slide tile extraction for WSIParser so they remain
fast and independent of real WSI files.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

import pyslyde.slide_parser as slide_parser

from pyslyde.masks.level0 import Level0Mask
from pyslyde.masks.mapped import MappedMask
from pyslyde.util.utilities import coord_to_name

# Helpers


class FakeSlide:
    """Minimal slide-like object for WSIParser tests."""

    def __init__(
        self,
        name="fake_slide",
        dims=(100, 80),  # (width, height)
        level_downsamples=(1.0, 4.0, 16.0),
        properties=None,
    ):
        self.name = name
        self.dims = dims
        self.level_downsamples = list(level_downsamples)
        self.properties = properties or {
            "openslide.mpp-x": "0.25",
            "openslide.mpp-y": "0.25",
        }


class FakeNormalizerBase:
    """Replacement base class for isinstance checks in tests."""


class FakeNormalizer(FakeNormalizerBase):
    """Simple deterministic normalizer."""

    def __init__(self, is_fitted=True, add_value=1):
        self.is_fitted = is_fitted
        self.add_value = add_value

    def normalize(self, tile: np.ndarray) -> np.ndarray:
        return tile + self.add_value


def fake_extract_tile_from_slide(slide, x, y, level, tile_dims):
    """
    Deterministic fake tile extractor.

    Returns an RGB tile whose constant value depends on x, y, and level,
    making behavior easy to assert without real WSI data.
    """
    w, h = tile_dims
    value = (x + y + level) % 255
    return np.full((h, w, 3), value, dtype=np.uint8)


def write_patch(
    tmp_path: Path, x: int, y: int, value: int, shape=(4, 4, 3), ext=".png"
):
    """Write a deterministic patch image to disk using coordinate-based naming."""
    arr = np.full(shape, value, dtype=np.uint8)
    path = tmp_path / f"{coord_to_name(x, y)}{ext}"
    ok = cv2.imwrite(str(path), arr)
    assert ok, f"Failed to write patch at {path}"
    return path


# Shared fixtures


@pytest.fixture
def fake_slide():
    """Provide a reusable fake slide."""
    return FakeSlide()


@pytest.fixture
def patch_wsi_dependencies(monkeypatch):
    """
    Patch external dependencies used by WSIParser:
    - StainNormalizer type check
    - extract_tile_from_slide tile reading
    """
    monkeypatch.setattr(slide_parser, "StainNormalizer", FakeNormalizerBase)
    monkeypatch.setattr(
        slide_parser,
        "extract_tile_from_slide",
        fake_extract_tile_from_slide,
    )


@pytest.fixture
def level0_mask(
    fake_slide,
):
    """
    Level-0 semantic mask covering the fake slide.

    Label 1 occupies the top-left 8 × 8 tile.
    Label 2 occupies the neighbouring tile to the right.
    """

    mask = np.zeros(
        (80, 100),
        dtype=np.uint8,
    )

    mask[0:8, 0:8] = 1
    mask[0:8, 8:16] = 2

    return Level0Mask(mask)


@pytest.fixture
def mapped_mask(
    fake_slide,
):
    """
    Mapped semantic mask stored at half the slide resolution.

    The mask represents the same labelled regions as ``level0_mask`` but is
    stored at one-half resolution in each dimension.
    """

    mask = np.zeros(
        (40, 50),
        dtype=np.uint8,
    )

    mask[0:4, 0:4] = 1
    mask[0:4, 4:8] = 2

    return MappedMask(
        mask=mask,
        slide_shape=(80, 100),
    )


@pytest.fixture
def multi_label_mask():
    """
    Whole-slide mask containing three vertical label regions.

    Labels 1, 2 and 3 occupy consecutive 8-pixel-wide bands from
    left to right.
    """

    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[:, :8] = 1
    mask[:, 8:16] = 2
    mask[:, 16:24] = 3

    return Level0Mask(mask)


@pytest.mark.slide_parser
class TestWSIParser:

    # Construction
    # ------------

    def test_init_level_mode_config(self, fake_slide):
        """Parser should initialize correctly in level mode."""
        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 32), (0, 32)],
            level=1,
        )

        assert parser.mode == "level"
        assert parser.tile_dims == (8, 8)
        assert parser.config["effective_level"] == 1
        assert parser.config["target_level"] == 1
        assert parser.config["base_mpp"] == pytest.approx(0.25)
        assert parser.config["effective_mpp"] == pytest.approx(1.0)
        assert parser.number == 0
        assert parser.tiles == []

    def test_init_target_mpp_mode_warns_when_level_also_given(
        self, fake_slide,
    ):
        """Providing both level and target_mpp should warn and prefer target_mpp."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parser = slide_parser.WSIParser(
                slide=fake_slide,
                tile_dim=8,
                border=[(0, 32), (0, 32)],
                level=2,
                target_mpp=1.0,
            )

        assert any("target_mpp" in str(w.message) for w in caught)
        assert parser.mode == "target_mpp"
        assert parser.config["target_mpp"] == pytest.approx(1.0)
        assert parser.config["effective_mpp"] == pytest.approx(1.0)
        assert parser.config["effective_level"] == 1

    def test_invalid_level_raises_key_error(self, fake_slide, patch_wsi_dependencies):
        """Invalid level index should raise KeyError."""
        with pytest.raises(KeyError, match="level must be in range"):
            slide_parser.WSIParser(
                slide=fake_slide,
                tile_dim=8,
                border=[(0, 32), (0, 32)],
                level=99,
            )

    def test_invalid_target_mpp_finer_than_base_raises(
        self, fake_slide, patch_wsi_dependencies
    ):
        """A target_mpp finer than slide base MPP should be rejected."""
        with pytest.raises(ValueError, match="finer than the slide base MPP"):
            slide_parser.WSIParser(
                slide=fake_slide,
                tile_dim=8,
                border=[(0, 32), (0, 32)],
                level=None,
                target_mpp=0.1,
            )

    def test_missing_level_when_target_mpp_is_none_raises(
        self, fake_slide, patch_wsi_dependencies
    ):
        """Level mode requires an explicit level."""
        with pytest.raises(ValueError, match="level must be provided"):
            slide_parser.WSIParser(
                slide=fake_slide,
                tile_dim=8,
                border=[(0, 32), (0, 32)],
                level=None,
                target_mpp=None,
            )

    # Tile generation
    # ---------------

    def test_tiler_generates_expected_tiles(self, fake_slide):
        """tiler should generate expected level-0 tile origins."""
        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )

        n = parser.tiler()

        assert n == 4
        assert parser.tiles == [(0, 0), (0, 8), (8, 0), (8, 8)]
        assert parser.number == 4

    def test_tiler_edge_cases_skips_partial_tiles(
        self, fake_slide, 
    ):
        """``edge_cases=True`` should exclude tiles whose footprint exceeds border."""
        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 18), (0, 18)],
            level=0,
        )

        n = parser.tiler(edge_cases=True)

        assert n == 4
        assert parser.tiles == [(0, 0), (0, 8), (8, 0), (8, 8)]

    def test_sample_tiles_is_reproducible_with_seed(
        self, fake_slide
    ):
        """``sample_tiles()`` should be deterministic when a seed is given."""
        tiles = [(0, 0), (0, 8), (8, 0), (8, 8), (16, 16)]

        parser1 = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 32), (0, 32)],
            level=0,
        )
        parser2 = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 32), (0, 32)],
            level=0,
        )

        parser1.tiles = tiles.copy()
        parser2.tiles = tiles.copy()

        parser1.sample_tiles(3, seed=123)
        parser2.sample_tiles(3, seed=123)

        assert parser1.tiles == parser2.tiles
        assert len(parser1.tiles) == 3

    # Mask extraction
    # ---------------

    def test_extract_mask_returns_requested_mask_region(
        self,
        fake_slide,
        level0_mask,
    ):
        """
        ``extract_mask()`` should return the mask region corresponding
        to the requested tile.
        """

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )

        region = parser.extract_mask(
            x=0,
            y=0,
            mask=level0_mask,
        )

        assert region.shape == (8, 8)
        assert np.all(region == 1)

    def test_extract_mask_preserves_multiple_labels_within_a_tile(
        self,
        fake_slide,
        level0_mask,
    ):
        """
        ``extract_mask()`` should preserve multiple semantic labels within a
        single extracted mask region.
        """

        # tile containing two labels
        level0_mask.array[0:8, 0:4] = 1
        level0_mask.array[0:8, 4:8] = 2

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )

        region = parser.extract_mask(
            x=0,
            y=0,
            mask=level0_mask,
        )

        assert region.shape == (8, 8)

        assert np.all(region[:, :4] == 1)
        assert np.all(region[:, 4:] == 2)

    def test_extract_masks_returns_one_mask_per_tile(
        self,
        fake_slide,
        level0_mask,
    ):
        """
        ``extract_masks()`` should yield one mask region per retained tile.
        """

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 24), (0, 8)],
            level=0,
        )

        parser.tiles = [
            (0, 0),
            (8, 0),
            (16, 0),
        ]

        masks = list(
            parser.extract_masks(
                mask=level0_mask,
            )
        )

        assert len(masks) == 3

        assert [coord for coord, _ in masks] == parser.tiles

        assert all(region.shape == (8, 8) for _, region in masks)

    def test_extract_masks_returns_expected_mask_regions(
        self,
        fake_slide,
        level0_mask,
    ):
        """
        ``extract_masks()`` should yield mask regions corresponding to each tile.
        """

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 8)],
            level=0,
        )

        parser.tiles = [
            (0, 0),
            (8, 0),
        ]

        regions = dict(
            parser.extract_masks(
                mask=level0_mask,
            )
        )

        assert np.all(regions[(0, 0)] == 1)
        assert np.all(regions[(8, 0)] == 2)

    # Tile filtering
    # --------------

    def test_filter_by_mask_keeps_only_tiles_above_threshold(
        self, 
        fake_slide, 
        level0_mask,
    ):
        """``filter_by_mask`` should retain only tiles meeting the requested 
        label coverage threshold.
        """
        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )

        parser.tiles = [
            (0, 0), 
            (8, 0)
        ]

        remaining = parser.filter_by_mask(
            mask=level0_mask, 
            labels=1, 
            threshold=0.5,
        )

        assert remaining == 1
        assert parser.tiles == [(0, 0)]

    def test_filter_by_mask_accepts_multiple_labels(
        self,
        fake_slide,
        level0_mask,
    ):
        """
        ``filter_by_mask()`` should retain tiles containing any requested label.
        """

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 24), (0, 8)],
            level=0,
        )

        parser.tiles = [
            (0, 0),
            (8, 0),
            (16, 0),
        ]

        remaining = parser.filter_by_mask(
            mask=level0_mask,
            labels=[1, 2],
            threshold=0.5,
        )

        assert remaining == 2
        assert parser.tiles == [
            (0, 0),
            (8, 0),
        ]

    def test_filter_by_mask_removes_tiles_without_requested_labels(
        self,
        fake_slide,
        level0_mask,
    ):
        """
        ``filter_by_mask()`` should remove every tile when none contains 
        the requested labels.
        """

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )

        parser.tiles = [
            (0, 0),
            (8, 0),
        ]

        remaining = parser.filter_by_mask(
            mask=level0_mask,
            labels=3,
            threshold=0.5,
        )

        assert remaining == 0
        assert parser.tiles == []

    def test_filter_by_mask_threshold_zero_retains_all_tiles(
        self,
        fake_slide,
        level0_mask,
    ):
        """
        ``filter_by_mask()`` should retain every tile when the threshold is zero.
        """

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 24), (0, 8)],
            level=0,
        )

        parser.tiles = [
            (0, 0),
            (8, 0),
            (16, 0),
        ]

        parser.filter_by_mask(
            mask=level0_mask,
            labels=1,
            threshold=0.0,
        )

        assert parser.tiles == [
            (0, 0),
            (8, 0),
            (16, 0),
        ]

    def test_filter_by_mask_threshold_one_requires_complete_tile_coverage(
        self,
        fake_slide,
    ):
        """
        ``filter_by_mask()`` should retain only tiles completely covered by the
        requested labels when the threshold is one.
        """

        mask = np.zeros((80, 100), dtype=np.uint8)

        mask[0:8, 0:8] = 1

        mask[0:8, 8:12] = 1

        mask = Level0Mask(mask)

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 8)],
            level=0,
        )

        parser.tiles = [
            (0, 0),
            (8, 0),
        ]

        parser.filter_by_mask(
            mask=mask,
            labels=1,
            threshold=1.0,
        )
        
        assert parser.tiles == [
            (0, 0),
        ]

    def test_filter_by_mask_rejects_invalid_threshold(
        self,
        fake_slide,
        level0_mask,
    ):
        """``filter_by_mask()`` should reject thresholds outside the interval [0, 1]."""

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )

        parser.tiles = [(0, 0)]

        for threshold in (-0.1, 1.1):

            with pytest.raises(
                ValueError,
                match="threshold",
            ):
                parser.filter_by_mask(
                    mask=level0_mask,
                    labels=1,
                    threshold=threshold,
                )

    def test_filter_by_mask_rejects_empty_label_collection(
        self,
        fake_slide,
        level0_mask,
    ):
        """``filter_by_mask()`` should reject an empty label collection."""

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )

        with pytest.raises(
            ValueError,
            match="At least one label must be specified",
        ):
            parser.filter_by_mask(
                mask=level0_mask,
                labels=[],
            )

    def test_filter_by_func_removes_matching_tiles(
        self, fake_slide, patch_wsi_dependencies
    ):
        """``filter_by_func`` should remove tiles for which filter_func returns True."""
        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 24), (0, 24)],
            level=0,
        )
        parser.tiles = [(0, 0), (8, 0), (16, 0)]

        def remove_dark_tiles(tile, threshold):
            return tile.mean() < threshold

        parser.filter_by_func(remove_dark_tiles, threshold=10)

        assert parser.tiles == [(16, 0)]

    # Tile extraction
    # ---------------

    def test_extract_tile_applies_mask(
            self, 
            fake_slide, 
            patch_wsi_dependencies,
            level0_mask,
    ):
        """``extract_tile()`` should replace masked pixels with ``bg_value``."""
        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )

        tile = parser.extract_tile(
            x=0,
            y=0,
            mask=level0_mask,
            bg_value=255,
        )

        assert tile.shape == (8, 8, 3)
        assert np.all(tile == 0)


    def test_extract_tile_retains_requested_label(
        self,
        fake_slide,
        patch_wsi_dependencies,
        multi_label_mask,
    ):
        """``extract_tile()`` should retain only the requested label."""

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 8)],
            level=0,
        )

        tile = parser.extract_tile(
            x=0,
            y=0,
            mask=multi_label_mask,
            labels=1,
            bg_value=255,
        )

        assert np.all(tile[:, :8] != 255)
        assert np.all(tile[:, 8:] == 255)

    def test_extract_tile_retains_multiple_labels(
        self,
        fake_slide,
        patch_wsi_dependencies,
        multi_label_mask,
    ):
        """``extract_tile()`` should retain all requested labels."""

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 24), (0, 8)],
            level=0,
        )

        tile = parser.extract_tile(
            x=0,
            y=0,
            mask=multi_label_mask,
            labels={1, 3},
            bg_value=255,
        )

        assert np.all(tile[:, :8] != 255)
        assert np.all(tile[:, 8:16] == 255)
        assert np.all(tile[:, 16:24] != 255)

    def test_extract_tile_retains_all_labels_when_labels_not_specified(
        self,
        fake_slide,
        patch_wsi_dependencies,
        multi_label_mask,
    ):
        """``extract_tile()`` should retain all non-zero labels by default."""

        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 24), (0, 8)],
            level=0,
        )

        tile = parser.extract_tile(
            x=0,
            y=0,
            mask=multi_label_mask,
            bg_value=255,
        )

        assert np.all(tile != 255)

    def test_extract_tile_raises_if_normalizer_missing(
        self, fake_slide, patch_wsi_dependencies
    ):
        """normalize=True requires a stain normalizer."""
        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )

        with pytest.raises(RuntimeError, match="No stain normalizer provided"):
            parser.extract_tile(0, 0, normalize=True)

    def test_extract_tile_raises_if_normalizer_not_fitted(
        self, fake_slide, patch_wsi_dependencies
    ):
        """normalize=True should fail when normalizer exists but is not fitted."""
        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
            stain_normalizer=FakeNormalizer(is_fitted=False),
        )

        with pytest.raises(RuntimeError, match="not fitted"):
            parser.extract_tile(0, 0, normalize=True)

    def test_extract_tile_normalizes_when_fitted(
        self, fake_slide, patch_wsi_dependencies
    ):
        """A fitted normalizer should be applied to extracted tiles."""
        parser_plain = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
        )
        parser_norm = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 16), (0, 16)],
            level=0,
            stain_normalizer=FakeNormalizer(is_fitted=True, add_value=1),
        )

        tile_plain = parser_plain.extract_tile(0, 0, normalize=False)
        tile_norm = parser_norm.extract_tile(0, 0, normalize=True)

        assert np.array_equal(tile_norm, tile_plain + 1)

    # Batch extraction
    # ----------------
    
    def test_extract_tiles_yields_all_tiles(self, fake_slide, patch_wsi_dependencies):
        """extract_tiles should yield one tile per stored coordinate."""
        parser = slide_parser.WSIParser(
            slide=fake_slide,
            tile_dim=8,
            border=[(0, 24), (0, 24)],
            level=0,
        )
        parser.tiles = [(0, 0), (8, 0), (16, 0)]

        out = list(parser.extract_tiles())

        assert len(out) == 3
        assert [coord for coord, _ in out] == parser.tiles
        assert all(tile.shape == (8, 8, 3) for _, tile in out)


@pytest.mark.slide_parser
class TestStitching:
    """Pytest suite for Stitching."""

    def test_patch_discovery_and_border_inference(self, tmp_path):
        """Stitching should discover patches and infer origin/coverage borders."""
        write_patch(tmp_path, 0, 0, 10)
        write_patch(tmp_path, 4, 0, 20)
        write_patch(tmp_path, 0, 4, 30)
        write_patch(tmp_path, 4, 4, 40)

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=4,
        )

        assert len(stitcher._patches()) == 4
        assert stitcher.origin_border == [(0, 4), (0, 4)]
        assert stitcher.coverage_border == [(0, 8), (0, 8)]
        assert stitcher.stride == 4

    def test_infer_stride_from_patch_origins(self, tmp_path):
        """Stride should be inferred from coordinate differences when not provided."""
        write_patch(tmp_path, 0, 0, 10)
        write_patch(tmp_path, 4, 0, 20)
        write_patch(tmp_path, 8, 0, 30)

        stitcher = slide_parser.Stitching(patch_path=str(tmp_path))

        assert stitcher.stride == 4

    def test_completeness_for_full_grid(self, tmp_path):
        """completeness should be 1.0 for a complete rectangular grid."""
        write_patch(tmp_path, 0, 0, 10)
        write_patch(tmp_path, 4, 0, 20)
        write_patch(tmp_path, 0, 4, 30)
        write_patch(tmp_path, 4, 4, 40)

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=4,
        )

        assert stitcher.completeness() == pytest.approx(1.0)

    def test_completeness_for_incomplete_grid(self, tmp_path):
        """completeness should reflect missing tiles in an otherwise regular grid."""
        write_patch(tmp_path, 0, 0, 10)
        write_patch(tmp_path, 4, 0, 20)
        write_patch(tmp_path, 0, 4, 30)

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=4,
        )

        assert stitcher.completeness() == pytest.approx(0.75)

    def test_stitch_overwrite_policy(self, tmp_path):
        """overwrite policy should place patches directly into expected quadrants."""
        write_patch(tmp_path, 0, 0, 10)
        write_patch(tmp_path, 4, 0, 20)
        write_patch(tmp_path, 0, 4, 30)
        write_patch(tmp_path, 4, 4, 40)

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=4,
            policy="overwrite",
        )
        canvas = stitcher.stitch()

        assert canvas.shape == (8, 8, 3)
        assert np.all(canvas[0:4, 0:4] == 10)
        assert np.all(canvas[0:4, 4:8] == 20)
        assert np.all(canvas[4:8, 0:4] == 30)
        assert np.all(canvas[4:8, 4:8] == 40)

    def test_stitch_max_policy_handles_overlap(self, tmp_path):
        """max policy should preserve the pixelwise maximum in overlapping regions."""
        write_patch(tmp_path, 0, 0, 50, shape=(4, 4, 3))
        write_patch(tmp_path, 2, 0, 100, shape=(4, 4, 3))

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=2,
            policy="max",
        )
        canvas = stitcher.stitch()

        assert canvas.shape == (4, 6, 3)
        assert np.all(canvas[:, 0:2] == 50)
        assert np.all(canvas[:, 2:4] == 100)
        assert np.all(canvas[:, 4:6] == 100)

    def test_stitch_resizes_output(self, tmp_path):
        """stitch(size=...) should resize the final stitched canvas."""
        write_patch(tmp_path, 0, 0, 10)
        write_patch(tmp_path, 4, 0, 20)

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=4,
            content_type="image",
        )
        canvas = stitcher.stitch(size=(16, 8))  # (width, height)

        assert canvas.shape == (8, 16, 3)

    def test_mask_content_type_defaults_to_nearest_interpolation(self, tmp_path):
        """Mask content should default to nearest-neighbour interpolation on resize."""
        write_patch(tmp_path, 0, 0, 1, shape=(4, 4), ext=".png")
        write_patch(tmp_path, 4, 0, 2, shape=(4, 4), ext=".png")

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=4,
            content_type="mask",
        )

        assert stitcher._get_resize_interpolation() == cv2.INTER_NEAREST
        canvas = stitcher.stitch(size=(16, 8))

        assert canvas.shape == (8, 16)

    def test_invalid_policy_raises(self, tmp_path):
        """Invalid stitching policy should raise ValueError at init."""
        with pytest.raises(ValueError, match="Invalid policy"):
            slide_parser.Stitching(
                patch_path=str(tmp_path),
                policy="bad_policy",
            )

    def test_invalid_content_type_raises(self, tmp_path):
        """Invalid content_type should raise ValueError at init."""
        with pytest.raises(ValueError, match="Invalid content_type"):
            slide_parser.Stitching(
                patch_path=str(tmp_path),
                content_type="bad_content",
            )

    def test_missing_patch_source_raises(self):
        """At least one patch source must be provided."""
        with pytest.raises(ValueError, match="Provide at least one"):
            slide_parser.Stitching(
                patch_path=None,
                patching=None,
            )

    def test_nonexistent_patch_path_raises_on_patch_access(self, tmp_path):
        """Accessing patches from a missing directory should raise FileNotFoundError."""
        missing = tmp_path / "does_not_exist"
        stitcher = slide_parser.Stitching(patch_path=str(missing))

        with pytest.raises(FileNotFoundError, match="patch_path does not exist"):
            stitcher._patches()

    def test_border_excluding_observed_coverage_raises(self, tmp_path):
        """Provided coverage border must fully contain all observed patch coverage."""
        write_patch(tmp_path, 0, 0, 10)
        write_patch(tmp_path, 4, 0, 20)

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=4,
            border=[(0, 6), (0, 4)],
        )

        with pytest.raises(ValueError, match="does not fully contain"):
            stitcher.stitch()

    def test_strict_stride_mismatch_raises(self, tmp_path):
        """strict_stride=True should reject disagreement between provided and inferred stride."""
        write_patch(tmp_path, 0, 0, 10)
        write_patch(tmp_path, 4, 0, 20)

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=8,
            strict_stride=True,
        )

        with pytest.raises(ValueError, match="does not match inferred stride"):
            stitcher.stitch()

    def test_inconsistent_patch_sizes_raise(self, tmp_path):
        """All patches must have consistent dimensions."""
        write_patch(tmp_path, 0, 0, 10, shape=(4, 4, 3))
        write_patch(tmp_path, 4, 0, 20, shape=(5, 4, 3))

        stitcher = slide_parser.Stitching(
            patch_path=str(tmp_path),
            stride=4,
        )

        with pytest.raises(ValueError, match="Inconsistent patch size"):
            stitcher.stitch()
