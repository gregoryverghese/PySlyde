"""
Pytest unit tests for pyslyde.masks.

These tests cover ``BaseMask``, ``Level0Mask`` and ``MappedMask``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyslyde.masks.base import BaseMask
from pyslyde.masks.level0 import Level0Mask
from pyslyde.masks.mapped import MappedMask


class DummyMask(BaseMask):
    """
    ``DummyMask`` implementation used solely for testing ``BaseMask``.

    Unlike the concrete subclasses, this implementation always returns an
    identity mapping irrespective of the supplied arguments.
    """

    def _resolve_scale(
        self,
        *,
        slide_shape: tuple[int, int],
        scale_x: float | None,
        scale_y: float | None,
    ) -> tuple[float, float]:
        """Return an identity coordinate mapping."""
        return 1.0, 1.0

    def read_region_native(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Return the requested region without resampling."""

        self._validate_region(
            x=x,
            y=y,
            width=width,
            height=height,
        )

        return self.array[
            y : y + height,
            x : x + width,
        ]

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Return the requested region."""

        return self.read_region_native(
            x=x,
            y=y,
            width=width,
            height=height,
        )


@pytest.fixture
def mask_array() -> np.ndarray:
    """
    Deterministic label mask.

    Returns:
        10 × 10 mask whose values increase row-wise from 0 to 99.
    """

    return np.arange(
        100,
        dtype=np.uint8,
    ).reshape(
        10,
        10,
    )


@pytest.fixture
def dummy_mask(
    mask_array: np.ndarray,
) -> DummyMask:
    """
    DummyMask constructed from the shared deterministic mask.
    """

    return DummyMask(
        mask=mask_array,
        slide_shape=mask_array.shape,
    )


@pytest.fixture
def level0_mask(
    mask_array: np.ndarray,
) -> Level0Mask:
    """
    Level0Mask constructed from the shared deterministic mask.
    """

    return Level0Mask(
        mask_array,
    )


@pytest.fixture
def mapped_mask(
    mask_array: np.ndarray,
) -> MappedMask:
    """
    MappedMask constructed from the shared deterministic mask.

    The stored mask has shape (10, 10) and represents a slide of
    shape (20, 20), resulting in a scale factor of 2 in both axes.
    """

    return MappedMask(
        mask=mask_array,
        slide_shape=(20, 20),
    )


class TestBaseMask:  

    def test_constructor_rejects_non_numpy_array(self):
        """Constructor should reject non-NumPy arrays."""

        with pytest.raises(
            TypeError,
            match="mask must be a numpy.ndarray",
        ):
            DummyMask(
                mask=[[1, 2], [3, 4]],
                slide_shape=(2, 2),
            )

    def test_constructor_rejects_non_two_dimensional_mask(self):
        """Constructor should reject masks that are not two-dimensional."""

        mask = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(
            ValueError,
            match="mask must be two-dimensional",
        ):
            DummyMask(
                mask=mask,
                slide_shape=(10, 10),
            )

    def test_constructor_rejects_empty_mask(self):
        """Constructor should reject masks having zero-sized dimensions."""

        mask = np.zeros((0, 10), dtype=np.uint8)

        with pytest.raises(
            ValueError,
            match="mask dimensions must be positive",
        ):
            DummyMask(
                mask=mask,
                slide_shape=(10, 10),
            )

    def test_constructor_rejects_non_positive_slide_dimensions(self):
        """Constructor should reject non-positive slide dimensions."""

        mask = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(
            ValueError,
            match="slide_shape must contain positive dimensions",
        ):
            DummyMask(
                mask=mask,
                slide_shape=(0, 10),
            )

    def test_properties_return_expected_values(
        self,
        dummy_mask: DummyMask,
        mask_array: np.ndarray,
    ):
        """Properties should expose the expected mask metadata."""

        assert dummy_mask.array is mask_array
        assert dummy_mask.dtype == np.uint8
        assert dummy_mask.shape == mask_array.shape
        assert dummy_mask.slide_shape == mask_array.shape
        assert dummy_mask.scale_x == pytest.approx(1.0)
        assert dummy_mask.scale_y == pytest.approx(1.0)

    
    def test_validate_region_accepts_valid_region(
        self,
        dummy_mask: DummyMask,
    ):
        """Region validation should accept regions contained within the slide."""

        dummy_mask._validate_region(
            x=2,
            y=3,
            width=4,
            height=5,
        )

    def test_validate_region_rejects_region_outside_slide(
        self,
        dummy_mask: DummyMask,
    ):
        """Region validation should reject regions extending beyond the slide."""

        with pytest.raises(
            ValueError,
            match="extends outside the slide",
        ):
            dummy_mask._validate_region(
                x=8,
                y=8,
                width=4,
                height=4,
            )

    def test_validate_region_rejects_negative_coordinates(
        self,
        dummy_mask: DummyMask,
    ):
        """Region validation should reject negative coordinates."""

        for x, y in [(-1, 0), (0, -1)]:
            with pytest.raises(
                ValueError,
                match="negative coordinates",
            ):
                dummy_mask._validate_region(
                    x=x,
                    y=y,
                    width=5,
                    height=5,
                )

    def test_validate_region_rejects_non_positive_dimensions(
        self,
        dummy_mask: DummyMask,
    ):
        """Region validation should reject non-positive dimensions."""

        for width, height in [(0, 5), (5, 0)]:
            with pytest.raises(
                ValueError,
                match="must be positive",
            ):
                dummy_mask._validate_region(
                    x=0,
                    y=0,
                    width=width,
                    height=height,
                )

    def test_repr_contains_mask_metadata(
        self,
        dummy_mask: DummyMask,
    ):
        """String representation should summarise the stored mask."""

        representation = repr(dummy_mask)

        assert "DummyMask" in representation
        assert "shape=(10, 10)" in representation
        assert "dtype=uint8" in representation
        assert "scale_x=1" in representation
        assert "scale_y=1" in representation


class TestLevel0Mask:

    def test_constructor_infers_slide_shape(
        self,
        level0_mask: Level0Mask,
        mask_array: np.ndarray,
    ):
        """Constructor should infer the slide shape from the stored mask."""

        assert level0_mask.slide_shape == mask_array.shape

    def test_scale_factors_are_identity(
        self,
        level0_mask: Level0Mask,
    ):
        """``Level0Mask`` should always expose an identity coordinate mapping."""

        assert level0_mask.scale_x == pytest.approx(1.0)
        assert level0_mask.scale_y == pytest.approx(1.0)

    def test_read_region_native_returns_requested_region(
        self,
        level0_mask: Level0Mask,
        mask_array: np.ndarray,
    ):
        """``read_region_native()`` should return the requested mask region."""

        region = level0_mask.read_region_native(
            x=2,
            y=3,
            width=4,
            height=2,
        )

        expected = mask_array[
            3:5,
            2:6,
        ]

        assert np.array_equal(
            region,
            expected,
        )

    def test_read_region_returns_level0_region(
        self,
        level0_mask: Level0Mask,
        mask_array: np.ndarray,
    ):
        """``read_region()`` should return the requested level-0 mask region."""

        region = level0_mask.read_region(
            x=2,
            y=3,
            width=4,
            height=2,
        )

        expected = mask_array[
            3:5,
            2:6,
        ]

        assert np.array_equal(
            region,
            expected,
        )

    def test_read_region_matches_native_region_for_level0_masks(
        self,
        level0_mask: Level0Mask,
    ):
        """``read_region()`` should be identical to ``read_region_native()``."""

        native = level0_mask.read_region_native(
            x=1,
            y=2,
            width=5,
            height=4,
        )

        level0 = level0_mask.read_region(
            x=1,
            y=2,
            width=5,
            height=4,
        )

        assert np.array_equal(
            native,
            level0,
        )


class TestMappedMask:

    def test_constructor_derives_scale_from_slide_shape(
        self,
        mapped_mask: MappedMask,
    ):
        """Constructor should derive coordinate scales from the slide shape."""

        assert mapped_mask.scale_x == pytest.approx(2.0)
        assert mapped_mask.scale_y == pytest.approx(2.0)

    def test_constructor_uses_explicit_scale(
        self,
        mask_array: np.ndarray,
    ):
        """Constructor should use explicitly supplied coordinate scales."""

        mask = MappedMask(
            mask=mask_array,
            slide_shape=(20, 20),
            scale_x=4.0,
            scale_y=5.0,
        )

        assert mask.scale_x == pytest.approx(4.0)
        assert mask.scale_y == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scale_x": 2.0},
            {"scale_y": 2.0},
        ],
    )
    def test_constructor_requires_both_scale_factors(
        self,
        mask_array: np.ndarray,
        kwargs: dict[str, float],
    ):
        """Constructor should require both scale factors together."""

        with pytest.raises(
            ValueError,
            match="Both scale_x and scale_y",
        ):
            MappedMask(
                mask=mask_array,
                slide_shape=(20, 20),
                **kwargs,
            )

    def test_constructor_rejects_non_positive_scale_factors(
        self,
        mask_array: np.ndarray,
    ):
        """Constructor should reject non-positive scale factors."""

        for scale_x, scale_y in [
            (0.0, 1.0),
            (-1.0, 1.0),
            (1.0, 0.0),
            (1.0, -1.0),
        ]:
            with pytest.raises(
                ValueError,
                match="must be positive",
            ):
                MappedMask(
                    mask=mask_array,
                    slide_shape=(20, 20),
                    scale_x=scale_x,
                    scale_y=scale_y,
                )

    def test_level0_to_mask_maps_coordinates(
        self,
        mapped_mask: MappedMask,
    ):
        """Level-0 coordinates should map to stored-mask coordinates."""

        x, y = mapped_mask._level0_to_mask(
            x=8,
            y=12,
        )

        assert x == pytest.approx(4.0)
        assert y == pytest.approx(6.0)

    def test_mask_roi_returns_expected_native_roi(
        self,
        mapped_mask: MappedMask,
    ):
        """Requested slide regions should map to the correct stored ROI."""

        roi = mapped_mask._mask_roi(
            x=4,
            y=6,
            width=8,
            height=4,
        )

        assert roi == (
            (2, 6),
            (3, 5),
        )

    def test_mask_roi_rounds_outwards_for_fractional_scales(
        self,
    ):
        """``_mask_roi()`` should round outward to ensure the stored-mask ROI
        fully covers the requested level-0 region.
        """
        
        mask = np.zeros(
            (10, 10),
            dtype=np.uint8,
        )

        mapped_mask = MappedMask(
            mask=mask,
            slide_shape=(15, 15),
        )

        roi = mapped_mask._mask_roi(
            x=2,
            y=2,
            width=5,
            height=5,
        )

        assert roi == (
            (1, 5),
            (1, 5),
        )

    def test_read_region_native_returns_native_resolution_region(
        self,
        mapped_mask: MappedMask,
        mask_array: np.ndarray,
    ):
        """``read_region_native()`` should return the stored-resolution region."""

        region = mapped_mask.read_region_native(
            x=4,
            y=6,
            width=8,
            height=4,
        )

        expected = mask_array[
            3:5,
            2:6,
        ]

        assert np.array_equal(
            region,
            expected,
        )

    def test_read_region_returns_requested_level0_shape(
        self,
        mapped_mask: MappedMask,
    ):
        """``read_region()`` should return the requested level-0 output shape."""

        region = mapped_mask.read_region(
            x=4,
            y=6,
            width=8,
            height=4,
        )

        assert region.shape == (4, 8)

    def test_read_region_preserves_label_values(
        self,
    ):
        """``read_region()`` should preserve label values during nearest-neighbour resampling."""

        mask = np.array(
            [
                [1, 2],
                [3, 4],
            ],
            dtype=np.uint8,
        )

        mapped_mask = MappedMask(
            mask=mask,
            slide_shape=(4, 4),
        )

        region = mapped_mask.read_region(
            x=0,
            y=0,
            width=4,
            height=4,
        )

        assert set(np.unique(region)) == {1, 2, 3, 4}

        assert np.all(region[:2, :2] == 1)
        assert np.all(region[:2, 2:] == 2)
        assert np.all(region[2:, :2] == 3)
        assert np.all(region[2:, 2:] == 4)

    def test_read_region_matches_native_when_no_resampling_is_required(
        self,
        mask_array: np.ndarray,
    ):
        """``read_region()`` should equal ``read_region_native()`` when no resampling is required."""

        mask = MappedMask(
            mask=mask_array,
            slide_shape=mask_array.shape,
        )

        native = mask.read_region_native(
            x=2,
            y=3,
            width=4,
            height=2,
        )

        level0 = mask.read_region(
            x=2,
            y=3,
            width=4,
            height=2,
        )

        assert np.array_equal(
            native,
            level0,
        )
