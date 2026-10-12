"""
Level-0 whole-slide mask implementation for PySlyde.

This module provides the canonical implementation of a whole-slide mask that
is already aligned to the slide's level-0 coordinate system.

Unlike MappedMask, this implementation performs no coordinate mapping or
resampling. Requested regions are returned directly from the stored NumPy array.
"""

from __future__ import annotations

import numpy as np

from pyslyde.masks.base import BaseMask

class Level0Mask(BaseMask):
    """
    Whole-slide mask stored directly in level-0 slide coordinates.

    No coordinate transformation or resampling is required, making
    ``read_region_native()`` and ``read_region()`` equivalent.

    Args:
        mask:
            Whole-slide mask aligned to level-0 slide coordinates.

    Notes:
        The corresponding slide dimensions are inferred from ``mask.shape``.
    """

    def __init__(
        self,
        mask: np.ndarray,
    ) -> None:
        super().__init__(
            mask=mask,
            slide_shape=mask.shape,
        )

    def _resolve_scale(
        self,
        *,
        slide_shape: tuple[int, int],
        scale_x: float | None,
        scale_y: float | None,
    ) -> tuple[float, float]:
        """
        Resolve the coordinate mapping between the slide and stored mask.

        Since a Level0Mask is already aligned to level-0 slide coordinates,
        the mapping is always the identity transformation.

        Args:
            slide_shape
                Level-0 slide dimensions as (height, width).

            scale_x
                Optional x-axis coordinate scale.

            scale_y
                Optional y-axis coordinate scale.

        Returns:
            tuple[float, float]
                Identity mapping ``(1.0, 1.0)``.

        Raises:
            ValueError
                If explicit scales other than 1.0 are supplied or if
                ``slide_shape`` does not match the stored mask dimensions.
        """

        if slide_shape != self.shape:
            raise ValueError(
                f"slide_shape {slide_shape} does not match "
                f"the stored mask shape {self.shape}."
            )

        if scale_x is not None and not np.isclose(scale_x, 1.0):
            raise ValueError(
                "Level0Mask cannot be constructed with scale_x != 1.0."
            )

        if scale_y is not None and not np.isclose(scale_y, 1.0):
            raise ValueError(
                "Level0Mask cannot be constructed with scale_y != 1.0."
            )

        return 1.0, 1.0

    def read_region_native(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Read a region from the stored mask.

        Since the mask is already stored in level-0 coordinates, the returned
        region has exactly the requested dimensions.

        Args:
            x, y
                Level-0 top-left coordinate.

            width, height
                Requested region dimensions.

        Returns:
            numpy.ndarray
                Mask region having shape ``(height, width)``.

        Raises:
            ValueError
                If the requested region lies outside the slide.
        """

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
        """
        Read a region aligned to level-0 output dimensions.

        Args:
            x, y
                Level-0 top-left coordinate.

            width, height
                Requested output dimensions.

        Returns:
            numpy.ndarray
                Mask region having shape ``(height, width)``.

        Raises:
            ValueError
                If the requested region lies outside the slide.
        """

        return self.read_region_native(
            x=x,
            y=y,
            width=width,
            height=height,
        )