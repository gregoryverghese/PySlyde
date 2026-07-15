"""
Coordinate-mapped whole-slide mask implementation for PySlyde.

This module provides the canonical implementation for masks whose stored
resolution differs from the slide's level-0 coordinate system.

Unlike Level0Mask, MappedMask stores the mask at an arbitrary resolution
while exposing the same level-0 coordinate interface. Requests are
translated into the stored mask coordinate system internally, allowing the
rest of PySlyde to remain agnostic to how masks are stored.

Scale Resolution
----------------
The relationship between level-0 slide coordinates and stored mask
coordinates may be:

- supplied explicitly through ``scale_x`` and ``scale_y``; or
- derived automatically from the relationship between the slide dimensions
  and the stored mask dimensions.

Construction fails if the coordinate mapping cannot be uniquely determined.
"""

from __future__ import annotations

import cv2
import numpy as np

from pyslyde.masks.base import BaseMask


class MappedMask(BaseMask):
    """
    Whole-slide mask whose stored resolution differs from the slide's
    level-0 coordinate system.

    Args:
        mask
            Stored mask.

        slide_shape
            Level-0 slide dimensions as (height, width).

        scale_x
            Optional mapping from level-0 x coordinates to stored mask
            coordinates.

        scale_y
            Optional mapping from level-0 y coordinates to stored mask
            coordinates.
    """

    def __init__(
        self,
        mask: np.ndarray,
        *,
        slide_shape: tuple[int, int],
        scale_x: float | None = None,
        scale_y: float | None = None,
    ) -> None:

        super().__init__(
            mask=mask,
            slide_shape=slide_shape,
            scale_x=scale_x,
            scale_y=scale_y,
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

        Coordinate scales are resolved in the following order:

        1. Explicit ``scale_x`` and ``scale_y``.
        2. Derived from the relationship between the slide and stored mask dimensions.

        Returns:
            Coordinate scales as ``(scale_x, scale_y)``.

        Raises:
            ValueError:
                If the supplied coordinate mapping is invalid.
        """

        if (scale_x is None) ^ (scale_y is None):
            raise ValueError(
                "Both scale_x and scale_y must be supplied together, or neither."
            )

        if scale_x is not None:

            if scale_x <= 0:
                raise ValueError(
                    "scale_x must be positive."
                )

            if scale_y <= 0:
                raise ValueError(
                    "scale_y must be positive."
                )

            return float(scale_x), float(scale_y)

        slide_h, slide_w = slide_shape
        mask_h, mask_w = self.shape

        derived_scale_x = slide_w / mask_w
        derived_scale_y = slide_h / mask_h

        return (
            float(derived_scale_x),
            float(derived_scale_y),
        )
    
    def read_region_native(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Read a region using the mask's native storage resolution.

        Args:
        x, y
            Level-0 top-left coordinate.

        width, height
            Level-0 region dimensions.

        Returns:
            numpy.ndarray
                Region extracted directly from the stored mask without
                interpolation.

        Raises:
            ValueError
                If the requested level-0 region lies outside the slide.

        Notes:
            The returned array is in the mask's native stored resolution.
            Consequently, its dimensions are generally different from the
            requested ``(height, width)`` while representing the same physical
            region.
        """

        self._validate_region(
            x=x,
            y=y,
            width=width,
            height=height,
        )
   
        (x_min, x_max), (y_min, y_max) = self._mask_roi(
            x,
            y,
            width,
            height,
        )

        return self.array[
            y_min:y_max,
            x_min:x_max,
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
                Mask region resampled to shape ``(height, width)`` using
                nearest-neighbour interpolation.

        Raises:
            ValueError
                If the requested region lies outside the slide.

        Notes:
            Nearest-neighbour interpolation is used to preserve 
            discrete mask labels during resampling.
        """

        roi = self.read_region_native(
            x=x,
            y=y,
            width=width,
            height=height,
        )

        if roi.shape == (height, width):
            return roi

        return cv2.resize(
            roi,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    def _level0_to_mask(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """
        Convert level-0 slide coordinates into the corresponding stored-mask
        coordinates.

        Args:
            x, y
                Level-0 slide coordinates.

        Returns:
            tuple[float, float]
                Corresponding coordinates in the stored mask coordinate system.
        """

        return (
            x / self.scale_x,
            y / self.scale_y,
        )

    def _mask_roi(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Return the stored-mask ROI corresponding to a level-0 slide region.

        Region starts are rounded down while region ends are rounded up to ensure
        that the returned mask region completely covers the requested physical
        region.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]:
                Stored mask ROI as ``((x_min, x_max), (y_min, y_max))``.        
        """

        x0, y0 = self._level0_to_mask(x, y)

        x1, y1 = self._level0_to_mask(
            x + width,
            y + height,
        )

        return (
            (int(np.floor(x0)), int(np.ceil(x1))),
            (int(np.floor(y0)), int(np.ceil(y1))),
        )