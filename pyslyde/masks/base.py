"""
Canonical whole-slide mask abstraction for PySlyde.

Architecture
============

PySlyde originally assumed that whole-slide masks were supplied as NumPy
arrays aligned to the slide's level-0 coordinate system. While simple, this
required external tools to upsample masks before they could be consumed by
PySlyde, leading to unnecessary storage requirements and memory usage.

This module introduces a storage-agnostic abstraction for whole-slide masks.

All coordinates exposed by the public interface are expressed in level-0
slide coordinates, regardless of the internal storage resolution of a mask.

Concrete implementations are responsible for translating between level-0
coordinates and their internal representation.

Design
------

The mask abstraction follows the same philosophy as the rest of PySlyde:

- Parser code operates exclusively in level-0 slide coordinates.
- If sufficient information exists to determine the coordinate mapping,
  derive it automatically.
- If the mapping cannot be uniquely determined, fail loudly with an
  informative exception rather than silently making assumptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class BaseMask(ABC):
    """
    Abstract representation of a whole-slide mask.

    A BaseMask represents a semantic segmentation or annotation mask covering
    an entire whole-slide image. Regardless of how the mask is stored
    internally, every implementation exposes a level-0 coordinate interface.

    All public methods operate in level-0 slide coordinates. Concrete
    subclasses are responsible for mapping those coordinates to their stored
    representation.

    Args:
        mask
            Stored mask array.

            The array may represent the slide at any resolution depending on the
            concrete subclass.

        slide_shape
            Level-0 slide dimensions as (height, width).

            Concrete subclasses may use this information to derive the mapping
            between slide coordinates and stored mask coordinates.

        scale_x
            Optional mapping from level-0 x coordinates to stored mask x coordinates.

        scale_y
            Optional mapping from level-0 y coordinates to stored mask y coordinates.
    """

    def __init__(
        self,
        mask: np.ndarray,
        *,
        slide_shape: tuple[int, int],
        scale_x: float | None = None,
        scale_y: float | None = None,
    ) -> None:
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                f"mask must be a numpy.ndarray, got {type(mask).__name__}."
            )

        if mask.ndim != 2:
            raise ValueError(
                f"mask must be two-dimensional, got shape {mask.shape}."
            )

        mask_h, mask_w = mask.shape

        if mask_h == 0 or mask_w == 0:
            raise ValueError(
                "mask dimensions must be positive."
            )

        slide_h, slide_w = slide_shape

        if slide_h <= 0 or slide_w <= 0:
            raise ValueError(
                "slide_shape must contain positive dimensions."
            )

        self._mask = mask
        self._slide_shape = slide_shape

        self._scale_x, self._scale_y = self._resolve_scale(
            slide_shape=slide_shape,
            scale_x=scale_x,
            scale_y=scale_y,
        )

    @property
    def array(self) -> np.ndarray:
        """
        Stored mask array.

        The array represents the internal storage of the mask and therefore
        may not be level-0 resolution.        

        Returns:
            numpy.ndarray        
        """
        return self._mask

    @property
    def dtype(self) -> np.dtype:
        """Return the underlying NumPy dtype."""
        return self._mask.dtype

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return stored mask dimensions.

        Returns:
            tuple[int, int]
                Stored mask dimensions as (height, width).
        """
        return self._mask.shape

    @property
    def slide_shape(self) -> tuple[int, int]:
        """
        Return the corresponding level-0 slide dimensions.

        Returns:
            tuple[int, int] or None
        """
        return self._slide_shape

    @property
    def scale_x(self) -> float:
        """
        Ratio between level-0 x coordinates and stored mask coordinates.
        """
        return self._scale_x

    @property
    def scale_y(self) -> float:
        """
        Ratio between level-0 y coordinates and stored mask coordinates.
        """
        return self._scale_y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"scale_x={self.scale_x:.6g}, "
            f"scale_y={self.scale_y:.6g})"
        )

    @abstractmethod
    def _resolve_scale(
        self,
        *,
        slide_shape: tuple[int, int],
        scale_x: float | None,
        scale_y: float | None,
    ) -> tuple[float, float]:
        """
        Resolve the coordinate mapping between the slide and stored mask.

        Examples:
            Level0Mask always returns (1.0, 1.0).

            MappedMask may use explicitly supplied scales, or derive scales 
            from the relationship between slide dimensions and mask dimensions.            

            Implementations must fail with an informative exception if the
            mapping cannot be uniquely determined.

        Returns:
            tuple[float, float]
                (scale_x, scale_y)
        """

    def _validate_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """
        Validate that a requested level-0 region lies entirely within the
        corresponding slide.

        Args:
            x, y
                Level-0 top-left coordinate.

            width, height
                Requested region dimensions.

        Raises:
            ValueError
                If the requested region lies outside the slide.
        """

        if width <= 0:
            raise ValueError("width must be positive.")

        if height <= 0:
            raise ValueError("height must be positive.")

        if x < 0 or y < 0:
            raise ValueError(
                "Requested region cannot have negative coordinates."
            )

        slide_h, slide_w = self.slide_shape

        if x + width > slide_w or y + height > slide_h:
            raise ValueError(
                "Requested region extends outside the slide."
            )

    @abstractmethod
    def read_region_native(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Read a region from the stored mask without resampling.

        The requested region is specified in level-0 slide coordinates but is
        returned in the mask's native storage resolution.

        Args:
            x, y:
                Level-0 top-left coordinate of the requested region.

            width, height:
                Dimensions of the requested region in level-0 pixels.

        Returns:
            Mask region at the native storage resolution. The returned array is 
            not guaranteed to have shape ``(height, width)`` but represents the 
            same physical region of the slide.

        Raises:
            ValueError:
                If the requested region lies outside the slide.
        """

    @abstractmethod
    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Read a region from the mask resampled to level-0 dimensions.

        The requested region is specified in level-0 slide coordinates and is
        returned with shape ``(height, width)``, making it suitable for
        pixel-wise correspondence with regions extracted from the slide.

        Args:
            x, y:
                Level-0 top-left coordinate of the requested region.

            width, height:
                Requested output dimensions in level-0 pixels.

        Returns:
            Mask region with shape ``(height, width)``.

        Raises:
            ValueError:
                If the requested region lies outside the slide.
        """