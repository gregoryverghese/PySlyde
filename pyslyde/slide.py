"""
slide.py: Slide abstraction, annotation handling, and API validation utilities.

This module provides:

- Slide:
    An extension of ``openslide.OpenSlide`` with annotation-aware
    region extraction and mask rasterisation capabilities.

- Annotations:
    A format-agnostic parser and container for polygon-based slide annotations.

- Validation exceptions:
    Custom exception classes used to signal invalid API arguments
    and contract violations within the Slide interface.
"""

import json
import operator as op
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from openslide import OpenSlide

from pyslyde.util.utilities import mask2rgb

_OPERATOR_FUNCS: Mapping[str, Callable[[int, int], bool]] = MappingProxyType(
    {
        ">": op.gt,
        ">=": op.ge,
        "=>": op.ge,
        "<": op.lt,
        "<=": op.le,
        "=<": op.le,
    }
)

ALLOWED_RESIZE_OPERATORS: tuple[str, ...] = tuple(_OPERATOR_FUNCS.keys())


class Slide(OpenSlide):
    """
    Wrapper around OpenSlide that adds annotation-aware utilities.

    The class can load polygon annotations and provides helpers to:
      - rasterise annotations into integer label masks (``generate_mask``)
      - extract image regions with aligned ROI masks (``generate_region``)
      - compute an annotation-derived bounding box (``get_border``)
      - save mask/visualisation/metadata artifacts (``save``)

    Coordinate conventions:
      - Annotation vertices are interpreted in level-0 (full-resolution) pixel
        space as(x, y) pairs; x is horizontal (column) and y is vertical (row).
      - Returned masks are NumPy arrays with shape (H, W) = (height, width).

    Attributes:
        level:
            Default OpenSlide magnificaiton level index stored on the instance.
        dims:
            Level-0 slide dimensions as (width, height).
        name:
            Filename (basename) of the slide.
        annotations:
            Optional loaded :class:``Annotations`` instance.
        _border:
            Cached annotation border in level-0 units as
            [(x_min, x_max), (y_min, y_max)].
    """

    MASK_SIZE: Tuple[int, int] = (2000, 2000)

    def __init__(
        self,
        filename: str,
        level: int = 0,
        annotations: Optional["Annotations"] = None,
        annotations_path: Optional[Union[str, List[str]]] = None,
        labels: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Create a Slide backed by an OpenSlide WSI and optionally load annotations.

        Only one annotation source is permitted:
        - pre-built ``Annotations`` instance via ``annotations``, or
        - ``annotations_path`` + ``source`` to load annotations from disk.

        Args:
            filename:
                Path to the whole-slide image file readable by OpenSlide.
            level:
                OpenSlide pyramid magnificaiton level index.
            annotations:
                Pre-loaded ``Annotations`` instance. If provided, ``annotations_path``
                and ``source`` are ignored.
            annotations_path:
                Path or list of paths to annotation file(s). Used only if
                ``annotations`` is not provided. Must be provided together with ``source``.
            labels:
                Optional list of annotation labels to keep when loading annotations
                from ``annotations_path``. Ignored when ``annotations`` is provided.
            source:
                Annotation loader identifier understood by ``Annotations``
                (e.g. "qupath", "imagej", "asap", "geojson", "csv", ...).
                Required if ``annotations_path`` is provided.

        Raises:
            ValueError:
                If ``annotations_path`` is provided without ``source`` (or vice versa),
                or if annotation loading fails.
        """
        super().__init__(filename)
        self.level: int = level
        self.dims: Tuple[int, int] = self.dimensions
        self.name: str = os.path.basename(filename)
        self._border: Optional[List[Tuple[int, int]]] = None
        self.annotations: Optional["Annotations"] = None

        if annotations is not None:
            self.annotations = annotations
        elif (annotations_path is None) ^ (source is None):
            raise ValueError(
                "Provide both ``annotations_path`` and ``source``, or neither."
            )
        elif annotations_path is not None:
            self.annotations = Annotations(
                annotations_path, source=source, labels=labels, encode=True
            )

    def generate_region(
        self,
        level: int = 0,
        x: Optional[Union[int, Tuple[int, int]]] = None,
        y: Optional[Union[int, Tuple[int, int]]] = None,
        x_size: Optional[int] = None,
        y_size: Optional[int] = None,
        scale_border: bool = False,
        factor: int = 1,
        threshold: Optional[int] = None,
        operator: str = "=>",
        *,
        labels: Optional[List[Union[int, str]]] = None,
        dtype: np.dtype = np.uint16,
        rounding: str = "round",
        clamp_to_level_bounds: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract an image region and its corresponding annotation mask.

        Region is defined in full-resolution coordinate space and is
        read from the slide at the requested magnification level. The
        returned annotation mask is rasterised only for the requested ROI and is
        aligned pixel-for-pixel with the returned image.

        Args:
            level:
                OpenSlide pyramid level index to read the image data from.
            x, y:
                ROI axis specifications in level-0 pixels. Each axis may be:
                - int: start coordinate (requires corresponding ``*_size``)
                - tuple: (min, max) bounds (size may be inferred if ``*_size`` is None)
                If ``x`` is None, the ROI defaults to the padded annotation border returned
                by :meth:``get_border`` (or the full slide if no annotations exist).
            x_size, y_size:
                ROI size in level-0 pixels. Required when the corresponding axis is specified
                as an int start coordinate.
            scale_border:
                If True, adjust the resolved ROI size using :meth:``Slide.resize_border``.
            factor, threshold, operator:
                Parameters forwarded to :meth:``Slide.resize_border`` when ``scale_border`` is True.
            labels:
                Optional subset of classes to include in the output mask. Elements may be
                class names (str) or class IDs (int). If None, all available labels are used.
            dtype:
                NumPy dtype of the output mask.
            rounding:
                Used when converting level-0 ROI sizes to ``level`` pixel dimensions.
                - round: default (nearest pixel grid)
                - floor: avoids over-requesting pixels
                - ceil: ensures coverage, may request slightly larger regions
            clamp_to_level_bounds:
                If True, clamps the requested (out_w, out_h) to the available level dimensions
                to avoid requesting pixels beyond slide bounds.

        Returns:
            image_rgb:
                RGB region as a NumPy array of shape (H, W, 3), corresponding to the ROI
                read at pyramid level ``level``.
            mask_roi:
                Integer label mask as a NumPy array of shape (H, W), aligned with ``image_rgb``.
                Background is 0; foreground pixels contain stable class IDs from
                :attr:``Annotations.class_key``.
        """
        level = self._validate_level(level)

        x_min, y_min, x_size, y_size = self._normalise_roi_level0(
            x,
            y,
            x_size,
            y_size,
            scale_border=scale_border,
            factor=factor,
            threshold=threshold,
            operator=operator,
        )

        ds = float(self.level_downsamples[level])
        out_w = max(Slide._round_dim(x_size / ds, rounding), 1)
        out_h = max(Slide._round_dim(y_size / ds, rounding), 1)

        if clamp_to_level_bounds:
            level_w, level_h = self.level_dimensions[level]
            out_w = min(out_w, int(level_w))
            out_h = min(out_h, int(level_h))

        region = self.read_region((x_min, y_min), level, (out_w, out_h))
        image_rgb = np.array(region.convert("RGB"))

        mask_roi = self._rasterise_roi_mask(
            x_min=x_min,
            y_min=y_min,
            x_size=x_size,
            y_size=y_size,
            level=level,
            out_w=out_w,
            out_h=out_h,
            ds=ds,
            labels=labels,
            dtype=dtype,
            rounding=rounding,
            clamp_to_level_bounds=clamp_to_level_bounds,
        )

        return image_rgb, mask_roi

    def generate_mask(
        self,
        size: Optional[Tuple[int, int]] = None,  # (width, height)
        labels: Optional[List[Union[int, str]]] = None,
        *,
        level: Optional[int] = None,
        dtype: np.dtype = np.uint16,
        full_res: bool = False,
        preserve_aspect: bool = False,
        aspect_rtol: float = 1e-3,
    ) -> np.ndarray:
        """
        Generate an annotation mask with *stable* integer IDs.

        Args:
            size:
                Output mask size (width, height). If provided, polygons are scaled into this size.
            labels:
                Subset of labels to include. Accepts label names (str) OR class IDs (int).
            level:
                Output OpenSlide level to generate at. Mutually exclusive with ``size``.
            dtype:
                dtype of the output mask (default uint16 to avoid overflow >255 classes).
            full_res:
                If True, allow full-resolution mask generation when neither ``size`` nor ``level``
                are provided. If False (default), raise instead of allocating huge arrays.
            preserve_aspect:
                If True and ``size`` is provided, validate that requested size preserves slide
                aspect ratio within ``aspect_rtol``. If violated, raise a clear ValueError.
                (This avoids geometric distortion of polygons.)
            aspect_rtol:
                Relative tolerance for aspect ratio validation when preserve_aspect=True.

        Returns:
            np.ndarray:
                2D mask array (height, width). Background is 0, foreground pixels are class IDs.
        """
        full_w, full_h = self.dims

        if size is not None and level is not None:
            raise ValueError("Provide only one of ``size`` or ``level``, not both.")

        if size is not None:
            level_w, level_h = size
            if not (isinstance(level_w, int) and isinstance(level_h, int)):
                raise ValueError(
                    f"``size`` must be (int width, int height), got {size!r}."
                )
            if level_w <= 0 or level_h <= 0:
                raise ValueError(f"``size`` must be positive, got {size!r}.")

            if preserve_aspect:
                slide_ar = full_w / full_h
                req_ar = level_w / level_h
                if not np.isclose(req_ar, slide_ar, rtol=aspect_rtol, atol=0.0):
                    raise ValueError(
                        "Requested mask size alters the slide aspect ratio, "
                        "which would distort annotation geometry. "
                        f"Slide aspect={slide_ar:.6f}, requested aspect={req_ar:.6f}, "
                        f"size={size}, slide_dims={(full_w, full_h)}. "
                        "Provide a size with a matching aspect ratio or set "
                        "``preserve_aspect=False`` to disable this validation."
                    )

            sx = level_w / full_w
            sy = level_h / full_h

        elif level is not None:
            level = self._validate_level(level)
            level_w, level_h = self.level_dimensions[level]
            ds = float(self.level_downsamples[level])
            sx = 1.0 / ds
            sy = 1.0 / ds

        else:
            if not full_res:
                raise ValueError(
                    "Full-resolution WSI mask generation is disabled by default "
                    "to prevent excessive memory allocation. Provide ``size=(w, h)`` "
                    "or ``level=<int>``, or set ``full_res=True`` to explicitly enable "
                    "full-resolution mask generation."
                )
            level_w, level_h = full_w, full_h
            sx = sy = 1.0

        slide_mask = np.zeros((level_h, level_w), dtype=dtype)

        if self.annotations is None or not self.annotations.annotations:
            return slide_mask

        coordinates = self.annotations.annotations
        class_key = self.annotations.class_key

        use_labels = self._select_labels(labels)
        if not use_labels:
            return slide_mask

        for label in use_labels:
            polys = coordinates.get(label)
            if not polys:
                continue

            k = class_key.get(str(label))
            if k is None:
                continue

            scaled_polys: List[np.ndarray] = []

            for poly in polys:
                arr = np.asarray(poly, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
                    continue

                arr[:, 0] *= sx
                arr[:, 1] *= sy

                arr = np.rint(arr).astype(np.int32)
                arr[:, 0] = np.clip(arr[:, 0], 0, level_w - 1)
                arr[:, 1] = np.clip(arr[:, 1], 0, level_h - 1)

                arr = self._validate_contour(arr)
                if arr is not None:
                    scaled_polys.append(arr)

            if scaled_polys:
                cv2.fillPoly(slide_mask, scaled_polys, color=(int(k),))

        return slide_mask

    def visualise_mask(
        self,
        *,
        size: Tuple[int, int] = MASK_SIZE,
        labels: Optional[List[Union[int, str]]] = None,
        dtype: np.dtype = np.uint16,
        preserve_aspect: bool = True,
        aspect_rtol: float = 1e-3,
    ) -> np.ndarray:
        """
        Generate an RGB visualisation of the annotation mask.

        Args:
            size:
                Output mask size specified as (width, height).
            labels:
                Optional subset of annotation labels to include. Elements may be
                label names (str) or integer IDs. If None, all available labels
                are rendered.
            dtype:
                NumPy dtype used for the intermediate mask before conversion to RGB.
                Default is np.uint16 to support more than 255 distinct labels.
            preserve_aspect:
                If True, validate that the requested size preserves the original
                aspect ratio within ``aspect_rtol`` for the slide.
            aspect_rtol:
                Relative tolerance used when validating aspect ratio preservation.

        Returns:
            RGB image of shape (H, W, 3) representing the visualised annotation mask.
        """
        mask = self.generate_mask(
            size=size,
            labels=labels,
            dtype=dtype,
            preserve_aspect=preserve_aspect,
            aspect_rtol=aspect_rtol,
        )
        return mask2rgb(mask)

    def save(
        self,
        save_dir: str,
        *,
        size: Tuple[int, int] = (2000, 2000),
        labels: Optional[List[Union[str, int]]] = None,
        dtype: np.dtype = np.uint16,
        save_mask: bool = True,
        save_vis: bool = True,
        save_meta: bool = True,
        overwrite: bool = False,
        basename: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save slide annotation artifacts into structured subdirectories:

            save_dir/
                masks/<basename>.npy
                vis/<basename>.png
                meta/<basename>.json

        Args:
            save_dir:
                Root output directory.
            size:
                Output mask size (width, height).
            labels:
                Optional subset of labels to include.
            dtype:
                NumPy dtype for mask.
            save_mask:
                Whether to save raw mask (.npy).
            save_vis:
                Whether to save visualization (.png).
            save_meta:
                Whether to save metadata (.json).
            overwrite:
                If False, raises error if files already exist.
            basename:
                Base filename (without extension) for saved artifacts.
                Defaults to the slide filename stem (self.name without extension).

        Returns:
            Dict[str, str]:
                Dictionary mapping artifact type -> file path for saved items.
                Keys are a subset of {"mask","vis","meta"} depending on flags.
        """
        return self._save_artifacts(
            save_dir,
            size=size,
            labels=labels,
            dtype=dtype,
            save_mask=save_mask,
            save_vis=save_vis,
            save_meta=save_meta,
            overwrite=overwrite,
            basename=basename,
        )

    def detect_components(
        self,
        level: int = 6,
        num_component: Optional[int] = None,
        min_size: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
        """
        Detect connected tissue components on a downsampled representation of the slide.

        For each detected component, a bounding rectangle is computed in thumbnail
        coordinates and mapped to full-resolution coordinates.

        Args:
            level:
                OpenSlide pyramid level used to generate the thumbnail for component detection.
                Higher levels are more downsampled and thus faster but less precise.
                Must be a valid level index for the slide.
            num_component:
                If provided, keep only the ``num_component`` largest components by contour area
                (after any ``min_size`` filtering). If ``None``, all detected components are kept.
            min_size:
                If provided, discard any component with contour area (in thumbnail pixel units)
                less than or equal to ``min_size``. If ``None``, no minimum area filtering is applied.

        Returns:
            components:
                A list of progressively accumulated overlay images (numpy.ndarray)
                where bounding rectangles are drawn around detected components.
                The last entry contains all detected components.
            borders:
                A list of bounding box coordinates in level-0 space, formatted as:
                [[(x1, x2), (y1, y2)], ...]
        """
        level_dims = self.level_dimensions[level]

        thumb_pil = self.get_thumbnail(level_dims).convert("RGB")
        image = np.array(thumb_pil)
        thumb_h, thumb_w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        blur = cv2.bilateralFilter(np.bitwise_not(gray), 9, 100, 100)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if num_component is not None:
            contours = sorted(contours, key=cv2.contourArea)
            contours = contours[-num_component:]

        if min_size is not None:
            contours = [c for c in contours if cv2.contourArea(c) > min_size]

        full_w, full_h = self.dims
        x_scale = full_w / thumb_w
        y_scale = full_h / thumb_h

        overlay = image.copy()
        components = []
        borders = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            components.append(overlay.copy())

            x1 = round(x_scale * x)
            x2 = round(x_scale * (x + w))
            y1 = round(y_scale * y)
            y2 = round(y_scale * (y + h))

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            borders.append([(x1, x2), (y1, y2)])

        return components, borders

    def get_border(
        self,
        padding: int = 100,
        *,
        level: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Generate border around min/max annotation points.

        Args:
            padding:
                Gap between max/min annotation point and border.
            level:
                If provided, return the border scaled into the
                coordinate system of this OpenSlide pyramid level.

        Returns:
            Border dimensions [(x_min, x_max), (y_min, y_max)].
        """
        if self.annotations is None or not self.annotations.annotations:
            border0 = [(0, self.dims[0]), (0, self.dims[1])]
        else:
            coordinates = self.annotations.annotations
            all_polys = list(chain(*list(coordinates.values())))
            all_pts = list(chain(*all_polys))
            xs, ys = zip(*all_pts) if all_pts else ([0], [0])
            border0 = [
                (int(min(xs) - padding), int(max(xs) + padding)),
                (int(min(ys) - padding), int(max(ys) + padding)),
            ]

            border0 = [
                (max(border0[0][0], 0), min(border0[0][1], self.dims[0])),
                (max(border0[1][0], 0), min(border0[1][1], self.dims[1])),
            ]

        self._border = border0

        if level is not None:
            level = self._validate_level(level)
            ds = float(self.level_downsamples[level])
            return [
                (int(border0[0][0] / ds), int(border0[0][1] / ds)),
                (int(border0[1][0] / ds), int(border0[1][1] / ds)),
            ]

        return border0

    def _select_labels(self, labels: Optional[List[Union[int, str]]]) -> List[str]:
        """
        Resolve a label subset specification, if any, into canonical label names.

        Args:
            labels
                Optional subset of classes to include. If None, all available annotation
                labels are returned. If provided, elements may be strings or integers.

        Returns:
            List of label names to rasterise. Labels not present in the annotation
            set are silently ignored.
        """
        if self.annotations is None or not self.annotations.annotations:
            return []

        coordinates = self.annotations.annotations
        id_to_label = self.annotations.id_to_label

        if labels is None:
            return list(coordinates.keys())

        use_labels: List[str] = []
        for lb in labels:
            if isinstance(lb, int):
                name = id_to_label.get(lb)
                if name is not None:
                    use_labels.append(name)
            elif isinstance(lb, str):
                use_labels.append(lb)
        return use_labels

    def _validate_level(self, level: int) -> int:
        """
        Validate magnification level index and return it.

        Args:
            level: OpenSlide magnification level index.

        Returns:
            Validated magnification level.
        """
        if not isinstance(level, int):
            raise ValueError(f"``level`` must be an int, got {type(level).__name__}.")

        n_levels = getattr(self, "level_count", None)
        if n_levels is None:
            raise ValueError(
                "Slide object has no ``level_count``; cannot validate ``level``."
            )

        if not (0 <= level < n_levels):
            dims = list(getattr(self, "level_dimensions", []))
            downs = list(getattr(self, "level_downsamples", []))
            raise ValueError(
                f"Invalid ``level``={level}. Valid levels are 0..{n_levels - 1}. "
                f"Available level_dimensions={dims} and level_downsamples={downs}."
            )

        return level

    def _validate_contour(self, arr: np.ndarray) -> Optional[np.ndarray]:
        """
        Clean and validate a contour for cv2.fillPoly.

        Removes explicit ring closure if present and rejects
        non-area geometries such as LineStrings.

        Args:
            arr: Polygon array.

        Returns:
            Cleaned polygon array.
        """
        if arr.shape[0] >= 2 and np.array_equal(arr[0], arr[-1]):
            arr = arr[:-1]

        if arr.shape[0] < 3:
            return None

        if cv2.contourArea(arr) <= 0:
            return None

        return arr

    def _infer_default_roi(
        self,
        x: Optional[Union[int, Tuple[int, int]]],
        y: Optional[Union[int, Tuple[int, int]]],
    ) -> Tuple[Union[int, Tuple[int, int]], Union[int, Tuple[int, int]]]:
        """
        Infer a complete ROI specification when the caller does not fully define one.

        All coordinates are interpreted in full-resolution pixel space.

        Args:
            x:
                ROI specification for the x-axis. May be:
                - int: start coordinate
                - (min, max) tuple
                - None
            y:
                ROI specification for the y-axis. Same allowed formats as ``x``.

        Returns:
            Tuple (x_spec, y_spec)
                A pair of ROI axis specifications suitable for downstream parsing, where
                each axis is either an int start coordinate or a (min, max) tuple.
        """
        if x is not None:
            return x, y if y is not None else (0, self.dims[1])

        border = self.get_border()
        if border and len(border) >= 2:
            return border[0], border[1]
        return (0, self.dims[0]), (0, self.dims[1])

    def _parse_axis(
        self,
        v: Union[int, Tuple[int, int]],
        size: Optional[int],
        axis_name: str,
    ) -> Tuple[int, int, int]:
        """
        Normalise a single axis ROI specification into explicit bounds and size (at level-0).

        Args:
            v:
                Axis ROI specification: an int start coordinate or a (min, max) tuple.
            size:
                Requested axis length in pixels (level-0 units). May be None when v
                is provided as a (min, max) tuple.
            axis_name:
                Name of the axis (e.g., "x" or "y") used for user-facing error messages.

        Returns:
            v_min:
                Inclusive start coordinate in level-0 pixels.
            v_max:
                Exclusive end coordinate in level-0 pixels.
            size:
                Axis length in level-0 pixels (``v_max - v_min``).
        """
        if isinstance(v, tuple):
            v_min, v_max = v
            if size is None:
                size = v_max - v_min
                return v_min, v_max, size
            else:
                v_min = v_min
                v_max = v_min + size
                return v_min, v_max, size

        if isinstance(v, int):
            if size is None:
                raise ValueError(
                    f"{axis_name}_size must be provided when {axis_name} is an int."
                )
            v_min = v
            v_max = v_min + size
            return v_min, v_max, size

        raise ValueError(f"Invalid {axis_name} spec: expected int or (min,max) tuple.")

    def _normalise_roi_level0(
        self,
        x: Optional[Union[int, Tuple[int, int]]],
        y: Optional[Union[int, Tuple[int, int]]],
        x_size: Optional[int],
        y_size: Optional[int],
        *,
        scale_border: bool,
        factor: int,
        threshold: Optional[int],
        operator: str,
    ) -> Tuple[int, int, int, int]:
        """
        Normalise, validate, and finalise an ROI in level-0 coordinate space.

        It:
        - Infers a default ROI from annotations if none was explicitly provided.
        - Parses axis specifications (int start or (min, max) tuple)
          into explicit (min, size) form.
        - Optionally adjusts ROI dimensions using ``resize_border``.
        - Clamps the ROI to slide bounds to prevent out-of-range reads.
        - Validates that the resolved ROI has positive dimensions.

        All coordinates are expressed in full-resolution pixel space.

        Args:
            x, y:
                ROI axis specifications. Each axis may be:
                - int: start coordinate in level-0 pixels (requires corresponding *_size)
                - tuple: (min, max) bounds in level-0 pixels (size may be inferred)
                - None: allowed only for x; triggers default ROI selection
            x_size, y_size:
                ROI size in pixels (level-0 units). Required if the corresponding axis
                is specified as an int start coordinate.
            scale_border:
                If True, adjust x_size and y_size using :meth:``Slide.resize_border``.
            factor, threshold, operator:
                Parameters forwarded to :meth:``Slide.resize_border`` when ``scale_border`` is True.

        Returns:
            x_min, y_min, x_size, y_size
                ROI origin and size in level-0 pixels.
        """
        x, y = self._infer_default_roi(x, y)

        x_min, x_max, x_size = self._parse_axis(x, x_size, "x")
        y_min, y_max, y_size = self._parse_axis(y, y_size, "y")

        if scale_border:
            x_size = Slide.resize_border(x_size, factor, threshold, operator)
            y_size = Slide.resize_border(y_size, factor, threshold, operator)

        full_w, full_h = self.dimensions
        if (x_min + x_size) > full_w:
            x_size = full_w - x_min
        if (y_min + y_size) > full_h:
            y_size = full_h - y_min

        if x_size <= 0 or y_size <= 0:
            raise ValueError(
                f"Resolved ROI is empty or invalid: x_size={x_size}, y_size={y_size}."
            )

        return x_min, y_min, x_size, y_size

    def _rasterise_roi_mask(
        self,
        *,
        x_min: int,
        y_min: int,
        x_size: int,
        y_size: int,
        level: int,
        out_w: int,
        out_h: int,
        ds: float,
        labels: Optional[List[Union[int, str]]],
        dtype: np.dtype,
        rounding: str = "round",
        clamp_to_level_bounds: bool = True,
    ) -> np.ndarray:
        """
        Rasterise annotation polygons into a mask for a specific ROI.

        Converts annotation polygons (in level-0 coordinates) into a discrete label
        mask, aligned pixel-for-pixel with an ROI image read at OpenSlide pyramid
        level ``level``.

        Args:
            x_min, y_min:
                ROI origin in level-0 pixels.
            x_size, y_size:
                ROI size in level-0 pixels.
            level:
                OpenSlide pyramid level index that defines the target pixel grid.
            out_w, out_h:
                ROI size in the coordinate system of pyramid level ``level``.
                These are the mask dimensions and must match the ROI image dimensions.
            ds:
                Downsample factor for pyramid level ``level`` relative to level 0.
            labels:
                Optional subset of labels/IDs to rasterise.
            dtype:
                NumPy dtype for the output mask (e.g., np.uint16).
            rounding:
                Used when converting level-0 ROI sizes to ``level`` pixel dimensions.
                - round: default (nearest pixel grid)
                - floor: avoids over-requesting pixels
                - ceil: ensures coverage, may request slightly larger regions
            clamp_to_level_bounds:
                If True, clamps the requested (out_w, out_h) to the available level dimensions
                to avoid requesting pixels beyond slide bounds.
        Returns:
            Integer label mask of shape (out_h, out_w). Background is 0 and
            foreground pixels are assigned class IDs.
        """
        expected_ds = float(self.level_downsamples[level])
        if not np.isfinite(ds) or ds <= 0:
            raise ValueError(f"``ds`` must be a positive finite float, got {ds!r}.")
        if not np.isclose(ds, expected_ds, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"Downsample mismatch for level={level}: got ds={ds}, expected {expected_ds}."
            )

        expected_out_w = max(Slide._round_dim(x_size / expected_ds, rounding), 1)
        expected_out_h = max(Slide._round_dim(y_size / expected_ds, rounding), 1)

        if clamp_to_level_bounds:
            level_w, level_h = self.level_dimensions[level]
            expected_out_w = min(expected_out_w, int(level_w))
            expected_out_h = min(expected_out_h, int(level_h))

        if (out_w, out_h) != (expected_out_w, expected_out_h):
            raise ValueError(
                f"Output size mismatch for level={level}: got (out_w, out_h)=({out_w}, {out_h}), "
                f"expected ({expected_out_w}, {expected_out_h}) from ROI and ds, rounding={rounding!r}, "
                f"clamp_to_level_bounds={clamp_to_level_bounds}."
            )

        mask_roi = np.zeros((out_h, out_w), dtype=dtype)

        if self.annotations is None or not self.annotations.annotations:
            return mask_roi

        coordinates = self.annotations.annotations
        class_key = self.annotations.class_key

        use_labels = self._select_labels(labels)
        if not use_labels:
            return mask_roi

        for label in use_labels:
            polys = coordinates.get(label)
            if not polys:
                continue

            k = class_key.get(str(label))
            if k is None:
                continue

            scaled_polys: List[np.ndarray] = []

            for poly in polys:
                arr = np.asarray(poly, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
                    continue

                arr[:, 0] -= float(x_min)
                arr[:, 1] -= float(y_min)

                minx, miny = arr.min(axis=0)
                maxx, maxy = arr.max(axis=0)
                if maxx < 0 or maxy < 0 or minx >= x_size or miny >= y_size:
                    continue

                arr[:, 0] /= expected_ds
                arr[:, 1] /= expected_ds

                arr = np.rint(arr).astype(np.int32)
                arr[:, 0] = np.clip(arr[:, 0], 0, out_w - 1)
                arr[:, 1] = np.clip(arr[:, 1], 0, out_h - 1)

                arr = self._validate_contour(arr)
                if arr is not None:
                    scaled_polys.append(arr)

            if scaled_polys:
                cv2.fillPoly(mask_roi, scaled_polys, color=(int(k),))

        return mask_roi

    def _save_artifacts(
        self,
        save_dir: str,
        *,
        size: Tuple[int, int] = (2000, 2000),
        labels: Optional[List[Union[str, int]]] = None,
        dtype: np.dtype = np.uint16,
        save_mask: bool = True,
        save_vis: bool = True,
        save_meta: bool = True,
        overwrite: bool = False,
        basename: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Internal implementation for saving slide annotation artifacts.

        See ``save()`` for the public API documentation.
        """
        if self.annotations is None or not self.annotations.annotations:
            raise ValueError("Cannot save mask artifacts: no annotations loaded.")

        if not (isinstance(size, tuple) and len(size) == 2):
            raise ValueError(f"``size`` must be a (width, height) tuple, got {size!r}.")
        if not (isinstance(size[0], int) and isinstance(size[1], int)):
            raise ValueError(f"``size`` must be ints, got {size!r}.")
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"``size`` must be positive, got {size!r}.")

        if not (save_mask or save_vis or save_meta):
            raise ValueError(
                "Nothing to save: all of save_mask/save_vis/save_meta are False."
            )

        base = basename or os.path.splitext(self.name)[0]

        masks_dir = os.path.join(save_dir, "masks")
        vis_dir = os.path.join(save_dir, "vis")
        meta_dir = os.path.join(save_dir, "meta")

        if save_mask:
            os.makedirs(masks_dir, exist_ok=True)
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)
        if save_meta:
            os.makedirs(meta_dir, exist_ok=True)

        paths: Dict[str, str] = {}

        mask = None
        if save_mask or save_vis:
            mask = self.generate_mask(size=size, labels=labels, dtype=dtype)

        if save_mask:
            mask_path = os.path.join(masks_dir, f"{base}.npy")
            if not overwrite and os.path.exists(mask_path):
                raise FileExistsError(f"{mask_path} already exists.")
            np.save(mask_path, mask)
            paths["mask"] = mask_path

        if save_vis:
            vis_path = os.path.join(vis_dir, f"{base}.png")
            if not overwrite and os.path.exists(vis_path):
                raise FileExistsError(f"{vis_path} already exists.")
            rgb = mask2rgb(mask)
            ok = cv2.imwrite(vis_path, rgb)
            if not ok:
                raise IOError(f"Failed to write visualization to {vis_path}.")
            paths["vis"] = vis_path

        if save_meta:
            meta_path = os.path.join(meta_dir, f"{base}.json")
            if not overwrite and os.path.exists(meta_path):
                raise FileExistsError(f"{meta_path} already exists.")

            meta = {
                "slide": self.name,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "mask_size": [int(size[0]), int(size[1])],  # (w, h)
                "dtype": str(dtype),
                "labels": labels,
                "class_map": self.annotations.id_to_label,
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            paths["meta"] = meta_path

        return paths

    @staticmethod
    def _round_dim(value: float, rounding: str) -> int:
        """
        Apply the requested rounding policy to a floating-point dimension.

        Args:
            value:
                Floating-point dimension (e.g., x_size / ds).
            rounding:
                One of {"round", "floor", "ceil"}.

        Returns:
            int:
                Rounded integer dimension.
        """
        if rounding == "round":
            return int(round(value))
        if rounding == "floor":
            return int(np.floor(value))
        if rounding == "ceil":
            return int(np.ceil(value))
        raise InvalidRoundingPolicyError(rounding)

    @staticmethod
    def resize_border(
        dim: int,
        factor: int = 1,
        threshold: Optional[int] = None,
        operator: str = "=>",
    ) -> int:
        """
        Round ``dim`` to the nearest multiple of ``factor``, subject to a threshold constraint.

        The function generates multiples of ``factor`` and selects the one closest to ``dim``
        among those satisfying the constraint defined by (``operator``, ``threshold``).

        The constraint determines which multiples are considered valid:

            ">"        : strictly greater than ``threshold``
            ">=" or "=>": greater than or equal to ``threshold``
            "<"        : strictly less than ``threshold``
            "<=" or "=<": less than or equal to ``threshold``

        Args:
            dim:
                Original dimension to adjust.
            factor:
                Positive integer increment defining allowed multiples.
            threshold:
                Boundary value used to filter valid multiples. If None,
                defaults to ``dim``.
            operator:
                String specifying the comparison rule applied between each
                multiple and ``threshold``.

        Returns:
                The adjusted dimension, equal to the multiple of ``factor``
                closest to ``dim`` that satisfies the specified constraint.
        """
        if threshold is None:
            threshold = dim

        if operator not in _OPERATOR_FUNCS:
            raise InvalidResizeBorderOperatorError(operator)

        op_func = _OPERATOR_FUNCS[operator]

        if factor <= 0:
            raise InvalidResizeBorderFactorError(factor)

        max_i = max(int(max(dim, threshold) / factor) + 100, 100)
        multiples = [factor * i for i in range(max_i) if op_func(factor * i, threshold)]

        if not multiples:
            return int(round(dim / factor) * factor)

        new_dim = min(multiples, key=lambda m: abs(dim - m))
        return int(new_dim)


class Annotations:
    """
    Container and parser for slide annotation data.

    Loads annotation files from one or more supported formats
    (e.g., QuPath, ImageJ, ASAP, CSV, GeoJSON) and stores them in a
    canonical in-memory representation:

        {label: [polygon1, polygon2, ...]}

    where each polygon is defined as a list of integer pixel coordinates:

        [[x1, y1], [x2, y2], ...]

    The class provides utilities for:
    - Merging annotations from multiple files
    - Filtering or renaming labels
    - Encoding labels as integer IDs
    - Exporting annotations to DataFrame or GeoJSON format

    Note
    ----
    All coordinates are normalised to integer pixel units.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        source: str,
        labels: Optional[List[str]] = None,
        class_map: Optional[Dict[str, int]] = None,
        encode: bool = False,
    ) -> None:
        """
        Initialise an Annotations object by loading annotation files.

        Args:
            path:
                Path or list of paths to annotation files.
                Multiple files will be parsed and merged.
            source:
                Annotation format identifier. Determines which internal loader
                is used. Examples include:
                - "qupath"
                - "imagej"
                - "asap"
                - "geojson"
                - "csv"
            labels:
                Optional subset of annotation labels to retain after loading.
                If provided, only these labels will be included.
            class_map:
                Optional explicit mapping from label name (str) to integer ID.
                If provided, this mapping will be used when encoding labels.
                Otherwise, IDs are generated automatically in sorted label order.
            encode:
                If True, labels are encoded to integer IDs internally.
                If False, original label names are preserved.
        """
        self.paths: List[str] = path if isinstance(path, list) else [path]
        self.source: str = source
        self.labels: Optional[List[str]] = labels
        self.encode: bool = encode
        self.class_map = class_map
        self._annotations: Optional[Dict[Union[str, int], List[List[List[int]]]]] = None
        self._generate_annotations()

    def __repr__(self) -> str:
        if self._annotations is None:
            return "Annotations(empty)"
        classes = list(self._annotations.keys())
        numbers = [len(self._annotations[k]) for k in classes]
        df = pd.DataFrame({"classes": classes, "number": numbers})
        return str(df)

    @property
    def annotations(self) -> Optional[Dict[Union[str, int], List[List[List[int]]]]]:
        return self._annotations

    @property
    def keys(self) -> List[Union[str, int]]:
        if self._annotations is None:
            return []
        return list(self._annotations.keys())

    @property
    def values(self) -> List[List[List[List[int]]]]:
        if self._annotations is None:
            return []
        return list(self._annotations.values())

    @property
    def numbers(self) -> Dict[str, int]:
        if not self._annotations:
            return {}
        return {k: len(v) for k, v in self._annotations.items()}

    @property
    def df(self) -> pd.DataFrame:
        return self.to_df()

    @property
    def class_key(self) -> Dict[str, int]:
        """
        Mapping from annotation labels to integer IDs.

        Returns:
            Mapping of label name (str) -> integer ID (>=1).
            Empty if no annotations are loaded.
        """
        if not self._annotations:
            return {}

        if self.class_map is not None:
            return dict(self.class_map)

        classes = sorted(map(str, self._annotations.keys()))
        return {lbl: i + 1 for i, lbl in enumerate(classes)}

    @property
    def id_to_label(self) -> Dict[int, str]:
        ck = self.class_key
        return {v: k for k, v in ck.items()}

    @property
    def encoded_annotations(self) -> Dict[int, List[List[List[int]]]]:
        return self.encode_keys()

    def to_df(self) -> pd.DataFrame:
        """
        DataFrame view of annotations, in a tabular vertex
        representation that round-trips with _csv().

        Returns:
            pd.DataFrame with columns:
            - label: str
            - polygon_id: int (index of polygon within label)
            - vertex_id: int (index of vertex within polygon)
            - x: int
            - y: int
        """
        if not self._annotations:
            return pd.DataFrame(columns=["label", "polygon_id", "vertex_id", "x", "y"])

        rows = []
        for label, polygons in self._annotations.items():
            label_str = str(label)

            for polygon_id, poly in enumerate(polygons):
                if not isinstance(poly, (list, tuple)):
                    raise ValueError(
                        f"Invalid annotation polygon for label='{label_str}': "
                        f"expected list of points, got {type(poly).__name__}."
                    )

                for vertex_id, pt in enumerate(poly):
                    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                        raise ValueError(
                            f"Invalid vertex for label='{label_str}', "
                            f"polygon_id={polygon_id}: expected [x, y], got {pt!r}."
                        )

                    x, y = pt
                    rows.append(
                        {
                            "label": label_str,
                            "polygon_id": polygon_id,
                            "vertex_id": vertex_id,
                            "x": self._to_pixel(x),
                            "y": self._to_pixel(y),
                        }
                    )

        return pd.DataFrame(
            rows, columns=["label", "polygon_id", "vertex_id", "x", "y"]
        )

    def to_geojson(
        self,
        *,
        close_rings: bool = True,
        fail_on_invalid: bool = True,
    ) -> Dict[str, Any]:
        """
        Export annotations to a GeoJSON FeatureCollection.

        Expected structure:
            self._annotations: {label: [polygon1, polygon2, ...]}
            polygon: [[x, y], [x, y], ...]

        Args:
            close_rings:
                If True, ensure polygon rings are closed (first==last).
            fail_on_invalid:
                If True, raise on invalid polygons (<3 vertices).
                If False, skip them.

        Returns:
            GeoJSON dict with structure:
            {"type": "FeatureCollection", "features": [...]}
        """
        if not self._annotations:
            return {"type": "FeatureCollection", "features": []}

        features: List[Dict[str, Any]] = []

        for label, polygons in self._annotations.items():
            label_str = str(label)

            if not isinstance(polygons, (list, tuple)):
                raise ValueError(
                    f"Invalid polygons container for label='{label_str}': "
                    f"expected list of polygons, got {type(polygons).__name__}."
                )

            for polygon_id, poly in enumerate(polygons):
                if not isinstance(poly, (list, tuple)):
                    raise ValueError(
                        f"Invalid polygon for label='{label_str}', polygon_id={polygon_id}: "
                        f"expected list of points, got {type(poly).__name__}."
                    )

                coords: List[List[int]] = []
                for v_idx, pt in enumerate(poly):
                    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                        raise ValueError(
                            f"Invalid vertex for label='{label_str}', polygon_id={polygon_id}, "
                            f"vertex_id={v_idx}: expected [x, y], got {pt!r}."
                        )
                    x, y = pt
                    coords.append([self._to_pixel(x), self._to_pixel(y)])

                if len(coords) < 3:
                    msg = (
                        f"Polygon too small for label='{label_str}', polygon_id={polygon_id}: "
                        f"{len(coords)} vertices (need >= 3)."
                    )
                    if fail_on_invalid:
                        raise ValueError(msg)
                    else:
                        continue

                if close_rings and coords[0] != coords[-1]:
                    coords = coords + [coords[0]]

                feature = {
                    "type": "Feature",
                    "properties": {
                        "label": label_str,
                        "polygon_id": polygon_id,
                        "n_vertices": len(coords) - (1 if close_rings else 0),
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords],
                    },
                }
                features.append(feature)

        return {"type": "FeatureCollection", "features": features}

    def save_csv(self, path: str, *, overwrite: bool = False) -> None:
        """
        Save annotations as a CSV file.

        Args:
            path:
                Output file path (must end with .csv).
            overwrite:
                If False, raise FileExistsError if path exists.
        """
        if os.path.splitext(path)[1].lower() != ".csv":
            raise ValueError(f"CSV output path must end with '.csv', got: {path!r}")

        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if not overwrite and os.path.exists(path):
            raise FileExistsError(f"{path} already exists.")

        self.df.to_csv(path, index=False)

    def save_geojson(
        self,
        path: str,
        *,
        close_rings: bool = True,
        fail_on_invalid: bool = True,
        indent: Optional[int] = 2,
        overwrite: bool = False,
    ) -> None:
        """
        Save annotations as a GeoJSON FeatureCollection to disk.

        Args:
            path:
                Output file path (must end with .geojson or .json).
            close_rings:
                Ensure polygon rings are closed (first == last).
            fail_on_invalid:
                Raise if invalid polygons encountered (<3 vertices).
            indent:
                Pretty-print indentation (None for compact).
            overwrite:
                If False, raise FileExistsError if path exists.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext not in {".geojson", ".json"}:
            raise ValueError(
                f"GeoJSON output path must end with '.geojson' or '.json', got: {path!r}"
            )

        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if not overwrite and os.path.exists(path):
            raise FileExistsError(f"{path} already exists.")

        geojson_obj = self.to_geojson(
            close_rings=close_rings,
            fail_on_invalid=fail_on_invalid,
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(geojson_obj, f, ensure_ascii=False, indent=indent)

    def save(
        self,
        path: str,
        *,
        format: str = "csv",
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """
        Save annotations as a CSV or GeoJSON file.

        This enforces that ``path`` has a suffix consistent with ``format``.

        Args:
            path:
                Output file path.
            format:
                "csv" or "geojson" (also accepts "json" meaning GeoJSON).
            overwrite:
                If False, raise FileExistsError if file exists.
            **kwargs:
                Forwarded to save_geojson() for geojson/json format
                (e.g., close_rings, fail_on_invalid, indent).
        """
        fmt = format.lower().strip()
        ext = os.path.splitext(path)[1].lower()

        if fmt == "csv":
            if ext != ".csv":
                raise ValueError(
                    f"Path extension {ext!r} does not match format {format!r}. "
                    "Expected '.csv'."
                )
            self.save_csv(path, overwrite=overwrite)

        elif fmt in {"geojson", "json"}:
            if ext not in {".geojson", ".json"}:
                raise ValueError(
                    f"Path extension {ext!r} does not match format {format!r}. "
                    "Expected '.geojson' or '.json'."
                )
            self.save_geojson(path, overwrite=overwrite, **kwargs)

        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'csv' or 'geojson'.")

    def filter_labels(
        self,
        labels: List[str],
        annotations: Optional[Dict[Union[str, int], List[List[List[int]]]]] = None,
    ) -> Dict[Union[str, int], List[List[List[int]]]]:
        """
        Return a filtered copy of annotations containing only requested labels.
        """
        src = self._annotations if annotations is None else annotations
        if not src:
            return {}
        wanted = set(labels)
        return {k: v for k, v in src.items() if k in wanted}

    def rename_labels(
        self,
        names: Dict[str, str],
        annotations: Optional[Dict[Union[str, int], List[List[List[int]]]]] = None,
    ) -> Dict[Union[str, int], List[List[List[int]]]]:
        """
        Return a copy of annotations with renamed labels.
        """
        src = self._annotations if annotations is None else annotations
        if not src:
            return {}
        out = {}
        for k, v in src.items():
            out[names.get(k, k)] = v
        return out

    def encode_keys(self) -> Dict[int, List[List[List[int]]]]:
        """
        Encode labels as integer values.

        Returns:
            dict: Annotations with integer keys.
        """
        if self._annotations is None:
            return {}
        annotations = {self.class_key[k]: v for k, v in self._annotations.items()}
        return annotations

    def _generate_annotations(self) -> None:
        """
        Load and merge annotations from all paths using the selected loader,
        and apply optional label filtering.
        """
        annotations: Dict[str, List[List[List[int]]]] = {}

        if not self.source:
            self._annotations = {}
            return

        loader_name = f"_{self.source}"
        loader = getattr(self, loader_name, None)
        if loader is None:
            raise ValueError(
                f"Unknown annotation source '{self.source}'. "
                f"No loader '{loader_name}' found."
            )

        for p in self.paths:
            loaded = loader(p)

            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Loader '{loader_name}' returned {type(loaded).__name__}, "
                    "expected dict."
                )

            for k, v in loaded.items():
                if k in annotations:
                    annotations[k].extend(v)
                else:
                    annotations[k] = v

        if self.labels:
            annotations = self.filter_labels(self.labels, annotations=annotations)

        self._annotations = annotations

    @staticmethod
    def _to_pixel(v: Union[float, int, str]) -> int:
        return int(round(float(v)))

    @staticmethod
    def _clean_id(value: Union[str, int]) -> Union[str, int]:
        if isinstance(value, str):
            return value.strip()
        return value

    def _imagej(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse ImageJ XML annotation files.

        Args:
            path (str): Path to the XML file.

        Returns:
            dict: Annotations dictionary.
        """
        tree = ET.parse(path)
        root = tree.getroot()

        ann_elements = root.findall(".//Annotation")
        annotations: Dict[str, List[List[List[int]]]] = {}

        for ann_idx, ann in enumerate(ann_elements):
            label = ann.attrib.get("Name", "undefined")
            label = str(label).strip() if label is not None else "undefined"

            vertices_groups = ann.findall(".//Vertices")
            if not vertices_groups:
                continue

            for vg_idx, vg in enumerate(vertices_groups):
                vertex_elements = vg.findall(".//Vertex")
                if not vertex_elements:
                    continue

                points: List[List[int]] = []
                for v_idx, ve in enumerate(vertex_elements):
                    if "X" not in ve.attrib or "Y" not in ve.attrib:
                        raise ValueError(
                            f"Missing X or Y attribute in ImageJ XML '{path}': "
                            f"Annotation='{label}', annotation_index={ann_idx}, "
                            f"vertices_group_index={vg_idx}, vertex_index={v_idx}. "
                            f"Found attributes: {list(ve.attrib.keys())}"
                        )
                    try:
                        x = self._to_pixel(ve.attrib["X"])
                        y = self._to_pixel(ve.attrib["Y"])
                    except Exception as e:
                        raise ValueError(
                            f"Invalid X/Y values in ImageJ XML '{path}': "
                            f"Annotation='{label}', annotation_index={ann_idx}, "
                            f"vertices_group_index={vg_idx}, vertex_index={v_idx}. "
                            f"X={ve.attrib.get('X')}, Y={ve.attrib.get('Y')}"
                        ) from e

                    points.append([x, y])
                annotations.setdefault(label, []).append(points)

        return annotations

    def _asap(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse ASAP XML annotation files.

        Args:
            path (str): Path to the XML file.

        Returns:
            dict: Annotations dictionary.
        """
        tree = ET.parse(path)
        root = tree.getroot()

        annotations: Dict[str, List[List[List[int]]]] = {}
        ann_elements = root.findall(".//Annotation")

        if not ann_elements:
            return annotations

        for idx, ann in enumerate(ann_elements):
            label = ann.attrib.get("PartOfGroup", "undefined")
            label = str(label).strip()

            coordinates = ann.findall(".//Coordinate")

            if not coordinates:
                continue

            points = []
            for c in coordinates:
                if "X" not in c.attrib or "Y" not in c.attrib:
                    raise ValueError(
                        f"Missing X or Y attribute in ASAP annotation index {idx} "
                        f"in file '{path}'."
                    )

                try:
                    x = self._to_pixel(c.attrib["X"])
                    y = self._to_pixel(c.attrib["Y"])
                except Exception as e:
                    raise ValueError(
                        f"Invalid coordinate values in ASAP annotation index {idx} "
                        f"in file '{path}'. X={c.attrib.get('X')}, Y={c.attrib.get('Y')}"
                    ) from e

                points.append([x, y])

            if label not in annotations:
                annotations[label] = []

            annotations[label].append(points)

        return annotations

    def _qupath(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse QuPath annotation JSON files.

        Args:
            path (str): Path to the JSON file.

        Returns:
            dict: Annotations dictionary.
        """
        annotations: Dict[str, List[List[List[int]]]] = {}

        with open(path) as json_file:
            j = json.load(json_file)

        if not isinstance(j, list):
            raise ValueError(
                f"Invalid QuPath JSON in '{path}': "
                f"expected a list of annotations, got {type(j).__name__}."
            )

        for idx, a in enumerate(j):
            if not isinstance(a, dict):
                raise ValueError(
                    f"Invalid QuPath annotation at index {idx} in '{path}': "
                    f"expected dict, got {type(a).__name__}."
                )

            c = (
                a.get("properties", {})
                .get("classification", {})
                .get("name", "undefined")
            )
            c = str(c).strip() if c is not None else "undefined"

            geom = a.get("geometry")
            if not isinstance(geom, dict):
                raise ValueError(
                    f"Missing/invalid 'geometry' for QuPath annotation "
                    f"index {idx} in '{path}'."
                )

            geometry = geom.get("type")
            coordinates = geom.get("coordinates")

            if geometry is None or coordinates is None:
                raise ValueError(
                    f"Missing 'geometry.type' or 'geometry.coordinates' "
                    f"for QuPath annotation index {idx} in '{path}'."
                )

            if c not in annotations:
                annotations[c] = []

            if geometry == "LineString":
                points = [
                    [self._to_pixel(i[0]), self._to_pixel(i[1])] for i in coordinates
                ]
                annotations[c].append(points)

            elif geometry == "Polygon":
                for ring in coordinates:
                    points = [
                        [self._to_pixel(i[0]), self._to_pixel(i[1])] for i in ring
                    ]
                    annotations[c].append(points)

            elif geometry == "MultiPolygon":
                for poly in coordinates:
                    for ring in poly:
                        points = [
                            [self._to_pixel(i[0]), self._to_pixel(i[1])] for i in ring
                        ]
                        annotations[c].append(points)

            else:
                raise ValueError(
                    f"Unsupported geometry type '{geometry}' "
                    f"for QuPath annotation index {idx} in '{path}'."
                )

        return annotations

    def _json(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse custom JSON annotation files with a specific structure.

        Expected structure (strict):
            {
                "<label>": {
                    "<polygon_id>": [
                        {"x": <num>, "y": <num>},
                        {"x": <num>, "y": <num>},
                        ...
                    ],
                    ...
                },
                ...
            }

        Args:
            path (str): Path to the JSON file.

        Returns:
            dict: Annotations dictionary.
        """
        with open(path) as json_file:
            json_annotations = json.load(json_file)

        if not isinstance(json_annotations, dict):
            raise ValueError(
                f"Invalid JSON annotation format in '{path}': expected a "
                f"top-level object/dict, got {type(json_annotations).__name__}."
            )

        annotations: Dict[str, List[List[List[int]]]] = {}

        for k, v in json_annotations.items():
            if not isinstance(v, dict):
                raise ValueError(
                    f"Invalid JSON annotation format for label='{k}' in '{path}': "
                    f"expected a dictmapping polygon_id -> list of vertices, "
                    f"got {type(v).__name__}."
                )

            polygons: List[List[List[int]]] = []

            for poly_id, v2 in v.items():
                if not isinstance(v2, list):
                    raise ValueError(
                        f"Invalid polygon vertex list for label='{k}', "
                        f"polygon_id='{poly_id}' in '{path}': expected "
                        f"a list of vertices, got {type(v2).__name__}."
                    )

                points: List[List[int]] = []
                for idx, i in enumerate(v2):
                    if not isinstance(i, dict):
                        raise ValueError(
                            f"Invalid vertex for label='{k}', polygon_id='{poly_id}' "
                            f"in '{path}': expected a dict with keys 'x' and 'y', "
                            f"got {type(i).__name__} at vertex index {idx}."
                        )

                    if "x" not in i or "y" not in i:
                        raise ValueError(
                            f"Missing 'x' or 'y' in vertex for label='{k}', "
                            f"polygon_id='{poly_id}' in '{path}': vertex index "
                            f"{idx}. Found keys: {list(i.keys())}"
                        )

                    try:
                        x = self._to_pixel(i["x"])
                        y = self._to_pixel(i["y"])
                    except Exception as e:
                        raise ValueError(
                            f"Invalid coordinate values for label='{k}', "
                            f"polygon_id='{poly_id}' in '{path}': vertex index "
                            f"{idx}. x={i.get('x')}, y={i.get('y')}"
                        ) from e

                    points.append([x, y])
                polygons.append(points)
            annotations[k] = polygons

        return annotations

    def _geojson(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse GeoJSON annotations into the canonical in-memory representation.

        This loader targets general GeoJSON exported by annotation tools.
        It accepts either a ``FeatureCollection`` or a single ``Feature``
        as the top-level container. Each feature is mapped to a label using
        ``feature.properties["label"]`` (default: ``"undefined"``).

        Supported geometry types are converted into the package's canonical structure:
        - ``Polygon``: exterior ring only.
        - ``MultiPolygon``: exterior ring of each polygon only.
        - ``LineString``: stored as a vertex sequence.
        - ``MultiLineString``: each line stored as a vertex sequence.

        Explicitly ignored geometry types (valid GeoJSON but not processed yet):
        - ``Point`` / ``MultiPoint`` (ignored to avoid surprising border expansion)
        - ``GeometryCollection`` (ignored; requires recursive policy decisions)

        Args:
            path: Path to a GeoJSON file.

        Returns:
            Dict[str, List[List[List[int]]]]
                Mapping ``label -> [sequence, sequence, ...]``, each sequence
                is a list of integer vertices ``[[x, y], ...]``.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        def _as_feature_list(obj: Any) -> List[Dict[str, Any]]:
            """
            Normalise a GeoJSON document into a list of Feature objects.

            Supported containers:
            - ``FeatureCollection``: returns its ``features`` list.
            - ``Feature``: wraps it in a list.

            Args:
                obj: Parsed JSON root object.

            Returns:
                list of dict: List of GeoJSON Feature dicts.
            """
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Invalid GeoJSON file '{path}': expected a JSON object/dict."
                )

            obj_type = obj.get("type")
            if obj_type == "FeatureCollection":
                feats = obj.get("features", [])
                if not isinstance(feats, list):
                    raise ValueError(
                        f"Invalid GeoJSON file '{path}': 'features' must be a list."
                    )
                return feats

            if obj_type == "Feature":
                return [obj]

            raise ValueError(
                f"Invalid GeoJSON file '{path}': expected 'FeatureCollection' or 'Feature', "
                f"got {obj_type!r}."
            )

        def _parse_point(coord: Any, *, feature_index: int) -> List[int]:
            """
            Parse a GeoJSON position into an integer pixel coordinate.

            GeoJSON positions are typically ``[x, y]`` and may optionally include additional
            elements (e.g., z). This parser uses the first two elements.

            Args:
                coord:
                    GeoJSON position (list/tuple with length >= 2).
                feature_index:
                    Feature index for error reporting.

            Returns:
                list[int]: Two-element integer coordinate ``[x, y]``.
            """
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                raise ValueError(
                    f"Invalid coordinate in feature index {feature_index} in '{path}': "
                    f"expected [x, y], got {coord!r}."
                )
            return [self._to_pixel(coord[0]), self._to_pixel(coord[1])]

        def _parse_linestring(coords: Any, *, feature_index: int) -> List[List[int]]:
            """
            Parse GeoJSON LineString coordinates into a vertex sequence.

            Args:
                coords:
                    List of GeoJSON positions.
                feature_index:
                    Feature index for error reporting.

            Returns:
                list[list[int]]: Vertex sequence ``[[x, y], ...]``.
            """
            if not isinstance(coords, list):
                raise ValueError(
                    f"Invalid coordinates for LineString in feature index {feature_index} "
                    f"in '{path}': expected a list, got {type(coords).__name__}."
                )
            return [_parse_point(c, feature_index=feature_index) for c in coords]

        def _parse_polygon_exterior(
            coords: Any, *, feature_index: int
        ) -> List[List[int]]:
            """
            Parse GeoJSON Polygon coordinates and return the exterior ring only.

            GeoJSON Polygon coordinates are ``[ring0, ring1, ...]`` where ring0 is the
            exterior ring and ring1..n are interior rings (holes). This loader ignores
            holes and returns ring0 only.

            Args:
                coords:
                    Polygon coordinates (non-empty list of rings).
                feature_index:
                    Feature index for error reporting.

            Returns:
                list[list[int]]: Exterior ring as a vertex sequence ``[[x, y], ...]``.
            """
            if not isinstance(coords, list) or not coords:
                raise ValueError(
                    f"Invalid coordinates for Polygon in feature index {feature_index} "
                    f"in '{path}':expected a non-empty list of rings."
                )
            ring0 = coords[0]
            if not isinstance(ring0, list):
                raise ValueError(
                    f"Invalid exterior ring for Polygon in feature index {feature_index} "
                    f"in '{path}':expected a list, got {type(ring0).__name__}."
                )
            return [_parse_point(c, feature_index=feature_index) for c in ring0]

        def _extract_sequences(
            gtype: str,
            coords: Any,
            *,
            feature_index: int,
        ) -> List[List[List[int]]]:
            """
            Convert a supported geometry into one or more vertex sequences.

            Some geometry types map to multiple sequences (e.g., MultiPolygon), so the
            return type is always a list of sequences.

            Supported:
            - Polygon: [exterior_ring]
            - MultiPolygon: [exterior_ring_0, exterior_ring_1, ...]
            - LineString: [line]
            - MultiLineString: [line_0, line_1, ...]

            Explicitly ignored (returns empty list):
            - Point, MultiPoint, GeometryCollection

            Args:
                gtype:
                    Geometry type string from ``geometry["type"]``.
                coords:
                    Geometry coordinate payload from ``geometry["coordinates"]``.
                feature_index:
                    Feature index for error reporting.

            Returns:
                list[list[list[int]]]: List of vertex sequences.
            """
            if gtype == "Polygon":
                return [_parse_polygon_exterior(coords, feature_index=feature_index)]

            if gtype == "MultiPolygon":
                if not isinstance(coords, list):
                    raise ValueError(
                        f"Invalid coordinates for MultiPolygon in feature index {feature_index} "
                        f"in '{path}':expected a list, got {type(coords).__name__}."
                    )
                return [
                    _parse_polygon_exterior(poly, feature_index=feature_index)
                    for poly in coords
                ]

            if gtype == "LineString":
                return [_parse_linestring(coords, feature_index=feature_index)]

            if gtype == "MultiLineString":
                if not isinstance(coords, list):
                    raise ValueError(
                        f"Invalid coordinates for MultiLineString in feature index {feature_index} "
                        f"in '{path}': expected a list, got {type(coords).__name__}."
                    )
                return [
                    _parse_linestring(line, feature_index=feature_index)
                    for line in coords
                ]

            if gtype in {"Point", "MultiPoint", "GeometryCollection"}:
                return []

            raise ValueError(
                f"Unsupported geometry type '{gtype}' in feature index {feature_index} in '{path}'."
            )

        features = _as_feature_list(data)
        annotations: Dict[str, List[List[List[int]]]] = {}

        for idx, feature in enumerate(features):
            if not isinstance(feature, dict):
                raise ValueError(f"Invalid feature at index {idx} in '{path}'.")

            if feature.get("type") != "Feature":
                raise ValueError(
                    f"Invalid GeoJSON feature at index {idx} in '{path}': "
                    " expected type='Feature'."
                )

            properties = feature.get("properties", {}) or {}
            if not isinstance(properties, dict):
                raise ValueError(
                    f"Invalid properties in feature index {idx} in '{path}': "
                    "expected an object/dict."
                )

            label = properties.get("label", "undefined")
            label = str(label).strip() if label is not None else "undefined"

            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                raise ValueError(
                    f"Missing/invalid geometry in feature index {idx} in '{path}'."
                )

            gtype = geometry.get("type")
            if gtype is None:
                raise ValueError(
                    f"Missing geometry.type in feature index {idx} in '{path}'."
                )

            coords = geometry.get("coordinates")
            if str(gtype) != "GeometryCollection" and coords is None:
                raise ValueError(
                    f"Missing geometry.coordinates in feature index {idx} in '{path}'."
                )

            sequences = _extract_sequences(str(gtype), coords, feature_index=idx)
            for seq in sequences:
                annotations.setdefault(label, []).append(seq)

        return annotations

    def _csv(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse CSV annotation files with a specific structure.

        Args:
            path (str):
                Path to the CSV file. CSV must contain following fields:
                - label: str
                - polygon_id: str or int
                - vertex_id: integer-like
                - x: float/int
                - y: float/int

        Returns:
            dict: Annotations dictionary.
        """
        df = pd.read_csv(path)

        required = {"label", "polygon_id", "vertex_id", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV annotation file must contain columns {sorted(required)}. "
                f"Missing: {sorted(missing)}. Found: {list(df.columns)}"
            )

        vertex_df = df[["label", "polygon_id", "vertex_id", "x", "y"]].copy()
        vertex_df["label"] = vertex_df["label"].fillna("undefined").astype(str)
        vertex_df["x"] = pd.to_numeric(vertex_df["x"], errors="coerce")
        vertex_df["y"] = pd.to_numeric(vertex_df["y"], errors="coerce")
        vertex_df = vertex_df.dropna(subset=["x", "y"])

        if vertex_df["polygon_id"].isna().any():
            bad_n = int(vertex_df["polygon_id"].isna().sum())
            raise ValueError(
                f"CSV contains {bad_n} rows with missing polygon_id. "
                "polygon_id is mandatory."
            )

        vertex_df["polygon_id"] = vertex_df["polygon_id"].map(self._clean_id)

        vertex_df["vertex_id"] = pd.to_numeric(vertex_df["vertex_id"], errors="coerce")
        if vertex_df["vertex_id"].isna().any():
            bad_n = int(vertex_df["vertex_id"].isna().sum())
            raise ValueError(
                f"CSV contains {bad_n} rows with non-numeric or missing vertex_id. "
                "vertex_id must be integer-like (0, 1, 2, ...)."
            )

        frac = (vertex_df["vertex_id"] % 1).abs()
        if (frac > 1e-9).any():
            bad_verts = vertex_df.loc[frac > 1e-9, "vertex_id"].head(5).tolist()
            raise ValueError(
                "vertex_id values must be integers. Found non-integer-like values: "
                f"{bad_verts}"
            )

        vertex_df["vertex_id"] = vertex_df["vertex_id"].astype(np.int64)

        labels = list(vertex_df["label"].drop_duplicates())
        annotations: Dict[str, List[List[List[int]]]] = {lbl: [] for lbl in labels}

        for (lbl, pid), g in vertex_df.groupby(["label", "polygon_id"], sort=False):
            if g["vertex_id"].duplicated().any():
                dup = g.loc[g["vertex_id"].duplicated(), "vertex_id"].iloc[0]
                raise ValueError(
                    f"Duplicate vertex_id={dup} found within label='{lbl}', "
                    f"polygon_id='{pid}'. vertex_id must be unique per polygon."
                )

            g = g.sort_values("vertex_id", kind="mergesort")
            points = [
                [self._to_pixel(x), self._to_pixel(y)]
                for x, y in zip(g["x"].to_numpy(), g["y"].to_numpy())
            ]
            if len(points) >= 3:
                annotations[lbl].append(points)

        return annotations


class InvalidRoundingPolicyError(ValueError):
    """
    Raised when an invalid rounding policy is requested.

    This is used by Slide._round_dim(...) and any API that accepts the
    `rounding` argument (e.g., Slide.generate_region).
    """

    def __init__(
        self, rounding: str, *, allowed: tuple[str, ...] = ("round", "floor", "ceil")
    ):
        self.rounding = rounding
        self.allowed = allowed
        super().__init__(f"Invalid rounding policy: {rounding!r}. Allowed: {allowed}.")


class InvalidResizeBorderOperatorError(ValueError):
    def __init__(
        self,
        operator: str,
        *,
        allowed: tuple[str, ...] = ALLOWED_RESIZE_OPERATORS,
    ) -> None:
        self.operator = operator
        self.allowed = allowed
        super().__init__(f"Invalid operator {operator!r}. Allowed: {allowed}.")


class InvalidResizeBorderFactorError(ValueError):
    def __init__(self, factor: int):
        self.factor = factor
        super().__init__(f"factor must be positive, got {factor}.")
