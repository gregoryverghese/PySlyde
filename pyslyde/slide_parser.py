"""
slide_parser.py: Whole-slide image (WSI) parser and stitching utilities for PySlyde."""

import os
import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from collections.abc import Collection

import cv2
import numpy as np
import pandas as pd
from openslide import OpenSlide

from pyslyde.encoders.feature_extractor import FeatureGenerator
from pyslyde.io.disk_io import DiskWrite
from pyslyde.io.lmdb_io import LMDBWrite
from pyslyde.normalization import StainNormalizer
from pyslyde.slide import Slide
from pyslyde.util.utilities import (
    coord_to_name,
    extract_tile_from_slide,
    name_to_coord,
    round_dim,
)
from pyslyde.masks.base import BaseMask


class WSIParser:
    """
    Parse a whole-slide image into tiles, mask regions, and tile-level features.

    This class provides utilities to:
        - generate tile origins over a specified border (``tiler``)
        - extract image tiles (``extract_tile``, ``extract_tiles``)
        - extract mask regions corresponding to tiles (``extract_mask``, ``extract_masks``)
        - filter tiles using masks or custom functions
        - save tiles or features to disk / LMDB / RocksDB

    Border convention
    -----------------
    ``border`` is interpreted as a level-0 coverage border in the form:

        [(x_min, x_max), (y_min, y_max)]

    where:
        - x_min / y_min are inclusive starts
        - x_max / y_max are exclusive ends

    Coordinate convention
    ---------------------
    Tile origins are stored as ``(x, y)`` coordinates in level-0 slide space.
    All internally generated tile coordinates (``self._tiles``) and all
    operations involving ``BaseMask``implementations use level-0 coordinates.

    Mask handling
    -------------
    Mask construction is intentionally decoupled from ``WSIParser``.

    Methods requiring a mask accept any implementation of ``BaseMask``.
    The parser interacts with masks exclusively through the ``BaseMask``
    interface, allowing masks to be stored at arbitrary resolutions while
    keeping all parser operations in level-0 slide coordinates.    

    Processing modes
    ----------------
    This class supports two mutually exclusive extraction modes:

    1. Level-based mode
       - user specifies ``level``
       - tiles are read directly from that slide pyramid level
       - behaviour is backward compatible with the legacy implementation

    2. Target-MPP mode
       - user specifies ``target_mpp``
       - the parser computes the required level-0 downsample relative to the
         slide base MPP
       - it then selects the closest finer-or-equal pyramid level to avoid
         unnecessary upsampling
       - the read tile is resized to the requested output dimensions if needed

    Tile size and footprint
    -----------------------
    ``tile_dim`` defines the output tile size in pixels.

    The physical / level-0 footprint of each tile depends on the active mode:
        - in level-based mode, footprint is determined by the chosen pyramid level
        - in target-MPP mode, footprint is determined by the requested target MPP

    Thus, output tile dimensions remain fixed, while the covered tissue area is
    controlled by the active read mode.

    Notes
    -----
    - ``target_mpp`` requires valid slide MPP metadata.
    - Requests for a target MPP finer than the slide base MPP are rejected.
    - Tile coordinates remain in level-0 space in both modes, even when tiles
      are read from a lower-resolution pyramid level.
    - Coordinate transformations required by a mask are handled by the
      supplied ``BaseMask`` implementation rather than by ``WSIParser``.      
    """

    def __init__(
        self,
        slide: Slide,
        tile_dim: int,
        border: List[Tuple[int, int]],
        level: Optional[int] = 0,
        target_mpp: Optional[float] = None,
        stain_normalizer: Optional[StainNormalizer] = None,
    ) -> None:
        """
        Initialize the WSI parser.

        Args:
            slide:
                OpenSlide object representing the whole slide image.

            tile_dim:
                Output tile dimension in pixels (square tiles).

            border:
                Level-0 coverage border in the form
                [(x_min, x_max), (y_min, y_max)].

            level:
                Slide pyramid (resolution) level to extract tile in level-based mode.
            
            stain_normalizer:
                Optional stain normalizer object  (must subclass StainNormalizer).
            
            target_mpp:
                Optional target microns-per-pixel. If provided, tiles are
                extracted at a consistent physical resolution across slides.
                If None, the parser uses the pyramid level-based behavior.
        """
        super().__init__()
        self.slide = slide
        self.level = level
        self.target_mpp = target_mpp
        self.tile_dims = (int(tile_dim), int(tile_dim))
        self.border = border

        self._x_min = int(self.border[0][0])
        self._x_max = int(self.border[0][1])
        self._y_min = int(self.border[1][0])
        self._y_max = int(self.border[1][1])

        if stain_normalizer is not None and not isinstance(
            stain_normalizer, StainNormalizer
        ):
            raise TypeError(
                f"stain_normalizer must be a subclass of StainNormalizer, got {type(stain_normalizer)}"
            )
        self.stain_normalizer = stain_normalizer

        self._base_mpp = self._get_base_mpp()
        self._validate_mode_args()
        self._configure_read_mode()

        self._tiles: List[Tuple[int, int]] = []
        self._features: List[np.ndarray] = []

    @property
    def number(self) -> int:
        """Return the number of tiles."""
        return len(self._tiles)

    @property
    def tiles(self) -> List[Tuple[int, int]]:
        """Return list of tuples with (x, y) coordinates."""
        return self._tiles

    @tiles.setter
    def tiles(self, value: List[Tuple[int, int]]) -> None:
        """Set tiles with list of tuples (x, y) int coordinates."""
        self._tiles = value

    @property
    def features(self) -> List[np.ndarray]:
        """Return list of numpy arrays."""
        return self._features

    @property
    def mode(self) -> str:
        """Return the active tile read mode."""
        return self._mode

    @property
    def config(self) -> Dict[str, Any]:
        """
        Return a dictionary describing the parser configuration and the
        effective tile extraction parameters.

        Fields
        ------
        name:
            Name of the slide (typically the slide filename).

        mode:
            Active extraction mode. One of:
                - "level": tiles are extracted using the user-specified pyramid level
                - "target_mpp": tiles are extracted to match a target physical resolution

        target_level:
            User-specified pyramid level (``level`` argument).
            Ignored when ``mode="target_mpp"``.

        effective_level:
            Actual pyramid level used for reading tiles from the slide.
            - In level mode: equal to ``target_level``
            - In target-MPP mode: automatically selected to best match the target scale

        target_mpp:
            Requested microns-per-pixel (µm/pixel).
            Only relevant when ``mode="target_mpp"``.

        base_mpp:
            Microns-per-pixel (µm/pixel) at level 0 of the slide, derived from slide metadata.
            May be None if unavailable.

        effective_mpp:
            Effective microns-per-pixel (µm/pixel) of the extracted tiles.
            - In level mode: computed as base_mpp × level downsample (if base_mpp available)
            - In target-MPP mode: equal to ``target_mpp``

        residual_scale:
            Ratio between the downsample of the selected read level and the exact
            downsample required by ``target_mpp``:

                residual_scale = read_downsample / target_downsample

            Interpretation:
                - 1.0 : exact match between requested and available resolution
                - < 1.0 : tiles are read from a finer level and downsampled
                - > 1.0 : tiles are read from a coarser level and upsampled

            In level mode, this is always 1.0.

        tile_size:
            Output tile dimensions as (width, height) in pixels.

        border:
            Level-0 coverage border used for tiling, in the form:
                [(x_min, x_max), (y_min, y_max)]

        number:
            Number of tile origins currently stored in the parser.
        """
        return {
            "name": self.slide.name,
            "mode": self.mode,
            "target_level": self.level,
            "effective_level": self._read_level,
            "target_mpp": self.target_mpp,
            "base_mpp": self._base_mpp,
            "effective_mpp": self._effective_mpp,
            "residual_scale": self._residual_scale,
            "tile_size": self.tile_dims,
            "border": self.border,
            "number": len(self._tiles),
        }

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return str(self.config)

    def _validate_mode_args(self) -> None:
        if self.target_mpp is not None:
            if self.level is not None:
                warnings.warn(
                    "Both `level` and `target_mpp` were provided. "
                    "`target_mpp` will be used and `level` ignored.",
                    UserWarning,
                )

            if not isinstance(self.target_mpp, (int, float)):
                raise TypeError(
                    f"target_mpp must be a number, got {type(self.target_mpp)}"
                )

            if self.target_mpp <= 0:
                raise ValueError(f"target_mpp must be positive, got {self.target_mpp}")

            if self._base_mpp is None:
                raise ValueError(
                    "target_mpp was provided, but slide MPP metadata is unavailable. "
                    "Cannot harmonize tiles by physical resolution."
                )
            if self.target_mpp < self._base_mpp:
                raise ValueError(
                    f"Requested target_mpp ({self.target_mpp}) is finer than the slide "
                    f"base MPP ({self._base_mpp}). This slide cannot provide that level "
                    "of detail."
                )

        else:
            if self.level is None:
                raise ValueError("level must be provided when target_mpp is None.")

            if not isinstance(self.level, int):
                raise TypeError(f"level must be an integer, got {type(self.level)}")

            if self.level < 0 or self.level >= len(self.slide.level_downsamples):
                raise KeyError(
                    f"level must be in range 0 - {len(self.slide.level_downsamples) - 1}"
                )

    def _get_base_mpp(self) -> Optional[float]:
        """
        Get the level-0 microns-per-pixel for the slide.

        Returns:
            Base MPP if available, otherwise None.

        Notes:
            Prefers averaging MPP-X and MPP-Y when both are available.
        """
        props = getattr(self.slide, "properties", {}) or {}

        mpp_x = props.get("openslide.mpp-x")
        mpp_y = props.get("openslide.mpp-y")

        vals: List[float] = []
        for v in (mpp_x, mpp_y):
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass

        if vals:
            return float(sum(vals) / len(vals))

        return None

    def _choose_read_level(self, target_downsample: float) -> int:
        """
        Select the most appropriate pyramid level for reading tiles for a given
        target level-0 downsample factor.

        This method prioritizes preserving image detail by selecting a pyramid
        level whose downsample is less than or equal to `target_downsample`
        (i.e., a finer or equal resolution than the requested target). Among
        such levels, it chooses the one with the largest downsample, which is
        the closest finer-or-equal match to the target.

        Args:
            target_downsample: 
                Desired downsample factor relative to level 0.

        Returns:
            Index of the selected pyramid level.
        """
        downsamples = [float(d) for d in self.slide.level_downsamples]

        candidates = [
            (i, d) for i, d in enumerate(downsamples) if d <= target_downsample
        ]

        if candidates:
            return max(candidates, key=lambda x: x[1])[0]

        return 0

    def _configure_read_mode(self) -> None:
        """
        Configure how tiles are sampled.

        In level-based mode:
            - use the user-specified self.level directly

        In target-MPP mode:
            - determine the required level-0 downsample from target MPP
            - select the closest finer-or-equal pyramid level
            - configure the level-0 tile footprint and read level
            - compute residual scale
              Ratio between the chosen read level downsample and the exact target
              downsample implied by target_mpp.

              Interpretation:
              residual_scale == 1.0 : exact match
              residual_scale < 1.0  : read from a finer level, then downsample
              residual_scale > 1.0  : read from a coarser level, then upsample

              With the current level-selection strategy and validated inputs,
              target-MPP mode should typically yield residual_scale <= 1.0.
        """
        if self.target_mpp is None:
            self._mode = "level"
            self._read_level = self.level
            self._target_downsample = float(self.slide.level_downsamples[self.level])
            self._read_downsample = float(
                self.slide.level_downsamples[self._read_level]
            )
            self._residual_scale = 1.0
            self._effective_mpp = (
                None
                if self._base_mpp is None
                else self._base_mpp * self._target_downsample
            )
        else:
            self._mode = "target_mpp"
            self._target_downsample = float(self.target_mpp) / float(self._base_mpp)
            self._read_level = self._choose_read_level(self._target_downsample)
            self._read_downsample = float(
                self.slide.level_downsamples[self._read_level]
            )
            self._residual_scale = self._read_downsample / self._target_downsample
            self._effective_mpp = float(self.target_mpp)

        self._x_dim = max(
            round_dim(self.tile_dims[0] * self._target_downsample, "round"), 1
        )
        self._y_dim = max(
            round_dim(self.tile_dims[1] * self._target_downsample, "round"), 1
        )

    def _read_tile_by_mode(self, x: int, y: int) -> np.ndarray:
        """
        Read a tile according to the active sampling mode.

        In level mode:
            - behaves like legacy code

        In target_mpp mode:
            - reads at the best pyramid level
            - uses read dimensions corresponding to the exact level-0 footprint
            - resizes to the requested tile_dims if needed
        """
        if self._mode == "level":
            return extract_tile_from_slide(
                slide=self.slide,
                x=x,
                y=y,
                level=self._read_level,
                tile_dims=self.tile_dims,
            )

        read_w = max(round_dim(self._x_dim / self._read_downsample, "round"), 1)
        read_h = max(round_dim(self._y_dim / self._read_downsample, "round"), 1)

        tile = extract_tile_from_slide(
            slide=self.slide,
            x=x,
            y=y,
            level=self._read_level,
            tile_dims=(read_w, read_h),
        )

        if tile.shape[1] != self.tile_dims[0] or tile.shape[0] != self.tile_dims[1]:
            tile = cv2.resize(
                tile,
                self.tile_dims,
                interpolation=cv2.INTER_LINEAR,
            )

        return tile

    def _remove_edge_case(self, x: int, y: int) -> bool:
        """
        Remove edge cases based on dimensions of patch.

        Args:
            x: 
                Level-0 x-coordinate of the tile origin.

            y: 
                Level-0 y-coordinate of the tile origin.

        Returns:
            Whether to remove patch or not.
        """
        remove = False
        if x + self._x_dim > self._x_max:
            remove = True
        if y + self._y_dim > self._y_max:
            remove = True
        return remove
    
    def _normalize_labels(
        self,
        labels: Union[int, Collection[int]],
    ) -> set[int]:
        """
        Normalize one or more labels into a set.

        Args:
            labels:
                Label or collection of labels.

        Returns:
            Normalized set of labels.

        Raises:
            ValueError:
                If no labels are supplied.
        """

        if isinstance(labels, int):
            labels = {labels}
        else:
            labels = set(labels)

        if not labels:
            raise ValueError(
                "At least one label must be specified."
            )

        return labels

    def _retain_labels(
        self,
        label_mask: np.ndarray,
        labels: Union[int, Collection[int]],
    ) -> np.ndarray:
        """
        Retain only the specified labels within a label mask.

        Pixels belonging to all other labels are reassigned to the
        background label (0).

        Args:
            label_mask:
                Label mask.

            labels:
                Label or collection of labels to retain.

        Returns:
            Label mask in which pixels belonging to the specified labels
            retain their original label values, while all remaining
            pixels are reassigned to the background label (0).
        """

        labels = self._normalize_labels(labels)

        return np.where(
            np.isin(label_mask, list(labels)),
            label_mask,
            0,
        )

    def tiler(
            self, 
            stride: Optional[int] = None, 
            edge_cases: bool = False
    ) -> int:
        """
        Generate tile coordinates based on border, level, and stride.

        Args:
            stride: 
                Step size for tiling.

            edge_cases: 
                Whether to handle edge cases.

        Returns:
            Number of patches generated.
        """
        stride = self.tile_dims[0] if stride is None else stride
        stride_l0 = max(round_dim(stride * self._target_downsample, "round"), 1)

        self._tiles = []
        for x in range(self._x_min, self._x_max, stride_l0):
            for y in range(self._y_min, self._y_max, stride_l0):
                if edge_cases and self._remove_edge_case(x, y):
                    continue
                self._tiles.append((x, y))

        return len(self._tiles)

    def extract_features(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        normalize: bool = False,
        mask: Optional[BaseMask] = None,
        labels: Optional[Union[int, Collection[int]]] = None,
        bg_value: int = 255,        
        force_hf_login: bool = False,
    ) -> Generator[Tuple[Tuple[int, int], np.ndarray], None, None]:
        """
        Extract features from tiles using a specified model.

        Args:
            model_name:
                Name of the feature extractor model.

                Depending on `model_name`, model weights are loaded from
                torchvision/timm-style sources or downloaded from Hugging Face
                through `FeatureGenerator`.

                For gated Hugging Face models, the user may need to expose
                `HUGGINGFACE_TOKEN` as an environment variable containing a valid
                Hugging Face access token. Once weights are cached locally,
                subsequent runs on the same machine typically do not require
                re-authentication unless the cache has been cleared.

            model_path:
                Optional path to a user-provided checkpoint.

                Reserved for future development. The current
                `FeatureGenerator` implementation does not yet support loading
                custom checkpoints through this argument.

            normalize:
                Whether to apply stain normalization to each tile before feature
                extraction.

            mask:
                Optional whole-slide mask used to mask the extracted tile.

            labels:
                Optional label or collection of labels to retain before
                applying the mask. If None, all non-zero labels are retained.

            bg_value:
                Pixel value assigned to pixels excluded by the mask, if masking
                is applied. Default is 255 (white).                
                
            force_hf_login:
                Whether to force Hugging Face authentication before loading a model.

        Yields:
            Tuple of tile coordinates and extracted feature vector.
        """
        if model_path is not None:
            raise NotImplementedError(
                "Providing `model_path` is not supported yet by FeatureGenerator."
            )

        encode = FeatureGenerator(
            model_name=model_name, 
            force_hf_login=force_hf_login
        )

        count = 0
        for x, y in self._tiles:
            tile = self.extract_tile(
                x=x, 
                y=y, 
                normalize=normalize,
                mask=mask,
                labels=labels,
                bg_value=bg_value,            
            )
            feature_vec = encode.forward_pass(tile)
            feature_vec = feature_vec.detach().cpu().numpy()
            count += 1
            yield (x, y), feature_vec

    def filter_by_mask(
        self,
        mask: BaseMask,
        labels: Union[int, Collection[int]],
        threshold: float = 0.5,
    ) -> int:
        """
        Filter retained tiles according to the proportion of a label 
        within each tile's corresponding mask region.

        Args:
            mask: 
                Whole-slide mask used for filtering.

            labels: 
                Label or collection of labels to filter tiles by.

            threshold: 
                Minimum proportion of the specified label required 
                for a tile to be retained.

        Returns:
            Number of retained tiles.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        
        labels = self._normalize_labels(labels)

        retained_tiles = []

        for x, y in self._tiles:

            roi = mask.read_region_native(
                x=x,
                y=y,
                width=self._x_dim,
                height=self._y_dim,
            )

            label_mask = np.isin(roi, list(labels))
            proportion = np.count_nonzero(label_mask) / roi.size

            if proportion >= threshold:
                retained_tiles.append((x, y))

        self._tiles = retained_tiles

        return len(self._tiles)

    def filter_by_func(
        self, filter_func: Callable[[np.ndarray], bool], *args, **kwargs
    ) -> None:
        """
        Filter tiles using a filtering function.

        Args:
            filter_func: Python function that takes a tile and returns a boolean.
            *args: Additional arguments for the filter function.
            **kwargs: Additional keyword arguments for the filter function.
        """
        tiles = self._tiles.copy()

        for i, (t, tile) in enumerate(self.extract_tiles()):
            if filter_func(tile, *args, **kwargs):
                tiles.remove(t)

        self._tiles = tiles.copy()

    def sample_tiles(self, n: int, seed: int | None = None) -> None:
        """
        Optional seed allows to always get the same subset of tiles
        """
        n = min(n, len(self._tiles))
        rng = random.Random(seed) if seed is not None else random
        sample_tiles = rng.sample(self._tiles, n)
        self._tiles = sample_tiles

    def extract_mask(
        self,
        x: int,
        y: int,
        mask: BaseMask,
    ) -> np.ndarray:
        """
        Extract the mask corresponding to a tile.

        Args:
            x:
                Level-0 x-coordinate of the tile origin.

            y:
                Level-0 y-coordinate of the tile origin.

            mask:
                Whole-slide mask from which to extract the tile mask.

        Returns:
            Mask corresponding to the requested tile.
        """

        return mask.read_region(
            x=x,
            y=y,
            width=self._x_dim,
            height=self._y_dim,
        )

    def extract_masks(
        self,
        mask: BaseMask,
    ) -> Generator[tuple[tuple[int, int], np.ndarray], None, None]:
        """
        Extract the corresponding mask for each retained tile.

        Args:
            mask:
                Whole-slide mask.

        Yields:
            Tile coordinates together with the corresponding mask.
        """

        for x, y in self._tiles:

            yield (
                (x, y),
                self.extract_mask(
                    x=x,
                    y=y,
                    mask=mask,
                ),
            )

    def extract_tile(
        self,
        x: int,
        y: int,
        normalize: bool = False,
        mask: Optional[BaseMask] = None,
        labels: Optional[Union[int, Collection[int]]] = None,
        bg_value: int = 255,
    ) -> np.ndarray:
        """
        Extract a single tile from the slide.

        The tile is read according to the active parser mode using ``(x, y)`` 
        as the level-0 coordinates of the tile's top-left corner.

        - In level-based mode, tiles are read directly from `self.level`.
        - In target-MPP mode, tiles are read from an automatically selected
          pyramid level and resized to the requested output size if needed.

        Optional masking and stain normalization can then be applied.

        Args:
            x:
                Level-0 x-coordinate of the tile origin.

            y:
                Level-0 y-coordinate of the tile origin.

            normalize:
                If True, apply stain normalization to the extracted tile.

            mask:
                Optional whole-slide mask used to mask the extracted tile.

            labels:
                Optional label or collection of labels to retain before
                applying the mask. If None, all non-zero labels are retained.

            bg_value:
                Pixel value assigned to pixels excluded by the mask, if masking
                is applied. Default is 255 (white).

        Returns:
            RGB tile array of shape `(height, width, 3)`.
        """
        tile = self._read_tile_by_mode(x, y)

        if mask is not None:

            label_mask = self.extract_mask(
                x=x,
                y=y,
                mask=mask,
            )

            if labels is not None:
                label_mask = self._retain_labels(
                    label_mask,
                    labels,
                )

            binary_mask = (label_mask > 0).astype(np.uint8)

            tile = tile.copy()            
            tile[binary_mask == 0] = bg_value

        if normalize:
            if self.stain_normalizer is None:
                raise RuntimeError(
                    "No stain normalizer provided. Pass a fitted StainNormalizer "
                    "to WSIParser or set normalize=False."
                )
            if not self.stain_normalizer.is_fitted:
                raise RuntimeError(
                    "StainNormalizer is not fitted. Run stain_normalizer.fit() "
                    "before calling this function."
                )
            tile = self.stain_normalizer.normalize(tile)

        return tile

    def extract_tiles(
        self,
        normalize: bool = False,
        mask: Optional[BaseMask] = None,
        labels: Optional[Union[int, Collection[int]]] = None,
        bg_value: int = 255,
    ) -> Generator[Tuple[Tuple[int, int], np.ndarray], None, None]:
        """
        Generator that yields tiles for all tiles currently retained by the parser.

        Args:
            normalize:
                If True, apply stain normalization to each tile.

            mask:
                Optional whole-slide mask used to mask the extracted tile.

            labels:
                Optional label or collection of labels to retain before applying
                the mask. If None, all non-zero labels are retained.

            bg_value:
                Pixel value assigned to pixels excluded by the mask, if masking
                is applied. Default is 255 (white).

        Yields:
            A tuple containing the tile coordinate `(x, y)` and the
            corresponding extracted tile array.
        """
        for x, y in self._tiles:
            tile = self.extract_tile(
                x=x,
                y=y,
                normalize=normalize,
                mask=mask,
                labels=labels,
                bg_value=bg_value,
            )

            yield (x, y), tile

    @staticmethod
    def _save_to_disk(
        image: np.ndarray, 
        path: str, 
        x: int, 
        y: int,
    ) -> bool:
        """
        Save tile to disk.

        Args:
            image: 
                Tile image as numpy array.

            path: 
                Path to save the image.

            x: 
                Level-0 x-coordinate of the tile origin used when
                constructing the filename.

            y: 
                Level-0 y-coordinate of the tile origin used when
                constructing the filename.

        Returns:
            Whether the tile was written successfully.
        """
        filename = coord_to_name(x, y)
        image_path = os.path.join(path, filename + ".png")
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        status = cv2.imwrite(image_path, image)
        return status

    def _save_tiles(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        tile_path: str,
        label_dir: bool = False,
        label_csv: bool = False,
    ) -> None:
        """
        Save the extracted tiles to disk.

        Args:
            func:
                Generator function that yields (coordinates, tile) tuples.

            tile_path:
                Base directory where tiles will be saved.

            label_dir:
                If True, saves tiles in subdirectories based on their label.

            label_csv:
                If True, saves tile metadata in a CSV file.
        """
        os.makedirs(tile_path, exist_ok=True)

        metadata = []

        for (x, y), tile in func:
            save_dir = tile_path

            if label_dir:
                save_dir = os.path.join(tile_path, os.path.basename(tile_path))
                os.makedirs(save_dir, exist_ok=True)

            self._save_to_disk(tile, save_dir, x, y)

            if label_csv:
                filename = coord_to_name(x, y) + ".png"
                metadata.append(
                    {
                        "x": x,
                        "y": y,
                        "path": os.path.join(save_dir, filename),
                    }
                )

        if label_csv:
            df = pd.DataFrame(metadata)
            df.to_csv(tile_path + "_metadata.csv", index=False)

    def save_tiles(
        self,
        tile_path: str,        
        normalize: bool = False,
        mask: Optional[BaseMask] = None,
        labels: Optional[Union[int, Collection[int]]] = None,
        bg_value: int = 255,   
        label_dir: bool = False,
        label_csv: bool = False,             
    ) -> None:
        """
        Wrapper to extract tiles and save them to disk.

        Wraps the two-step workflow:

            func = parser.extract_tiles(...)
            parser._save_tiles(func, tile_path, ...)

        Args:
            tile_path:
                Base directory where tiles will be saved.

            normalize:
                Whether to stain-normalize tiles during extraction.
            
            mask:
                Optional whole-slide mask used to mask the extracted tile.
            
            labels:
                Optional label or collection of labels to retain before
                applying the mask. If None, all non-zero labels are retained.

            bg_value:
                Pixel value assigned to pixels excluded by the mask, if masking
                is applied. Default is 255 (white).

            label_dir:
                If True, saves tiles in a (single) subdirectory named after tile_path basename.
            
            label_csv:
                If True, writes a CSV with tile metadata (x, y, path).                
        """

        func = self.extract_tiles(
            normalize=normalize,
            mask=mask,
            labels=labels,
            bg_value=bg_value,
        )

        self._save_tiles(
            func,
            tile_path,
            label_dir=label_dir,
            label_csv=label_csv,
        )

    def to_lmdb(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        db_path: str,
        map_size: int,
        write_frequency: int = 10,
    ) -> None:
        """
        Save to LMDB database.

        Args:
            func: 
                Generator function that yields (coordinates, tile) tuples.

            db_path: 
                Base directory where tiles or features will be saved.

            map_size: 
                Map size for LMDB.

            write_frequency: 
                Controls batch commit of a transaction.
        """
        os.makedirs(db_path, exist_ok=True)
        lmdb_writer = LMDBWrite(db_path, map_size, write_frequency)
        lmdb_writer.write(func)

    def to_rocksdb(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        db_path: str,
        write_frequency: int = 10,
    ) -> None:
        """
        Save to RocksDB database.

        Args:
            func: 
                Generator function that yields (coordinates, tile) tuples.
            
            db_path: 
                Base directory where tiles or features will be saved.
            
            write_frequency: 
                Controls batch commit of a transaction.
        """
        try:
            from pyslyde.io.rocksdb_io import RocksDBWrite
        except ImportError:
            raise ImportError(
                "RocksDB is not installed. Install it with: pip install pyslyde[rocksdb]"
            )

        os.makedirs(db_path, exist_ok=True)
        rocksdb_writer = RocksDBWrite(db_path, write_frequency)
        rocksdb_writer.write(func)

    def feat_to_disk(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        path: str,
        write_frequency: int = 10,
    ) -> None:
        """
        Save features to disk.

        Args:
            func: 
                Generator function that yields (coordinates, feature) tuples.
            
            path: 
                Path to save the features.

            write_frequency: 
                Controls batch commit of a transaction.
        """
        os.makedirs(path, exist_ok=True)
        disk_writer = DiskWrite(path, write_frequency)
        disk_writer.write(func)


class Stitching:
    """
    Reconstruct a 2D canvas from saved patch files.

    Public border convention
    ------------------------
    This class treats `border` as a coverage border in level-0 coordinates:

        [(x_min, x_max), (y_min, y_max)]

    where:
        - x_min / y_min are inclusive starts
        - x_max / y_max are exclusive ends
        - width  = x_max - x_min
        - height = y_max - y_min

    This matches the preferred public convention used elsewhere in the codebase.

    Internal origin-border convention
    ---------------------------------
    For grid reasoning, the class also uses an internal origin-border:

        [(x_min_origin, x_max_origin), (y_min_origin, y_max_origin)]

    where x/y maxima refer to the maximum observed patch origins, not full
    covered extents.

    Patch-source contract
    ---------------------
    Current implementation supports file-based patches in `patch_path`,
    with filenames encoding patch origins via `name_to_coord(...)`.

    Future support may allow an in-memory `patching` provider yielding
    `((x, y), patch)` directly, but that is not implemented yet.

    Placement contract
    ------------------
    Patches are placed using their actual coordinates, not grid indices.
    This keeps reconstruction correct even if tiles are sparse, filtered,
    or non-contiguous.

    Border inference contract
    -------------------------
    If `border` is not provided, the stitched canvas is inferred from the
    observed patches only. Therefore, if outer-edge patches were filtered
    out before stitching, the inferred stitched extent will shrink to the
    remaining observed patch set.

    Overlap policies
    ----------------
    - "overwrite": later patches overwrite earlier ones
    - "max": pixelwise maximum in overlapping regions

    Potential future extensions could include "average" or blending-based
    overlap resolution.

    Resizing
    --------
    If `size` is passed to `stitch()`, the final canvas is resized using:
        - `resize_interpolation` if explicitly provided
        - otherwise:
            - cv2.INTER_NEAREST for `content_type="mask"`
            - cv2.INTER_LINEAR  for `content_type="image"`

    Notes
    -----
    - Missing patches leave zero-valued gaps.
    - `stride` is used for completeness estimation and validation, not
      for primary placement.
    """

    VALID_POLICIES = {"overwrite", "max"}
    VALID_CONTENT_TYPES = {"image", "mask"}

    def __init__(
        self,
        patch_path: Optional[str],
        *,
        slide: Optional[OpenSlide] = None,
        patching: Optional[Any] = None,
        name: Optional[str] = None,
        stride: Optional[int] = None,
        border: Optional[List[Tuple[int, int]]] = None,
        level: int = 0,
        policy: str = "overwrite",
        content_type: str = "image",
        resize_interpolation: Optional[int] = None,
        strict_stride: bool = False,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ) -> None:
        """
        Initialize the Stitching object.

        Args:
            patch_path:
                Directory containing patch files whose filenames encode
                patch-origin coordinates. May be None only for future support
                of in-memory `patching`.
            slide:
                Optional OpenSlide object. Currently used only for metadata such
                as pyramid downsample.
            patching:
                Reserved for future in-memory patch providers that yield
                `((x, y), patch)` directly. Not implemented yet.
            name:
                Optional slide or sample name.
            stride:
                Optional declared patch stride in the coordinate space of patch
                origins. Used for validation and completeness estimation, not
                for placement.
            border:
                Optional level-0 coverage border override in the form:
                    [(x_min, x_max), (y_min, y_max)]
                where x_max/y_max are exclusive ends.
                If omitted, coverage is inferred from observed patches only.
            level:
                Pyramid level metadata.
            policy:
                Overlap policy. One of {"overwrite", "max"}.
            content_type:
                Type of stitched content. One of {"image", "mask"}.
            resize_interpolation:
                Optional OpenCV interpolation flag (e.g., cv2.INTER_LINEAR),
                                used when `size` is passed to `stitch()`.
                This must be one of the cv2.INTER_* constants.
                If None, a default is selected based on `content_type`:
                    - "mask"  -> cv2.INTER_NEAREST
                    - "image" -> cv2.INTER_LINEAR
            strict_stride:
                If True, raise an error when provided `stride` disagrees with
                inferred stride.
            image_extensions:
                Tuple of allowed image file extensions for patch loading.
                Extensions are matched case-insensitively against file suffixes.
                Defaults to (".png", ".jpg", ".jpeg", ".tif", ".tiff").
        """
        self.patch_path = patch_path
        self.slide = slide
        self.patching = patching
        self.name = name
        self._stride = stride
        self.border = border
        self.level = level
        self.policy = policy
        self.content_type = content_type
        self.resize_interpolation = resize_interpolation
        self.strict_stride = strict_stride
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

        if self.policy not in self.VALID_POLICIES:
            raise ValueError(
                f"Invalid policy {self.policy!r}. "
                f"Expected one of {sorted(self.VALID_POLICIES)}."
            )

        if self.content_type not in self.VALID_CONTENT_TYPES:
            raise ValueError(
                f"Invalid content_type {self.content_type!r}. "
                f"Expected one of {sorted(self.VALID_CONTENT_TYPES)}."
            )

        if self.patch_path is None and self.patching is None:
            raise ValueError("Provide at least one of `patch_path` or `patching`.")

    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "name": self.name,
            "level": self.level,
            "stride": self.stride,
            "origin_border": self.origin_border,
            "coverage_border": self.coverage_border,
            "policy": self.policy,
            "content_type": self.content_type,
            "resize_interpolation": self.resize_interpolation,
            "number": len(self._patch_records()),
        }

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return str(self.config)

    @property
    def stride(self) -> Optional[int]:
        """
        Return user-provided or inferred patch stride.

        Returns:
            User-provided stride if available; otherwise inferred stride.
        """
        return self._stride if self._stride is not None else self._infer_stride()

    @property
    def downsample(self) -> float:
        """
        Downsample factor for the current pyramid level.

        Returns:
            float: Downsample factor relative to level-0.

        Raises:
            ValueError: If no slide is attached or level is invalid.
        """
        if self.slide is None:
            raise ValueError("A slide is required to determine level downsample.")

        if self.level < 0 or self.level >= len(self.slide.level_downsamples):
            raise ValueError(
                f"Invalid level {self.level}. "
                f"Valid range is [0, {len(self.slide.level_downsamples) - 1}]."
            )

        return float(self.slide.level_downsamples[self.level])

    def _patches(self) -> List[str]:
        """
        Return image patch filenames from `patch_path`.

        Raises:
            ValueError: If file-based patch loading is unavailable.
            FileNotFoundError: If patch_path does not exist.
            NotADirectoryError: If patch_path is not a directory.
        """
        if self.patch_path is None:
            raise ValueError("`patch_path` is not set for file-based patch loading.")

        if not os.path.exists(self.patch_path):
            raise FileNotFoundError(f"patch_path does not exist: {self.patch_path!r}")

        if not os.path.isdir(self.patch_path):
            raise NotADirectoryError(
                f"patch_path is not a directory: {self.patch_path!r}"
            )

        return [
            f
            for f in os.listdir(self.patch_path)
            if Path(f).suffix.lower() in self.image_extensions
        ]

    def _patch_records(self) -> List[Tuple[Tuple[int, int], str]]:
        """
        Return patch metadata records from the active source.

        Returns:
            List of ((x, y), source_name) records.

        Raises:
            NotImplementedError:
                If `patching` is provided, since in-memory stitching is not yet
                implemented.
        """
        if self.patching is not None:
            raise NotImplementedError(
                "In-memory `patching` sources are not implemented yet. "
                "Current Stitching supports file-based patches only."
            )

        records: List[Tuple[Tuple[int, int], str]] = []
        for fname in self._patches():
            try:
                x, y = name_to_coord(fname)
                records.append(((x, y), fname))
            except ValueError:
                continue
        return records

    def _get_coords(self) -> List[Tuple[int, int]]:
        """Return patch-origin coordinates from patch records."""
        return [coord for coord, _ in self._patch_records()]

    def _read_patch(self, source_name: str) -> np.ndarray:
        """
        Read a patch from disk.

        Args:
            source_name:
                Patch filename.

        Returns:
            Patch array loaded using cv2.IMREAD_UNCHANGED.
        """
        if self.patch_path is None:
            raise ValueError("`patch_path` is not set for file-based patch loading.")

        path = os.path.join(self.patch_path, source_name)
        patch = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if patch is None:
            raise ValueError(f"Failed to read patch image: {path}")
        return patch

    def _get_first_patch_info(self) -> Tuple[int, int, Optional[int], np.dtype]:
        """
        Inspect the first patch to determine shape metadata.

        Returns:
            (patch_h, patch_w, channels, dtype)
            where channels is None for 2D patches.
        """
        records = self._patch_records()
        if not records:
            raise ValueError("No valid patch records found.")

        _, source_name = records[0]
        patch = self._read_patch(source_name)

        if patch.ndim == 2:
            patch_h, patch_w = patch.shape
            channels = None
        else:
            patch_h, patch_w, channels = patch.shape

        return patch_h, patch_w, channels, patch.dtype

    @property
    def origin_border(self) -> List[Tuple[int, int]]:
        """
        Return internal patch-origin border.

        Returns:
            [(x_min_origin, x_max_origin), (y_min_origin, y_max_origin)]

        Notes:
            Maxima refer to maximum patch origins, not full coverage extents.
        """
        coords = self._get_coords()
        if not coords:
            return [(0, 0), (0, 0)]

        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]

        return [
            (min(x_coords), max(x_coords)),
            (min(y_coords), max(y_coords)),
        ]

    @property
    def coverage_border(self) -> List[Tuple[int, int]]:
        """
        Return full stitched coverage border.

        Returns:
            [(x_min, x_max), (y_min, y_max)]
            where x_max/y_max are exclusive ends.

        Notes:
            - If `self.border` is provided, it is treated as authoritative.
            - Otherwise coverage is inferred from observed patches only.
        """
        if self.border is not None:
            return self.border

        origin_border = self.origin_border
        if origin_border == [(0, 0), (0, 0)]:
            return [(0, 0), (0, 0)]

        patch_h, patch_w, _, _ = self._get_first_patch_info()
        x_min, x_max_origin = origin_border[0]
        y_min, y_max_origin = origin_border[1]

        return [
            (x_min, x_max_origin + patch_w),
            (y_min, y_max_origin + patch_h),
        ]

    def _infer_stride(self) -> Optional[int]:
        """
        Infer patch stride from observed coordinate differences.

        Returns:
            Minimum positive difference across sorted unique x and y origins,
            or None if stride cannot be inferred robustly.
        """
        coords = self._get_coords()
        if len(coords) < 2:
            return None

        x_coords = sorted(set(c[0] for c in coords))
        y_coords = sorted(set(c[1] for c in coords))

        x_diffs = [x_coords[i + 1] - x_coords[i] for i in range(len(x_coords) - 1)]
        y_diffs = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]

        diffs = [d for d in (x_diffs + y_diffs) if d > 0]
        return min(diffs) if diffs else None

    def _validate_stride(self) -> None:
        """
        Validate provided and/or inferred stride.

        Raises:
            TypeError:
                If a provided stride is not an integer.
            ValueError:
                If a provided or inferred stride is not positive, or if
                `strict_stride=True` and provided/inferred strides disagree.
        """
        if self._stride is not None:
            if not isinstance(self._stride, int):
                raise TypeError(
                    f"stride must be an integer if provided, got {type(self._stride)}"
                )
            if self._stride <= 0:
                raise ValueError(f"stride must be positive, got {self._stride}")

        inferred = self._infer_stride()
        if inferred is not None and inferred <= 0:
            raise ValueError(f"Inferred stride must be positive, got {inferred}")

        if self._stride is not None and inferred is not None and self.strict_stride:
            if inferred != self._stride:
                raise ValueError(
                    f"Provided stride ({self._stride}) does not match inferred "
                    f"stride ({inferred})."
                )

    def completeness(self) -> float:
        """
        Estimate rectangular grid completeness from observed patch origins.

        Returns:
            actual_patch_count / expected_patch_count

        Notes:
            This is a heuristic for regular grids. It is computed from the
            internal origin-border and stride, not the public coverage border.
        """
        border = self.origin_border
        stride = self.stride

        if not border or not stride:
            return 0.0

        x_min, x_max = border[0]
        y_min, y_max = border[1]

        expected_x = ((x_max - x_min) // stride) + 1
        expected_y = ((y_max - y_min) // stride) + 1
        expected_patches = expected_x * expected_y
        actual_patches = len(self._patch_records())

        return actual_patches / expected_patches if expected_patches > 0 else 0.0

    def _make_canvas(
        self,
        canvas_h: int,
        canvas_w: int,
        channels: Optional[int],
        dtype: np.dtype,
    ) -> np.ndarray:
        """Create an empty canvas of the requested shape."""
        if channels is None:
            return np.zeros((canvas_h, canvas_w), dtype=dtype)
        return np.zeros((canvas_h, canvas_w, channels), dtype=dtype)

    def _paste_overwrite(
        self,
        canvas: np.ndarray,
        patch: np.ndarray,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> None:
        """Apply overwrite stitching policy."""
        canvas[y0:y1, x0:x1] = patch

    def _paste_max(
        self,
        canvas: np.ndarray,
        patch: np.ndarray,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> None:
        """Apply pixelwise maximum stitching policy."""
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], patch)

    def _get_resize_interpolation(self) -> int:
        """
        Resolve interpolation used when resizing stitched output.

        Returns:
            OpenCV interpolation flag.
        """
        if self.resize_interpolation is not None:
            return self.resize_interpolation

        if self.content_type == "mask":
            return cv2.INTER_NEAREST

        return cv2.INTER_LINEAR

    def _resize_output(
        self,
        canvas: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Resize stitched output.

        Args:
            canvas:
                Stitched canvas.
            size:
                Target size as (width, height).

        Returns:
            Resized canvas.
        """
        return cv2.resize(
            canvas,
            size,
            interpolation=self._get_resize_interpolation(),
        )

    def stitch(self, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Stitch patch arrays back into a single canvas.

        Args:
            size:
                Optional final output size as (width, height). If provided,
                the stitched canvas is resized at the end.

        Returns:
            np.ndarray:
                Reconstructed 2D or 3D canvas.

        Raises:
            ValueError:
                If no patches are found, patch sizes are inconsistent, or the
                provided coverage border is invalid relative to observed patches.
            NotImplementedError:
                If `patching` is provided.
        """
        self._validate_stride()

        records = self._patch_records()
        if not records:
            raise ValueError("No valid patch images found for stitching.")

        patch_h, patch_w, channels, dtype = self._get_first_patch_info()
        origin_border = self.origin_border
        coverage_border = self.coverage_border

        x_min_cov, x_max_cov = coverage_border[0]
        y_min_cov, y_max_cov = coverage_border[1]

        canvas_w = x_max_cov - x_min_cov
        canvas_h = y_max_cov - y_min_cov

        if canvas_w <= 0 or canvas_h <= 0:
            raise ValueError(
                f"Invalid coverage border {coverage_border}: "
                "stitched canvas must have positive width and height."
            )

        obs_x_min, obs_x_max_origin = origin_border[0]
        obs_y_min, obs_y_max_origin = origin_border[1]
        obs_x_max_cov = obs_x_max_origin + patch_w
        obs_y_max_cov = obs_y_max_origin + patch_h

        if obs_x_min < x_min_cov or obs_y_min < y_min_cov:
            raise ValueError(
                "Provided coverage border does not include the minimum observed "
                "patch origin."
            )
        if obs_x_max_cov > x_max_cov or obs_y_max_cov > y_max_cov:
            raise ValueError(
                "Provided coverage border does not fully contain the observed "
                "patch coverage."
            )

        canvas = self._make_canvas(canvas_h, canvas_w, channels, dtype)

        for (x, y), source_name in records:
            patch = self._read_patch(source_name)

            if patch.shape[:2] != (patch_h, patch_w):
                raise ValueError(
                    f"Inconsistent patch size for {source_name}: "
                    f"expected {(patch_h, patch_w)}, got {patch.shape[:2]}"
                )

            x0 = x - x_min_cov
            y0 = y - y_min_cov
            x1 = x0 + patch_w
            y1 = y0 + patch_h

            if x0 < 0 or y0 < 0 or x1 > canvas_w or y1 > canvas_h:
                raise ValueError(
                    f"Patch {source_name} at {(x, y)} falls outside the "
                    f"stitched coverage border {coverage_border}."
                )

            if self.policy == "overwrite":
                self._paste_overwrite(canvas, patch, x0, x1, y0, y1)
            elif self.policy == "max":
                self._paste_max(canvas, patch, x0, x1, y0, y1)
            else:
                raise ValueError(f"Unsupported stitching policy: {self.policy}")

        if size is not None:
            canvas = self._resize_output(canvas, size)

        return canvas
