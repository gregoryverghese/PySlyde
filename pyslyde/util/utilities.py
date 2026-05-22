"""Utility functions for PySlyde.

This module contains various utility functions for image processing,
mask operations, and data manipulation in the PySlyde package.
"""

import glob
import os
import random
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openslide import OpenSlide
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters.rank import entropy as skimage_entropy
from skimage.morphology import closing, disk, footprint_rectangle, opening

from pyslyde.exceptions import InvalidRoundingPolicyError


def round_dim(value: float, rounding: str) -> int:
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


def extract_tile_from_slide(
    slide: Union[OpenSlide, object],
    x: int,
    y: int,
    level: int,
    tile_dims: Tuple[int, int],
) -> np.ndarray:
    """
    Extract a single RGB tile directly from a slide-like object.

    Args:
        slide:
            An OpenSlide-compatible object exposing `read_region`.
        x:
            Level-0 x coordinate of the tile origin.
        y:
            Level-0 y coordinate of the tile origin.
        level:
            Pyramid level at which to read the tile.
        tile_dims:
            Output tile size as (width, height).

    Returns:
        np.ndarray:
            RGB tile array of shape (height, width, 3).
    """
    if not hasattr(slide, "read_region"):
        raise TypeError(
            "slide must provide a read_region((x, y), level, tile_dims) method"
        )

    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError(f"x and y must be integers, got {type(x)} and {type(y)}")

    if x < 0 or y < 0:
        raise ValueError(f"x and y must be non-negative, got {(x, y)}")

    if not isinstance(level, int):
        raise TypeError(f"level must be an integer, got {type(level)}")

    if (
        not isinstance(tile_dims, tuple)
        or len(tile_dims) != 2
        or not all(isinstance(v, int) for v in tile_dims)
    ):
        raise ValueError("tile_dims must be a tuple of two integers: (width, height)")

    width, height = tile_dims
    if width <= 0 or height <= 0:
        raise ValueError(f"tile_dims values must be positive, got {(width, height)}")

    tile = slide.read_region((x, y), level, (width, height))
    return np.array(tile.convert("RGB"))


def coord_to_name(x: int, y: int, sep: str = "_") -> str:
    """
    Convert tile coordinates to a canonical tile name.

    Coordinates are serialized in `x_y` order.

    Args:
        x:
            X coordinate.
        y:
            Y coordinate.
        sep:
            Separator between coordinate fields.

    Returns:
        str:
            Tile name in the form ``"{x}{sep}{y}"``.
    """
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError(f"x and y must be integers, got {type(x)} and {type(y)}")
    return f"{x}{sep}{y}"


def name_to_coord(name: str, sep: str = "_") -> Tuple[int, int]:
    """
    Parse tile coordinates from a canonical tile name.

    Accepts either a bare stem such as ``"100_200"`` or a filename such
    as ``"100_200.png"``. Coordinates are interpreted in `x_y` order.

    Args:
        name:
            Tile name or filename.
        sep:
            Separator between coordinate fields.

    Returns:
        Tuple[int, int]:
            Parsed `(x, y)` coordinates.

    Raises:
        ValueError:
            If the name does not match the expected `x_y` format.
    """
    stem = Path(name).stem
    parts = stem.split(sep)

    if len(parts) < 2:
        raise ValueError(
            f"Could not parse tile coordinates from {name!r}. "
            f"Expected format like 'x{sep}y' or 'x{sep}y.ext'."
        )

    try:
        x = int(parts[-2])
        y = int(parts[-1])
    except ValueError as e:
        raise ValueError(
            f"Could not parse integer tile coordinates from {name!r}."
        ) from e

    return x, y


class TissueDetect:
    """
    Tissue detection utility for whole-slide images or image arrays.

    Supported inputs
    ----------------
    - str:
        Path to a whole-slide image readable by OpenSlide.
    - OpenSlide:
        An already opened OpenSlide object.
    - np.ndarray:
        An image array in either grayscale (H, W) or color (H, W, 3) format.

    Backend behavior
    ----------------
    - For OpenSlide inputs, tissue detection is performed on a thumbnail-level
      RGB image and masks/borders are mapped back to full slide coordinates.
    - For ndarray inputs, the provided array is treated as the full-resolution
      image coordinate space.

    Border convention
    -----------------
    Borders are returned as:

        [(x_min, x_max), (y_min, y_max)]

    where x_max / y_max are exclusive ends.
    """

    bilateral_args = [
        {"d": 90, "sigmaColor": 5000, "sigmaSpace": 5000},
        {"d": 90, "sigmaColor": 5000, "sigmaSpace": 5000},
        {"d": 90, "sigmaColor": 10000, "sigmaSpace": 10000},
        {"d": 90, "sigmaColor": 10000, "sigmaSpace": 100},
    ]

    thresh_args = [
        {"thresh": 0, "maxval": 255, "type": cv2.THRESH_TRUNC + cv2.THRESH_OTSU},
        {"thresh": 0, "maxval": 255, "type": cv2.THRESH_OTSU},
    ]

    def __init__(self, slide: Union[str, OpenSlide, np.ndarray]) -> None:
        """
        Initialize the tissue detector.

        Args:
            slide:
                Either:
                - path to a whole-slide image,
                - an OpenSlide object,
                - or a numpy image array.
        """
        if isinstance(slide, str):
            self.slide = OpenSlide(slide)
            self._backend = "openslide"
        elif isinstance(slide, OpenSlide):
            self.slide = slide
            self._backend = "openslide"
        elif isinstance(slide, np.ndarray):
            self.slide = slide
            self._backend = "ndarray"
        else:
            raise TypeError(
                "slide must be a path, an OpenSlide object, or a numpy.ndarray"
            )

        self.tissue_mask: Optional[np.ndarray] = None
        self.contour_mask: Optional[np.ndarray] = None
        self.contours: Optional[List[np.ndarray]] = None
        self._border: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    def _get_full_dimensions(self) -> Tuple[int, int]:
        """
        Return full-resolution dimensions as (width, height).

        Returns:
            Tuple[int, int]:
                Full image dimensions in x/y order.
        """
        if self._backend == "openslide":
            return self.slide.dimensions
        return self.slide.shape[1], self.slide.shape[0]

    def _choose_thumbnail_level(self, downsample: int = 32) -> int:
        """
        Choose a thumbnail level for OpenSlide processing.

        Args:
            downsample:
                Desired approximate downsample factor.

        Returns:
            int:
                Chosen OpenSlide level index.

        Raises:
            RuntimeError:
                If called for a non-OpenSlide backend.
        """
        if self._backend != "openslide":
            raise RuntimeError("_choose_thumbnail_level is only valid for OpenSlide")

        ds = [int(d) for d in self.slide.level_downsamples]
        return ds.index(downsample) if downsample in ds else len(ds) - 1

    def _resize_array_thumbnail(
        self, image: np.ndarray, downsample: int = 32
    ) -> np.ndarray:
        """
        Create a thumbnail-like version of a numpy image.

        Args:
            image:
                Input image (H, W, C)
            downsample:
                Downsampling factor (similar to OpenSlide level downsample)

        Returns:
            np.ndarray:
                Downsampled image
        """
        h, w = image.shape[:2]
        new_w = max(1, w // downsample)
        new_h = max(1, h // downsample)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _coerce_array_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Coerce an ndarray input to uint8 RGB.

        Supported inputs:
        - grayscale: (H, W)
        - RGB-like:  (H, W, 3)

        This method does not try to infer BGR vs RGB provenance. It assumes
        the array should be treated as display/processing RGB once converted.

        Args:
            image:
                Input image array.

        Returns:
            np.ndarray:
                RGB uint8 array of shape (H, W, 3).
        """
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 3:
            pass
        else:
            raise ValueError("ndarray input must have shape (H, W) or (H, W, 3)")

        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0 if image.max() <= 1.0 else 255.0)
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def _get_detection_image(self) -> np.ndarray:
        """
        Return the RGB image used for tissue detection/contour generation.

        Args:
            downsample:
                Approximate downsampling factor for detection.

        Returns:
            np.ndarray:
                RGB image used as the working image for detection.
        """
        if self._backend == "openslide":
            level = self._choose_thumbnail_level(downsample=32)
            image = self.slide.get_thumbnail(self.slide.level_dimensions[level])
            return np.array(image.convert("RGB"))
        elif self._backend == "ndarray":
            MIN_SIZE = 256
            image = self._coerce_array_to_rgb(self.slide)
            h, w = image.shape[:2]
            downsample = min(32, max(1, min(w // MIN_SIZE, h // MIN_SIZE)))
            return self._resize_array_thumbnail(image, downsample=downsample)
        else:
            raise RuntimeError(f"Unsupported backend: {self._backend}")

    def _get_display_image(self) -> np.ndarray:
        """
        Return the image used for contour visualization.

        Returns:
            np.ndarray:
                RGB image for visualization.

        Notes:
            For OpenSlide input this is a thumbnail image.
            For ndarray input this is the array itself coerced to RGB.
        """
        return self._get_detection_image().copy()

    @property
    def tissue_thumbnail(self) -> np.ndarray:
        """
        Get a visualization image with tissue contours and bounding box drawn.

        Returns:
            np.ndarray:
                RGB image with contours and border overlay.

        Notes:
            - For OpenSlide input, this is a thumbnail-level visualization.
            - For ndarray input, this is based on the input array itself.
        """
        image = self._get_display_image()
        contours = self._generate_tissue_contour()

        if not contours:
            return image

        cv2.drawContours(image, contours, -1, (0, 255, 0), 5)
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
        return image

    def mask_image(self, thumb: np.ndarray) -> np.ndarray:
        """
        Apply the current contour mask to an image by whitening background.

        Args:
            thumb:
                RGB image to mask.

        Returns:
            np.ndarray:
                Masked image.

        Raises:
            ValueError:
                If contour_mask has not been generated yet.
        """
        if self.contour_mask is None:
            raise ValueError(
                "contour_mask is not available. Run _generate_tissue_contour() first."
            )

        thumb = thumb.copy()
        thumb[:, :, 0][self.contour_mask == 0] = 255
        thumb[:, :, 1][self.contour_mask == 0] = 255
        thumb[:, :, 2][self.contour_mask == 0] = 255

        return thumb

    def border(
        self, mask: Optional[np.ndarray] = None
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get border coordinates from a tissue/contour mask.

        Args:
            mask:
                Optional mask to use instead of `self.contour_mask`.

        Returns:
            Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
                Border coordinates as:
                    ((x_min, x_max), (y_min, y_max))
                where x_max / y_max are exclusive ends.

        Notes:
            If the provided mask is not already full-resolution, it is resized
            to the full dimensions of the active backend before contour
            extraction so that returned coordinates are in full image space.
        """
        if mask is None and self.contour_mask is None:
            return None

        mask = self.contour_mask if mask is None else mask
        if mask is None:
            return None

        width, height = self._get_full_dimensions()

        if mask.shape[:2] != (height, width):
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask_resized = mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            self._border = ((0, width), (0, height))
            return self._border

        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        self._border = ((x, x + w), (y, y + h))
        return self._border

    def detect_tissue(self) -> np.ndarray:
        """
        Detect tissue regions and return a full-resolution binary tissue mask.

        Returns:
            np.ndarray:
                Tissue mask in full image coordinates with dtype uint8.

        Notes:
            - For OpenSlide input, tissue is detected on a thumbnail image and
              resized back to full slide dimensions.
            - For ndarray input, detection is performed directly on the array
              image and returned in array coordinates.
        """
        image = self._get_detection_image()
        width, height = self._get_full_dimensions()

        gray = rgb2gray(image)
        gray_f = gray.flatten()

        pixels_int = gray_f[np.logical_and(gray_f > 0.1, gray_f < 0.98)]

        if pixels_int.size == 0:
            self.tissue_mask = np.zeros(gray.shape, dtype=np.uint8)
            return cv2.resize(
                self.tissue_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        t = threshold_otsu(pixels_int)
        thresh = np.logical_and(gray_f < t, gray_f > 0.1).reshape(gray.shape)

        mask = opening(
            closing(thresh, footprint=footprint_rectangle((2, 2))),
            footprint=footprint_rectangle((2, 2)),
        )

        self.tissue_mask = mask.astype(np.uint8)
        return cv2.resize(
            self.tissue_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    def _generate_tissue_contour(self) -> List[np.ndarray]:
        """
        Generate tissue contours from the detection image.

        Returns:
            List[np.ndarray]:
                List of contours found in the working image.

        Notes:
            The generated `contour_mask` is in the coordinate space of the
            working detection image:
            - thumbnail space for OpenSlide input
            - array space for ndarray input

            Use `border()` to obtain full-image coordinates.
        """
        slide = self._get_detection_image()

        img_hsv = cv2.cvtColor(slide, cv2.COLOR_RGB2HSV)
        lower_red = np.array([120, 0, 0])
        upper_red = np.array([180, 255, 255])
        mask = cv2.inRange(img_hsv, lower_red, upper_red)

        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        m = cv2.bitwise_and(slide, slide, mask=mask)
        im_fill = np.where(m == 0, 233, m)

        gray = cv2.cvtColor(im_fill.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        for b in TissueDetect.bilateral_args:
            gray = cv2.bilateralFilter(np.bitwise_not(gray), **b)
        blur = 255 - gray

        for t in TissueDetect.thresh_args:
            _, blur = cv2.threshold(blur, **t)

        self.contour_mask = blur.astype(np.uint8)
        contours, _ = cv2.findContours(
            self.contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        self.contours = contours
        return self.contours


def mask2rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert a mask to RGB representation.

    Args:
        mask: Input mask as numpy array.

    Returns:
        np.ndarray: RGB mask with colors assigned to each class.
    """
    n_classes = len(np.unique(mask))
    colors = sns.color_palette("hls", n_classes)
    rgb_mask = np.zeros(mask.shape + (3,))
    for c in range(1, n_classes + 1):
        t = mask == c
        rgb_mask[:, :, 0][t] = colors[c - 1][0]
        rgb_mask[:, :, 1][t] = colors[c - 1][1]
        rgb_mask[:, :, 2][t] = colors[c - 1][2]
    return rgb_mask


def draw_boundary(
    annotations: Dict[str, List[List[List[int]]]], offset: int = 100
) -> List[Tuple[int, int]]:
    """
    Draw boundary around annotations.

    Args:
        annotations: Dictionary of annotations.
        offset: Offset from the boundary.

    Returns:
        List of boundary coordinates.
    """
    annotations = list(chain(*[annotations[f] for f in annotations]))
    coords = list(chain(*annotations))
    boundaries = list(
        map(lambda x: (min(x) - offset, max(x) + offset), list(zip(*coords)))
    )
    return boundaries


def oneHotToMask(onehot: np.ndarray, background: str | None = None) -> np.ndarray:
    """
    Convert one-hot encoded mask to RGB mask.

    Args:
        onehot: One-hot encoded mask.

    Returns:
        np.ndarray: RGB mask.
    """
    n_classes = onehot.shape[-1]
    idx = np.argmax(onehot, axis=-1)
    colors = sns.color_palette("hls", n_classes)

    multimask = np.take(colors, idx, axis=0)

    if background is not None:
        if background.lower() == "black":
            multimask[idx == 0] = (0, 0, 0)
        elif background.lower() == "white":
            multimask[idx == 0] = (255, 255, 255)
        else:
            raise ValueError("background must be 'black', 'white', or None")

    return multimask


def sample_patches(patch: Any, n: int, replacement: bool = False) -> Any:
    """
    Sample patches from a patch object.

    Args:
        patch: Patch object to sample from.
        n: Number of patches to sample.
        replacement: Whether to sample with replacement.

    Returns:
        New patch object with sampled patches.
    """
    if replacement:
        patches = random.choices(patch._patches, k=n)
    else:
        patches = random.sample(patch._patches, n)

    new_patch = type(patch)(
        patch.slide, patch.size, patch.mag_level, patch.border, patch.step
    )

    new_patch.patches = patches
    return new_patch


def get_x_y_from_0(slide, point_0, level, integer=True):
    """
    Given a point point_0 = (x0, y0) at level 0, this function will return
    the coordinates associated to the level 'level' of this point point_l = (x_l, y_l).
    Inverse function of get_x_y

    Args:
        slide : Openslide object from which we extract.
        point_0 : A tuple, or tuple like object of size 2 with integers.
        level : Integer, level to convert to.
        integer : Boolean, by default True. Wether or not to round
                  the output.

    Returns:
        A tuple corresponding to the converted coordinates, point_l.
    """
    x_0, y_0 = point_0
    size_x_l = slide.level_dimensions[level][0]
    size_y_l = slide.level_dimensions[level][1]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])

    x_l = x_0 * size_x_l / size_x_0
    y_l = y_0 * size_y_l / size_y_0
    if integer:
        point_l = (round(x_l), round(y_l))
    else:
        point_l = (x_l, y_l)
    return point_l


def get_size(slide, size_from, level_from, level_to, rounding="round"):
    """
    Given a size (size_from) at a certain level (level_from), this function will return
    a new size (size_to) but at a different level (level_to).
    Args:
        slide : Openslide object from which we extract.
        size_from : A tuple, or tuple like object of size 2 with integers.
        level_from : Integer, initial level.
        level_to : Integer, final level.
        integer : One of {"round", "floor", "ceil", None}.
                  If None, return floating-point dimensions unchanged.
        Returns:
            A tuple, or tuple like object of size 2 with integers corresponding
            to the new size at level level_to. Or size_to.
    """
    size_x, size_y = size_from
    downsamples = slide.level_downsamples
    scale_factor = float(downsamples[level_from]) / downsamples[level_to]
    size_x_scaled = float(size_x) * scale_factor
    size_y_scaled = float(size_y) * scale_factor

    if rounding is None:
        return size_x_scaled, size_y_scaled

    return round_dim(size_x_scaled, rounding), round_dim(size_y_scaled, rounding)


def visualise_wsi_tiling(
    wsi,
    tiler,
    save_path,
    level=3,
    plot_args={"color": "red", "size": (10, 10), "title": ""},
):
    """
    Visualise tile layout on a WSI thumbnail.

    Args:
        wsi:
            OpenSlide object.
        tiler:
            Object containing tile coordinates and dimensions.
        save_path:
            Path to save the output image.
        level:
            OpenSlide level used for thumbnail generation.
        plot_args:
            Dictionary with keys:
                - color: rectangle color
                - size: figure size
                - title: plot title

    Notes:
        - Tiles are drawn as rectangles on the thumbnail.
        - Coordinates are scaled according to level downsampling.
    """
    mpl.use("Agg")
    wsi_thumb = wsi.get_thumbnail(wsi.level_dimensions[level])
    wsi_thumb = np.array(wsi_thumb.convert("RGB"))
    plt.figure(figsize=plot_args["size"])
    plt.imshow(wsi_thumb)
    print("_x_dims", tiler._x_dim)
    ax = plt.gca()

    for t_xy in tiler.tiles:
        x = int(t_xy[0] / wsi.level_downsamples[level])
        y = int(t_xy[1] / wsi.level_downsamples[level])
        w = int(tiler._x_dim / wsi.level_downsamples[level])
        h = int(tiler._y_dim / wsi.level_downsamples[level])
        patch = patches.Rectangle(
            (y, x), w, h, fill=False, edgecolor=plot_args["color"]
        )
        ax.add_patch(patch)

    ax.set_title(plot_args["title"], size=20)
    print("saving where", save_path)
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


def low_entropy(tile, threshold):
    """
    Check whether a tile has low entropy.

    Args:
        tile:
            Input image (grayscale or RGB).
        threshold:
            Entropy threshold below which the tile is considered low entropy.

    Returns:
        bool: True if average entropy is below the threshold, else False.
    """
    avg_entropy = image_entropy(tile)
    return avg_entropy < threshold


def image_entropy(gray, neighborhood=10):
    """
    Compute the average local entropy of an image.

    Args:
        gray:
            Grayscale image.
        neighborhood:
            Radius of the disk-shaped neighborhood used for entropy calculation.

    Returns:
        float: Mean entropy value across the image.
    """
    if gray.ndim != 2:
        raise ValueError("image_entropy expects a 2D grayscale image")
    if gray.dtype != np.uint8:
        raise ValueError("image_entropy expects uint8 input")
    return float(np.mean(skimage_entropy(gray, disk(neighborhood))))


def tile_intensity(tile, threshold, channel=None):
    """
    Check whether a tile exceeds a mean intensity threshold.

    Args:
        tile:
            Input image array (H, W, C) or (H, W).
        threshold:
            Intensity threshold.
        channel:
            Optional channel index. If provided, only that channel is evaluated.

    Returns:
        bool: True if mean intensity exceeds the threshold, else False.
    """
    if channel is not None:
        return np.mean(tile[:, :, channel]) > threshold
    return np.mean(tile) > threshold


def calculate_std_mean(patch_path, channel=True, norm=True):
    """
    returns standard deviation and mean of patches
    :param patch_path: path to patches
    :param channel: boolean default value True
    :param norm: normalize values 0-255->0-1
    :return mean: list of channel means
    :return std: list of channel std
    """
    if patch_path is not None:
        patches = glob.glob(os.path.join(patch_path, "*"))
    shape = cv2.imread(patches[0]).shape
    channels = shape[-1]
    chnl_values = np.zeros((channels))
    chnl_values_sqrt = np.zeros((channels))
    pixel_nums = len(patches) * shape[0] * shape[1]
    print("total number pixels: {}".format(pixel_nums))
    axis = (0, 1, 2) if not channel else (0, 1)
    divisor = 1.0 if not norm else 255.0
    for path in patches:
        patch = cv2.imread(path)
        patch = (patch / divisor).astype("float64")
        chnl_values += np.sum(patch, axis=axis, dtype="float64")
    mean = chnl_values / pixel_nums
    for path in patches:
        patch = cv2.imread(path)
        patch = (patch / divisor).astype("float64")
        chnl_values_sqrt += np.sum(np.square(patch - mean), axis=axis, dtype="float64")
    std = np.sqrt(chnl_values_sqrt / pixel_nums, dtype="float64")
    print("mean: {}, std: {}".format(mean, std))
    return mean, std


def calculate_weights(mask_path, num_cls):
    """
    Compute inverse frequency class weights from segmentation masks.

    Args:
        mask_path:
            Path to directory containing mask images.
        num_cls:
            Number of classes.

    Returns:
        List[float]: List of inverse-frequency weights for each class.

    Notes:
        - Assumes masks are stored as images readable by OpenCV.
        - Pixel values are treated as class labels.
    """
    print("Calculating weights")
    if mask_path is not None:
        mask_files = glob.glob(os.path.join(mask_path, "*"))
    cls_nums = {c: 0 for c in range(num_cls)}

    for f in mask_files:
        mask = cv2.imread(f)
        pixels = mask.reshape(-1)
        classes = np.unique(pixels, return_counts=True)
        pixelDict = dict(list(zip(*classes)))
        for k, v in pixelDict.items():
            cls_nums[k] = cls_nums[k] + v

    total = sum(list(cls_nums.values()))
    weights = [v / total for v in list(cls_nums.values())]
    print(weights)
    weights = [1 / w for w in weights]
    print(weights)
    return weights
