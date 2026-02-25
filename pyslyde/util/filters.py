"""Image filtering and processing utilities for PySlyde."""

import cv2
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import entropy as skimage_entropy
from typing import Optional, List, Tuple, Any


def image_entropy(patch: np.ndarray) -> float:
    """
    Calculate the entropy of an image patch.
    
    Args:
        patch: Input image patch as RGB array.
        
    Returns:
        float: Average entropy value of the patch.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    entr = skimage_entropy(gray, disk(10))
    avg_entr = np.mean(entr)
    return avg_entr


def entropy(tile: np.ndarray, threshold: float) -> Optional[bool]:
    """
    Check if tile entropy is below threshold.
    
    Args:
        tile: Input image tile.
        threshold: Entropy threshold value.
        
    Returns:
        bool: True if entropy is below threshold, None otherwise.
    """
    avg_entropy = image_entropy(tile)
    if avg_entropy < threshold:
        return True
    return None


def tile_intensity(tile: np.ndarray, threshold: float, 
                   channel: Optional[int] = None) -> Optional[bool]:
    """
    Check if tile intensity is above threshold.
    
    Args:
        tile: Input image tile.
        threshold: Intensity threshold value.
        channel: Specific channel to check (0=red, 1=green, 2=blue).
        
    Returns:
        bool: True if intensity is above threshold, None otherwise.
    """
    if channel is not None:
        if np.mean(tile[:, :, channel]) > threshold:
            return True
    elif channel is None:
        if np.mean(tile) > threshold:
            return True
    return None


def remove_black(patch: Any,
                 threshold: int = 60,
                 max_value: int = 255,
                 area_thresh: float = 0.2) -> Any:
    """
    Remove patches that are predominantly black.
    
    Args:
        patch: Patch object containing patches to filter.
        threshold: Threshold for black pixel detection.
        max_value: Maximum value for thresholding.
        area_thresh: Area threshold for black pixel proportion.
        
    Returns:
        Patch object with black patches removed.
    """
    n = len(patch._patches)
    print('n', n)
    patches = patch._patches.copy()
    
    for image, p in patch.extract_patches():
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
        values, counts = np.unique(thresh, return_counts=True)
        
        if len(values) == 2:
            area_proportion = counts[0] / (counts[0] + counts[1])
            if area_proportion > area_thresh:
                patches.remove(p)
        elif len(values) == 1 and values[0] == 0:
            patches.remove(p)
        elif len(values) == 1 and values[0] == 255:
            continue

    patch._patches = patches
    n_removed = n - len(patch._patches)
    print(f'Black: N patches removed: {n_removed}')
    return patch


def remove_blue(patch: Any,
                area_thresh: float = 0.2,
                lower_blue: List[int] = [100, 150, 0],
                upper_blue: List[int] = [130, 255, 255]) -> Any:
    """
    Remove patches that are predominantly blue.
    
    Args:
        patch: Patch object containing patches to filter.
        area_thresh: Area threshold for blue pixel proportion.
        lower_blue: Lower HSV bounds for blue color.
        upper_blue: Upper HSV bounds for blue color.
        
    Returns:
        Patch object with blue patches removed.
    """
    n = len(patch._patches)
    patches = patch._patches.copy()
    
    for image, p in patch.extract_patches():
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(lower_blue), np.array(upper_blue))
        values, counts = np.unique(mask, return_counts=True)

        if len(values) == 2:
            area_proportion = counts[1] / (counts[0] + counts[1])
            if area_proportion > area_thresh:
                patches.remove(p)
        elif len(values) == 1 and values[0] == 255:
            patches.remove(p)
        elif len(values) == 1 and values[0] == 0:
            continue

    patch._patches = patches
    n_removed = n - len(patch._patches)
    print(f'Blue: N patches removed: {n_removed}')
    return patch

