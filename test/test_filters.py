# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 12:06:32 2025

@author: PiazzeseC
"""

import numpy as np
import cv2
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk

from pyslyde.util import filters
import os
from contextlib import redirect_stdout


def make_rgb_image(color=(128, 128, 128), size=(64, 64)):
    """Create a solid RGB image."""
    img = np.ones((*size, 3), dtype=np.uint8)
    img[..., 0] *= color[0]
    img[..., 1] *= color[1]
    img[..., 2] *= color[2]
    return img

class DummyPatch:
    """
    Dummy patch container for testing.
    Stores original patch objects and simulates extract_patches().
    """
    def __init__(self, patches):
        # Store original objects
        self._patches = patches

    def extract_patches(self):
        """
        Simulate the real patch extraction.
        Yields (image, patch_object) for each patch.
        Here, patch_object is the original object itself.
        """
        for patch in self._patches:
            yield patch, patch

def test_image_entropy():
    # Create a uniform gray image — entropy should be ~0
    img = make_rgb_image(color=(100, 100, 100), size=(64, 64))
    result = filters.image_entropy(img)
    
    # Check types and expected range
    assert isinstance(result, float)
    assert 0 <= result < 8  # typical entropy range for 8-bit images
    
    # For uniform image, mean entropy should be near zero
    assert result < 1e-3, f"Expected near-zero entropy, got {result}"
    
def test_entropy():
    
    # General test
    img = make_rgb_image(color=(50, 50, 50))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    expected = np.mean(sk_entropy(gray, disk(10)))
    got = filters.image_entropy(img)
    assert np.isclose(got, expected, atol=1e-6)
    
    
    # test entropy is true_when below the threshold
    filters.image_entropy = lambda x: 0.1
    tile = make_rgb_image()
    assert filters.entropy(tile, threshold=0.5) is True

    # test entropy is true_when above the threshold
    filters.image_entropy = lambda x: 1.0
    tile = make_rgb_image()
    assert filters.entropy(tile, threshold=0.5) is None

def test_tile_intensity():
    # global threshold
    tile = make_rgb_image(color=(200, 200, 200))
    assert filters.tile_intensity(tile, threshold=100) is True
    assert filters.tile_intensity(tile, threshold=250) is None

    # channel threshold
    tile = make_rgb_image(color=(10, 200, 10))
    # channel 1 (G) mean is 200
    assert filters.tile_intensity(tile, threshold=100, channel=1) is True
    assert filters.tile_intensity(tile, threshold=250, channel=1) is None
    
    assert filters.tile_intensity(tile, threshold=50, channel=0) is None
    
def test_remove_black():
    # Removing dark patches
    black = make_rgb_image(color=(0, 0, 0))
    white = make_rgb_image(color=(255, 255, 255))
    patch = DummyPatch([black, white])
    result = filters.remove_black(patch, threshold=60, max_value=255, area_thresh=0.2)
    # black should be removed
    assert len(result._patches) == 1
    assert np.array_equal(result._patches[0], white)


    # Keeping bright patches
    white = make_rgb_image(color=(255, 255, 255))
    patch = DummyPatch([white])
    result = filters.remove_black(patch)
    assert len(result._patches) == 1  # nothing removed  
    assert np.array_equal(result._patches[0], white)
    
    # partially black patch above area threshold
    partially_dark = make_rgb_image(color=(10, 10, 10))
    patch = DummyPatch([partially_dark])
    result = filters.remove_black(patch, threshold=50, max_value=255, area_thresh=0.1)
    assert len(result._patches) == 0

    # partially black patch below area threshold
    slightly_dark = make_rgb_image(color=(100, 100, 100))
    patch = DummyPatch([slightly_dark])
    result = filters.remove_black(patch, threshold=50, max_value=255, area_thresh=0.5)
    # Should remain because dark area proportion < 0.5
    assert len(result._patches) == 1
    assert np.array_equal(result._patches[0], slightly_dark)

    # empty patch list
    patch = DummyPatch([])
    result = filters.remove_black(patch)
    assert len(result._patches) == 0  # nothing to remove, should not error
    
def test_remove_blue():
    # removes strong blue patches
    blue = make_rgb_image(color=(0, 0, 255))
    green = make_rgb_image(color=(0, 255, 0))
    patch = DummyPatch([blue, green])
    result = filters.remove_blue(patch, area_thresh=0.2)
    # blue patch removed
    assert len(result._patches) == 1
    assert result._patches[0] is green # patch is exactly the original green object

    # keeps non blue patches
    yellow = make_rgb_image(color=(255, 255, 0))
    patch = DummyPatch([yellow])
    result = filters.remove_blue(patch)
    assert len(result._patches) == 1
    assert result._patches[0] is yellow # patch is exactly the original yellow object
    
    
# ---------- UNITTST RUNNER ----------
if __name__ == "__main__":
    print("Running individual test functions...\n")
    
    # List all your test functions
    test_functions = [
        test_image_entropy,
        test_entropy,
        test_tile_intensity,
        test_remove_black,
        test_remove_blue
    ]
    
    # Run each test
    for item in test_functions:
        if isinstance(item, tuple):
            test_func, arg = item
            print(f"Running {test_func.__name__} with argument {arg}...")
            try:
                # Suppress print statements inside the test function
                with open(os.devnull, "w") as f, redirect_stdout(f):
                    test_func(arg)
                print(f"{test_func.__name__}: PASS\n")
            except AssertionError as e:
                print(f"{test_func.__name__}: FAIL\n{e}\n")
            except Exception as e:
                print(f"{test_func.__name__}: ERROR\n{e}\n")
        else:
            test_func = item
            print(f"Running {test_func.__name__}...")
            try:
                # Suppress print statements inside the test function
                with open(os.devnull, "w") as f, redirect_stdout(f):
                    test_func()
                print(f"{test_func.__name__}: PASS\n")
            except AssertionError as e:
                print(f"{test_func.__name__}: FAIL\n{e}\n")
            except Exception as e:
                print(f"{test_func.__name__}: ERROR\n{e}\n")