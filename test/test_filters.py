# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 12:06:32 2025

@author: PiazzeseC
"""

import numpy as np
import cv2
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
import pytest
from pyslyde.util import filters


@pytest.fixture
def make_rgb_image():
    """Return a function to create a solid RGB image."""
    def _make(color=(128, 128, 128), size=(64, 64)):
        img = np.ones((*size, 3), dtype=np.uint8)
        img[..., 0] *= color[0]
        img[..., 1] *= color[1]
        img[..., 2] *= color[2]
        return img
    return _make

@pytest.fixture
def dummy_patch():
    class DummyPatch:
        """
        Dummy patch container for testing.
        Stores original patch objects and simulates extract_patches().
        """
        def __init__(self, patches):
            self._patches = patches

        # Filters expect list of tuples (image, metadata)
        def extract_patches(self):
            return [(p, p) for p in self._patches]

    return DummyPatch

def test_image_entropy(make_rgb_image):
    # Create a uniform gray image — entropy should be ~0
    img = make_rgb_image(color=(100, 100, 100))
    result = filters.image_entropy(img)
    
    # Check types and expected range
    assert isinstance(result, float)
    assert 0 <= result < 8  # typical entropy range for 8-bit images
    
    # For uniform image, mean entropy should be near zero
    assert result < 1e-3, f"Expected near-zero entropy, got {result}"
    
    
def test_image_entropy_general(make_rgb_image):
    # General test for entropy
    img = make_rgb_image(color=(50, 50, 50))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    expected = np.mean(sk_entropy(gray, disk(10)))
    got = filters.image_entropy(img)
    assert np.isclose(got, expected, atol=1e-6)
    
    
def test_entropy_below_threshold(make_rgb_image, monkeypatch):
    # Test that entropy returns True when below the threshold
    # Using monkeypatch to temporarily replace a function only for the duration of a single test. 
    monkeypatch.setattr(filters, "image_entropy", lambda x: 0.1)
    
    tile = make_rgb_image()
    assert filters.entropy(tile, threshold=0.5) is True


def test_entropy_above_threshold(make_rgb_image, monkeypatch):
    # Test that entropy returns None when above the threshold
    filters.image_entropy = lambda x: 1.0
    tile = make_rgb_image()
    assert filters.entropy(tile, threshold=0.5) is None


def test_tile_intensity_light_gray(make_rgb_image):
    # global threshold
    tile = make_rgb_image(color=(200, 200, 200))
    assert filters.tile_intensity(tile, threshold=100) is True
    assert filters.tile_intensity(tile, threshold=250) is None


def test_tile_intensity_bright_green(make_rgb_image):
    # channel threshold
    tile = make_rgb_image(color=(10, 200, 10))
    # channel 1 (G) mean is 200
    assert filters.tile_intensity(tile, threshold=100, channel=1) is True
    assert filters.tile_intensity(tile, threshold=250, channel=1) is None
    assert filters.tile_intensity(tile, threshold=50, channel=0) is None
    
    
def test_remove_black(make_rgb_image, dummy_patch):
    # Removing dark patches
    black = make_rgb_image(color=(0, 0, 0))
    white = make_rgb_image(color=(255, 255, 255))
    patch = dummy_patch([black, white])
    result = filters.remove_black(patch, threshold=60, max_value=255, area_thresh=0.2)
    # black should be removed
    assert len(result._patches) == 1
    assert np.array_equal(result._patches[0], white)


def test_remove_white(make_rgb_image, dummy_patch):
    # Keeping bright patches
    white = make_rgb_image(color=(255, 255, 255))
    patch = dummy_patch([white])
    result = filters.remove_black(patch)
    assert len(result._patches) == 1  # nothing removed  
    assert np.array_equal(result._patches[0], white)
    
    
def test_remove_partially_black_above_threshold(make_rgb_image, dummy_patch):    
    # partially black patch above area threshold
    partially_dark = make_rgb_image(color=(10, 10, 10))
    patch = dummy_patch([partially_dark])
    result = filters.remove_black(patch, threshold=50, max_value=255, area_thresh=0.1)
    assert len(result._patches) == 0


def test_remove_partially_black_below_threshold(make_rgb_image, dummy_patch): 
    # partially black patch below area threshold
    slightly_dark = make_rgb_image(color=(100, 100, 100))
    patch = dummy_patch([slightly_dark])
    result = filters.remove_black(patch, threshold=50, max_value=255, area_thresh=0.5)
    # Should remain because dark area proportion < 0.5
    assert len(result._patches) == 1
    assert np.array_equal(result._patches[0], slightly_dark)


def test_remove_empty_patch(dummy_patch): 
    # empty patch list
    patch = dummy_patch([])
    result = filters.remove_black(patch)
    assert len(result._patches) == 0  # nothing to remove, should not error
    
    
def test_remove_blue(make_rgb_image, dummy_patch):
    # removes strong blue patches
    blue = make_rgb_image(color=(0, 0, 255))
    green = make_rgb_image(color=(0, 255, 0))
    patch = dummy_patch([blue, green])
    result = filters.remove_blue(patch, area_thresh=0.2)
    # blue patch removed
    assert len(result._patches) == 1
    assert result._patches[0] is green # patch is exactly the original green object


def test_remove_blue_keeps_non_blue(make_rgb_image, dummy_patch):
    # keeps non blue patches
    yellow = make_rgb_image(color=(255, 255, 0))
    patch = dummy_patch([yellow])
    result = filters.remove_blue(patch)
    assert len(result._patches) == 1
    assert result._patches[0] is yellow # patch is exactly the original yellow object
    
    