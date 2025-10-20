# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:46:25 2025

@author: PiazzeseC
"""

import unittest
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
from pyslyde.util import utilities
import seaborn as sns
from sklearn.decomposition import IncrementalPCA
import random
from pathlib import Path
from openslide import OpenSlide
from PIL import Image
import cv2
from contextlib import redirect_stdout


### ---------- TEST: mask2rgb ----------
def test_mask2rgb_shape_and_values():
    
    # Create a mask with 3 classes: 0 (background), 1, 2
    mask = np.array([
        [0, 1, 2],
        [2, 1, 0],
        [1, 2, 0]
    ])

    rgb = utilities.mask2rgb(mask)
    
    # Checking the values 
    n_classes = len(np.unique(mask))
    palette = sns.color_palette('hls', n_classes)
    expected = np.zeros_like(rgb)
    expected[mask == 1] = palette[1]
    expected[mask == 2] = palette[2]
    # class 0 (background) remains black [0,0,0]

    # Assert all values match
    colors = sns.color_palette('hls', 3)  # 3 classes
    expected_rgb = np.zeros(mask.shape + (3,), dtype=float)

    # Map classes 1 and 2 to colors; class 0 stays black
    expected_rgb[mask == 1] = colors[0]  # class 1
    expected_rgb[mask == 2] = colors[1]  # class 2
    
    np.testing.assert_allclose(rgb, expected_rgb, atol=1e-6)
    

    # Check shape
    assert rgb.shape == (3, 3, 3)

    # Should not be all zeros
    assert not np.all(rgb == 0)
    
    # Edge case: mask contains only background
    mask = np.zeros((4, 4), dtype=int)
    rgb = utilities.mask2rgb(mask)
    assert np.all(rgb == 0)
    
    # Edge case: mask contains only one foreground class
    mask = np.ones((2, 2), dtype=int)
    rgb = utilities.mask2rgb(mask)
    assert rgb.shape == (2, 2, 3)
    assert not np.all(rgb == 0)    

### ---------- TEST: oneHotToMask ----------
def test_oneHotToMask_output():
    onehot = np.zeros((2, 2, 3))
    onehot[0, 0, 0] = 1
    onehot[0, 1, 1] = 1
    onehot[1, 0, 2] = 1
    onehot[1, 1, 1] = 1
    
    # Reference colors from seaborn (scaled to uint8)
    colors = (np.array(sns.color_palette('hls', 3)) * 255).astype(np.uint8)

    # ---------------- black background ----------------
    mask_black = utilities.oneHotToMask(onehot, background="black")
    expected_black = np.zeros_like(mask_black)
    expected_black[0, 0] = (0, 0, 0)       # background forced to black
    expected_black[0, 1] = colors[1]       # class 1
    expected_black[1, 0] = colors[2]       # class 2
    expected_black[1, 1] = colors[1]       # class 1

    np.testing.assert_array_equal(mask_black, expected_black)
    np.testing.assert_allclose(mask_black, expected_black, atol=1e-6)
    
    # Check shape and dtype
    assert mask_black.shape == (2, 2, 3)
    assert mask_black.dtype == np.uint8
    
    # Should not be all zeros
    assert not np.all(mask_black == 0)
    
    # Check float case
    mask_black_float = utilities.oneHotToMask(onehot, background="black").astype(float) / 255.0
    
    # Get colors dynamically, same as in function
    colors_float = np.array(sns.color_palette('hls', 3))  # floats [0,1]

    expected_black_float = np.zeros((2, 2, 3), dtype=float)
    expected_black_float[0, 0] = [0.0, 0.0, 0.0]  # background forced to black
    expected_black_float[0, 1] = colors_float[1]        # class 1
    expected_black_float[1, 0] = colors_float[2]        # class 2
    expected_black_float[1, 1] = colors_float[1]        # class 1
    
    np.testing.assert_array_equal(np.round(mask_black_float, 2), np.round(expected_black_float, 2))
    
    # Check shape and dtype
    assert mask_black_float.shape == (2, 2, 3)
    assert mask_black_float.dtype == np.float64
    
    # Should not be all zeros
    assert not np.all(mask_black_float == 0)
    
    # ---------------- white background ----------------
    mask_white = utilities.oneHotToMask(onehot, background="white")
    expected_white = expected_black.copy()
    expected_white[0, 0] = (255, 255, 255)  # background forced to white

    np.testing.assert_array_equal(mask_white, expected_white)
    np.testing.assert_allclose(mask_white, expected_white, atol=1e-6)
    
    # Check shape and dtype
    assert mask_white.shape == (2, 2, 3)
    assert mask_white.dtype == np.uint8
    
    # Should not be all zeros
    assert not np.all(mask_white == 0)
    
    # Check float case
    mask_white_float = utilities.oneHotToMask(onehot, background="white").astype(float) / 255.0
    
    expected_white_float = np.zeros((2, 2, 3), dtype=float)
    expected_white_float[0, 0] = [1, 1, 1]  # background forced to black
    expected_white_float[0, 1] = colors_float[1]        # class 1
    expected_white_float[1, 0] = colors_float[2]        # class 2
    expected_white_float[1, 1] = colors_float[1]        # class 1
    
    np.testing.assert_array_equal(np.round(mask_white_float, 2), np.round(expected_white_float, 2))
    
    # Check shape and dtype
    assert mask_white_float.shape == (2, 2, 3)
    assert mask_white_float.dtype == np.float64
    
    # Should not be all zeros
    assert not np.all(mask_white_float == 0)
    
    # ---------------- palette background (None) ----------------
    mask_palette = utilities.oneHotToMask(onehot, background=None)
    expected_palette = expected_black.copy()
    expected_palette[0, 0] = colors[0]  # class 0 gets palette color

    np.testing.assert_array_equal(mask_palette, expected_palette)
    np.testing.assert_allclose(mask_palette, expected_palette, atol=1e-6)
    
    # Check shape and dtype
    assert mask_palette.shape == (2, 2, 3)
    assert mask_palette.dtype == np.uint8
    
    # Should not be all zeros
    assert not np.all(mask_palette == 0)
    
    # Check float case
    mask_palette_float = utilities.oneHotToMask(onehot, background=None).astype(float) / 255.0
    
    expected_palette_float = expected_black_float.copy()
    expected_palette_float[0, 0] = colors_float[0]  # class 0 gets palette color
    
    np.testing.assert_array_equal(np.round(mask_palette_float, 2), np.round(expected_palette_float, 2))
    
    # Check shape and dtype
    assert mask_palette_float.shape == (2, 2, 3)
    assert mask_palette_float.dtype == np.float64

    # ---------------- invalid background ----------------
    with pytest.raises(ValueError, match="background must be 'black', 'white', or None"):
        utilities.oneHotToMask(onehot, background="blue")

### ---------- TEST: draw_boundary ----------
def test_draw_boundary_output():
    annotations = {
        "region1": [[[10, 10], [20, 20]]],
        "region2": [[[30, 30], [40, 40]]]
    }
    bounds = utilities.draw_boundary(annotations, offset=5)
    assert isinstance(bounds, list)
    assert len(bounds) == 2
    assert bounds[0][0] <= bounds[0][1]  # x-min < x-max
    
    result = utilities.draw_boundary(annotations, offset=5)
    assert result == [(5, 45), (5, 45)]

### ---------- TEST: get_x_y_from_0 ----------
def test_get_x_y_from_0_scaling():
    class MockSlide:
        level_dimensions = [(1000, 1000), (500, 500)]

    point = (250, 250)
    level = 1
    converted = utilities.get_x_y_from_0(MockSlide(), point, level)
    assert converted == (125, 125)

### ---------- TEST: get_size ----------
def test_get_size_scaling():
    class MockSlide:
        level_downsamples = [1.0, 4.0]

    size = (400, 400)
    new_size = utilities.get_size(MockSlide(), size, 0, 1)
    assert new_size == (100, 100)

### ---------- TEST: calculate_std_mean ----------
@patch('cv2.imread')
def test_calculate_std_mean(mock_imread, tmp_path):
    # Create dummy patch directory with fake images
    dummy_image = np.ones((10, 10, 3), dtype=np.uint8) * 100
    mock_imread.return_value = dummy_image
    mean, std = utilities.calculate_std_mean("dummy_path", norm=True)
    assert len(mean) == 3
    assert all([0 <= m <= 1 for m in mean])
     
def test_entropy_functions():
    """Combined tests for image_entropy and entropy functions."""
    blank_tile = np.zeros((50, 50), dtype=np.uint8)
    white_tile = np.ones((50, 50), dtype=np.uint8)
    random_tile = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

    # --- image_entropy tests ---
    avg_entropy_blank = utilities.image_entropy(blank_tile)
    avg_entropy_white = utilities.image_entropy(white_tile)
    avg_entropy_random = utilities.image_entropy(random_tile)
    

    assert avg_entropy_blank < 0.1
    assert avg_entropy_white < 0.1
    assert avg_entropy_random > 0.5

    # --- entropy (low-information detection) tests ---
    assert  utilities.entropy(blank_tile, threshold=0.1)
    assert not  utilities.entropy(random_tile, threshold=0.1)

def test_tile_intensity():
    """Tests for the tile_intensity function."""

    # Grayscale tiles
    blank_tile = np.zeros((50, 50), dtype=np.uint8)
    random_tile = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

    # Color tile
    color_tile = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

    # --- Grayscale tests ---
    assert not utilities.tile_intensity(blank_tile, threshold=10), "Blank tile should be below threshold"
    assert utilities.tile_intensity(random_tile, threshold=0), "Random tile should be above threshold"

    # --- Color channel tests ---
    for c in range(3):
        assert utilities.tile_intensity(color_tile, threshold=0, channel=c), f"Channel {c} should be above threshold"

    # --- No channel, full tile mean ---
    assert utilities.tile_intensity(color_tile, threshold=0), "Mean of color tile should be above threshold"
    assert not utilities.tile_intensity(np.zeros((50,50,3)), threshold=0.1), "Zero tile should be below threshold"

def test_calculate_std_mean():
    # Prepare fake image patches
    patches_rgb = [
        np.ones((4, 4, 3), dtype=np.uint8) * 50,
        np.ones((4, 4, 3), dtype=np.uint8) * 100,
        np.ones((4, 4, 3), dtype=np.uint8) * 150
    ]

    # Function to mock cv2.imread
    def fake_imread(path):
        mapping = {
            'patch1.png': patches_rgb[0],
            'patch2.png': patches_rgb[1],
            'patch3.png': patches_rgb[2],
        }
        return mapping[path].copy()

    # Patch glob.glob to return fake file paths, and cv2.imread to return the fake images
    with patch('glob.glob', return_value=['patch1.png', 'patch2.png', 'patch3.png']), \
         patch('cv2.imread', side_effect=fake_imread):
        
        mean, std = utilities.calculate_std_mean('dummy_path', channel=True, norm=True)

    # Expected values
    expected_mean = np.array([100/255]*3)
    expected_std = np.sqrt(np.mean([(50/255-100/255)**2,
                                    (100/255-100/255)**2,
                                    (150/255-100/255)**2])) * np.ones(3)

    assert np.allclose(mean, expected_mean, atol=1e-6)
    assert np.allclose(std, expected_std, atol=1e-6)


    # ----- Test 2: Grayscale images without normalization -----
    # Prepare fake grayscale patches
    imgs_gray = [
        np.ones((4, 4, 1), dtype=np.uint8) * 20,
        np.ones((4, 4, 1), dtype=np.uint8) * 40
    ]

    # Function to mock cv2.imread
    def fake_imread(path):
        mapping = {
            'patch1.png': imgs_gray[0],
            'patch2.png': imgs_gray[1],
        }
        # If the path isn't in mapping, return the first image as fallback
        return mapping.get(path, imgs_gray[0]).copy()

    # Patch glob.glob to return 2 paths (simulate 2 patches)
    with patch('glob.glob', return_value=['patch1.png', 'patch2.png']), \
         patch('cv2.imread', side_effect=fake_imread):
        
        mean, std = utilities.calculate_std_mean('dummy_path', channel=False, norm=False)

    # Expected mean and std (no normalization)
    expected_mean = np.array([30.0])
    expected_std = np.array([10.0])

    # Assertions
    assert np.allclose(mean, expected_mean, atol=1e-6)
    assert np.allclose(std, expected_std, atol=1e-6)

def test_get_pca():

    # ----- Prepare fake .npy data -----
    fake_arrays = [
        np.random.rand(5, 3),
        np.random.rand(3),       # 1D array
        np.zeros((4, 3)),        # zero array, should be skipped
        np.random.rand(5, 3)
    ]
    fake_files = ['a.npy', 'b.npy', 'c.npy', 'd.npy']

    with patch('glob.glob', return_value=fake_files), \
         patch('numpy.load', side_effect=fake_arrays), \
         patch('builtins.print'):

        # Call function
        ipca = utilities.get_pca()

        # Should return IncrementalPCA
        assert isinstance(ipca, IncrementalPCA)

def test_sample_patches():

    # ----- Dummy Patch class  -----
    class DummyPatch:
        def __init__(self, slide, size, mag_level, border, step):
            self.slide = slide
            self.size = size
            self.mag_level = mag_level
            self.border = border
            self.step = step
            self._patches = [f"patch_{i}" for i in range(10)]
            self.patches = []

    # Create an instance of DummyPatch
    patch_obj = DummyPatch("slide_1", (64, 64), 20, 4, 2)

    # ----- Test 1: Without replacement -----
    with patch("random.sample", side_effect=random.sample) as mock_sample:
        new_patch = utilities.sample_patches(patch_obj, n=5, replacement=False)

    # Check properties
    assert isinstance(new_patch, DummyPatch)
    assert len(new_patch.patches) == 5
    assert all(p in patch_obj._patches for p in new_patch.patches)
    mock_sample.assert_called_once()

    # ----- Test 2: With replacement -----
    with patch("random.choices", side_effect=random.choices) as mock_choices:
        new_patch2 = utilities.sample_patches(patch_obj, n=7, replacement=True)

    assert isinstance(new_patch2, DummyPatch)
    assert len(new_patch2.patches) == 7
    assert all(p in patch_obj._patches for p in new_patch2.patches)
    mock_choices.assert_called_once()

    # ----- Test 3: Ensure independence -----
    assert new_patch is not new_patch2
    assert patch_obj.patches == []

def test_visualise_wsi_tiling(tmp_path):
    
    tmp_path = Path(tmp_path)
    save_path = tmp_path / "output.png"

    # ----- Dummy WSI -----
    class DummyImage:
        def convert(self, mode):
            assert mode == 'RGB'
            return np.zeros((100, 100, 3), dtype=np.uint8)

    class DummyWSI:
        def __init__(self):
            self.level_dimensions = [(1000, 1000)] * 4
            self.level_downsamples = [1, 2, 4, 8]
        def get_thumbnail(self, dim):
            return DummyImage()

    # ----- Dummy tiler -----
    class DummyTiler:
        def __init__(self):
            self.tiles = [(0, 0), (100, 200)]
            self._x_dim = 256
            self._y_dim = 256

    wsi = DummyWSI()
    tiler = DummyTiler()

    # ----- Patch objects in the module where the function is defined (utilities.py) -----
    with patch("pyslyde.util.utilities.patches.Rectangle") as mock_rect, \
     patch("pyslyde.util.utilities.plt") as mock_plt, \
     patch("pyslyde.util.utilities.mpl") as mock_mpl:

        utilities.visualise_wsi_tiling(
            wsi=wsi,
            tiler=tiler,
            save_path=str(save_path),
            viewing_res=3,
            plot_args={'color': 'red', 'size': (12, 12), 'title': ''}
        )

    # ----- Assertions -----
    mock_mpl.use.assert_called_once_with('Agg')  # Matplotlib backend set to Agg
    mock_plt.figure.assert_called_once()          # Figure created
    mock_plt.imshow.assert_called_once()          # Thumbnail plotted
    mock_plt.savefig.assert_called_once_with(str(save_path))  # Image saved
    mock_plt.close.assert_called_once()           # Figure closed

#####################################################################
#####################   class TissueDetect ##########################
#####################################################################

def dummy_slide(as_pil=True):
    mock_slide = MagicMock()
    mock_slide.level_downsamples = [1, 2, 4, 8, 16, 32]
    mock_slide.level_dimensions = [(100, 100)] * 6
    mock_slide.shape = [100, 100, 3]
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    if as_pil:
        pil_image = Image.fromarray(dummy_image)
        mock_slide.get_thumbnail.return_value = pil_image
    else:
        mock_slide.get_thumbnail.return_value = dummy_image
        
    return mock_slide

def dummy_slide_OpenSlide(as_pil=True):
    mock_slide = MagicMock(spec=OpenSlide)
    mock_slide.level_downsamples = [1, 2, 4, 8, 16, 32]
    mock_slide.level_dimensions = [(100, 100)] * 6
    mock_slide.dimensions = (100, 100)
    
    dummy_image_os = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    if as_pil:
        pil_image = Image.fromarray(dummy_image_os)
        # return the real PIL image directly
        mock_slide.get_thumbnail.return_value = pil_image
    else:
        mock_slide.shape = [100, 100, 3]
        mock_slide.get_thumbnail.return_value = dummy_image_os
    
    return mock_slide

def test_slide_object():
    # --- Case 1: slide is a NumPy array (not OpenSlide) ---
    slide_obj = dummy_slide(as_pil=False)  # call function to get mock slide
    td = utilities.TissueDetect(slide_obj)
    assert td.slide == slide_obj   # same object
    assert td.slide.level_downsamples == [1, 2, 4, 8, 16, 32]
    assert td.slide.level_dimensions == [(100, 100)] * 6
    assert td.slide.shape == [100, 100, 3]
    assert np.array_equal(td.slide.get_thumbnail(), np.ones((100, 100, 3), dtype=np.uint8) * 128)
    assert td.tissue_mask is None
    assert td.contour_mask is None
    
    # --- Case 2: slide is OpenSlide ---
    slide_obj_os = dummy_slide_OpenSlide(as_pil=False)  # call function to get mock slide
    td_os = utilities.TissueDetect(slide_obj_os)
    assert td_os.slide == slide_obj_os  # same object
    assert td_os.slide.level_downsamples == [1, 2, 4, 8, 16, 32]
    assert td_os.slide.level_dimensions == [(100, 100)] * 6
    assert td_os.slide.dimensions == (100, 100)
    assert np.array_equal(td_os.slide.get_thumbnail(), np.ones((100, 100, 3), dtype=np.uint8) * 128)
    assert td_os.tissue_mask is None
    assert td_os.contour_mask is None
 
def test_mask_image():
        
    td = utilities.TissueDetect(dummy_slide())
    td.contour_mask = np.ones((100, 100), dtype=np.uint8)
    thumb = np.ones((100, 100, 3), dtype=np.uint8) * 100
    masked = td.mask_image(thumb)
    assert isinstance(masked, np.ndarray)
    assert masked.shape == thumb.shape
    
    td_op = utilities.TissueDetect(dummy_slide_OpenSlide())
    td_op.contour_mask = np.ones((100, 100), dtype=np.uint8)
    thumb_op = np.ones((100, 100, 3), dtype=np.uint8) * 100
    masked_op = td_op.mask_image(thumb_op)
    assert isinstance(masked_op, np.ndarray)
    assert masked_op.shape == thumb.shape

def test_detect_tissue():
    # NumPy array slide
    slide_np = np.ones((100, 100, 3), dtype=np.uint8) * 128
    td_np = utilities.TissueDetect(slide_np)
    mask_np = td_np.detect_tissue()
    
    assert isinstance(mask_np, np.ndarray)
    # mask should be 2D (height, width)
    assert mask_np.shape == slide_np.shape[:2]
    assert td_np.tissue_mask.shape == slide_np.shape[:2]

    # OpenSlide mock
    slide_os = dummy_slide_OpenSlide(as_pil=True)
    td_os = utilities.TissueDetect(slide_os)
    mask_os = td_os.detect_tissue()
    
    assert isinstance(mask_os, np.ndarray)
    # For OpenSlide, mask shape is (height, width) as well
    expected_shape = (slide_os.dimensions[1], slide_os.dimensions[0])
    assert mask_os.shape == expected_shape
    assert td_os.tissue_mask.shape == expected_shape

    # Check mask values
    assert set(np.unique(mask_np)).issubset({0, 1})
    assert set(np.unique(mask_os)).issubset({0, 1})

def test_generate_tissue_contour():
    slide_np = np.ones((100, 100, 3), dtype=np.uint8) * 128
    td_np = utilities.TissueDetect(slide_np)
    contours_np = td_np._generate_tissue_contour()

    # Check that contours is iterable
    assert hasattr(contours_np, '__iter__')
    assert td_np.contour_mask is not None
    assert td_np.contour_mask.shape[:2] == slide_np.shape[:2]

    for c in contours_np:
        assert isinstance(c, np.ndarray)
        assert c.shape[1] == 2

    # OpenSlide mock
    slide_os = dummy_slide_OpenSlide(as_pil=True)
    td_os = utilities.TissueDetect(slide_os)
    contours_os = td_os._generate_tissue_contour()
    assert hasattr(contours_os, '__iter__')
    assert td_os.contour_mask is not None
    assert td_os.contour_mask.shape[:2] == slide_os.dimensions
    for c in contours_os:
        assert isinstance(c, np.ndarray)
        assert c.shape[1] == 2

def test_border():
    # --- NumPy slide ---
    slide_np = np.ones((100, 100, 3), dtype=np.uint8) * 128
    td_np = utilities.TissueDetect(slide_np)
    td_np._generate_tissue_contour()
    
    border_coords = td_np.border()
    assert isinstance(border_coords, tuple)
    assert len(border_coords) == 2
    (x_min, y_min), (x_max, y_max) = border_coords
    # Use slide_np.shape[:2] for height, width
    height, width = slide_np.shape[:2]
    assert 0 <= x_min < x_max <= width
    assert 0 <= y_min < y_max <= height

    # --- OpenSlide mock ---
    slide_os = dummy_slide_OpenSlide(as_pil=True)  # PIL mock, has .dimensions
    td_os = utilities.TissueDetect(slide_os)
    td_os._generate_tissue_contour()
    
    border_coords_os = td_os.border()
    assert isinstance(border_coords_os, tuple)
    assert len(border_coords_os) == 2
    (x_min, y_min), (x_max, y_max) = border_coords_os
    # Use mock slide dimensions
    width, height = slide_os.dimensions
    assert 0 <= x_min < x_max <= width
    assert 0 <= y_min < y_max <= height

def test_tissue_thumbnail():
    # Mock slide
    mock_slide = MagicMock()
    
    # Mock slide.level_downsamples and level_dimensions
    mock_slide.level_downsamples = [1, 2, 4, 8, 16, 32]
    mock_slide.level_dimensions = [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16), (8, 8)]
    
    # Mock get_thumbnail to return a PIL Image with correct size
    from PIL import Image
    mock_slide.get_thumbnail.side_effect = lambda size: Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    
    # Instantiate TissueDetect with the mock slide
    td = utilities.TissueDetect(mock_slide)
    
    # Mock _generate_tissue_contour to return a simple contour
    td._generate_tissue_contour = MagicMock(return_value=[np.array([[0,0],[0,10],[10,10],[10,0]])])
    
    # Call the property
    thumbnail = td.tissue_thumbnail
    
    # Assertions
    assert isinstance(thumbnail, np.ndarray), "Thumbnail should be a numpy array"
    assert thumbnail.shape[2] == 3, "Thumbnail should have 3 channels (RGB)"
    assert np.all(thumbnail >= 0) and np.all(thumbnail <= 255), "Pixel values should be in 0-255 range"
    
    
# ---------- UNITTST RUNNER ----------
if __name__ == "__main__":
    print("Running individual test functions...\n")
    
    # List all your test functions
    test_functions = [
        test_mask2rgb_shape_and_values,
        test_oneHotToMask_output,
        test_draw_boundary_output,
        test_get_x_y_from_0_scaling,
        test_get_size_scaling,
        test_calculate_std_mean,
        test_entropy_functions,
        test_tile_intensity,
        test_get_pca,
        test_sample_patches,
        (test_visualise_wsi_tiling, '/PySlyde/test'),
        test_slide_object,
        test_mask_image,
        test_detect_tissue,
        test_generate_tissue_contour,
        test_border,
        test_tissue_thumbnail
    ]
    
    # Run each test
    for item in test_functions:
        if isinstance(item, tuple):
            test_func, arg = item
            print(f"Running {test_func.__name__} with argument {arg}...")
            try:
                # Suppress all print output inside the test function
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
                # Suppress all print output inside the test function
                with open(os.devnull, "w") as f, redirect_stdout(f):
                    test_func()
                print(f"{test_func.__name__}: PASS\n")
            except AssertionError as e:
                print(f"{test_func.__name__}: FAIL\n{e}\n")
            except Exception as e:
                print(f"{test_func.__name__}: ERROR\n{e}\n")