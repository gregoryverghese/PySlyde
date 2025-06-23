"""Whole Slide Image (WSI) parser and stitching utilities for PySlyde."""

import os
import glob
import json
import random
import operator as op
from typing import List, Tuple, Optional, Generator, Callable, Dict, Any
from itertools import chain

import numpy as np
import cv2
import seaborn as sns
from matplotlib.path import Path
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd

from pyslyde.io.lmdb_io import LMDBWrite
from pyslyde.io.disk_io import DiskWrite
from pyslyde.encoders.feature_extractor import FeatureGenerator


class WSIParser:
    """
    Whole Slide Image parser for extracting tiles and features.
    
    This class provides functionality to parse whole slide images,
    extract tiles, and generate features from those tiles.
    """
    
    def __init__(
            self,
            slide: OpenSlide,
            tile_dim: int,
            border: List[Tuple[int, int]],
            mag_level: int = 0,
            stain_normalizer: Optional[Any] = None
    ) -> None:
        """
        Initialize the WSI parser.
        
        Args:
            slide: OpenSlide object representing the whole slide image.
            tile_dim: Dimension of tiles to extract.
            border: Border coordinates as list of tuples.
            mag_level: Magnification level to work at.
            stain_normalizer: Optional stain normalizer object.
        """
        super().__init__()
        self.slide = slide
        self.mag_level = mag_level
        self.tile_dims = (tile_dim, tile_dim)
        self.border = border

        self._x_min = int(self.border[0][1])
        self._x_max = int(self.border[1][1])
        self._y_min = int(self.border[0][0])
        self._y_max = int(self.border[1][0])
        self._downsample = int(slide.level_downsamples[mag_level])
        print('downsample', self._downsample)
        self._x_dim = int(tile_dim * self._downsample)
        self._y_dim = int(tile_dim * self._downsample)
        self._tiles: List[Tuple[int, int]] = []
        self._features: List[np.ndarray] = []
        self._number = len(self._tiles)

        self.stain_normalizer = stain_normalizer

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
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        config = {
            'name': self.slide.name,
            'mag': self.mag_level,
            'size': self.tile_dims,
            'border': self.border,
            'number': self._number
        }
        return config

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return str(self.config)

    def _remove_edge_case(self, x: int, y: int) -> bool:
        """
        Remove edge cases based on dimensions of patch.
        
        Args:
            x: Base x coordinate to test.
            y: Base y coordinate to test.
            
        Returns:
            bool: Whether to remove patch or not.
        """
        remove = False
        if x + self._x_dim > self._x_max:
            remove = True
        if y + self._y_dim > self._y_max:
            remove = True
        return remove

    def _tile_downsample(self, image: np.ndarray, ds: int) -> np.ndarray:
        """
        Downsample an image by a factor.
        
        Args:
            image: Input image as numpy array.
            ds: Downsample factor.
            
        Returns:
            np.ndarray: Downsampled image.
        """
        if ds:
            x, y, _ = image.shape
            image = cv2.resize(image, (int(y / ds), int(x / ds)))
            print(f"Downsampled to shape {image.shape}")
        return image

    def tiler(
            self, 
            stride: Optional[int] = None, 
            edge_cases: bool = False
    ) -> int:
        """
        Generate tile coordinates based on border, mag_level, and stride.
        
        Args:
            stride: Step size for tiling.
            edge_cases: Whether to handle edge cases.
            
        Returns:
            int: Number of patches generated.
        """
        stride = self.tile_dims[0] if stride is None else stride
        stride = stride * self._downsample
        
        self._tiles = []
        for x in range(self._x_min, self._x_max, stride):
            for y in range(self._y_min, self._y_max, stride):
                # if self._remove_edge_case(x, y):
                    # continue
                self._tiles.append((x, y))

        self._number = len(self._tiles)
        return self._number

    def extract_features(
            self,
            model_name: str,
            model_path: str,
            device: Optional[str] = None,
            downsample: Optional[int] = None,
            normalize: bool = False
    ) -> Generator[Tuple[Tuple[int, int], np.ndarray], None, None]:
        """
        Extract features from tiles using a specified model.
        
        Args:
            model_name: Name of the model to use.
            model_path: Path to the model weights.
            device: Device to run the model on.
            downsample: Optional downsample factor.
            normalize: Whether to normalize the tiles.
            
        Yields:
            Tuple of tile coordinates and feature vector.
        """
        encode = FeatureGenerator(model_name, model_path)
        print(f'Extracting features...')
        print(f'checking again... {len(self.tiles)}')

        for i, t in enumerate(self.tiles):
            tile = self.extract_tile(t[0], t[1])
            if downsample:
                tile = self._tile_downsample(tile, downsample)
            if normalize and self.stain_normalizer is not None:
                self.stain_normalizer.normalize(tile)

            feature_vec = encode.forward_pass(tile)
            feature_vec = feature_vec.detach().cpu().numpy()
            print(f'{i}')
            yield t, feature_vec

    def filter_tissue(
            self,
            slide_mask: np.ndarray,
            label: int,
            threshold: float = 0.5
    ) -> int:
        """
        Filter tiles based on tissue mask.
        
        Args:
            slide_mask: Mask of the slide.
            label: Label to filter for.
            threshold: Threshold for tissue proportion.
            
        Returns:
            int: Number of tiles remaining.
        """
        slide_mask[slide_mask != label] = 0
        slide_mask[slide_mask == label] = 1
        tiles = self._tiles.copy()
        for t in self._tiles:
            x, y = (t[0], t[1])
            t_mask = slide_mask[x:x + self._x_dim, y:y + self._y_dim]
            if np.sum(t_mask) < threshold * (self._x_dim * self._y_dim):
                tiles.remove(t)

        self._tiles = tiles
        return len(self._tiles)

    def filter_tiles(self, filter_func: Callable[[np.ndarray], bool], *args, **kwargs) -> None:
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

        print(f'Removed {self.number - len(tiles)} tiles')
        self._tiles = tiles.copy()

    def sample_tiles(self, n: int) -> None:
        """
        Sample a subset of tiles.
        
        Args:
            n: Number of tiles to sample.
        """
        n = len(self._tiles) if n > len(self._tiles) else n
        sample_tiles = random.sample(self._tiles, n)
        self._tiles = sample_tiles

    def extract_tile(
            self,
            x: Optional[int] = None,
            y: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract individual patch from WSI.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            
        Returns:
            np.ndarray: Extracted patch.
        """
        tile = self.slide.read_region(
            (y, x),
            self.mag_level, 
            self.tile_dims
        )
        tile = np.array(tile.convert('RGB'))
        return tile

    def extract_tiles(
            self,
            normalize: bool = False
    ) -> Generator[Tuple[Tuple[int, int], np.ndarray], None, None]:
        """
        Generator to extract all tiles.
        
        Args:
            normalize: Whether to normalize the tiles.
            
        Yields:
            Tuple of tile coordinates and tile array.
        """
        print('this is the final tile number', len(self._tiles))
        for t in self._tiles:
            tile = self.extract_tile(t[0], t[1])
            if normalize and self.stain_normalizer is not None:
                self.stain_normalizer.normalize(tile)
            yield t, tile

    @staticmethod
    def _save_to_disk(
            image: np.ndarray,
            path: str,
            x: Optional[int] = None,
            y: Optional[int] = None
    ) -> bool:
        """
        Save tile to disk.
        
        Args:
            image: Tile image as numpy array.
            path: Path to save the image.
            x: X coordinate for filename.
            y: Y coordinate for filename.
            
        Returns:
            bool: Success status.
        """
        assert isinstance(y, int) and isinstance(x, int)
        filename = '_' + str(y) + '_' + str(x)
        image_path = path + filename + '.png'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        status = cv2.imwrite(image_path, image)
        return status
    
    def save(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        tile_path: str,
        label_dir: bool = False,
        label_csv: bool = False,
        normalize: bool = False
    ) -> None:
        """
        Save the extracted tiles to disk.
        
        Args:
            func: Generator function that yields (coordinates, tile) tuples.
            tile_path: Base directory where tiles will be saved.
            label_dir: If True, saves tiles in subdirectories based on their label.
            label_csv: If True, saves tile metadata in a CSV file.
            normalize: If True, applies normalization to the tiles before saving.
        """
        os.makedirs(tile_path, exist_ok=True)

        metadata = []

        for (x, y), tile in func:
            if normalize and self.stain_normalizer is not None:
                self.stain_normalizer.normalize(tile)
            
            # Generate directory path
            save_dir = tile_path
            if label_dir:
                save_dir = os.path.join(tile_path, os.path.basename(tile_path))
                os.makedirs(save_dir, exist_ok=True)

            # Save the tile image
            self._save_to_disk(tile, save_dir, x, y)
            
            # Optionally save metadata
            if label_csv:
                metadata.append({"y": y, "x": x, "path": save_dir + f"_{y}_{x}.png"})

        if label_csv:
            df = pd.DataFrame(metadata)
            df.to_csv(tile_path + "_metadata.csv", index=False)

    def to_lmdb(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        db_path: str, 
        map_size: int,
        write_frequency: int = 10
    ) -> None:
        """
        Save to LMDB database.
        
        Args:
            func: Generator function that yields (coordinates, tile) tuples.
            db_path: Base directory where tiles or features will be saved.
            map_size: Map size for LMDB.
            write_frequency: Controls batch commit of a transaction.
        """
        os.makedirs(db_path, exist_ok=True)
        lmdb_writer = LMDBWrite(db_path, map_size, write_frequency)
        lmdb_writer.write(func)

    def to_rocksdb(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        db_path: str,
        write_frequency: int = 10
    ) -> None:
        """
        Save to RocksDB database.
        
        Args:
            func: Generator function that yields (coordinates, tile) tuples.
            db_path: Base directory where tiles or features will be saved.
            write_frequency: Controls batch commit of a transaction.
        """
        os.makedirs(db_path, exist_ok=True)
        # Note: RocksDBWrite import needs to be added
        # rocksdb_writer = RocksDBWrite(db_path, write_frequency)
        # rocksdb_writer.write(func)
        pass

    def feat_to_disk(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        path: str,
        write_frequency: int = 10
    ) -> None:
        """
        Save features to disk.
        
        Args:
            func: Generator function that yields (coordinates, feature) tuples.
            path: Path to save the features.
            write_frequency: Controls batch commit of a transaction.
        """
        os.makedirs(path, exist_ok=True)
        disk_writer = DiskWrite(path, write_frequency)
        disk_writer.write(func)


class Stitching:
    """
    Stitching class for reconstructing whole slide images from patches.
    """
    
    MAG_FACTORS: Dict[int, int] = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64}

    def __init__(self, patch_path: str,
                 slide: Optional[OpenSlide] = None,
                 patching: Optional[Any] = None,
                 name: Optional[str] = None,
                 step: Optional[int] = None,
                 border: Optional[List[Tuple[int, int]]] = None,
                 mag_level: int = 0) -> None:
        """
        Initialize the Stitching object.
        
        Args:
            patch_path: Path to the patches.
            slide: OpenSlide object.
            patching: Patching object.
            name: Name of the slide.
            step: Step size for stitching.
            border: Border coordinates.
            mag_level: Magnification level.
        """
        self.patch_path = patch_path
        self.slide = slide
        self.patching = patching
        self.name = name
        self.step = step
        self.border = border
        self.mag_level = mag_level

    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        config = {
            'name': self.name,
            'mag': self.mag_level,
            'step': self.step,
            'border': self.border,
            'number': len(self._patches())
        }
        return config

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return str(self.config)

    @property
    def step(self) -> Optional[int]:
        """Get the step size."""
        return self._get_step()

    @property
    def mag_factor(self) -> int:
        """Get the magnification factor."""
        return self.MAG_FACTORS[self.mag_level]

    def _get_coords(self) -> List[Tuple[int, int]]:
        """Extract coordinates from patch filenames."""
        coords = []
        for f in os.listdir(self.patch_path):
            if f.endswith('.png'):
                parts = f.split('_')
                if len(parts) >= 3:
                    y = int(parts[-2])
                    x = int(parts[-1].split('.')[0])
                    coords.append((x, y))
        return coords

    def _get_border(self) -> List[Tuple[int, int]]:
        """Calculate border from patch coordinates."""
        coords = self._get_coords()
        if not coords:
            return [(0, 0), (0, 0)]
        
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return [(y_min, y_max), (x_min, x_max)]

    def _get_step(self) -> Optional[int]:
        """Calculate step size from patch coordinates."""
        coords = self._get_coords()
        if len(coords) < 2:
            return None
        
        # Find the minimum difference between consecutive coordinates
        x_coords = sorted([c[0] for c in coords])
        y_coords = sorted([c[1] for c in coords])
        
        x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        
        if x_diffs and y_diffs:
            return min(min(x_diffs), min(y_diffs))
        return None

    def _completeness(self) -> float:
        """Calculate completeness of the stitched image."""
        if not self.border or not self.step:
            return 0.0
        
        expected_patches = ((self.border[1][1] - self.border[1][0]) // self.step + 1) * \
                          ((self.border[0][1] - self.border[0][0]) // self.step + 1)
        actual_patches = len(self._patches())
        
        return actual_patches / expected_patches if expected_patches > 0 else 0.0

    def _patches(self) -> List[str]:
        """Get list of patch filenames."""
        patches = []
        for f in os.listdir(self.patch_path):
            if f.endswith('.png'):
                patches.append(f)
        return patches

    def stitch(self, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Stitch patches together to create a complete image.
        
        Args:
            size: Size of the output image.
            
        Returns:
            np.ndarray: Stitched image.
        """
        # Implementation would go here
        # This is a placeholder for the actual stitching logic
        raise NotImplementedError("Stitching method not yet implemented")


