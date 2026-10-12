"""Disk I/O utilities for PySlyde."""

import os
import numpy as np

from typing import Generator, List, Optional, Tuple
from pyslyde.util.utilities import coord_to_name


class DiskWrite:
    """
    Disk writer for saving tiles and features to disk as NumPy `.npy` files.

    This class provides functionality to write tiles and features to disk
    in batches with configurable write frequency.
    """

    def __init__(self, path: str, write_frequency: int = 10) -> None:
        """
        Initialize the disk writer.

        Args:
            path: Directory path to save files.
            write_frequency: Number of items to buffer before writing to disk.
        """
        self.path = path
        self.write_frequency = write_frequency
        os.makedirs(path, exist_ok=True)

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"DiskWrite(path: {self.path})"

    def write(
        self, parser: Generator[Tuple[Tuple[int, int], np.ndarray], None, None]
    ) -> None:
        """
        Write tiles to disk in batches.

        Writes tiles to disk in batches after every self.write_frequency iterations.

        Args:
            parser: Generator that yields (coordinates, tile) tuples.
        """
        print("Beginning writing to disk...")

        tile_buffer: List[Tuple[str, np.ndarray]] = []

        n_items = 0

        for i, (p, tile) in enumerate(parser):
            x, y = p
            name = coord_to_name(x, y)
            tile_path = os.path.join(self.path, f"{name}.npy")
            tile_buffer.append((tile_path, tile))
         
            n_items = i + 1

            if n_items % self.write_frequency == 0:
                self._write_buffer(tile_buffer)
                tile_buffer = []

        if tile_buffer:
            self._write_buffer(tile_buffer)

        print(f"Finished writing to disk ({n_items} items).")

    def _write_buffer(
        self,
        tile_buffer: List[Tuple[str, np.ndarray]],
    ) -> None:
        """
        Write the contents of the buffers to disk.

        Args:
            tile_buffer: List of tuples (file path, tile).
        """
        for tile_path, tile in tile_buffer:
            try:
                np.save(tile_path, tile)
                
            except OSError as e:
                raise OSError(f"Failed to write tile: {tile_path}") from e            


class DiskRead:
    """
    Disk reader for reading tiles and features saved as NumPy `.npy` files.

    This class provides functionality to read tiles and features from disk.
    """

    def __init__(
        self, path: str, image_size: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Initialize the disk reader.

        Args:
            path: Directory path containing NumPy `.npy` tile files.
            image_size: Optional size of images to read.
        """
        self.path = path
        self.image_size = image_size

    @property
    def num_keys(self) -> int:
        """Get the number of tile files."""
        return len(self.get_keys())

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"DiskRead(path: {self.path})"

    def get_keys(self) -> List[str]:
        """
        Get all tile keys from disk.

        Returns:
            List of tile filenames without the `.npy` extension.
        """
        return [
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.path)
            if filename.endswith(".npy")
        ]

    def read_image(self, key: str) -> np.ndarray:
        """
        Read an image/tile from disk.

        Args:
            key: Tile key, usually generated from coordinates using coord_to_name.

        Returns:
            np.ndarray: The image as a NumPy array.
        """
        tile_path = os.path.join(self.path, f"{key}.npy")

        try:
            image = np.load(tile_path)

        except OSError as e:
            raise OSError(f"Failed to read tile: {tile_path}") from e

        except ValueError as e:
            raise ValueError(f"Corrupted NumPy file: {tile_path}") from e           
        return image