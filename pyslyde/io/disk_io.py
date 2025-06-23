"""Disk I/O utilities for PySlyde."""

import os
import pickle
from typing import Generator, Tuple, List, Dict, Any

import numpy as np


class DiskWrite:
    """
    Disk writer for saving tiles and features to disk.
    
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
        return f'DiskWrite(path: {self.path})'

    def _print_progress(self, i: int) -> None:
        """Print progress information."""
        print(f"Processed {i + 1} tiles", end='\r')

    def write(self, parser: Generator[Tuple[Tuple[int, int], np.ndarray], None, None]) -> None:
        """
        Write tiles to disk in batches.
        
        Writes tiles to disk in batches after every self.write_frequency iterations.
        
        Args:
            parser: Generator that yields (coordinates, tile) tuples.
        """
        tile_buffer: List[Tuple[str, np.ndarray]] = []
        meta_buffer: List[Tuple[str, Dict[str, Any]]] = []

        for i, (p, tile) in enumerate(parser):
            # Create file names for tile and metadata
            name = str(p[1]) + '_' + str(p[0])
            tile_path = os.path.join(self.path, f"{name}.npy")
            meta_path = os.path.join(self.path, f"{name}_meta.pkl")

            # Accumulate tile and metadata in buffers
            tile_buffer.append((tile_path, tile))
            meta_buffer.append((meta_path, {'size': tile.shape, 'dtype': tile.dtype}))

            # Write to disk if the buffer reaches the specified frequency
            if (i + 1) % self.write_frequency == 0:
                self._write_buffer(tile_buffer, meta_buffer)
                tile_buffer = []  # Clear the buffer after writing
                meta_buffer = []  # Clear the metadata buffer
                #self._print_progress(i)

        # Write any remaining tiles in the buffer after the loop completes
        if tile_buffer:
            self._write_buffer(tile_buffer, meta_buffer)

        print("\nFinished writing tiles to disk.")

    def _write_buffer(self, tile_buffer: List[Tuple[str, np.ndarray]], 
                     meta_buffer: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Write the contents of the buffers to disk.
        
        Args:
            tile_buffer: List of tuples (file path, tile).
            meta_buffer: List of tuples (file path, metadata).
        """
        # Write all tiles in the buffer to disk
        for tile_path, tile in tile_buffer:
            np.save(tile_path, tile)

        # Write corresponding metadata to disk
        for meta_path, metadata in meta_buffer:
            with open(meta_path, 'wb') as meta_file:
                pickle.dump(metadata, meta_file)

