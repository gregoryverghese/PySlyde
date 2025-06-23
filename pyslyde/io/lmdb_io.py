"""LMDB I/O utilities for PySlyde."""

import os
import glob
import pickle
from typing import Generator, Tuple, List, Optional, Any
from PIL import Image

import lmdb
import numpy as np
from torch.utils.data import DataLoader, Dataset


class NpyObject:
    """
    Wrapper class for numpy arrays to be stored in LMDB.
    
    This class provides serialization and deserialization of numpy arrays
    for storage in LMDB databases.
    """
    
    def __init__(self, ndarray: np.ndarray) -> None:
        """
        Initialize the NpyObject.
        
        Args:
            ndarray: Numpy array to wrap.
        """
        self.ndarray = ndarray.tobytes()
        #self.label = label
        self.size = ndarray.shape
        self.dtype = ndarray.dtype

    def get_ndarray(self) -> np.ndarray:
        """
        Reconstruct the numpy array from bytes.
        
        Returns:
            np.ndarray: The reconstructed numpy array.
        """
        ndarray = np.frombuffer(self.ndarray, dtype=self.dtype)
        return ndarray.reshape(self.size)


class LMDBWrite:
    """
    LMDB writer for saving tiles and features to LMDB database.
    
    This class provides functionality to write tiles and features to LMDB
    databases with configurable write frequency.
    """
    
    def __init__(self, db_path: str, map_size: int, write_frequency: int = 10) -> None:
        """
        Initialize the LMDB writer.
        
        Args:
            db_path: Path to the LMDB database.
            map_size: Size of the LMDB memory map.
            write_frequency: Number of items to buffer before committing.
        """
        self.db_path = db_path
        self.map_size = map_size
        print(f"db path and map_size: {self.db_path} {self.map_size}")
        self.env = lmdb.open(path=self.db_path, map_size=self.map_size, writemap=True)
        self.write_frequency = write_frequency

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f'LMDBWrite(size: {self.map_size}, path: {self.db_path})'
    

    def _print_progress(self, i: int, total: int) -> None:
        """Print progress information."""
        complete = float(i)/total
        print(f'\r- Progress: {complete:.1%}', end='\r')


    def write(self, parser: Generator[Tuple[Tuple[int, int], np.ndarray], None, None]) -> None:
        """
        Write tiles to LMDB database.
        
        Args:
            parser: Generator that yields (coordinates, tile) tuples.
        """
        print("Beginning writing to db ...")
        txn=self.env.begin(write=True)
        for i, (p, tile) in enumerate(parser):
            name = str(p[1])+'_'+str(p[0])
            key = f"{name}"
            value=NpyObject(tile)
            txn.put(key.encode("ascii"), pickle.dumps(value))
            #self._print_progress(i,len(patch._patches))
            if i % self.write_frequency == 0:
                txn.commit()
                txn = self.env.begin(write=True)
        txn.commit()
        self.env.close()
        print("Writing to db done")


    def write_image(self,image: np.ndarray,name: str) -> None: 
        txn=self.env.begin(write=True)
        value=NpyObject(image)
        b=pickle.dumps(value)
        key = f"{name}"
        txn.put(key.encode('ascii'), b)
        txn.commit()


    def close(self) -> None:
        """Close the LMDB environment."""
        self.env.close()


class LMDBRead:
    """
    LMDB reader for reading tiles and features from LMDB database.
    
    This class provides functionality to read tiles and features from LMDB
    databases.
    """
    
    def __init__(self, db_path: str, image_size: Optional[Tuple[int, ...]] = None) -> None:
        """
        Initialize the LMDB reader.
        
        Args:
            db_path: Path to the LMDB database.
            image_size: Optional size of images to read.
        """
        self.db_path=db_path
        self.env=lmdb.open(self.db_path,
                           readonly=True,
                           lock=False
                           )
        self.image_size = image_size


    @property
    def num_keys(self) -> int:
        """Get the number of keys in the database."""
        with self.env.begin() as txn:
            length = txn.stat()['entries']
        return length


    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f'LMDBRead(path: {self.db_path})'


    def get_keys(self) -> List[bytes]:
        """
        Get all keys from the database.
        
        Returns:
            List of keys as bytes.
        """
        txn = self.env.begin()
        keys = [k for k, _ in txn.cursor()]
        #self.env.close()
        return keys


    def read_image(self,key: bytes) -> np.ndarray:
        """
        Read an image from the database.
        
        Args:
            key: Key of the image to read.
            
        Returns:
            np.ndarray: The image as a numpy array.
        """
        txn = self.env.begin()
        data = txn.get(key)
        image = pickle.loads(data)
        image=image.get_ndarray()
        #image = np.frombuffer(image, dtype=np.uint8)
        #image = image.reshape(self.image_size)
        return image

