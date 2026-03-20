"""RocksDB I/O utilities for PySlyde using rocksdict."""

import pickle
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from pyslyde.util.utilities import coord_to_name

try:
    from rocksdict import Options, Rdict
except ImportError as e:
    raise ImportError(
        "RocksDB support requires the optional dependency 'rocksdict'. "
        "Install it with: pip install 'PySlyde[rocksdb]'"
    ) from e


class NpyObject:
    """
    Wrapper class for numpy arrays to be stored in RocksDB.

    This class serializes numpy arrays into raw bytes together with
    shape and dtype metadata so they can be reconstructed later.
    """

    def __init__(self, ndarray: np.ndarray) -> None:
        """
        Initialize the wrapper.

        Args:
            ndarray: Numpy array to store.
        """
        self.ndarray = ndarray.tobytes()
        self.size = ndarray.shape
        self.dtype = ndarray.dtype

    def get_ndarray(self) -> np.ndarray:
        """
        Reconstruct the stored numpy array.

        Returns:
            np.ndarray: Reconstructed array.
        """
        ndarray = np.frombuffer(self.ndarray, dtype=self.dtype)
        return ndarray.reshape(self.size)


class RocksDBWrite:
    """
    RocksDB writer for saving tiles or feature arrays.

    Arrays are stored under coordinate-based keys derived from `(x, y)`
    tile coordinates using the shared package naming convention.
    """

    def __init__(self, db_path: str, write_frequency: int = 10) -> None:
        """
        Initialize the RocksDB writer.

        Args:
            db_path:
                Path to the RocksDB database.
            write_frequency:
                Number of items to buffer in Python before writing to the DB.
        """
        self.db_path = db_path
        self.write_frequency = write_frequency
        print(f"DB Path: {self.db_path}")

        options = Options(raw_mode=True)
        self.db = Rdict(self.db_path, options=options)

    def __repr__(self) -> str:
        """Return string representation of the writer."""
        return f"RocksDBWrite(path: {self.db_path})"

    def _flush_buffer(self, buffer: Dict[bytes, bytes]) -> int:
        """
        Flush a buffered set of key/value pairs to RocksDB.

        Args:
            buffer: Mapping of serialized keys to serialized values.

        Returns:
            int: Number of items written.
        """
        for key, value in buffer.items():
            self.db[key] = value
        return len(buffer)

    def write(
        self,
        parser: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
    ) -> None:
        """
        Write arrays from a generator into RocksDB.

        Each yielded item must be of the form:
            ((x, y), array)

        Args:
            parser: Generator yielding `(coordinates, array)` tuples.
        """
        print("Beginning writing to RocksDB...")

        buffer: Dict[bytes, bytes] = {}
        total_written = 0

        for (x, y), tile in parser:
            key = coord_to_name(x, y).encode("ascii")
            value = pickle.dumps(NpyObject(tile))
            buffer[key] = value

            if len(buffer) >= self.write_frequency:
                total_written += self._flush_buffer(buffer)
                buffer.clear()

        if buffer:
            total_written += self._flush_buffer(buffer)
            buffer.clear()

        print(f"Finished writing {total_written} items to RocksDB.")

    def write_image(self, image: np.ndarray, name: str) -> None:
        """
        Write a single array using a caller-provided key name.

        Args:
            image:
                Array to store.
            name:
                Database key name.
        """
        key = name.encode("ascii")
        value = pickle.dumps(NpyObject(image))
        self.db[key] = value

    def close(self) -> None:
        """
        Close the RocksDB handle if supported.

        rocksdict manages resources internally, but if a close()
        method is available we call it explicitly.
        """
        close_fn = getattr(self.db, "close", None)
        if callable(close_fn):
            close_fn()
        del self.db


class RocksDBRead:
    """
    RocksDB reader for reading stored arrays from RocksDB.
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the RocksDB reader.

        Args:
            db_path: Path to the RocksDB database.
        """
        self.db_path = db_path
        options = Options(raw_mode=True)
        self.db = Rdict(self.db_path, options=options)

    @property
    def num_keys(self) -> int:
        """
        Count the number of keys in the database.

        Returns:
            int: Number of stored entries.
        """
        return sum(1 for _ in self.db.keys())  # len(self.db)

    def __repr__(self) -> str:
        """Return string representation of the reader."""
        return f"RocksDBRead(path: {self.db_path})"

    def get_keys(self) -> List[str]:
        """
        Return all keys stored in the database.

        Returns:
            List[str]: Decoded key strings.
        """
        return [key.decode("ascii") for key in self.db.keys()]

    def read_image(self, key: str) -> Optional[np.ndarray]:
        """
        Read an array from the database by key.

        Args:
            key: Database key string.

        Returns:
            Optional[np.ndarray]:
                Reconstructed numpy array if found, otherwise None.
        """
        value = self.db.get(key.encode("ascii"))
        if value is None:
            return None

        image = pickle.loads(value)
        return image.get_ndarray()

    def close(self) -> None:
        """
        Close the RocksDB handle if supported.
        """
        close_fn = getattr(self.db, "close", None)
        if callable(close_fn):
            close_fn()
        del self.db
