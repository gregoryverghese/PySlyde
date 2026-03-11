import pickle
import rocksdb
import numpy as np


class NpyObject:
    def __init__(self, ndarray):
        self.ndarray = ndarray.tobytes()
        self.size = ndarray.shape
        self.dtype = ndarray.dtype

    def get_ndarray(self):
        ndarray = np.frombuffer(self.ndarray, dtype=self.dtype)
        return ndarray.reshape(self.size)


class RocksDBWrite:
    def __init__(self, db_path, write_frequency=10):
        self.db_path = db_path
        print(f"DB Path: {self.db_path}")

        # Configure RocksDB options
        options = rocksdb.Options()
        options.create_if_missing = True
        options.write_buffer_size = 64 * 1024 * 1024  # 64MB
        options.max_write_buffer_number = 3
        options.target_file_size_base = 64 * 1024 * 1024
        self.db = rocksdb.DB(self.db_path, options)
        self.write_frequency = write_frequency

    def __repr__(self):
        return f"RocksDBWrite(path: {self.db_path})"

    def _print_progress(self, i, total):
        complete = float(i) / total
        print(f"\r- Progress: {complete:.1%}", end="\r")

    def write(self, parser):
        batch = rocksdb.WriteBatch()
        total = sum(1 for _ in parser)

        for i, (p, tile) in enumerate(parser):
            name = str(p[1]) + "_" + str(p[0])
            key = f"{name}".encode("ascii")
            value = NpyObject(tile)
            batch.put(key, pickle.dumps(value))

            # Commit every write_frequency entries
            if i % self.write_frequency == 0:
                self.db.write(batch)
                batch.clear()
                self._print_progress(i, total)

        # Commit any remaining items in the batch
        self.db.write(batch)

    def write_image(self, image, name):
        key = f"{name}".encode("ascii")
        value = NpyObject(image)
        self.db.put(key, pickle.dumps(value))

    def close(self):
        del self.db  # Close RocksDB by deleting the DB instance


class RocksDBRead:
    def __init__(self, db_path):
        self.db_path = db_path
        options = rocksdb.Options()
        options.create_if_missing = False
        self.db = rocksdb.DB(self.db_path, options, read_only=True)

    @property
    def num_keys(self):
        # Counting keys in RocksDB requires iterating through the database
        it = self.db.iterkeys()
        it.seek_to_first()
        return sum(1 for _ in it)

    def __repr__(self):
        return f"RocksDBRead(path: {self.db_path})"

    def get_keys(self):
        keys = []
        it = self.db.iterkeys()
        it.seek_to_first()
        for key in it:
            keys.append(key.decode("ascii"))
        return keys

    def read_image(self, key):
        value = self.db.get(key.encode("ascii"))
        if value:
            image = pickle.loads(value)
            return image.get_ndarray()
        else:
            return None
