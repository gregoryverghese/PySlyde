"""
Unit tests for DiskWrite, LMDB IO, RocksDB IO, TFRecordWrite, and tfrecord_write helpers.
Covers batching, path handling, metadata integrity, overwrite behavior, and basic TFRecords round-trips.
"""
import os
import io
import json
import shutil
import pickle
import tempfile
import unittest
import pytest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import tensorflow as tf

from pyslyde.io.disk_io import DiskWrite
from pyslyde.io.tfrecords_io import TFRecordWrite
from pyslyde.io.tfrecord_write import (
    getShardNumber,
    printProgress,
    wrapInt64,
    wrapFloat,
    wrapBytes,
    convert,
    doConversion,
    getFiles,
)

##################################################################
# Check dependencies (LMDB, RocksDB)
# Capture both import failures for lmdb and rocksdb, 
# as well as missing lmdb_io and rocksdb_io scripts if API changes.
# Assume tf is correctly installed.
##################################################################
try:
    from pyslyde.io.lmdb_io import NpyObject as LMDBNpyObject, LMDBWrite, LMDBRead
    HAS_LMDB = True
except Exception:
    LMDBNpyObject = LMDBWrite = LMDBRead = None
    HAS_LMDB = False

try:
    from pyslyde.io.rocksdb_io import NpyObject as RocksNpyObject, RocksDBWrite, RocksDBRead
    HAS_ROCKSDB = True
except Exception:
    RocksNpyObject = RocksDBWrite = RocksDBRead = None
    HAS_ROCKSDB = False


##################################################################
# Helpers
##################################################################

def _synthetic_parser(coords, shape=(8, 8, 3), dtype=np.uint8):
    """
    Yield deterministic ((x, y), tile) pairs for the given coordinates.
    Each tile is a constant array whose value depends on the coordinate, enabling content checks.
    """
    for (x, y) in coords:
        if np.issubdtype(dtype, np.integer):
            val = (x + 2 * y) % np.iinfo(dtype).max
            tile = np.full(shape, val, dtype=dtype)
        else:
            val = (x + 2.5 * y) / 255.0
            tile = np.full(shape, val, dtype=dtype)
        yield (x, y), tile


##################################################################
# Tests for pyslyde/io/disk_io.py
##################################################################

@pytest.mark.disk
class TestDiskWrite(unittest.TestCase):
    """
    Test suite for DiskWrite focusing on correctness of filenames, tiles/metadata,
    buffer flushing semantics, overwrite behavior, and directory creation.
    """

    @classmethod
    def setUpClass(cls):
        """
        Prepare reusable coordinate sets shared across tests to keep cases consistent.
        """
        cls.coords_small = [(0, 0), (1, 2), (5, 3)]
        cls.coords_many = [(i, i + 1) for i in range(5)]  # 5 tiles

    def setUp(self):
        """
        Create a fresh temporary output directory for isolation between tests.
        """
        self.tmp = tempfile.mkdtemp(prefix="pyslyde_diskio_")

    def tearDown(self):
        """
        Remove the temporary directory and all contents after each test.
        """
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _tile_path(self, base, x, y):
        """
        Return the expected .npy path for a tile saved by DiskWrite (y_x.npy naming).
        """
        return os.path.join(base, f"{y}_{x}.npy")

    def _meta_path(self, base, x, y):
        """
        Return the expected metadata pickle path for a tile (y_x_meta.pkl naming).
        """
        return os.path.join(base, f"{y}_{x}_meta.pkl")

    def test_repr_includes_path(self):
        """
        __repr__ should include the target path and initialization should create the directory.
        """
        dw = DiskWrite(self.tmp, write_frequency=10)
        self.assertIn(self.tmp, repr(dw))
        self.assertTrue(os.path.isdir(self.tmp))

    def test_write_flush_at_end_when_under_frequency(self):
        """
        When the number of tiles is below write_frequency, all tiles are written on final flush with correct metadata.
        """
        dw = DiskWrite(self.tmp, write_frequency=10)
        parser = _synthetic_parser(self.coords_small, shape=(8, 8, 3), dtype=np.uint8)

        f = io.StringIO()
        with redirect_stdout(f):
            dw.write(parser)

        for (x, y) in self.coords_small:
            tpath = self._tile_path(self.tmp, x, y)
            mpath = self._meta_path(self.tmp, x, y)
            self.assertTrue(os.path.isfile(tpath), f"Missing tile {tpath}")
            self.assertTrue(os.path.isfile(mpath), f"Missing meta {mpath}")

            tile_loaded = np.load(tpath)
            with open(mpath, "rb") as fh:
                meta = pickle.load(fh)

            expected = next(_synthetic_parser([(x, y)], shape=(8, 8, 3), dtype=np.uint8))[1]
            self.assertEqual(tuple(tile_loaded.shape), (8, 8, 3))
            self.assertTrue(np.array_equal(tile_loaded, expected))
            self.assertEqual(tuple(meta["size"]), (8, 8, 3))
            self.assertEqual(meta["dtype"], expected.dtype)

        out = f.getvalue()
        self.assertIn("Finished writing tiles to disk.", out)

    def test_write_batches_with_frequency(self):
        """
        With write_frequency < N, DiskWrite should flush multiple times and persist all tiles and metadata.
        """
        dw = DiskWrite(self.tmp, write_frequency=2)
        parser = _synthetic_parser(self.coords_many, shape=(10, 10, 3), dtype=np.uint8)
        dw.write(parser)

        for (x, y) in self.coords_many:
            tpath = self._tile_path(self.tmp, x, y)
            mpath = self._meta_path(self.tmp, x, y)
            self.assertTrue(os.path.exists(tpath))
            self.assertTrue(os.path.exists(mpath))
            with open(mpath, "rb") as fh:
                meta = pickle.load(fh)
            self.assertEqual(tuple(meta["size"]), (10, 10, 3))
            self.assertEqual(str(meta["dtype"]), "uint8")

    def test_empty_parser_writes_nothing(self):
        """
        An empty generator should not create any files in the output directory.
        """
        dw = DiskWrite(self.tmp, write_frequency=3)

        def empty_gen():
            """Yield nothing."""
            if False:
                yield  # pragma: no cover

        dw.write(empty_gen())
        self.assertEqual(os.listdir(self.tmp), [])

    def test_overwrite_existing_files(self):
        """
        Writing the same coordinate twice should overwrite both tile contents and metadata.
        """
        coord = (3, 5)
        tpath = self._tile_path(self.tmp, *coord)
        mpath = self._meta_path(self.tmp, *coord)

        dw1 = DiskWrite(self.tmp, write_frequency=1)
        parser1 = _synthetic_parser([coord], shape=(6, 6, 1), dtype=np.uint8)
        dw1.write(parser1)
        tile1 = np.load(tpath)

        dw2 = DiskWrite(self.tmp, write_frequency=1)
        parser2 = _synthetic_parser([coord], shape=(6, 6, 1), dtype=np.float32)
        dw2.write(parser2)
        tile2 = np.load(tpath)

        self.assertFalse(np.array_equal(tile1, tile2))
        with open(mpath, "rb") as fh:
            meta = pickle.load(fh)
        self.assertEqual(tuple(meta["size"]), (6, 6, 1))
        self.assertEqual(str(meta["dtype"]), "float32")

    def test_path_is_created_if_missing(self):
        """
        DiskWrite should create missing nested output directories and write tiles there successfully.
        """
        nested = os.path.join(self.tmp, "deep", "nested", "outdir")
        self.assertFalse(os.path.exists(nested))
        dw = DiskWrite(nested, write_frequency=1)
        self.assertTrue(os.path.isdir(nested))

        coord = (0, 1)
        dw.write(_synthetic_parser([coord]))
        self.assertTrue(os.path.isfile(self._tile_path(nested, *coord)))
        self.assertTrue(os.path.isfile(self._meta_path(nested, *coord)))

    def test_invalid_write_frequency_raises(self):
        """
        A zero write_frequency triggers a ZeroDivisionError during batching logic (current behavior).
        """
        dw = DiskWrite(self.tmp, write_frequency=0)
        parser = _synthetic_parser([(1, 1)])
        with self.assertRaises(ZeroDivisionError):
            dw.write(parser)


##################################################################
# Tests for pyslyde/io/lmdb_io.py
##################################################################

@pytest.mark.lmdb
@unittest.skipUnless(HAS_LMDB, "lmdb (and lmdb_io import) not available")
class TestNpyObject_LMDB(unittest.TestCase):
    """
    Tests for LMDB NpyObject wrapper to ensure dtype and shape are preserved across serialization.
    """
    def test_roundtrip_uint8(self):
        """
        NpyObject should reconstruct the original uint8 array with identical shape and values.
        """
        arr = np.full((7, 9, 3), 42, dtype=np.uint8)
        obj = LMDBNpyObject(arr)
        back = obj.get_ndarray()
        self.assertEqual(back.dtype, arr.dtype)
        self.assertEqual(tuple(back.shape), tuple(arr.shape))
        self.assertTrue(np.array_equal(back, arr))

    def test_roundtrip_float32(self):
        """
        NpyObject should reconstruct the original float32 array with identical shape and values.
        """
        arr = np.ones((4, 5), dtype=np.float32) * 3.14
        obj = LMDBNpyObject(arr)
        back = obj.get_ndarray()
        self.assertEqual(back.dtype, arr.dtype)
        self.assertEqual(tuple(back.shape), tuple(arr.shape))
        self.assertTrue(np.allclose(back, arr))


@pytest.mark.lmdb
@unittest.skipUnless(HAS_LMDB, "lmdb (and lmdb_io import) not available")
class TestLMDBReadWrite(unittest.TestCase):
    """
    Integration tests for LMDBWrite/LMDBRead: writing tiles/images, counting keys, and reading back arrays.
    """
    def setUp(self):
        """
        Create a temporary LMDB database directory and common test coordinates.
        """
        self.tmp = tempfile.mkdtemp(prefix="pyslyde_lmdbio_")
        self.db_path = os.path.join(self.tmp, "tiles_db")
        self.coords = [(0, 0), (2, 1), (5, 3), (7, 4), (9, 8)]
        self.map_size = 8 * 1024 * 1024

    def tearDown(self):
        """
        Remove the temporary LMDB database directory and all contents.
        """
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writer_repr_and_path(self):
        """
        __repr__ should include the database path and configured map size; path should be created.
        """
        writer = LMDBWrite(self.db_path, map_size=self.map_size, write_frequency=3)
        rep = repr(writer)
        self.assertIn(self.db_path, rep)
        self.assertIn(str(self.map_size), rep)
        self.assertTrue(os.path.isdir(self.db_path))
        writer.close()

    def test_write_tiles_and_read_back(self):
        """
        Writer should store all tiles under y_x keys; reader should return correct counts, keys, and data.
        """
        writer = LMDBWrite(self.db_path, map_size=self.map_size, write_frequency=2)
        tiles = list(_synthetic_parser(self.coords, shape=(10, 10, 3), dtype=np.uint8))

        f = io.StringIO()
        with redirect_stdout(f):
            def gen():
                for item in tiles:
                    yield item
            writer.write(gen())

        out = f.getvalue()
        self.assertIn("Beginning writing to db ...", out)
        self.assertIn("Writing to db done", out)

        reader = LMDBRead(self.db_path)
        self.assertEqual(reader.num_keys, len(self.coords))
        self.assertIn(self.db_path, repr(reader))

        keys = reader.get_keys()
        self.assertEqual(len(keys), len(self.coords))
        self.assertTrue(all(isinstance(k, bytes) for k in keys))

        expected_names = {f"{y}_{x}".encode("ascii") for (x, y), _ in tiles}
        self.assertSetEqual(set(keys), expected_names)

        sample_key = next(iter(expected_names))
        arr = reader.read_image(sample_key)
        y_str, x_str = sample_key.decode("ascii").split("_")
        x, y = int(x_str), int(y_str)
        expected = next(_synthetic_parser([(x, y)], shape=(10, 10, 3), dtype=np.uint8))[1]
        self.assertEqual(tuple(arr.shape), tuple(expected.shape))
        self.assertEqual(str(arr.dtype), str(expected.dtype))
        self.assertTrue(np.array_equal(arr, expected))

    def test_write_image_and_read_back(self):
        """
        write_image should store a single image under a given key retrievable by LMDBRead.read_image.
        """
        writer = LMDBWrite(self.db_path, map_size=self.map_size, write_frequency=10)
        image = np.arange(25, dtype=np.uint16).reshape(5, 5)
        writer.write_image(image, name="custom_key")
        writer.close()

        reader = LMDBRead(self.db_path)
        keys = reader.get_keys()
        self.assertIn(b"custom_key", keys)
        back = reader.read_image(b"custom_key")
        self.assertEqual(tuple(back.shape), (5, 5))
        self.assertEqual(str(back.dtype), "uint16")
        self.assertTrue(np.array_equal(back, image))

    def test_reader_num_keys_empty_db(self):
        """
        A newly created (but unused) LMDB should report zero entries via num_keys.
        """
        writer = LMDBWrite(self.db_path, map_size=self.map_size, write_frequency=5)
        writer.close()
        reader = LMDBRead(self.db_path)
        self.assertEqual(reader.num_keys, 0)


##################################################################
# Tests for pyslyde/io/rocksdb_io.py
##################################################################

@pytest.mark.rocksdb
@unittest.skipUnless(HAS_ROCKSDB, "rocksdb (and rocksdb_io import) not available")
class TestNpyObject_RocksDB(unittest.TestCase):
    """
    Tests for RocksDB NpyObject wrapper to ensure dtype and shape are preserved across serialization.
    """
    def test_roundtrip_uint8(self):
        """
        NpyObject should reconstruct the original uint8 array with identical shape and values.
        """
        arr = np.full((6, 4, 3), 13, dtype=np.uint8)
        obj = RocksNpyObject(arr)
        back = obj.get_ndarray()
        self.assertEqual(back.dtype, arr.dtype)
        self.assertEqual(tuple(back.shape), tuple(arr.shape))
        self.assertTrue(np.array_equal(back, arr))

    def test_roundtrip_float32(self):
        """
        NpyObject should reconstruct the original float32 array with identical shape and values.
        """
        arr = (np.arange(12, dtype=np.float32) / 10.0).reshape(3, 4)
        obj = RocksNpyObject(arr)
        back = obj.get_ndarray()
        self.assertEqual(back.dtype, arr.dtype)
        self.assertEqual(tuple(back.shape), tuple(arr.shape))
        self.assertTrue(np.allclose(back, arr))


@pytest.mark.rocksdb
@unittest.skipUnless(HAS_ROCKSDB, "rocksdb (and rocksdb_io import) not available")
class TestRocksDBReadWrite(unittest.TestCase):
    """
    Integration tests for RocksDBWrite/RocksDBRead: writing tiles/images, counting keys, and reading back arrays.
    """
    def setUp(self):
        """
        Create a temporary RocksDB database directory and common test coordinates.
        """
        self.tmp = tempfile.mkdtemp(prefix="pyslyde_rocksdbio_")
        self.db_path = os.path.join(self.tmp, "rocks_db")
        self.coords = [(0, 0), (2, 1), (5, 3), (7, 4), (9, 8)]

    def tearDown(self):
        """
        Remove the temporary RocksDB database directory and all contents.
        """
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writer_repr_and_path(self):
        """
        __repr__ should include the database path, and initialization should create DB files/dir.
        """
        writer = RocksDBWrite(self.db_path, write_frequency=3)
        rep = repr(writer)
        self.assertIn(self.db_path, rep)
        self.assertTrue(os.path.isdir(self.db_path))
        writer.close()

    def test_write_tiles_and_read_back(self):
        """
        Writer should store all tiles under y_x string keys; reader should return correct counts, keys, and data.
        """
        writer = RocksDBWrite(self.db_path, write_frequency=2)

        # Pass a list (re-iterable) because write() pre-counts with sum(1 for _ in parser).
        tiles = list(_synthetic_parser(self.coords, shape=(10, 10, 3), dtype=np.uint8))
        writer.write(tiles)

        reader = RocksDBRead(self.db_path)
        self.assertEqual(reader.num_keys, len(self.coords))
        self.assertIn(self.db_path, repr(reader))

        keys = reader.get_keys()
        self.assertEqual(len(keys), len(self.coords))
        self.assertTrue(all(isinstance(k, str) for k in keys))

        expected_names = {f"{y}_{x}" for (x, y), _ in tiles}
        self.assertSetEqual(set(keys), expected_names)

        sample_key = next(iter(expected_names))
        arr = reader.read_image(sample_key)
        y_str, x_str = sample_key.split("_")
        x, y = int(x_str), int(y_str)
        expected = next(_synthetic_parser([(x, y)], shape=(10, 10, 3), dtype=np.uint8))[1]
        self.assertEqual(tuple(arr.shape), tuple(expected.shape))
        self.assertEqual(str(arr.dtype), str(expected.dtype))
        self.assertTrue(np.array_equal(arr, expected))

    def test_write_image_and_read_back(self):
        """
        write_image should store a single image under a given key retrievable by RocksDBRead.read_image.
        """
        writer = RocksDBWrite(self.db_path, write_frequency=10)
        image = (np.arange(16, dtype=np.uint16).reshape(4, 4) * 2)
        writer.write_image(image, name="custom_key")
        writer.close()

        reader = RocksDBRead(self.db_path)
        keys = reader.get_keys()
        self.assertIn("custom_key", keys)
        back = reader.read_image("custom_key")
        self.assertEqual(tuple(back.shape), (4, 4))
        self.assertEqual(str(back.dtype), "uint16")
        self.assertTrue(np.array_equal(back, image))

    def test_reader_num_keys_empty_db_and_missing_key(self):
        """
        A newly created (but unused) RocksDB should report zero keys; missing keys return None on read.
        """
        writer = RocksDBWrite(self.db_path, write_frequency=5)
        writer.close()
        reader = RocksDBRead(self.db_path)
        self.assertEqual(reader.num_keys, 0)
        self.assertIsNone(reader.read_image("does_not_exist"))


##################################################################
# Tests for pyslyde/io/tfrecords_io.py
##################################################################

@pytest.mark.tfrecords
class TestTFRecordWrite(unittest.TestCase):
    """
    Tests for TFRecordWrite: shard/size properties and TFRecord round-trip with a dummy patcher.
    """
    def setUp(self):
        """
        Create a temporary output directory and a dummy patch provider with deterministic tiles.
        """
        self.tmp = tempfile.mkdtemp(prefix="pyslyde_tfrecordsio_")

        class DummyPatch:
            """Lightweight patch-like provider exposing extract_patches(), _patches, and size."""
            def __init__(self, patches, size=(16, 16)):
                self._patches = patches  # list of (np.ndarray, {'name': str})
                self.size = size

            def extract_patches(self):
                # Each call returns a fresh generator as expected by TFRecordWrite.
                for img, info in self._patches:
                    yield img, info

        # Build 4 tiny RGB tiles with names y_x
        coords = [(0, 0), (1, 2), (2, 1), (3, 3)]
        patches = []
        for (x, y) in coords:
            img = np.full((16, 16, 3), (x + 2 * y) % 255, dtype=np.uint8)
            patches.append((img, {'name': f'{y}_{x}'}))
        self.patch = DummyPatch(patches)

    def tearDown(self):
        """
        Remove the temporary directory and all contents.
        """
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _parse_examples(self, tfrecord_path, has_mask=False):
        """
        Read back TFRecord examples into a list of dicts with decoded PNGs and fields.
        """
        feats = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'dims': tf.io.FixedLenFeature([], tf.int64),
        }
        if has_mask:
            feats['mask'] = tf.io.FixedLenFeature([], tf.string)
            feats['imageName'] = tf.io.FixedLenFeature([], tf.string)
            feats['maskName'] = tf.io.FixedLenFeature([], tf.string)
        else:
            feats['name'] = tf.io.FixedLenFeature([], tf.string)

        out = []
        for raw in tf.data.TFRecordDataset([tfrecord_path]):
            ex = tf.io.parse_single_example(raw, feats)
            if has_mask:
                img = tf.image.decode_png(ex['image'])
                msk = tf.image.decode_png(ex['mask'])
                out.append({
                    'image': img.numpy(),
                    'mask': msk.numpy(),
                    'dims': int(ex['dims'].numpy()),
                    'imageName': ex['imageName'].numpy(),
                    'maskName': ex['maskName'].numpy(),
                })
            else:
                img = tf.image.decode_png(ex['image'])
                out.append({
                    'image': img.numpy(),
                    'dims': int(ex['dims'].numpy()),
                    'name': ex['name'].numpy(),
                })
        return out

    def test_properties_and_convert_single_shard(self):
        """
        With tiny tiles, shard_number should be 1 and convert() should write a single TFRecords file.
        """
        writer = TFRecordWrite(db_path=self.tmp, patch=self.patch)
        self.assertGreater(writer.mem_size, 0)
        self.assertEqual(writer.shard_number, 1)
        self.assertEqual(writer.img_num_per_shard, len(self.patch._patches) // 1)

        writer.convert()
        rec = os.path.join(self.tmp, '0.tfrecords')
        self.assertTrue(os.path.isfile(rec))

        examples = self._parse_examples(rec, has_mask=False)
        self.assertEqual(len(examples), len(self.patch._patches))
        for ex, (_, info) in zip(examples, self.patch._patches):
            self.assertEqual(ex['dims'], self.patch.size[0])
            self.assertEqual(ex['name'], info['name'].encode('utf8'))
            self.assertEqual(tuple(ex['image'].shape[:2]), tuple(self.patch.size))


##################################################################
# Tests for pyslyde/io/tfrecord_write.py
##################################################################

@pytest.mark.tfrecord_write
class TestTFRecordWriteScript(unittest.TestCase):
    """
    Tests for tfrecord_write helpers: wrapping utilities, conversion functions, and simple end-to-end sharding.
    """
    def setUp(self):
        """
        Create a temporary dataset of small PNG images and masks plus an output directory.
        """
        self.tmp = tempfile.mkdtemp(prefix="pyslyde_tfrecordwrite_")
        self.img_dir = os.path.join(self.tmp, "imgs")
        self.msk_dir = os.path.join(self.tmp, "msks")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.msk_dir, exist_ok=True)

        # Create 3 images and matching *_masks.png masks
        self.images, self.masks = [], []
        for name in ["imgA.png", "imgB.png", "imgC.png"]:
            arr = np.full((12, 12, 3), 127, dtype=np.uint8)
            msk = np.zeros((12, 12, 1), dtype=np.uint8)
            img_path = os.path.join(self.img_dir, name)
            msk_path = os.path.join(self.msk_dir, name.replace(".png", "_masks.png"))
            # Save via PIL through tf.keras (to match loader expectations)
            tf.keras.utils.save_img(img_path, arr, scale=False)
            tf.keras.utils.save_img(msk_path, msk, scale=False)
            self.images.append(img_path)
            self.masks.append(msk_path)

        # Minimal config to split one file to validation and one to test
        self.cfg_path = os.path.join(self.tmp, "config.json")
        with open(self.cfg_path, "w") as fh:
            json.dump({"validFiles": ["imgB"], "testFiles": ["imgC"]}, fh)

        self.out_path = os.path.join(self.tmp, "tfrecords_out")
        os.makedirs(self.out_path, exist_ok=True)

    def tearDown(self):
        """
        Remove the temporary directory tree and generated files.
        """
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _collect_tfrecords(self, subdir):
        """
        Collect TFRecord files under the given subdir if present.
        """
        d = Path(self.out_path) / subdir
        return sorted([str(p) for p in d.glob("*.tfrecords")]) if d.exists() else []

    def _parse_examples(self, tfrecord_path):
        """
        Parse TFRecord examples with image/mask and return decoded numpy arrays and fields.
        """
        feats = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
            'imageName': tf.io.FixedLenFeature([], tf.string),
            'maskName': tf.io.FixedLenFeature([], tf.string),
            'dims': tf.io.FixedLenFeature([], tf.int64),
        }
        out = []
        for raw in tf.data.TFRecordDataset([tfrecord_path]):
            ex = tf.io.parse_single_example(raw, feats)
            img = tf.image.decode_png(ex['image'])
            msk = tf.image.decode_png(ex['mask'])
            out.append({
                'image': img.numpy(),
                'mask': msk.numpy(),
                'imageName': ex['imageName'].numpy(),
                'maskName': ex['maskName'].numpy(),
                'dims': int(ex['dims'].numpy()),
            })
        return out

    def test_wrap_helpers(self):
        """
        wrapInt64/wrapFloat/wrapBytes should produce tf.train.Feature objects carrying the given values.
        """
        f_i = wrapInt64(7)
        f_f = wrapFloat(3.5)
        f_b = wrapBytes(tf.constant(b"abc"))
        self.assertIsInstance(f_i, tf.train.Feature)
        self.assertEqual(list(f_i.int64_list.value), [7])
        self.assertIsInstance(f_f, tf.train.Feature)
        self.assertAlmostEqual(list(f_f.float_list.value)[0], 3.5, places=5)
        self.assertIsInstance(f_b, tf.train.Feature)
        self.assertEqual(list(f_b.bytes_list.value)[0], b"abc")

    def test_convert_writes_single_tfrecord(self):
        """
        convert() should serialize image/mask pairs into a TFRecords file that decodes correctly.
        """
        rec_path = os.path.join(self.out_path, "single.tfrecords")
        convert(self.images, self.masks, rec_path, dim=None)
        self.assertTrue(os.path.isfile(rec_path))

        exs = self._parse_examples(rec_path)
        # Some masks may be skipped if missing; ensure at least one example exists
        self.assertGreaterEqual(len(exs), 1)
        for ex in exs:
            self.assertEqual(tuple(ex['image'].shape[:2]), (12, 12))
            self.assertEqual(tuple(ex['mask'].shape[:2]), (12, 12))
            self.assertEqual(ex['dims'], 12)

    def test_doConversion_one_shard(self):
        """
        doConversion() should split into shards and call convert(), creating TFRecords under out/train.
        """
        shard_num, per = getShardNumber(self.images, self.masks, shardSize=0.1)
        self.assertGreaterEqual(shard_num, 1)
        doConversion(self.images, self.masks, shard_num, per, self.out_path, 'train')
        files = self._collect_tfrecords('train')
        self.assertGreaterEqual(len(files), 1)

    def test_getFiles_end_to_end(self):
        """
        getFiles() should read config and write TFRecords to train/validation/test subdirectories.
        """
        getFiles(self.img_dir, self.msk_dir, self.out_path, self.cfg_path, shardSize=0.1)
        self.assertGreaterEqual(len(self._collect_tfrecords('train')), 1)
        self.assertGreaterEqual(len(self._collect_tfrecords('validation')), 1)
        self.assertGreaterEqual(len(self._collect_tfrecords('test')), 1)
