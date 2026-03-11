import os
import sys

import numpy as np
import tensorflow as tf


class TFRecordWrite:
    def __init__(self, db_path, patch, shard_size=0.01, unit=10**9):

        self.db_path = db_path
        self.patch = patch
        self.shard_size = shard_size
        self.unit = unit

    def _print_progress(self, i):
        complete = float(i) / self.img_num_per_shard
        print(f"\r- Progress: {complete:.1%}", end="\r")

    @staticmethod
    def _wrap_int64(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _wrap_bytes(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @property
    def mem_size(self):
        mem = sum(sys.getsizeof(p.tobytes()) for p, _ in self.patch.extract_patches())
        return mem / self.unit

    @property
    def shard_number(self):
        return int(np.ceil(self.mem_size / self.shard_size))

    @property
    def img_num_per_shard(self):
        return int(np.floor(len(self.patch._patches) / self.shard_number))

    def convert(self):
        it = iter(self.patch.extract_patches())
        for i in range(self.shard_number):
            path = os.path.join(self.db_path, str(i) + ".tfrecords")
            writer = tf.io.TFRecordWriter(path)
            for j in range(self.img_num_per_shard):
                image, p = next(it)
                self._print_progress(j)
                image = tf.image.encode_png(image)

                data = {
                    "image": self._wrap_bytes(image),
                    "name": self._wrap_bytes(p["name"].encode("utf8")),
                    "dims": self._wrap_int64(self.patch.size[0]),
                }

                features = tf.train.Features(feature=data)
                example = tf.train.Example(features=features)
                serialized = example.SerializeToString()
                writer.write(serialized)
