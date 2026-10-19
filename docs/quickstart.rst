Quick Start
===========

This page shows the shortest path from a whole-slide image to a set of tiles
ready for model training, on the :doc:`example dataset <example_data>`. All
code blocks on this page run in order, as written, in a fresh Python session
(they are executed by the test suite, see :doc:`testing`). For a complete,
explained walkthrough of everything else PySlyde can do, see the
:doc:`tutorial notebook <examples/pyslyde_tutorial>`.

Get the example data
--------------------

.. code-block:: python

   from pyslyde.datasets import example_data_dir

   data = example_data_dir()   # downloads the slide once (1.9 MB), then uses the cache
   slide_path = str(data / "wsi" / "CMU-1-Small-Region.svs")
   annotation_path = str(data / "annotations" / "example.geojson")

To use your own data, replace ``slide_path`` and ``annotation_path`` with the
paths to your files.

Load a slide and its annotations
--------------------------------

:class:`~pyslyde.slide.Slide` is a subclass of :class:`openslide.OpenSlide`.
Annotations can be read from QuPath, ImageJ, ASAP, GeoJSON, CSV or a custom
JSON format, selected with ``source``:

.. code-block:: python

   from pyslyde import Slide

   slide = Slide(slide_path, annotations_path=annotation_path, source="geojson")
   print(slide.dims)                         # (2220, 2967) - level-0 (width, height)
   print(slide.annotations.class_key)        # {'epithelium': 1, 'tissue': 2}

Tile the annotated area
-----------------------

:class:`~pyslyde.slide_parser.WSIParser` cuts the slide into tiles inside a
border, here the bounding box of the annotations. Bright, mostly-background
tiles are then dropped with :func:`pyslyde.util.filters.tile_intensity`:

.. code-block:: python

   from pyslyde import WSIParser
   from pyslyde.util import filters

   parser = WSIParser(slide=slide, tile_dim=256, border=slide.get_border(), level=0)
   print(parser.tiler(stride=256), "tiles")

   parser.filter_by_func(filters.tile_intensity, threshold=220)
   print(parser.number, "tiles after removing background")

Save the results
----------------

Save the tiles as PNG files, named by their level-0 coordinates, and the
annotation mask with its metadata:

.. code-block:: python

   parser.save_tiles(tile_path="quickstart_output/tiles")
   slide.save("quickstart_output/slide", size=(555, 742), overwrite=True)

Next steps
----------

* :doc:`examples/index` - tutorials covering tissue detection, masks and
  regions, tile filtering, stain normalisation, feature extraction, other
  output formats (LMDB, RocksDB) and stitching
* :doc:`api/index` - full API reference
* :doc:`example_data` - what the example dataset contains and how it is built
