PySlyde
=======

PySlyde is a lightweight Python toolkit for preprocessing pathology whole
slide images (WSIs). It wraps `OpenSlide <https://openslide.org/>`_ and adds
annotation parsing, mask generation, tissue detection, tiling, stain
normalisation and feature extraction with pathology foundation models.

Features
--------

* **WSI handling**: every slide format OpenSlide supports
* **Annotations**: QuPath, ImageJ, ASAP, GeoJSON, CSV and a custom JSON format
* **Masks and regions**: annotation masks at any size or pyramid level, plus
  aligned region and mask extraction
* **Tissue detection**: automatic tissue masks and borders
* **Tiling**: configurable tile size, stride and level; filtering by mask or
  any function
* **Stain normalisation**: Macenko, Reinhard and Vahadane
* **Feature extraction**: torchvision CNNs and pathology foundation models
  (UNI, Virchow, Prov-GigaPath, H-optimus, Phikon, ...)
* **Output formats**: disk, LMDB, RocksDB and TFRecords

Quick example
-------------

.. code-block:: python

   from pyslyde import Slide
   from pyslyde.datasets import example_data_dir

   data = example_data_dir()
   slide = Slide(
       str(data / "wsi" / "CMU-1-Small-Region.svs"),
       annotations_path=str(data / "annotations" / "example.geojson"),
       source="geojson",
   )
   mask = slide.generate_mask(size=(555, 742))
   region, region_mask = slide.generate_region(level=0, x=(600, 1600), y=(1200, 2200))

See the :doc:`quickstart` for a short example and the :doc:`tutorials <examples/index>` for the
full workflow.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   example_data
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   testing
   contributing

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
