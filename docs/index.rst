PySlyde Documentation
=====================

A comprehensive Python package for preprocessing pathology whole slide images (WSIs).

PySlyde is built as a wrapper around OpenSlide and provides powerful, user-friendly functionality for working with high-resolution pathology images, making it ideal for researchers and data scientists in the medical imaging domain.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api/index
   examples/index
   contributing
   changelog

Features
--------

* **WSI Handling**: Supports large pathology slides and other WSI formats via OpenSlide
* **Efficient Preprocessing**: Streamline tasks like cropping, resizing, and filtering at high performance
* **Annotation Support**: Easily integrate and visualize annotations from multiple formats (QuPath, ImageJ, ASAP, JSON, CSV)
* **Tiling and Patching**: Flexible tiling options for patch extraction, ideal for deep learning workflows
* **Image Metadata Extraction**: Retrieve and manage metadata from WSIs
* **Multiple Output Formats**: Save processed data to disk, LMDB, or RocksDB databases
* **Tissue Detection**: Automatic tissue region detection and masking
* **Feature Extraction**: Built-in support for extracting features from tiles using pre-trained models

Quick Installation
------------------

.. code-block:: bash

   pip install pyslyde

Quick Example
------------

.. code-block:: python

   from pyslyde import Slide, Annotations

   # Load a slide with annotations
   slide = Slide("path/to/slide.svs")
   annotations = Annotations("path/to/annotations.json", source="json")
   
   # Generate tissue mask
   mask = slide.generate_mask()
   
   # Extract a region
   region, region_mask = slide.generate_region(
       x=(1000, 2000),
       y=(1500, 2500),
       x_size=1000,
       y_size=1000
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 