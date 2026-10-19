Example data
============

PySlyde ships with a small example dataset that the :doc:`quickstart`, the
:doc:`tutorial notebooks <examples/index>` and the integration tests all use,
so every documented workflow can be reproduced from a fresh installation.

.. list-table::
   :widths: 25 75

   * - Slide
     - ``CMU-1-Small-Region.svs``: Aperio SVS, 2220 × 2967 px, a single
       pyramid level, 0.499 µm/px (20×), 1.9 MB.
   * - Source
     - `OpenSlide test data <https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/>`_
   * - Licence
     - CC0-1.0 (public domain dedication)
   * - SHA-256
     - ``ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7``
   * - Annotations
     - Synthetic, computer-generated ``tissue`` and ``epithelium`` polygons,
       shipped with PySlyde (MIT) in all six supported formats. They are
       **not** pathologist annotations.

Using it
--------

.. code-block:: python

   from pyslyde.datasets import example_data_dir, fetch_example_slide

   slide_path = fetch_example_slide()   # just the slide
   data = example_data_dir()            # slide + annotations, laid out as below

``example_data_dir()`` returns a directory laid out as::

   wsi/
       CMU-1-Small-Region.svs
   annotations/
       example.geojson        source="geojson"
       example.csv            source="csv"
       example_qupath.json    source="qupath"
       example_asap.xml       source="asap"
       example_imagej.xml     source="imagej"
       example_custom.json    source="json"

How it works
------------

* The slide is **not** stored in the PySlyde repository. On first use it is
  downloaded from the OpenSlide server, checked against the pinned SHA-256
  above and cached in ``~/.cache/pyslyde``. A download that does not match the
  checksum is rejected. Later calls reuse the cached copy, with no network
  access.
* The annotations are small text files in ``pyslyde/datasets/annotations``.
  They were derived from the slide with ``scripts/make_example_annotations.py``:

  - ``tissue`` is the largest foreground component (Otsu threshold on
    saturation).
  - ``epithelium`` is the three largest haematoxylin-rich regions inside the
    tissue (colour deconvolution).

  Re-running the script regenerates all six files from the same polygons.

Environment variables
---------------------

``PYSLYDE_CACHE_DIR``
   Cache location (default ``~/.cache/pyslyde`` or ``$XDG_CACHE_HOME/pyslyde``).
``PYSLYDE_DATA_DIR``
   Use an existing directory with the layout above instead of the example
   data. This is useful offline or to run the tutorials on your own slides.
