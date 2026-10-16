Installation
============

.. note::

   The dependency list is defined in ``pyproject.toml``, which is the single
   source of truth; this page does not repeat it. Optional extras are listed
   below.

Requirements
------------

* Python (see ``requires-python`` in ``pyproject.toml``)
* The **OpenSlide C library**. The ``openslide-python`` package installed
  with PySlyde is only a binding and needs the native library alongside it.
  Choose one of the options below.

OpenSlide native library
^^^^^^^^^^^^^^^^^^^^^^^^

Any platform, via pip (recommended):

.. code-block:: bash

   pip install openslide-bin

Ubuntu / Debian:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install libopenslide0

macOS (Homebrew):

.. code-block:: bash

   brew install openslide

Windows: use ``openslide-bin`` (above), or see the
`OpenSlide download page <https://openslide.org/download/>`_.

Install PySlyde
---------------

From PyPI:

.. code-block:: bash

   pip install pyslyde

From source:

.. code-block:: bash

   git clone https://github.com/PharosKCL/pyslyde.git
   cd pyslyde
   pip install -e .

Optional extras (see ``[project.optional-dependencies]`` in ``pyproject.toml``):

.. code-block:: bash

   pip install -e ".[feature-extractor]"   # deep-learning feature extraction (torch, timm, ...)
   pip install -e ".[tensorflow]"          # TensorFlow models / TFRecords output
   pip install -e ".[geospatial]"          # pyslyde.tools (geopandas, shapely, rasterio)
   pip install -e ".[rocksdb]"             # RocksDB output (rocksdict)
   pip install -e ".[all]"                 # all of the above
   pip install -e ".[dev]"                 # tests and linting (pytest, ruff, mypy, ...)
   pip install -e ".[docs]"                # build this documentation

The :doc:`quickstart` and :doc:`tutorials <examples/index>` use feature
extraction, so install ``".[all]"`` (or at least ``".[feature-extractor]"``)
to run them end to end.

Check the installation
----------------------

.. code-block:: python

   import openslide
   import pyslyde

   print(pyslyde.__version__)
   print(openslide.__library_version__)   # fails if the C library is missing

Then run the :doc:`quickstart`, which downloads a small example slide and
exercises the main workflow.

Troubleshooting
---------------

``OSError: ... libopenslide`` / ``ModuleNotFoundError: openslide``
   The OpenSlide C library is missing. Install it with one of the options
   above.

``libGL.so.1: cannot open shared object file`` (Linux servers / containers)
   Install ``libgl1``, or replace ``opencv-python`` with
   ``opencv-python-headless``.

GPU / CUDA
   Install the PyTorch build that matches your CUDA version
   (https://pytorch.org/get-started/locally/) before installing PySlyde.

Getting help
------------

Please open an issue on
`GitHub <https://github.com/PharosKCL/pyslyde/issues>`_ with your operating
system, Python version, the command you ran and the full error message.
