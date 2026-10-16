Running the tests
=================

Install the development extras, which include ``pytest`` and ``nbmake``, then
run the tests from the repository root:

.. code-block:: bash

   pip install -e ".[all,dev]"

Common commands
---------------

.. code-block:: bash

   # 1. Fast: unit tests only (mocked, no network)
   pytest -m "not integration and not models"

   # 2. Everything that needs no special set-up or access: unit, integration
   #    and open-model tests (network needed on the first run)
   RUN_NETWORK_TESTS=1 pytest -m "not gated and not bigmem"

   # 3. Everything: also gated and very large models, plus the tutorial notebooks
   RUN_NETWORK_TESTS=1 HUGGINGFACE_TOKEN=<token> pytest
   pytest --nbmake docs/examples/*.ipynb

With the example dataset available, ``pytest -m integration`` is expected to
report **0 skipped** tests. Optional backends that are not installed (e.g.
``rocksdict`` for the RocksDB writer) are skipped with a reason.

Test markers
------------

Tests are grouped with pytest markers. Unmarked tests are unit tests: they are
mocked, fast and need nothing. Combine markers with ``-m`` to choose what to
run.

.. list-table::
   :header-rows: 1
   :widths: 14 48 38

   * - Marker
     - What it covers
     - When it runs
   * - ``integration``
     - End-to-end tests on the :doc:`example dataset <example_data>`: every
       annotation format, masks, regions, saving, component detection, and
       the :doc:`quickstart` code blocks.
     - Always selected. Needs network on the first run only (1.9 MB slide
       plus torchvision ResNet-18 weights), then cached. See
       `Integration data`_.
   * - ``models``
     - Real feature-extractor weights and forward passes (ResNet, VGG, ...).
     - Skipped unless ``RUN_NETWORK_TESTS=1``. Needs network and a few GB of
       disk.
   * - ``gated``
     - Subset of ``models``: gated Hugging Face models.
     - Also needs ``HUGGINGFACE_TOKEN``, unless the weights are already in the
       local Hugging Face cache.
   * - ``bigmem``
     - Subset of ``models``: very large models.
     - Skipped unless enough RAM/VRAM is available (24 GB by default).

Environment variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Effect
   * - ``RUN_NETWORK_TESTS=1``
     - Enables the ``models`` tests (downloads model weights).
   * - ``HUGGINGFACE_TOKEN``
     - Access token for ``gated`` models.
   * - ``MIN_CPU_AVAIL_GB``, ``MIN_FREE_VRAM_GB``
     - Memory needed before a ``bigmem`` model is loaded (default 24).
   * - ``PYSLYDE_IT_STRICT=1``
     - Integration tests fail, instead of being skipped, when the data cannot
       be obtained. Continuous integration runs in strict mode.
   * - ``PYSLYDE_CACHE_DIR``
     - Cache location for the example slide (default ``~/.cache/pyslyde``).
   * - ``PYSLYDE_IT_DATA_DIR``, ``PYSLYDE_IT_DATA_URL``,
       ``PYSLYDE_IT_DATA_SHA256``
     - Run the integration tests on other data (see below).

Integration data
----------------

By default the integration tests use the :doc:`example dataset
<example_data>`, which is downloaded automatically on the first run. Nothing
needs to be configured.

Other data can be supplied as a directory (or ``.zip`` / ``.tar.gz``)
containing ``wsi/`` and ``annotations/``:

.. code-block:: bash

   PYSLYDE_IT_DATA_DIR=/path/to/data pytest -m integration
   PYSLYDE_IT_DATA_URL=https://.../data.zip PYSLYDE_IT_DATA_SHA256=<sha256> pytest -m integration

A checksum is mandatory for remote archives. If the data cannot be obtained
(for example, no network on the first run), the integration tests are
skipped with the reason, unless ``PYSLYDE_IT_STRICT=1`` is set.

Tutorial notebooks
------------------

``pytest --nbmake docs/examples/*.ipynb`` executes the tutorial notebooks end
to end on the example dataset.
