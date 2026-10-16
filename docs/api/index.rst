API Reference
=============

The reference is generated from the docstrings in the source code, one page
per module.

Core
----

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pyslyde.slide
   pyslyde.slide_parser
   pyslyde.exceptions
   pyslyde.datasets

Masks
-----

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pyslyde.masks.base
   pyslyde.masks.level0
   pyslyde.masks.mapped

Stain normalisation
-------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pyslyde.normalization.base
   pyslyde.normalization.factory
   pyslyde.normalization.macenko
   pyslyde.normalization.reinhard
   pyslyde.normalization.vahadane

Feature extraction
------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pyslyde.encoders.feature_extractor

Input / output
--------------

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pyslyde.io.disk_io
   pyslyde.io.lmdb_io
   pyslyde.io.rocksdb_io
   pyslyde.io.tfrecords_io
   pyslyde.io.tfrecord_write

Utilities
---------

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pyslyde.util.utilities
   pyslyde.util.filters
   pyslyde.util.pca
   pyslyde.tools
