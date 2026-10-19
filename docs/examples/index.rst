Tutorials
=========

The tutorials are Jupyter notebooks that run end to end on the
:doc:`example dataset <../example_data>`, with no configuration. Download them
from the `docs/examples folder on GitHub
<https://github.com/PharosKCL/pyslyde/tree/main/docs/examples>`_, or open the
rendered versions below.

To run them on your own slides, set ``PYSLYDE_DATA_DIR`` to a folder that
contains ``wsi/`` and ``annotations/`` before starting Jupyter.

.. toctree::
   :maxdepth: 1

   pyslyde_tutorial
   geopandas_polygons

The notebooks are executed in continuous integration (``pytest --nbmake``), so
they stay in step with the API.
