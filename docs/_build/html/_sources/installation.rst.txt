Installation
===========

PySlyde can be installed from PyPI or from source. This guide covers all installation methods and requirements.

Requirements
-----------

PySlyde requires Python 3.8 or higher and the following dependencies:

* `numpy <https://numpy.org/>`_ >= 1.20.0
* `opencv-python <https://opencv.org/>`_ >= 4.5.0
* `openslide-python <https://openslide.org/>`_ >= 1.1.0
* `pandas <https://pandas.pydata.org/>`_ >= 1.3.0
* `matplotlib <https://matplotlib.org/>`_ >= 3.3.0
* `seaborn <https://seaborn.pydata.org/>`_ >= 0.11.0
* `scikit-image <https://scikit-image.org/>`_ >= 0.18.0
* `scipy <https://scipy.org/>`_ >= 1.7.0
* `lmdb <https://lmdb.readthedocs.io/>`_ >= 1.2.0
* `einops <https://einops.rocks/>`_ >= 0.4.0
* `h5py <https://www.h5py.org/>`_ >= 3.1.0
* `huggingface_hub <https://huggingface.co/docs/huggingface_hub/>`_ >= 0.10.0
* `tensorflow <https://tensorflow.org/>`_ >= 2.8.0
* `torch <https://pytorch.org/>`_ >= 1.10.0
* `timm <https://github.com/huggingface/pytorch-image-models>`_ >= 0.6.0
* `webdataset <https://github.com/webdataset/webdataset>`_ >= 0.2.0

Installation from PyPI
---------------------

The easiest way to install PySlyde is using pip:

.. code-block:: bash

   pip install pyslyde

This will install PySlyde and all its dependencies automatically.

Installation from Source
-----------------------

To install from source, clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/gregoryverghese/PySlide.git
   cd PySlide
   pip install -e .

Development Installation
------------------------

For development work, install with development dependencies:

.. code-block:: bash

   git clone https://github.com/gregoryverghese/PySlide.git
   cd PySlide
   pip install -e ".[dev]"

This includes additional tools for development:

* `pytest <https://pytest.org/>`_ - Testing framework
* `pytest-cov <https://pytest-cov.readthedocs.io/>`_ - Coverage reporting
* `black <https://black.readthedocs.io/>`_ - Code formatting
* `flake8 <https://flake8.pycqa.org/>`_ - Linting
* `mypy <https://mypy.readthedocs.io/>`_ - Type checking
* `pre-commit <https://pre-commit.com/>`_ - Git hooks

System Dependencies
------------------

Some dependencies may require system-level libraries:

Ubuntu/Debian:
^^^^^^^^^^^^^

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install libopenslide-dev libgl1-mesa-glx libglib2.0-0

macOS:
^^^^^^

.. code-block:: bash

   brew install openslide

Windows:
^^^^^^^^

For Windows, most dependencies are available as pre-compiled wheels. If you encounter issues with OpenSlide, you may need to install it manually from the `OpenSlide website <https://openslide.org/download/>`_.

Verifying Installation
---------------------

To verify that PySlyde is installed correctly, run:

.. code-block:: python

   import pyslyde
   print(pyslyde.__version__)

You should see the version number printed without any errors.

Troubleshooting
--------------

Common Installation Issues
^^^^^^^^^^^^^^^^^^^^^^^^^

1. **OpenSlide not found**: Make sure you have the system-level OpenSlide library installed.

2. **CUDA issues**: If you're using GPU acceleration, ensure you have the correct CUDA version installed for your PyTorch/TensorFlow version.

3. **Memory issues**: PySlyde works with large images, so ensure you have sufficient RAM (8GB+ recommended).

4. **Permission errors**: On some systems, you may need to use `pip install --user` or install with sudo.

Getting Help
-----------

If you encounter installation issues:

1. Check the `GitHub issues <https://github.com/gregoryverghese/PySlide/issues>`_ for similar problems
2. Create a new issue with your system details and error messages
3. Contact the maintainer at gregory.verghese@gmail.com 