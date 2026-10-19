# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import re
import sys

# Import pyslyde from the source tree (no install needed to build the docs).
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)


def _read_version() -> str:
    """Read ``__version__`` from pyslyde/__init__.py without importing it."""
    with open(os.path.join(REPO_ROOT, "pyslyde", "__init__.py"), encoding="utf-8") as f:
        m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', f.read(), re.M)
    return m.group(1) if m else "unknown"


# -- Project information -----------------------------------------------------

project = "PySlyde"
author = "Gregory Verghese and the PySlyde contributors"
copyright = f"2024-{datetime.date.today().year}, {author}"
release = _read_version()
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",  # Markdown (MyST) pages + Jupyter notebooks
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "examples/*_output",
]
root_doc = "index"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- Notebooks (myst-nb) -----------------------------------------------------
# Notebooks are rendered with their stored outputs; they are executed in CI
# (pytest --nbmake) rather than during the docs build.
nb_execution_mode = "off"
myst_heading_anchors = 3

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autoclass_content = "class"

# Heavy / optional dependencies are mocked so the API reference can be built
# without installing deep-learning frameworks (e.g. on Read the Docs).
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "timm",
    "transformers",
    "huggingface_hub",
    "tensorflow",
    "tf_keras",
    "rocksdict",
    "rasterio",
    "geopandas",
    "shapely",
    "sklearn",
    "webdataset",
    "einops",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "openslide": ("https://openslide.org/api/python/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_logo = "_static/logoV2.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "logo_only": True,
}
html_context = {
    "display_github": True,
    "github_user": "PharosKCL",
    "github_repo": "pyslyde",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
