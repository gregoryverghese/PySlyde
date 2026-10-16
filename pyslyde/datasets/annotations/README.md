# PySlyde example data

This folder contains the annotations of the PySlyde example dataset: a small
public slide with synthetic annotations in every supported format. The dataset
is used by the integration tests, the tutorial notebooks and the code examples
in the documentation.

| Item | Details |
|---|---|
| Slide | `CMU-1-Small-Region.svs` (Aperio SVS, 2220 × 2967 px, single level, 0.499 µm/px, 20×, 1.9 MB) |
| Slide source | OpenSlide test data – https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/ |
| Slide licence | CC0-1.0 (public domain dedication) |
| Slide SHA-256 | `ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7` |
| Annotations | Synthetic, computer-generated polygons (classes `tissue`, `epithelium`) – **not** pathologist annotations |
| Annotation licence | Same as PySlyde (MIT) |
| Regenerate | `python scripts/make_example_annotations.py` |

## Getting the slide

The slide is not stored in this repository. `pyslyde.datasets.fetch_example_slide()`
downloads it, checks it against the SHA-256 above and caches it in
`~/.cache/pyslyde` (override with `PYSLYDE_CACHE_DIR`). Later calls use the
cached copy, so the download happens only once.

You do not need to run it yourself in these cases:

- **Tests:** `pytest -m integration` downloads the slide automatically on the first
  run. See [Running the tests](../../../docs/testing.rst).
- **Tutorials:** each notebook in [`docs/examples/`](../../../docs/examples/)
  fetches the slide in its first cell.

To use the example data in your own code, call the function directly:

```python
from pyslyde.datasets import fetch_example_slide

slide_path = fetch_example_slide()  # path to the cached CMU-1-Small-Region.svs
```

## Annotations

The same polygons are written in every annotation format PySlyde reads:

| `source=` | File |
|---|---|
| `geojson` | `example.geojson` (FeatureCollection, label in `properties.label`) |
| `csv` | `example.csv` (`label, polygon_id, vertex_id, x, y`) |
| `qupath` | `example_qupath.json` (QuPath feature list, `properties.classification.name`) |
| `asap` | `example_asap.xml` |
| `imagej` | `example_imagej.xml` |
| `json` | `example_custom.json` (`{label: {polygon_id: [{x, y}, ...]}}`) |

`tissue` is the largest foreground component (Otsu threshold on saturation).
`epithelium` is the three largest haematoxylin-rich regions inside the tissue
(colour deconvolution, 65th-percentile threshold). Coordinates are level-0 pixels.
