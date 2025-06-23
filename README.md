# PySlyde

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/pyslyde.svg)](https://badge.fury.io/py/pyslyde)

PySlyde is a comprehensive Python package for preprocessing pathology whole slide images (WSIs). Built as a wrapper around OpenSlide, it provides powerful, user-friendly functionality for working with high-resolution pathology images, making it ideal for researchers and data scientists in the medical imaging domain.

## Features

- **WSI Handling**: Supports large pathology slides and other WSI formats via OpenSlide
- **Efficient Preprocessing**: Streamline tasks like cropping, resizing, and filtering at high performance
- **Annotation Support**: Easily integrate and visualize annotations from multiple formats (QuPath, ImageJ, ASAP, JSON, CSV)
- **Tiling and Patching**: Flexible tiling options for patch extraction, ideal for deep learning workflows
- **Image Metadata Extraction**: Retrieve and manage metadata from WSIs
- **Multiple Output Formats**: Save processed data to disk, LMDB, or RocksDB databases
- **Tissue Detection**: Automatic tissue region detection and masking
- **Feature Extraction**: Built-in support for extracting features from tiles using pre-trained models

## Installation

### From PyPI

```bash
pip install pyslyde
```

### From Source

```bash
git clone https://github.com/gregoryverghese/PySlide.git
cd PySlide
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/gregoryverghese/PySlide.git
cd PySlide
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from pyslyde import Slide, Annotations

# Load a slide
slide = Slide("path/to/your/slide.svs")

# Load annotations
annotations = Annotations("path/to/annotations.json", source="json")

# Create slide with annotations
slide_with_annotations = Slide(
    "path/to/your/slide.svs",
    annotations=annotations
)

# Generate mask
mask = slide_with_annotations.generate_mask()

# Extract a region
region, region_mask = slide_with_annotations.generate_region(
    x=(1000, 2000),
    y=(1500, 2500),
    x_size=1000,
    y_size=1000
)
```

### Tiling and Feature Extraction

```python
from pyslyde import WSIParser

# Create parser
parser = WSIParser(
    slide=slide,
    tile_dim=256,
    border=slide.get_border(),
    mag_level=0
)

# Generate tiles
num_tiles = parser.tiler(stride=128)

# Extract features
for coords, features in parser.extract_features(
    model_name="resnet50",
    model_path="path/to/model.pth"
):
    print(f"Tile {coords}: {features.shape}")

# Save tiles to disk
parser.save(
    parser.extract_tiles(),
    tile_path="output/tiles/"
)
```

### Tissue Detection

```python
from pyslyde.util.utilities import TissueDetect

# Detect tissue regions
detector = TissueDetect("path/to/slide.svs")
tissue_mask = detector.detect_tissue()

# Get tissue border
border = detector.border()

# Visualize tissue regions
thumbnail = detector.tissue_thumbnail
```

## Documentation

For detailed documentation, please visit the [PySlyde documentation](https://github.com/gregoryverghese/PySlide#readme).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Gregory Verghese
- **Email**: gregory.verghese@gmail.com
- **Project Link**: [https://github.com/gregoryverghese/PySlide](https://github.com/gregoryverghese/PySlide)

## Citation

If you use PySlyde in your research, please cite:

```bibtex
@software{pyslyde2024,
  title={PySlyde: A Python package for preprocessing pathology whole slide images},
  author={Verghese, Gregory},
  year={2024},
  url={https://github.com/gregoryverghese/PySlide}
}
```






