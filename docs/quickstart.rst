Quick Start Guide
================

This guide will help you get started with PySlyde quickly. We'll cover the basic usage patterns and common workflows.

Basic Usage
----------

Loading a Slide
^^^^^^^^^^^^^^

The core of PySlyde is the `Slide` class, which wraps OpenSlide functionality:

.. code-block:: python

   from pyslyde import Slide

   # Load a whole slide image
   slide = Slide("path/to/your/slide.svs")
   
   # Access basic properties
   print(f"Slide dimensions: {slide.dims}")
   print(f"Slide name: {slide.name}")
   print(f"Number of levels: {slide.level_count}")

Working with Annotations
^^^^^^^^^^^^^^^^^^^^^^^

PySlyde supports multiple annotation formats:

.. code-block:: python

   from pyslyde import Annotations

   # Load annotations from different sources
   qupath_annotations = Annotations("path/to/qupath.json", source="qupath")
   imagej_annotations = Annotations("path/to/imagej.xml", source="imagej")
   asap_annotations = Annotations("path/to/asap.xml", source="asap")
   json_annotations = Annotations("path/to/annotations.json", source="json")
   csv_annotations = Annotations("path/to/annotations.csv", source="csv")

   # Create slide with annotations
   slide_with_annotations = Slide(
       "path/to/slide.svs",
       annotations=qupath_annotations
   )

Generating Masks
^^^^^^^^^^^^^^^

Create masks from annotations:

.. code-block:: python

   # Generate a mask with default size (2000x2000)
   mask = slide_with_annotations.generate_mask()
   
   # Generate a mask with custom size
   mask = slide_with_annotations.generate_mask(size=(1000, 1000))
   
   # Generate a mask for specific labels
   mask = slide_with_annotations.generate_mask(labels=["tumor", "stroma"])

Extracting Regions
^^^^^^^^^^^^^^^^^

Extract specific regions from the slide:

.. code-block:: python

   # Extract a region with specified coordinates and size
   region, region_mask = slide_with_annotations.generate_region(
       x=(1000, 2000),  # x range
       y=(1500, 2500),  # y range
       x_size=1000,     # width
       y_size=1000      # height
   )

Tiling and Feature Extraction
----------------------------

Creating Tiles
^^^^^^^^^^^^^

Use the `WSIParser` for advanced tiling operations:

.. code-block:: python

   from pyslyde import WSIParser

   # Create a parser
   parser = WSIParser(
       slide=slide_with_annotations,
       tile_dim=256,  # tile size
       border=slide_with_annotations.get_border(),
       mag_level=0    # magnification level
   )

   # Generate tiles
   num_tiles = parser.tiler(stride=128)
   print(f"Generated {num_tiles} tiles")

Extracting Features
^^^^^^^^^^^^^^^^^^

Extract features from tiles using pre-trained models:

.. code-block:: python

   # Extract features from all tiles
   for coords, features in parser.extract_features(
       model_name="resnet50",
       model_path="path/to/model.pth"
   ):
       print(f"Tile {coords}: {features.shape}")

Saving Results
^^^^^^^^^^^^^

Save tiles and features in different formats:

.. code-block:: python

   # Save tiles to disk
   parser.save(
       parser.extract_tiles(),
       tile_path="output/tiles/",
       label_dir=True,
       label_csv=True
   )

   # Save to LMDB database
   parser.to_lmdb(
       parser.extract_tiles(),
       db_path="output/tiles.lmdb",
       map_size=1024*1024*1024  # 1GB
   )

Tissue Detection
---------------

Automatic tissue detection:

.. code-block:: python

   from pyslyde.util.utilities import TissueDetect

   # Detect tissue regions
   detector = TissueDetect("path/to/slide.svs")
   tissue_mask = detector.detect_tissue()
   
   # Get tissue border
   border = detector.border()
   
   # Visualize tissue regions
   thumbnail = detector.tissue_thumbnail

Image Filtering
--------------

Apply filters to remove unwanted regions:

.. code-block:: python

   from pyslyde import filters

   # Remove black patches
   filtered_patches = filters.remove_black(
       patches,
       threshold=60,
       area_thresh=0.2
   )

   # Remove blue patches (staining artifacts)
   filtered_patches = filters.remove_blue(
       patches,
       area_thresh=0.2
   )

Complete Example
---------------

Here's a complete example that demonstrates a typical workflow:

.. code-block:: python

   from pyslyde import Slide, Annotations, WSIParser
   from pyslyde.util.utilities import TissueDetect

   # 1. Load slide and annotations
   slide = Slide("path/to/slide.svs")
   annotations = Annotations("path/to/annotations.json", source="json")
   slide_with_annotations = Slide("path/to/slide.svs", annotations=annotations)

   # 2. Detect tissue regions
   detector = TissueDetect("path/to/slide.svs")
   tissue_mask = detector.detect_tissue()
   border = detector.border()

   # 3. Create parser for tiling
   parser = WSIParser(
       slide=slide_with_annotations,
       tile_dim=256,
       border=border,
       mag_level=0
   )

   # 4. Generate tiles
   num_tiles = parser.tiler(stride=128)
   print(f"Generated {num_tiles} tiles")

   # 5. Extract features
   for coords, features in parser.extract_features(
       model_name="resnet50",
       model_path="path/to/model.pth"
   ):
       print(f"Processed tile {coords}")

   # 6. Save results
   parser.save(
       parser.extract_tiles(),
       tile_path="output/tiles/",
       label_dir=True
   )

Next Steps
----------

Now that you have the basics, you can explore:

* :doc:`user_guide/index` - Detailed user guide
* :doc:`api/index` - Complete API reference
* :doc:`examples/index` - More examples and tutorials

For more advanced usage patterns and best practices, see the :doc:`user_guide/index`. 