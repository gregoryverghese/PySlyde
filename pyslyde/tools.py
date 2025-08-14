"""
Tools PySlyde.

Anthony Baptista
14/08/2025
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, shape
from skimage import measure
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from shapely.validation import make_valid
import rasterio
from rasterio.features import rasterize, shapes
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def polygons_from_mask(mask):
    """
    Convert a labeled mask into a GeoDataFrame of polygons, 
    applying contour extraction, convex hull simplification, and watershed segmentation 
    for invalid polygons.

    Args:
        mask (numpy.ndarray): 2D array where each non-zero value represents a distinct labeled region.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the extracted polygons with additional attributes:
            - geometry (shapely.geometry.Polygon): Polygon geometry of each region.
            - area (float): Polygon area in coordinate units.
            - perimeter (float): Polygon perimeter length.
            - centroid (shapely.geometry.Point): Centroid point of the polygon.

    Notes:
        - Background pixels (value 0) are ignored.
        - Invalid polygons are rasterized and split using watershed segmentation 
          based on the distance transform.
        - CRS is set to EPSG:4326 by default.
    """

    # Extract polygons for each labeled region
    polygons = []
    for region_value in np.unique(mask):
        if region_value == 0:  # skip background
            continue

        # Binary mask for this cell
        region_mask = (mask == region_value)

        # Find contours
        contours = measure.find_contours(region_mask, 0)

        for contour in contours:
            poly = Polygon([(x, y) for y, x in contour])
            if poly.is_valid and poly.area > 0:
                # Get minimal convex polygon
                poly_min = poly.convex_hull
                polygons.append(poly_min)
            
            elif poly.is_valid==False and poly.area > 0:
                resolution = 1  # cell size
                minx, miny, maxx, maxy = poly.bounds
                width = int((maxx - minx) / resolution)
                height = int((maxy - miny) / resolution)

                # Rasterize polygon
                transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
                raster = rasterize(
                    [(poly, 1)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8)

                # Compute distance transform
                distance = ndi.distance_transform_edt(raster)
                # Get coordinates of local maxima
                coordinates = peak_local_max(distance, labels=raster, min_distance=10)
                # Create an empty marker array
                markers = np.zeros_like(distance, dtype=int)
                # Label each peak with a unique integer
                for i, coord in enumerate(coordinates, 1):
                    markers[coord[0], coord[1]] = i

                # Apply watershed
                labels = watershed(-distance, markers, mask=raster)
                for geom, value in shapes(labels.astype(np.int32), mask=(labels>0), transform=transform):
                    polygons.append(shape(geom))

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326") 
    gdf['area'] = gdf.area
    gdf['perimeter'] = gdf.length
    gdf['centroid'] = gdf.centroid
    
    return gdf


def preprocess_mask(mask):
    """
    Preprocess a mask by extracting a specific class and filling internal holes.

    Args:
        mask (numpy.ndarray): 2D array representing a categorical mask.
                              Pixels with value 2 are considered the target class.

    Returns:
        numpy.ndarray: Binary mask (0 and 1) of the target class after hole filling.

    Notes:
        - The function extracts only pixels with value 2.
        - Internal holes within the target regions are filled using a binary morphological operation.
    """
    
    binary_mat = (mask==2).astype(np.uint8)
    binary_mat = binary_fill_holes(binary_mat).astype(np.uint8)

    return binary_mat