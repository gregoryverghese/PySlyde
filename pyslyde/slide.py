#!/usr/bin/env python3

"""
slide.py: Contains the Slide and Annotations classes.

Slide class: Wrapper around openslide.OpenSlide with annotation overlay and mask generation.
Annotations class: Parses annotation files from QuPath, ImageJ, and ASAP.
"""

import os
import glob
import json
import itertools
import operator as op
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Sequence
from itertools import chain

import numpy as np
import cv2
import openslide
from openslide import OpenSlide
import pandas as pd
from matplotlib.path import Path
import seaborn as sns

from pyslyde.util.utilities import mask2rgb


__author__ = 'Gregory Verghese'
__email__ = 'gregory.verghese@gmail.com'


class Slide(OpenSlide):
    """
    Whole Slide Image (WSI) object that enables annotation overlay as a wrapper around
    openslide.OpenSlide. Generates annotation mask.

    Attributes:
        mag (int): Magnification level.
        dims (tuple): Dimensions of the WSI.
        name (str): Name of the slide file.
        draw_border (bool): Whether to generate border based on annotations.
        _border (list): List of border coordinates [(x1, y1), (x2, y2)].
    """
    MAG_FACTORS: Dict[int, int] = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32}
    MASK_SIZE: Tuple[int, int] = (2000, 2000)

    def __init__(self,
                 filename: str,
                 mag: int = 0,
                 annotations: Optional['Annotations'] = None,
                 annotations_path: Optional[Union[str, List[str]]] = None,
                 labels: Optional[List[str]] = None,
                 source: Optional[str] = None) -> None:
        super().__init__(filename)
        self.mag: int = mag
        self.dims: Tuple[int, int] = self.dimensions
        self.name: str = os.path.basename(filename)
        self._border: Optional[List[Tuple[int, int]]] = None
        self.annotations: Optional['Annotations'] = None

        if annotations is not None:
            self.annotations = annotations
        elif annotations_path is not None and source is not None:
            self.annotations = Annotations(
                annotations_path,
                source=source,
                labels=labels,
                encode=True
            )

    @property
    def slide_mask(self) -> np.ndarray:
        """Get the slide mask as an RGB array."""
        mask = self.generate_mask((Slide.MASK_SIZE))
        mask = mask2rgb(mask)
        return mask

    def generate_mask(self, size: Optional[Tuple[int, int]] = None, 
                     labels: Optional[List[Union[int, str]]] = None) -> np.ndarray:
        """
        Generate a mask representation of annotations.

        Args:
            size (tuple, optional): Dimensions of the mask.
            labels (list, optional): List of labels to include in the mask.

        Returns:
            np.ndarray: Single-channel mask with integer for each class.
        """
        x, y = self.dims[0], self.dims[1]
    
        slide_mask = np.zeros((y, x), dtype=np.uint8)
        
        if self.annotations is None:
            return slide_mask
        
        self.annotations.encode = True
        coordinates = self.annotations.annotations
        if coordinates is None:
            return slide_mask
            
        keys = sorted(list(coordinates.keys()))
        if labels:
            # Convert string labels to integer keys if needed
            label_keys = []
            for l in labels:
                if isinstance(l, str) and l in self.annotations.class_key:
                    label_keys.append(self.annotations.class_key[l])
                elif isinstance(l, int):
                    label_keys.append(l)
            labels = label_keys
        else:
            labels = keys
        
        for k in keys:
            if k in labels:
                v = coordinates[k]
                v = [np.array(a) for a in v]
                cv2.fillPoly(slide_mask, v, color=(int(k),))
        
        if size is not None:
            slide_mask = cv2.resize(slide_mask, size)
    
        return slide_mask

    @staticmethod
    def resize_border(dim: int, factor: int = 1, threshold: Optional[int] = None, 
                     operator: str = '=>') -> int:
        """
        Resize and redraw annotation border. Useful to trim WSI and mask to a specific size.

        Args:
            dim (int): Dimension to resize.
            factor (int): Border increments.
            threshold (int, optional): Minimum/maximum size.
            operator (str): Threshold limit operator.

        Returns:
            int: New border dimension.
        """
        if threshold is None:
            threshold = dim

        operator_dict: Dict[str, Callable] = {'>': op.gt, '=>': op.ge, '<': op.lt, '=<': op.lt}
        op_func = operator_dict[operator]
        multiples = [factor * i for i in range(100000)]
        multiples = [m for m in multiples if op_func(m, threshold)]
        diff = list(map(lambda x: abs(dim - x), multiples))
        new_dim = multiples[diff.index(min(diff))]
        return new_dim

    def get_border(self, space: int = 100) -> List[Tuple[int, int]]:
        """
        Generate border around max/min annotation points.

        Args:
            space (int): Gap between max/min annotation point and border.

        Returns:
            list: Border dimensions [(x1, y1), (x2, y2)].
        """
        if self.annotations is None:
            self._border = [(0, self.dims[0]), (0, self.dims[1])]
        else:
            coordinates = self.annotations.annotations
            if coordinates is None:
                self._border = [(0, self.dims[0]), (0, self.dims[1])]
            else:
                coordinates = list(chain(*list(coordinates.values())))
                coordinates = list(chain(*coordinates))
                f = lambda x: (min(x) - space, max(x) + space)
                self._border = list(map(f, list(zip(*coordinates))))

        mag_factor = Slide.MAG_FACTORS[self.mag]
        f = lambda x: (int(x[0] / mag_factor), int(x[1] / mag_factor))
        self._border = list(map(f, self._border))

        return self._border

    def detect_components(self, level_dims: int = 6, num_component: Optional[int] = None, 
                         min_size: Optional[int] = None) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
        """
        Find the largest section on the slide.

        Args:
            level_dims (int): Level of downsampling.
            num_component (int, optional): Number of components to keep.
            min_size (int, optional): Minimum size of component.

        Returns:
            tuple: (List of images with contours, list of border coordinates)
        """
        new_dims = self.level_dimensions[6]
        image = np.array(self.get_thumbnail(self.level_dimensions[6]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(np.bitwise_not(gray), 9, 100, 100)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if num_component is not None:
            idx = sorted([(cv2.contourArea(c), i) for i, c in enumerate(contours)])
            contours = [contours[i] for c, i in idx]
            contours = contours[-num_component:]

        if min_size is not None:
            contours = list(map(lambda x: cv2.contourArea(x), contours))
            contours = [c for c in contours if c > min_size]

        borders: List[List[Tuple[int, int]]] = []
        components: List[np.ndarray] = []
        image_new = image.copy()
        
        for c in contours:
            if isinstance(c, (list, np.ndarray)):
                x, y, w, h = cv2.boundingRect(np.array(c))
            else:
                continue
                
            x_scale = self.dims[0] / new_dims[0]
            y_scale = self.dims[1] / new_dims[1]
            x1 = round(x_scale * x)
            x2 = round(x_scale * (x + w))
            y1 = round(y_scale * y)
            y2 = round(y_scale * (y - h))
            self._border = [(x1, x2), (y1, y2)]
            image_new = cv2.rectangle(image_new, (x, y), (x + w, y + h), (0, 255, 0), 2)
            components.append(image_new)
            borders.append([(x1, x2), (y1, y2)])

        return components, borders

    def generate_region(self,
                        mag: int = 0,
                        x: Optional[Union[int, Tuple[int, int]]] = None,
                        y: Optional[Union[int, Tuple[int, int]]] = None,
                        x_size: Optional[int] = None,
                        y_size: Optional[int] = None,
                        scale_border: bool = False,
                        factor: int = 1,
                        threshold: Optional[int] = None,
                        operator: str = '=>') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a specific region of the slide.

        Args:
            mag (int): Magnification level.
            x (int or tuple, optional): Minimum x coordinate or (x_min, x_max).
            y (int or tuple, optional): Minimum y coordinate or (y_min, y_max).
            x_size (int, optional): Width of the region.
            y_size (int, optional): Height of the region.
            scale_border (bool): Whether to resize the border.
            factor (int): Factor for resizing.
            threshold (int, optional): Threshold for resizing.
            operator (str): Operator for threshold.

        Returns:
            tuple: (Extracted region as RGB ndarray, mask)
        """
        x_min: int = 0
        y_min: int = 0
        x_max: int = 0
        y_max: int = 0
        
        if x is None:
            border = self.get_border()
            if border and len(border) >= 2:
                x, y = border[0], border[1]
            else:
                x, y = (0, self.dims[0]), (0, self.dims[1])
                
        if x is not None:
            if isinstance(x, tuple):
                if x_size is None:
                    x_min, x_max = x
                    x_size = x_max - x_min
                elif x_size is not None:
                    x_min = x[0]
                    x_max = x_min + x_size
            elif isinstance(x, int):
                x_min = x
                x_max = x + (x_size or 0)
                
        if y is not None:
            if isinstance(y, tuple):
                if y_size is None:
                    y_min, y_max = y
                    y_size = y_max - y_min
                elif y_size is not None:
                    y_min = y[0]
                    y_max = y_min + y_size
            elif isinstance(y, int):
                y_min = y
                y_max = y + (y_size or 0)

        if scale_border and x_size is not None and y_size is not None:
            x_size = Slide.resize_border(x_size, factor, threshold, operator)
            y_size = Slide.resize_border(y_size, factor, threshold, operator)
            
        if x_size is not None and (x_min + x_size) > self.dimensions[0]:
            x_size = self.dimensions[0] - x_min
        if y_size is not None and (y_min + y_size) > self.dimensions[1]:
            y_size = self.dimensions[1] - y_min

        if x_size is None or y_size is None:
            raise ValueError("x_size and y_size must be specified")

        x_size_adj = int(x_size / Slide.MAG_FACTORS[mag])
        y_size_adj = int(y_size / Slide.MAG_FACTORS[mag])
        region = self.read_region((x_min, y_min), mag, (x_size_adj, y_size_adj))
        mask = self.generate_mask()[y_min:y_min + y_size, x_min:x_min + x_size]

        return np.array(region.convert('RGB')), mask

    def save(self, path: str, size: Tuple[int, int] = (2000, 2000), mask: bool = False) -> None:
        """
        Save a thumbnail of the slide as an image file.

        Args:
            path (str): Path to save the image.
            size (tuple): Size of the thumbnail.
            mask (bool): Whether to save the mask instead of the image.
        """
        if mask:
            cv2.imwrite(path, self.slide_mask)
        else:
            image = self.get_thumbnail(size)
            image = image.convert('RGB')
            image = np.array(image)
            cv2.imwrite(path, image)


class Annotations:
    """
    Parses annotation files in XML or JSON format and returns a dictionary
    containing x, y coordinates for each region of interest (ROI).

    Args:
        path (str or list): Path(s) to annotation file(s).
        source (str): Annotation source type (e.g., 'qupath', 'imagej', 'asap').
        labels (list, optional): List of ROI names.
        encode (bool): Whether to encode labels as integers.
    """

    def __init__(self, path: Union[str, List[str]], source: str, 
                 labels: Optional[List[str]] = None, encode: bool = False) -> None:
        self.paths: List[str] = path if isinstance(path, list) else [path]
        self.source: str = source
        self.labels: Optional[List[str]] = labels
        self.encode: bool = encode
        self._annotations: Optional[Dict[Union[str, int], List[List[List[int]]]]] = None
        self._generate_annotations()

    def __repr__(self) -> str:
        if self._annotations is None:
            return "Annotations(empty)"
        numbers = [len(v) for k, v in self._annotations.items()]
        df = pd.DataFrame({"classes": self.labels or [], "number": numbers})
        return str(df)

    @property
    def keys(self) -> List[Union[str, int]]:
        if self._annotations is None:
            return []
        return list(self._annotations.keys())

    @property
    def values(self) -> List[List[List[List[int]]]]:
        if self._annotations is None:
            return []
        return list(self._annotations.values())

    @property
    def annotations(self) -> Optional[Dict[Union[str, int], List[List[List[int]]]]]:
        if self.encode:
            annotations = self.encode_keys()
            self.encode = False
        else:
            annotations = self._annotations
        return annotations

    @property
    def class_key(self) -> Dict[str, int]:
        if self.labels is None:
            self.labels = list(self._annotations.keys()) if self._annotations else []
        class_key = {l: i + 1 for i, l in enumerate(self.labels)}
        return class_key

    @property
    def numbers(self) -> Dict[str, int]:
        if self._annotations is None:
            return {}
        numbers = [len(v) for k, v in self._annotations.items()]
        return dict(zip(self.labels or [], numbers))

    def _generate_annotations(self) -> None:
        """
        Call the appropriate method for the file type and generate annotations.
        """
        self._annotations = {}
        if not isinstance(self.paths, list):
            self._paths = [self.paths]
        if self.source is not None:
            for p in self.paths:
                annotations = getattr(self, '_' + self.source)(p)
                for k, v in annotations.items():
                    if k in self._annotations:
                        self._annotations[k].append(v)
                    else:
                        self._annotations[k] = v
        if len(self.labels or []) > 0:
            self._annotations = self.filter_labels(self.labels or [])
        else:
            self.labels = list(self._annotations.keys())

    def filter_labels(self, labels: List[str]) -> Dict[Union[str, int], List[List[List[int]]]]:
        """
        Remove labels from annotations.

        Args:
            labels (list): Label list to keep.

        Returns:
            dict: Filtered annotation dictionary.
        """
        self.labels = labels
        if self._annotations is None:
            return {}
        keys = list(self._annotations.keys())
        for k in keys:
            if k not in labels:
                del self._annotations[k]
        return self._annotations

    def rename_labels(self, names: Dict[str, str]) -> None:
        """
        Rename annotation labels.

        Args:
            names (dict): Mapping from current labels to new labels.
        """
        if self._annotations is None:
            return
        for k, v in names.items():
            self._annotations[v] = self._annotations.pop(k)
        self.labels = list(self._annotations.keys())

    def encode_keys(self) -> Dict[int, List[List[List[int]]]]:
        """
        Encode labels as integer values.

        Returns:
            dict: Annotations with integer keys.
        """
        if self._annotations is None:
            return {}
        annotations = {self.class_key[k]: v for k, v in self._annotations.items()}
        return annotations

    def _imagej(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse ImageJ XML annotation files.

        Args:
            path (str): Path to the XML file.

        Returns:
            dict: Annotations dictionary.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        anns = root.findall('Annotation')
        labels = list(root.iter('Annotation'))
        labels = list(set([i.attrib['Name'] for i in labels]))
        annotations = {l: [] for l in labels}
        for i in anns:
            label = i.attrib['Name']
            instances = list(i.iter('Vertices'))
            for j in instances:
                coordinates = list(j.iter('Vertex'))
                coordinates = [[c.attrib['X'], c.attrib['Y']] for c in coordinates]
                coordinates = [[round(float(c[0])), round(float(c[1]))] for c in coordinates]
                annotations[label] = annotations[label] + [coordinates]
        return annotations

    def _asap(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse ASAP XML annotation files.

        Args:
            path (str): Path to the XML file.

        Returns:
            dict: Annotations dictionary.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        ns = root[0].findall('Annotation')
        labels = list(root.iter('Annotation'))
        labels = list(set([i.attrib['PartOfGroup'] for i in labels]))
        annotations = {l: [] for l in labels}
        for i in ns:
            coordinates = list(i.iter('Coordinate'))
            coordinates = [[float(c.attrib['X']), float(c.attrib['Y'])] for c in coordinates]
            coordinates = [[round(c[0]), round(c[1])] for c in coordinates]
            label = i.attrib['PartOfGroup']
            annotations[label] = annotations[label] + [coordinates]
        return annotations

    def _qupath(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse QuPath annotation JSON files.

        Args:
            path (str): Path to the JSON file.

        Returns:
            dict: Annotations dictionary.
        """
        annotations: Dict[str, List[List[List[int]]]] = {}
        with open(path) as json_file:
            j = json.load(json_file)
        for a in j:
            c = a['properties']['classification']['name']
            geometry = a['geometry']['type']
            coordinates = a['geometry']['coordinates']
            if c not in annotations:
                annotations[c] = []
            if geometry == "LineString":
                points = [[int(i[0]), int(i[1])] for i in coordinates]
                annotations[c].append(points)
            elif geometry == "Polygon":
                for a2 in coordinates:
                    points = [[int(i[0]), int(i[1])] for i in a2]
                    annotations[c].append(points)
            elif geometry == "MultiPolygon":
                for a2 in coordinates:
                    for a3 in a2:
                        points = [[int(i[0]), int(i[1])] for i in a3]
                        annotations[c].append(points)
        return annotations

    def _json(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse JSON annotation files with a specific structure.

        Args:
            path (str): Path to the JSON file.

        Returns:
            dict: Annotations dictionary.
        """
        with open(path) as json_file:
            json_annotations = json.load(json_file)
        
        labels = list(json_annotations.keys())
        if self.labels is None:
            self.labels = []
        self.labels.extend(labels)
        annotations = {k: [[[int(i['x']), int(i['y'])] for i in v2] 
                       for v2 in v.values()] for k, v in json_annotations.items()}
        return annotations

    def _dataframe(self, path: str) -> None:
        """
        Parse a DataFrame with a specific structure.
        """
        anns_df = pd.read_csv(path)
        anns_df.fillna('undefined', inplace=True)
        anns_df.set_index('labels', drop=True, inplace=True)
        self.labels = list(set(anns_df.index))
        annotations: Dict[str, List[List[List[int]]]] = {}
        for l in self.labels:
            coords = list(zip(anns_df.loc[l].x, anns_df.loc[l].y))
            annotations[l] = [[[int(x), int(y)] for x, y in coords]]

        self._annotations = annotations

    def _csv(self, path: str) -> Dict[str, List[List[List[int]]]]:
        """
        Parse CSV annotation files with a specific structure.

        Args:
            path (str): Path to the CSV file.

        Returns:
            dict: Annotations dictionary.
        """
        anns_df = pd.read_csv(path)
        anns_df.fillna('undefined', inplace=True)
        anns_df.set_index('labels', drop=True, inplace=True)
        labels = list(set(anns_df.index))
        annotations: Dict[str, List[List[List[int]]]] = {}
        for l in labels:
            coords = list(zip(anns_df.loc[l].x, anns_df.loc[l].y))
            annotations[l] = [[[int(x), int(y)] for x, y in coords]]

        self._annotations = annotations
        return annotations

    def df(self) -> pd.DataFrame:
        """
        Return a DataFrame of annotations.

        Returns:
            pd.DataFrame: DataFrame of annotations.
        """
        if self._annotations is None:
            return pd.DataFrame()
        labels = [[l] * len(self._annotations[l][0]) for l in self._annotations.keys()]
        labels = list(chain(*labels))
        x_values = [xi[0] for x in list(self._annotations.values()) for xi in x[0]]
        y_values = [yi[1] for y in list(self._annotations.values()) for yi in y[0]]
        df = pd.DataFrame({'labels': list(labels), 'x': x_values, 'y': y_values})

        return df

    def save(self, save_path: str) -> None:
        """
        Save annotations as a CSV file.

        Args:
            save_path (str): Path to save the annotations.
        """
        self.df().to_csv(save_path)



