"""PySlyde: A Python package for whole slide image processing and annotation."""

from pyslyde.slide import Annotations, Slide
from pyslyde.slide_parser import Stitching, WSIParser

__version__ = "0.1.0"
__author__ = "Gregory Verghese"
__email__ = "gregory.verghese@gmail.com"

__all__ = ["Annotations", "Slide", "Stitching", "WSIParser"]
