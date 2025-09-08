"""PySlyde: A Python package for whole slide image processing and annotation."""

from pyslyde.slide import Slide, Annotations
from pyslyde.slide_parser import WSIParser, Stitching

__version__ = "0.1.0"
__author__ = "Gregory Verghese"
__email__ = "gregory.verghese@gmail.com"

__all__ = ["Slide", "Annotations", "WSIParser", "Stitching"]



