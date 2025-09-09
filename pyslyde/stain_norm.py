"""
    Stain Normalisation using Macenko, Vahadane and Reinhard
    Code inspired by staintools and tiatoolbox

    author: Holly Rafique
    date: 09/09/2025
"""
import numpy as np
try:
    import cv2  # optional; only used in helpers
except Exception:
    cv2 = None

class StainNormalizer:


    def __init__(
        self,
        method: str = 'macenko'
    ):
        self.method = method

    # ---------- Public API ----------
    
    def fit(self, target_rgb: np.ndarray) -> "StainNormalizer":
        """Learn target stain matrix and robust concentration percentiles from a target RGB image."""

        return self

    def normalize(self, source_rgb: np.ndarray) -> "StainNormalizer":
        """Transform towards the target image."""
        return self
    

