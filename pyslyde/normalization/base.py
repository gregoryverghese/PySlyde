"""
    Stain Normalisation abstract base class
    Code inspired by staintools and tiatoolbox TODO - add references here

    author: Holly Rafique
    date: 09/09/2025
"""
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

try:
    import cv2  # optional, only for load/save helpers
except Exception:
    cv2 = None
    

class StainNormalizer(ABC):
    """
    Abstract base for stain normalization algorithms.

    Subclasses must implement:
      - fit(target_tile)
      - normalize(source_tile)
      - get_profile() / set_profile(profile)  (for serialization)
    """

    @abstractmethod
    def fit(self, target_tile: np.ndarray) -> "StainNormalizer":
        """Learn target-specific parameters (e.g., stain basis, stats)."""
        raise NotImplementedError

    @abstractmethod
    def normalize(self, source_tile: np.ndarray) -> np.ndarray:
        """Normalize a source tile using learned target parameters."""
        raise NotImplementedError


    # ----- Serialization hooks -----

    @abstractmethod
    def get_profile(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of learned parameters."""
        raise NotImplementedError

    @abstractmethod
    def set_profile(self, profile: Dict[str, Any]) -> None:
        """Load learned parameters from a dict."""
        raise NotImplementedError

    def save_profile(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_profile(), f)

    def load_profile(self, path: str) -> "StainNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            prof = json.load(f)
        self.set_profile(prof)
        return self

    # ----- Shared small utilities -----

    @staticmethod
    def rgb2od(I: np.ndarray, I0: float | None = None) -> np.ndarray:
        I = I.astype(np.float32, copy=False)
        # Auto-select I0 if not provided: 1.0 for float images in [0,1], else 255.0
        if I0 is None:
            I0 = 1.0 if I.max() <= 1.0 + 1e-6 else 255.0
        # epsilon relative to I0 to avoid log(0) without crushing dynamic range
        eps = np.finfo(np.float32).eps * I0 * 10.0  # ~1e-6 of I0
        I_clamped = np.clip(I, eps, I0)
        return -np.log(I_clamped / I0)

    @staticmethod
    def od2rgb(OD: np.ndarray, I0: float = 255.0) -> np.ndarray:
        I = I0 * np.exp(-OD)
        # If I0==1.0 (float image), keep float output; else return uint8
        if I0 <= 1.0 + 1e-6:
            return np.clip(I, 0.0, 1.0).astype(np.float32)
        return np.clip(I, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _normalize_columns(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(M, axis=0) + eps
        return M / norms
