"""
Stain Normalisation abstract base class
Code inspired by staintools and tiatoolbox TODO - add references here

author: Holly Rafique
date: 09/09/2025
"""

import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict

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

    def __init__(self) -> None:
        self._fitted: bool = False

    @abstractmethod
    def fit(self, target_tile: np.ndarray) -> "StainNormalizer":
        """Learn target-specific parameters (e.g., stain basis, stats)."""
        self._fitted = True
        raise NotImplementedError

    @abstractmethod
    def normalize(self, source_tile: np.ndarray) -> np.ndarray:
        """Normalize a source tile using learned target parameters."""
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """Whether the normalizer has been fitted with a target."""
        return self._fitted

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
    def rgb2od(
        I: np.ndarray, I0: float | None = None, beta: float | None = None
    ) -> np.ndarray:
        """
        Convert RGB to optical density (OD).

        Parameters
        ----------
        I : np.ndarray
            Input RGB image.
        I0 : float, optional
            Reference intensity. Defaults to 1.0 if max <= 1, else 255.0.
        beta : float, optional
            If provided, pixels with OD norm <= beta are treated as background
            and excluded from the returned OD array.
        """
        I = I.astype(np.float32, copy=False)
        # Auto-select I0 if not provided: 1.0 for float images in [0,1], else 255.0
        if I0 is None:
            I0 = 1.0 if I.max() <= 1.0 + 1e-6 else 255.0
        # epsilon relative to I0 to avoid log(0) without crushing dynamic range
        eps = np.finfo(np.float32).eps * I0 * 10.0  # ~1e-6 of I0
        I_clamped = np.clip(I, eps, I0)
        OD = -np.log(I_clamped / I0)

        # Optional masking
        if beta is not None:
            mask = np.linalg.norm(OD, axis=-1) > beta
            return OD[mask]

        return OD

    @staticmethod
    def od2rgb(
        OD: np.ndarray, I0: float | None = None, ref_dtype: np.dtype = np.uint8
    ) -> np.ndarray:
        """
        Convert optical density (OD) back to RGB.

        Parameters
        ----------
        OD : np.ndarray
            Optical density array.
        I0 : float, optional
            Reference intensity. Defaults to 255 if ref_dtype is uint8, else 1.0.
        ref_dtype : np.dtype, optional
            Desired output dtype (usually source_tile.dtype).
            - np.uint8 → output in [0, 255]
            - np.float32/64 → output in [0, 1]
        """
        # print("OD max:",OD.max())
        # Auto-select I0 if not provided: 1.0 for float images in [0,1], else 255.0
        if I0 is None:
            # print("autoselect I0")
            I0 = 255.0 if ref_dtype == np.uint8 else 1.0
            # print("I0",I0)

        # I = I0 * np.exp(-OD) #I = I0 * np.exp(-np.clip(OD, 0, 2.5))
        I = I0 * np.exp(-np.clip(OD, 0, 2.5))

        # --- Convert to expected dtype ---
        if ref_dtype == np.uint8:
            return np.clip(I, 0.0, 255.0).astype(np.uint8)
        else:
            return np.clip(I / I0, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _normalize_columns(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(M, axis=0) + eps
        return M / norms

    @staticmethod
    def tissue_mask(OD: np.ndarray, beta: float = 0.15) -> np.ndarray:
        """Return mask of tissue pixels, ignoring white background."""
        return (OD > beta).any(axis=-1)
