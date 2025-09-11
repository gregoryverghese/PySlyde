# stain_normalization/reinhard.py
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional
from .base import StainNormalizer


class ReinhardStainNormalizer(StainNormalizer):
    """
    Reinhard color normalization in CIE Lab (D65) with minimal dependencies.

    - Converts RGB↔Lab using sRGB + D65 matrices (NumPy only).
    - fit(): compute target Lab per-channel mean/std
    - normalize(): standardize source Lab to its own stats, then re-scale to target stats

    Notes
    -----
    * Input/Output: RGB uint8 or float-like arrays shaped (H, W, 3). Output is uint8.
    * This implementation uses global image statistics (no tissue mask).
    """

    def __init__(self, eps: float = 1e-6, clip_rgb: bool = True):
        if eps <= 0:
            raise ValueError(f"eps must be > 0. Got {eps}")
        self.eps = float(eps)

        self.clip_rgb = bool(clip_rgb)
        self.mu_lab: Optional[np.ndarray] = None   # (3,)
        self.std_lab: Optional[np.ndarray] = None  # (3,)

    # ---------- Public API ----------

    def fit(self, target_tile: np.ndarray) -> "ReinhardStainNormalizer":
        lab = self._rgb_to_lab(target_tile)

        mu = lab.reshape(-1, 3).mean(axis=0)
        sd = lab.reshape(-1, 3).std(axis=0)

        if np.any(sd < self.eps):
            raise ValueError(
                "Target tile has near-zero variance in Lab space — cannot compute normalization profile."
            )

        self.mu_lab = mu
        self.std_lab = sd + self.eps
        return self


    def normalize(self, source_tile: np.ndarray) -> np.ndarray:
        if self.mu_lab is None or self.std_lab is None:
            raise RuntimeError("Not fitted. Call fit(target_tile) first or use fit_normalize().")
        lab = self._rgb_to_lab(source_tile)

        # Source stats
        mu_s = lab.reshape(-1, 3).mean(axis=0)
        std_s = lab.reshape(-1, 3).std(axis=0) + self.eps

        # Standardize then re-scale to target stats (per-channel in Lab)
        lab_norm = (lab - mu_s) / std_s
        lab_tgt = lab_norm * self.std_lab + self.mu_lab

        out = self._lab_to_rgb(lab_tgt)
        return out

    # ---------- Serialization ----------

    def get_profile(self) -> Dict[str, Any]:
        return {
            "algo": "Reinhard",
            "eps": self.eps,
            "clip_rgb": self.clip_rgb,
            "mu_lab": None if self.mu_lab is None else self.mu_lab.tolist(),
            "std_lab": None if self.std_lab is None else self.std_lab.tolist(),
        }

    def set_profile(self, profile: Dict[str, Any]) -> None:
        eps_val = float(profile.get("eps", self.eps))
        if eps_val <= 0:
            raise ValueError(f"eps must be > 0. Got {eps_val}")
        self.eps = eps_val
        
        self.clip_rgb = bool(profile.get("clip_rgb", self.clip_rgb))
        mu = profile.get("mu_lab")
        sd = profile.get("std_lab")
        self.mu_lab = None if mu is None else np.asarray(mu, dtype=np.float32)
        self.std_lab = None if sd is None else np.asarray(sd, dtype=np.float32)

    # ---------- Color space utilities (NumPy-only) ----------

    @staticmethod
    def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
        # x in [0,1]
        a = 0.055
        return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

    @staticmethod
    def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
        # x in [0,1] (may be slightly out of bounds before clipping)
        a = 0.055
        x = np.clip(x, 0, 1)
        return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)


    @staticmethod
    def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
        # Expect HxWx3, uint8 or float. Convert to float32 in [0,1] sRGB.
        x = rgb.astype(np.float32)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.max() > 1.0:
            x = x / 255.0
        x = ReinhardStainNormalizer._srgb_to_linear(x)

        # sRGB D65 transform
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
        xyz = np.tensordot(x, M.T, axes=1)
        # Scale to typical Lab reference (Y scaled to 100)
        return xyz * 100.0

    @staticmethod
    def _xyz_to_rgb(xyz: np.ndarray, clip: bool = True) -> np.ndarray:
        # xyz with Y~[0,100], convert back to sRGB uint8
        xyz = xyz / 100.0
        M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660,  1.8760108,  0.0415560],
                          [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
        lin_rgb = np.tensordot(xyz, M_inv.T, axes=1)
        srgb = ReinhardStainNormalizer._linear_to_srgb(lin_rgb)

        if clip:
            srgb = np.clip(srgb, 0.0, 1.0)
        rgb8 = (srgb * 255.0 + 0.5).astype(np.uint8)
        return rgb8

    @staticmethod
    def _f_lab(t: np.ndarray) -> np.ndarray:
        # Helper for XYZ->Lab
        delta = 6.0 / 29.0
        return np.where(t > (delta ** 3), np.cbrt(t), (t / (3 * delta ** 2)) + (4.0 / 29.0))

    @staticmethod
    def _finv_lab(t: np.ndarray) -> np.ndarray:
        # Helper for Lab->XYZ
        delta = 6.0 / 29.0
        return np.where(t > delta, t ** 3, 3 * (delta ** 2) * (t - 4.0 / 29.0))

    @staticmethod
    def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
        # D65 reference white (CIE 1931 2°)
        Xn, Yn, Zn = 95.047, 100.000, 108.883
        xr = xyz[..., 0] / Xn
        yr = xyz[..., 1] / Yn
        zr = xyz[..., 2] / Zn

        fx = ReinhardStainNormalizer._f_lab(xr)
        fy = ReinhardStainNormalizer._f_lab(yr)
        fz = ReinhardStainNormalizer._f_lab(zr)

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        lab = np.stack([L, a, b], axis=-1).astype(np.float32)
        return lab

    @staticmethod
    def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
        # D65 reference white
        Xn, Yn, Zn = 95.047, 100.000, 108.883

        L = lab[..., 0]
        a = lab[..., 1]
        b = lab[..., 2]

        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        xr = ReinhardStainNormalizer._finv_lab(fx)
        yr = ReinhardStainNormalizer._finv_lab(fy)
        zr = ReinhardStainNormalizer._finv_lab(fz)

        X = xr * Xn
        Y = yr * Yn
        Z = zr * Zn
        xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        return xyz

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        return self._xyz_to_lab(self._rgb_to_xyz(rgb))

    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        xyz = self._lab_to_xyz(lab)
        return self._xyz_to_rgb(xyz, clip=self.clip_rgb)
