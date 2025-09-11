from .base import StainNormalizer
from typing import Any, Dict, Optional
import numpy as np

class MacenkoStainNormalizer(StainNormalizer):
    """
    Macenko stain normalization.
    Lightweight Macenko stain normalizer (NumPy only).

    - Stain matrix estimated via SVD + angular percentiles.
    - Concentrations solved with pseudo-inverse + clamp (fast, no extra deps).
    - Stain intensity scaled by robust percentile (default 95).

    Parameters
    ----------
    alpha : float
        Percent cutoff (0–100) for extreme angles in projected OD space.
    beta : float
        OD-norm threshold to exclude near-background pixels.
    percentile : float
        Robust max percentile for concentrations (e.g., 95).
    I0 : float
        Illumination white for RGB<->OD transforms (255 for 8-bit).
    """
    def __init__(
        self,
        alpha: float = 10.0,              # starting angular percentile cutoff
        beta: float = 0.15,               # OD-norm background threshold
        percentile: float = 95.0,         # scaling percentile for concentrations
        # --- new guardrail knobs ---
        cos_similarity_max: float = 0.98,
        scale_clip: tuple[float, float] = (1e-2, 1e2),
    ):
        # ---- validate alpha ----
        if not (0 < alpha <= 50):
            raise ValueError(
                f"alpha must be between 0 and 50 (exclusive 0). Got {alpha}."
            )
        self.alpha = float(alpha)

        # ---- validate beta ----
        if not (0 < beta < 3):
            raise ValueError(
                f"alpha must be between 0 and 50 (exclusive 0). Got {alpha}."
            )
        self.beta = float(beta)

        # ---- validate percentile ----
        if not (0 < percentile < 100):
            raise ValueError(
                f"percentile must be between 0 and 100 (exclusive). Got {percentile}."
            )
        self.percentile = float(percentile)

        #self.I0 = 255.0     #intensity of white pixels

        # build private fallback ladder for adaptive-α (not user-facing)
        self._alpha_steps = self._build_alpha_steps(self.alpha)
        
        # guardrail knobs that make sense to expose
        self.cos_similarity_max = float(cos_similarity_max) # if cos(stain1, stain2) > this, widen alpha
        self.scale_clip = (float(scale_clip[0]), float(scale_clip[1])) # clamp per-stain scale factors

        self.W_target: Optional[np.ndarray] = None
        self.H_target_pct: Optional[np.ndarray] = None

    @staticmethod
    def _build_alpha_steps(alpha: float) -> tuple[float, ...]:
        """
        Build a 6-element sequence of candidate alpha values,
        centered on the chosen alpha and spaced by 5.
        Example: alpha=20 -> (10, 15, 20, 25, 30, 35)
        """
        step = 5
        half = 3  # 3 below and 2 above gives 6 total
        alphas = [alpha + (i - half) * step for i in range(6)]
        # ensure all values are positive
        alphas = [min(50.0, max(1.0, round(a, 2))) for a in alphas]
        return tuple(alphas)

    # ---- Base interface ----

    def fit(self, target_tile: np.ndarray) -> "MacenkoStainNormalizer":
        """Estimate target stain basis and per-stain percentile intensities."""
        W = self._estimate_stain_matrix(target_tile)
        C = self._concentrations(target_tile, W)
        self.W_target = W
        self.H_target_pct = np.percentile(C, self.percentile, axis=0)
        return self

    def normalize(self, source_tile: np.ndarray) -> np.ndarray:
        """Normalize a source tile using fitted target statistics."""
        if self.W_target is None or self.H_target_pct is None:
            raise RuntimeError("Not fitted. Call fit(target_tile) first or use fit_normalize().")

        W_src = self._estimate_stain_matrix(source_tile)

        OD = self.rgb2od(source_tile).reshape(-1, 3)
        C = self._solve_concentrations_fast(OD, W_src)

        H_src_pct = np.percentile(C, self.percentile, axis=0)
        H_src_pct = np.maximum(H_src_pct, 1e-8)  # protect against tiny denominators

        scale = (self.H_target_pct + 1e-8) / H_src_pct
        # --- guardrail 2: clamp scales ---
        lo, hi = self.scale_clip
        scale = np.clip(scale, lo, hi)

        C_scaled = C * scale
        OD_norm = C_scaled @ self.W_target.T
        return self.od2rgb(OD_norm.reshape(source_tile.shape))

    def get_profile(self) -> Dict[str, Any]:
        return {
            "algo": "Macenko",
            "alpha": self.alpha,
            "beta": self.beta,
            "percentile": self.percentile,
            #"I0": self.I0,
            "alpha_steps": list(self._alpha_steps),
            "cos_similarity_max": self.cos_similarity_max,
            "scale_clip": list(self.scale_clip),
            "W_target": None if self.W_target is None else self.W_target.tolist(),
            "H_target_pct": None if self.H_target_pct is None else self.H_target_pct.tolist(),
        }
    
    def set_profile(self, profile: Dict[str, Any]) -> None:
        self.alpha = float(profile.get("alpha", self.alpha))
        self.beta = float(profile.get("beta", self.beta))
        self.percentile = float(profile.get("percentile", self.percentile))
        #self.I0 = float(profile.get("I0", self.I0))
        self._alpha_steps = tuple(profile.get("alpha_steps", self.alpha_steps))
        self.cos_similarity_max = float(profile.get("cos_similarity_max", self.cos_similarity_max))
        sc = profile.get("scale_clip", self.scale_clip)
        self.scale_clip = (float(sc[0]), float(sc[1]))
        Wt = profile.get("W_target"); Ht = profile.get("H_target_pct")
        self.W_target = None if Wt is None else np.asarray(Wt, dtype=np.float32)
        self.H_target_pct = None if Ht is None else np.asarray(Ht, dtype=np.float32)

    # ---- Macenko internals ----
    

    def _estimate_stain_matrix(self, I: np.ndarray) -> np.ndarray:
        if I.ndim != 3 or I.shape[2] != 3:
            raise ValueError("Expected RGB image (H, W, 3).")
        OD = self.rgb2od(I).reshape(-1, 3)
        OD = OD[np.linalg.norm(OD, axis=1) > self.beta]
        if OD.size == 0:
            raise ValueError("All pixels filtered as background; try lowering beta.")

        # PCA via SVD on covariance; project to top-2 PC plane
        _, _, Vt = np.linalg.svd(np.cov(OD.T))
        P2 = Vt[:2, :].T                        # 3x2 plane basis
        proj = OD @ P2                          # (N,2)
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        def stain_dirs(alpha_pct: float) -> np.ndarray:
            a = np.percentile(phi, alpha_pct)
            b = np.percentile(phi, 100.0 - alpha_pct)
            v1 = P2 @ np.array([np.cos(a), np.sin(a)], dtype=np.float32)
            v2 = P2 @ np.array([np.cos(b), np.sin(b)], dtype=np.float32)
            W = np.stack([v1, v2], axis=1)
            return self._normalize_columns(W)

        # initial attempt with current alpha
        W = stain_dirs(self.alpha)

        # --- guardrail 1: adaptive alpha if stains too similar ---
        def cos_sim(u, v) -> float:
            return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

        if cos_sim(W[:, 0], W[:, 1]) > self.cos_similarity_max:
            for a in self._alpha_steps:
                W_try = stain_dirs(a)
                if cos_sim(W_try[:, 0], W_try[:, 1]) <= self.cos_similarity_max:
                    W = W_try
                    break
            else:
                # last resort: enforce orthogonal direction within the PCA plane
                v1p = (P2.T @ W[:, 0]); v1p /= (np.linalg.norm(v1p) + 1e-12)
                v2p = np.array([-v1p[1], v1p[0]], dtype=np.float32)  # 90° rotation
                W = np.stack([W[:, 0], P2 @ v2p], axis=1)
                W = self._normalize_columns(W)

        # heuristic: keep hematoxylin-ish first
        if W[2, 0] < W[2, 1]:
            W = W[:, [1, 0]]
        return W

    def _concentrations(self, I: np.ndarray, W: np.ndarray) -> np.ndarray:
        OD = self.rgb2od(I).reshape(-1, 3)
        return self._solve_concentrations_fast(OD, W)

    @staticmethod
    def _solve_concentrations_fast(OD: np.ndarray, W: np.ndarray) -> np.ndarray:
        C = OD @ np.linalg.pinv(W.T)
        return np.clip(C, 0, None)
    

