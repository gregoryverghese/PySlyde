from .base import StainNormalizer
from typing import Any, Dict, Optional
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import nnls

class VahadaneStainNormalizer(StainNormalizer):
    """
    Vahadane et al. (2016) stain normalization using Sparse NMF.

    Reference:
        Vahadane et al., "Structure-preserving color normalization
        and sparse stain separation for histological images", ISBI 2016.

    Parameters
    ----------
    n_stains : int
        Number of stains (usually 2: H&E). Must be >= 2.
    alpha : float
        Sparsity regularization for W in NMF. Must be >= 0.
    l1_ratio : float
        Balance between L1 (1.0) and L2 (0.0) regularization. Must be in [0,1].
    max_iter : int
        Maximum iterations for NMF solver. Must be > 0.
    verbose : bool
        Print debug messages and guardrail warnings.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        max_iter: int = 500,
        n_stains: int = 2,
        verbose: bool = True,
    ):
        # ---- validate n_stains ----
        if not isinstance(n_stains, int) or n_stains < 2:
            raise ValueError(f"n_stains must be an integer >= 2. Got {n_stains}.")
        self.n_stains = n_stains

        # ---- validate alpha ----
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative. Got {alpha}.")
        self.alpha = float(alpha)

        # ---- validate l1_ratio ----
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError(f"l1_ratio must be between 0 and 1. Got {l1_ratio}.")
        self.l1_ratio = float(l1_ratio)

        # ---- validate max_iter ----
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(f"max_iter must be a positive integer. Got {max_iter}.")
        self.max_iter = max_iter

        self.verbose = bool(verbose)

        self.W_target: Optional[np.ndarray] = None  # target stain basis (3xk)

    # ---------- Base API ----------

    def fit(self, target_tile: np.ndarray) -> "VahadaneStainNormalizer":
        """Estimate target stain basis using sparse NMF."""
        self.W_target = self._estimate_stain_matrix(target_tile)
        return self

    def normalize(self, source_tile: np.ndarray) -> np.ndarray:
        """Normalize source tile using target stain basis."""
        if self.W_target is None:
            raise RuntimeError("Not fitted. Call fit(target_tile) first.")

        # Estimate source stain basis
        W_src = self._estimate_stain_matrix(source_tile)

        # Solve concentrations for source given W_src
        C = self._concentrations(source_tile, W_src)

        # Reconstruct OD using target basis
        OD_norm = C @ self.W_target.T
        return self.od2rgb(OD_norm.reshape(source_tile.shape))

    def get_profile(self) -> Dict[str, Any]:
        return {
            "algo": "Vahadane",
            "n_stains": self.n_stains,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "W_target": None if self.W_target is None else self.W_target.tolist(),
        }

    def set_profile(self, profile: Dict[str, Any]) -> None:
        self.n_stains = int(profile.get("n_stains", self.n_stains))
        self.alpha = float(profile.get("alpha", self.alpha))
        self.l1_ratio = float(profile.get("l1_ratio", self.l1_ratio))
        self.max_iter = int(profile.get("max_iter", self.max_iter))
        Wt = profile.get("W_target")
        self.W_target = None if Wt is None else np.asarray(Wt, dtype=np.float32)

    # ---------- Internals ----------

    def _estimate_stain_matrix(self, I: np.ndarray) -> np.ndarray:
        """Estimate stain basis matrix using sparse NMF."""
        if I.ndim != 3 or I.shape[2] != 3:
            raise ValueError("Expected RGB image (H, W, 3).")

        OD = self.rgb2od(I).reshape(-1, 3)
        OD = OD[~np.any(OD < 0.15, axis=1)]  # remove near-background

        if OD.size == 0:
            raise ValueError("No valid OD pixels found for stain estimation.")

        model = NMF(
            n_components=self.n_stains,
            init="random",
            alpha_W=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=42,
        )
        W = model.fit_transform(OD)  # (N, k)
        H = model.components_        # (k, 3)

        # Normalize stain vectors
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)

        # Guardrail: warn if stains nearly collinear
        if self.n_stains == 2:
            cos_sim = float(np.dot(H[0], H[1]) / (np.linalg.norm(H[0]) * np.linalg.norm(H[1]) + 1e-12))
            if cos_sim > 0.95 and self.verbose:
                print(f"[Vahadane] Warning: stain basis nearly collinear (cos_sim={cos_sim:.3f})")

        return H.T  # shape (3, k)

    def _concentrations(self, I: np.ndarray, W: np.ndarray) -> np.ndarray:
        if W is None or not isinstance(W, np.ndarray):
            raise ValueError("Estimated stain matrix W is invalid.")
        if W.shape[0] != 3 or W.shape[1] != self.n_stains:
            raise ValueError(f"Unexpected W shape {W.shape}, expected (3, {self.n_stains})")

        # Flatten image → optical density
        OD = self.rgb2od(I).reshape(-1, 3)  # (n_pixels, 3)


        # Solve NNLS for each pixel
        C = np.zeros((OD.shape[0], self.n_stains), dtype=np.float32)

        for i, od in enumerate(OD):
            C[i, :], _ = nnls(W, od)

        return C