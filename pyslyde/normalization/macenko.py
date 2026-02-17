from .base import StainNormalizer
from typing import Any, Dict, Optional
import numpy as np

class MacenkoStainNormalizer(StainNormalizer):
    """
    Macenko stain normalization.
    Lightweight Macenko stain normalizer (NumPy only).

    - Stain matrix estimated via SVD + angular percentiles.
    - Concentrations solved with pseudo-inverse  clamp (fast, no extra deps).
    - Stain intensity scaled by robust percentile (default 95).

    Parameters
    ----------
    alpha : float
        Percent cutoff (0–100) for extreme angles in projected OD space.
    beta : float
        OD-norm threshold to exclude near-background pixels.
    percentile : float
        Robust max percentile for concentrations (e.g., 95).
    """
    def __init__(
                    self,
                    alpha: float = 10.0,              # starting angular percentile cutoff
                    beta: float = 0.15,               # OD-norm background threshold
                    percentile: float = 95.0,         # scaling percentile for concentrations
                    # --- new guardrail knobs ---
                    cos_similarity_max: float = 0.95,
                    scale_clip: tuple[float, float] = (0.5, 3.0),
                    allow_fallback: bool = True,
                    verbose: bool = False,
                ):
        super().__init__()   # calls StainNormalizer.__init__
        # ---- validate alpha ----
        if not (0 < alpha <= 50):
            raise ValueError(
                f"alpha must be between 0 and 50 (exclusive 0). Got {alpha}."
            )
        # ---- validate beta ----
        if not (0 < beta < 3):
            raise ValueError(
                f"beta must be between 0 and 3. Got {beta}."
            )
        # ---- validate percentile ----
        if not (0 < percentile < 100):
            raise ValueError(
                f"percentile must be between 0 and 100 (exclusive). Got {percentile}."
            )

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.percentile = float(percentile)
        # build private fallback ladder for adaptive-α (not user-facing)
        self._alpha_steps = self._build_alpha_steps(self.alpha)
        
        # guardrail knobs that make sense to expose
        self.cos_similarity_max = float(cos_similarity_max) # if cos(stain1, stain2) > this, widen alpha
        self.scale_clip = (float(scale_clip[0]), float(scale_clip[1])) # clamp per-stain scale factors
        self.verbose = bool(verbose)
        self.allow_fallback = bool(allow_fallback) 

        self.W_target: Optional[np.ndarray] = None
        self.W_src: Optional[np.ndarray] = None
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


    @staticmethod
    def _solve_concentrations_fast(OD: np.ndarray, W: np.ndarray) -> np.ndarray:
        C = OD @ np.linalg.pinv(W.T)
        return np.clip(C, 0, None)
    
    @staticmethod
    def _tissue_mask_rgb(I: np.ndarray, beta: float = 0.15) -> np.ndarray:
        """Hybrid tissue mask using brightness + OD norm."""
        if I.ndim != 3 or I.shape[2] != 3:
            raise ValueError("Expected RGB (H, W, 3).")

        # normalize to float [0,1]
        I_float = I.astype(np.float32)
        if I_float.max() > 1.0:
            I_float /= 255.0

        # brightness mask (mean intensity, not single channel)
        brightness = I_float.mean(axis=2)
        mask_bright = brightness < 0.9

        # OD-norm mask
        OD = StainNormalizer.rgb2od(I).reshape(-1, 3)
        mask_od = (np.linalg.norm(OD, axis=1) > beta).reshape(I.shape[:2])

        return mask_bright & mask_od
    
    def _sanitize_basis(self, W: np.ndarray) -> np.ndarray:
        # Normalize columns
        W = self._normalize_columns(W)
        # Enforce non-negativity
        W = np.abs(W)
        # Re-normalize after abs
        W = self._normalize_columns(W)

        # Guardrail: enforce separation
        cos_sim = float(np.dot(W[:, 0], W[:, 1]) /
                        (np.linalg.norm(W[:, 0]) * np.linalg.norm(W[:, 1]) + 1e-12))
        
        if cos_sim > self.cos_similarity_max:
            if self.verbose:
                print(f"[Macenko] Stains too similar (cos={cos_sim:.3f}), enforcing PCA-orthogonal fallback.")
            # orthogonalize v2 with respect to v1 using Gram-Schmidt
            v1 = W[:, 0]
            v2 = W[:, 1] - np.dot(W[:, 1], v1) * v1
            if np.linalg.norm(v2) < 1e-12:  # degenerate case
                # if v2 collapsed, just pick any orthogonal unit vector
                idx = np.argmin(np.abs(v1))
                v2 = np.zeros_like(v1)
                v2[idx] = 1.0
                v2 -= np.dot(v2, v1) * v1
            v2 /= (np.linalg.norm(v2) + 1e-12)
            W = np.stack([v1, v2], axis=1)
        return self._normalize_columns(W)
    

    # ---- Base interface ----

    def fit(self, target_tile: np.ndarray | list[np.ndarray],
            percentile: float | None = None) -> "MacenkoStainNormalizer":
        """
        Estimate target stain basis and per-stain percentile intensities.

        Parameters
        ----------
        target : np.ndarray or list of np.ndarray
            A single RGB tile, or a list of tiles/patches.
        percentile : float, optional
            Override percentile for robust scaling (default = self.percentile).
        """
        perc = self.percentile if percentile is None else percentile

        # --- collect concentrations across one or many tiles ---
        if isinstance(target_tile, list):
            Cs = []
            for tile in target_tile:
                W = self._estimate_stain_matrix(tile)
                Cs.append(self._concentrations(tile, W))
            C_all = np.vstack(Cs)
        else:
            W = self._estimate_stain_matrix(target_tile)
            C_all = self._concentrations(target_tile, W)

        #W = self._estimate_stain_matrix(target_tile)
        #C = self._concentrations(target_tile, W)

        # --- per-channel percentile over tissue pixels only ---
        H_target_pct = []
        for j in range(C_all.shape[1]):
            vals = C_all[:, j][C_all[:, j] > 0]
            if vals.size == 0:
                H_target_pct.append(1e-8)
            else:
                H_target_pct.append(np.percentile(vals, perc))

        H_target_pct = np.array(H_target_pct, dtype=np.float32)

        # --- floor to avoid collapse (staintools/tiatoolbox style) ---
        H_target_pct = np.maximum(H_target_pct, 0.1)

        # --- reorder stains using blue:red ratio heuristic ---
        # Hematoxylin usually contributes more to blue channel relative to red.
        if target_tile is not None:
            W = self._sanitize_basis(W)
            br_ratio = W[2, :] / (W[0, :] + 1e-12)  # blue:red ratio for each stain
            if br_ratio[1] > br_ratio[0]:
                W = W[:, [1, 0]]      # swap so H is first, E is second

        self.H_target_pct = H_target_pct
        self.W_target = W
        self._fitted = True
        
        if self.verbose:
            cos_sim = float(np.dot(W[:, 0], W[:, 1]) /
                            (np.linalg.norm(W[:, 0]) * np.linalg.norm(W[:, 1]) + 1e-12))
            print(f"[Macenko/fit] W_target:\n{W}")
            print(f"[Macenko/fit] Cosine similarity = {cos_sim:.3f}")
            for j, val in enumerate(self.H_target_pct):
                print(f"[Macenko/fit] H_target_pct[{j}] = {val:.4f}")
            
        return self
    
    
    def fit_source(self, source: Any, thumb_size: int = 1024) -> "MacenkoStainNormalizer":
        """
        Estimate stain basis (W_src) for a source slide or image.

        Parameters
        ----------
        source : OpenSlide object or np.ndarray
            Whole-slide image (OpenSlide) or RGB array.
        thumb_size : int
            Thumbnail size (for OpenSlide only).
        """
        if hasattr(source, "get_thumbnail"):  # OpenSlide case
            thumb = source.get_thumbnail((thumb_size, thumb_size))
            I = np.array(thumb.convert("RGB"))
        elif isinstance(source, np.ndarray):
            I = source
        else:
            raise TypeError("source must be an OpenSlide object or an RGB NumPy array.")
        # --- apply tissue mask ---
        mask = self._tissue_mask_rgb(I, beta=self.beta)
        if mask.mean() < 0.05:  # less than 5% tissue pixels
            raise RuntimeError("Source slide has too little tissue for reliable fitting.")

        # only keep tissue pixels
        #I_tissue = I[mask]
        # keep shape, just whiten background
        I_tissue = I.copy()
        I_tissue[~mask] = 255

        
        # Estimate stain matrix on tissue pixels
        self.W_src = self._estimate_stain_matrix(I_tissue)
        if self.verbose:
            print("[Macenko] Fitted source W_src:\n", self.W_src)

        return self
    

    def normalize(self, source_tile: np.ndarray) -> np.ndarray:
        """Normalize source tile using target stain basis."""

        if self.W_target is None or self.H_target_pct is None:
            raise RuntimeError("Not fitted. Call fit(target_tile) first or use fit_normalize().")

        if self.W_src is not None:
            W_src = self.W_src
        else:
            if not self.allow_fallback:
                raise RuntimeError("W_src not fitted at slide-level. "
                                    "Call fit_source(slide) before normalize().")
    
            if self.verbose:
                print("[Macenko] Warning: W_src not fitted at slide-level. "
                    "Estimating from this tile (may be unstable).")
            W_src = self._estimate_stain_matrix(source_tile)

        C = self._concentrations(source_tile, W_src)

        # --- per-channel percentile scaling ---
        H_src_pct = []
        for j in range(C.shape[1]):
            vals = C[:, j][C[:, j] > 0]
            if vals.size == 0:
                H_src_pct.append(1e-8)
            else:
                H_src_pct.append(np.percentile(vals, self.percentile))
        H_src_pct = np.array(H_src_pct, dtype=np.float32)

        scale = (self.H_target_pct + 1e-8) / H_src_pct
        lo, hi = self.scale_clip
        scale_clipped = np.clip(scale, lo, hi)

        if self.verbose and not np.allclose(scale, scale_clipped):
            print(f"[Macenko] Scale factors {scale} clipped to {scale_clipped}")
            
        if self.verbose:
            for j, (src, tgt, s, sc) in enumerate(zip(H_src_pct, self.H_target_pct, scale, scale_clipped)):
                print(f"[Macenko/normalize] Channel {j}: H_src_pct={src:.4f}, "
                    f"H_target_pct={tgt:.4f}, scale={s:.4f}, clipped={sc:.4f}")

        C_scaled = C * scale_clipped
        OD_norm = C_scaled @ self.W_target.T
        OD_norm *= 2.5 / np.percentile(OD_norm, 99)
        #return self.od2rgb(OD_norm.reshape(source_tile.shape), ref_dtype=source_tile.dtype)
        H, W, _ = source_tile.shape
        OD_reshaped = OD_norm.reshape(H, W, 3)
        return self.od2rgb(OD_reshaped, I0=255.0, ref_dtype=np.uint8)



    def get_profile(self) -> Dict[str, Any]:
        return {
            "algo": "Macenko",
            "alpha": self.alpha,
            "beta": self.beta,
            "percentile": self.percentile,
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
        
        # --- 1. Convert to OD and mask background ---
        mask = self._tissue_mask_rgb(I, beta=self.beta)
        OD = self.rgb2od(I).reshape(-1, 3)[mask.reshape(-1)]
        if OD.size == 0:
            raise ValueError("All pixels filtered as background; try lowering beta.")
        
        # --- 2. Row-normalize OD (important for angular analysis) ---
        OD = OD / (np.linalg.norm(OD, axis=1, keepdims=True) + 1e-12)

        # --- 3. PCA via SVD, project to top-2 plane ---
        # _, _, Vt = np.linalg.svd(np.cov(OD.T))
        # P2 = Vt[:2, :].T                        # 3x2 plane basis
        # proj = OD @ P2                          # (N,2)
        # 
        U, S, Vt = np.linalg.svd(OD, full_matrices=False)
        P2 = Vt[:2, :].T
        proj = OD @ P2
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        def stain_dirs(alpha_pct: float) -> np.ndarray:
            a = np.percentile(phi, alpha_pct)
            b = np.percentile(phi, 100.0 - alpha_pct)
            v1 = P2 @ np.array([np.cos(a), np.sin(a)], dtype=np.float32)
            v2 = P2 @ np.array([np.cos(b), np.sin(b)], dtype=np.float32)
            W = np.stack([v1, v2], axis=1)
            return self._normalize_columns(W)

        # --- 4. Start with current alpha ---
        # initial attempt with current alpha
        W = stain_dirs(self.alpha)

        # --- 5. guardrail: adaptive alpha if stains too similar ---
        def cos_sim(u, v) -> float:
            return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

        if cos_sim(W[:, 0], W[:, 1]) > self.cos_similarity_max:
            if self.verbose:
                print(f"[Macenko] Stains too similar with alpha={self.alpha}, widening...")
            for a in self._alpha_steps:
                W_try = stain_dirs(a)
                if cos_sim(W_try[:, 0], W_try[:, 1]) <= self.cos_similarity_max:
                    if self.verbose:
                        print(f"[Macenko] Using widened alpha={a}")
                    W = W_try
                    break
            else:
                if self.verbose:
                    print("[Macenko] Using orthogonal fallback for stain separation")
                # last resort: enforce orthogonal direction within the PCA plane
                v1p = (P2.T @ W[:, 0])
                v1p /= (np.linalg.norm(v1p) + 1e-12)
                v2p = np.array([-v1p[1], v1p[0]], dtype=np.float32)  # 90° rotation
                W = np.stack([W[:, 0], P2 @ v2p], axis=1)
                W = self._normalize_columns(W)

        # --- 6. Heuristic: hematoxylin (darker) first ---
        if W[2, 0] < W[2, 1]:
            W = W[:, [1, 0]]

        # --- 7. Sanitise: enforce positivity & re-normalize ---
        W = self._sanitize_basis(W)
                
        if self.verbose:
            cs = cos_sim(W[:, 0], W[:, 1])
            print(f"[Macenko] Final stain basis cos-sim = {cs:.3f}")

        return W

    def _concentrations(self, I: np.ndarray, W: np.ndarray) -> np.ndarray:
        OD = self.rgb2od(I).reshape(-1, 3)
        mask = np.linalg.norm(OD, axis=1) > self.beta
        C = np.zeros((OD.shape[0], W.shape[1]), dtype=np.float32)
        if np.any(mask):
            C_masked = self._solve_concentrations_fast(OD[mask], W)
            C[mask] = C_masked
        return C
    



