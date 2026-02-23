"""
test_feature_extractor_integration.py

Integration tests for pyslyde.encoders.feature_extractor.FeatureGenerator.

Explanation
----------- 
- Unit tests (mocked) already verify internal logic deterministically.
- These integration tests verify: real model load + forward_pass + output shape/format.
- By default:
    * If RUN_NETWORK_TESTS != 1 -> skip models that may need downloads.
    * Gated HF models additionally require HUGGINGFACE_TOKEN when downloads are allowed.

How to run on terminal
----------------------
  - Navigate to script's location

  - Run integration tests (will skip unless you opt-in to downloads)
    # For all integration tests
    pytest -m integration -v
    pytest -v test_feature_extractor_integration.py

  - Allow downloads + run all models (gated models also require token):
    # For all integration tests
    RUN_NETWORK_TESTS=1 HUGGINGFACE_TOKEN=... pytest -m integration -v

    # For this specific integration test
    RUN_NETWORK_TESTS=1 HUGGINGFACE_TOKEN=... pytest -v test_feature_extractor_integration.py

  - Run only those integration tests that may hit network:
    RUN_NETWORK_TESTS=1 HUGGINGFACE_TOKEN=... pytest -m "integration and network" -v

  - Run integration tests specifying minimum memory requirements for ``HUGE_MODELS``
    RUN_NETWORK_TESTS=1 \
    MIN_CPU_AVAIL_GB=24 \
    MIN_FREE_VRAM_GB=24 \
    HUGGINGFACE_TOKEN=... \
    pytest -v test_feature_extractor_integration.py    

Notes
-----
 - HUGGINGFACE_TOKEN: Hugging Face access token to authenticate access to gated models.

 - MIN_CPU_AVAIL_GB and MIN_FREE_VRAM_GB (default 24GB): Minimum RAM/VRAM required by 
   ``HUGE_MODELS``, for loading/inference (otherwise, relevant tests are skipped).    
"""

from __future__ import annotations

import importlib.util
import os
from typing import Optional

import numpy as np
import pytest
import torch

from pyslyde.encoders.feature_extractor import (
    EXPECTED_DIMS,
    GATED_HF_MODELS,
    FeatureGenerator,
)

RUN_NETWORK = os.getenv("RUN_NETWORK_TESTS") in ("1", "true", "True")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
ALL_MODELS = sorted(EXPECTED_DIMS.keys())
HUGE_MODELS = {"uni", "uni2", "gigapath", "hoptimus0", "hoptimus1"}


def _dummy_rgb_image(model_name: str) -> np.ndarray:
    """
    Generates a synthetic RGB image  suitable for end-to-end integration tests.

    Returns a NumPy array of shape (H, W, 3).
    """
    h, w = 224, 224
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _skip_reason(model_name: str) -> Optional[str]:
    """
    Decides if this model's integration test should run in the current environment.

    Behaviour:
    - If RUN_NETWORK_TESTS != 1 (default), skip models that might require downloads.
    - For gated HF models: require both RUN_NETWORK_TESTS=1 and HUGGINGFACE_TOKEN.
    - For TensorFlow-backed model(s): require tensorflow installed.
    """
    if not RUN_NETWORK:
        return "RUN_NETWORK_TESTS != 1; skipping tests that may download model weights."

    if model_name in GATED_HF_MODELS:
        repo = FeatureGenerator.__new__(FeatureGenerator)._model_repo_id(model_name)
        cached = FeatureGenerator.__new__(FeatureGenerator)._hf_cache_exists(repo)
        if not cached and os.getenv("HUGGINGFACE_TOKEN") is None:
            return f"{model_name} is gated on Hugging Face. Set HUGGINGFACE_TOKEN or cache the model first."

    if model_name == "pathfm":
        if not _has_tensorflow():
            return "pathfm requires tensorflow. Install tensorflow to run this test."
        if not _has_tf_keras():
            return "pathfm requires tf_keras. Install tf_keras to run this test."

    if model_name in HUGE_MODELS:
        min_cpu_avail_gb = float(os.getenv("MIN_CPU_AVAIL_GB", "24"))
        min_free_vram_gb = float(os.getenv("MIN_FREE_VRAM_GB", "24"))

        cpu_avail = _available_cpu_ram_gb()
        if cpu_avail is None:
            return f"{model_name}: cannot determine available CPU RAM; skipping to avoid OOM."
        if cpu_avail < min_cpu_avail_gb:
            return (
                f"{model_name} may OOM during load; need >= {min_cpu_avail_gb:.1f}GB "
                f"available CPU RAM, found {cpu_avail:.1f}GB."
            )

        free_vram = _free_vram_gb()
        if free_vram is not None and free_vram < min_free_vram_gb:
            return (
                f"{model_name} may OOM on GPU; need >= {min_free_vram_gb:.1f}GB "
                f"free VRAM, found {free_vram:.1f}GB."
            )

    return None


def _has_tensorflow() -> bool:
    """
    Checks whether TensorFlow is available in the current Python environment.

    Performs a lightweight availability check without importing TensorFlow,
    avoiding unnecessary side effects or startup overhead.
    """
    return importlib.util.find_spec("tensorflow") is not None


def _has_tf_keras() -> bool:
    """
    Check whether the legacy tf_keras package is available.

    pathfm requires tf_keras (Keras 2 compatibility layer) in addition to TensorFlow.
    This function performs a lightweight availability check without importing tf_keras.
    """
    return importlib.util.find_spec("tf_keras") is not None


def _expected_dim(model_name: str) -> int:
    """
    Returns the expected feature dimensionality for a given model.

    Raises KeyError if the model name is not present in EXPECTED_DIMS
    """
    if model_name not in EXPECTED_DIMS:
        raise KeyError(
            f"{model_name} not present in EXPECTED_DIMS; update EXPECTED_DIMS or test list."
        )
    return EXPECTED_DIMS[model_name]


def _available_cpu_ram_gb() -> float | None:
    """Return available system RAM in GB, or None if it cannot be determined."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass

    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except Exception:
        pass

    return None


def _free_vram_gb() -> float | None:
    """Return free VRAM (GB) for CUDA:0, or None if CUDA isn't usable."""
    try:
        import torch

        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            return None
        free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
        return free_bytes / (1024**3)
    except Exception:
        return None


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_real_model_load_and_forward_pass(model_name: str):
    """
    End-to-end integration test for real model loading and inference.

    Tests the full FeatureGenerator pipeline for each supported
    model, including model construction, preprocessing, inference, and
    postprocessing. When not skipped, it uses the actual model implementation
    (and cached or downloaded weights, if permitted) rather than mocks.

    Validates the following runtime contracts:
    - forward_pass returns a torch.Tensor
    - the output is a single-sample (1D) embedding
    - all values in the embedding are finite
    - the embedding dimensionality matches the expected value defined in
      EXPECTED_DIMS for the given model

    Models are conditionally skipped when required dependencies or credentials
    (e.g., gated Hugging Face access) are not available in the current environment.
    """
    reason = _skip_reason(model_name)
    if reason:
        pytest.skip(reason)

    fg = FeatureGenerator(model_name=model_name)

    img = _dummy_rgb_image(model_name)
    out = fg.forward_pass(img)

    assert isinstance(out, torch.Tensor), (
        f"{model_name}: expected torch.Tensor, got {type(out)}"
    )
    assert out.ndim == 1, (
        f"{model_name}: expected 1D output, got shape {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), f"{model_name}: output contains non-finite values"
    assert out.shape[0] == _expected_dim(model_name), (
        f"{model_name}: expected dim {_expected_dim(model_name)}, got {out.shape[0]}"
    )


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("model_name", sorted(GATED_HF_MODELS))
def test_gated_model_cache_miss_requires_token(model_name: str, monkeypatch):
    """
    Verifies authentication requirements for gated Hugging Face models on cache miss.

    Expected behavior:
    - Model initialization raises a RuntimeError
    - The error message explicitly references the missing HUGGINGFACE_TOKEN

    The test is skipped unless RUN_NETWORK_TESTS=1, as the gating logic is only
    relevant when network-backed model loading is permitted.
    """
    if not RUN_NETWORK:
        pytest.skip("RUN_NETWORK_TESTS != 1")

    monkeypatch.setattr(FeatureGenerator, "_hf_cache_exists", lambda self, repo: False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    with pytest.raises(RuntimeError) as e:
        FeatureGenerator(model_name=model_name)

    assert "HUGGINGFACE_TOKEN" in str(e.value)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", sorted(GATED_HF_MODELS))
def test_gated_model_cache_hit_does_not_require_token(model_name: str, monkeypatch):
    """
    Verifies that gated Hugging Face models do not require authentication
    when a local cache is already available.

    Expected behavior:
    - Model initialization succeeds without raising an error
    - No call is made to the Hugging Face login mechanism
    """
    monkeypatch.setattr(FeatureGenerator, "_hf_cache_exists", lambda self, repo: True)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("_hf_login should not be called when cache exists")

    monkeypatch.setattr(FeatureGenerator, "_hf_login", _raise_if_called)

    monkeypatch.setattr(
        FeatureGenerator,
        f"_{model_name}",
        lambda self, _mn=model_name: object(),
    )

    fg = FeatureGenerator(model_name=model_name)
    assert fg.model is not None
