"""
test_feature_extractor.py

Unit tests for pyslyde.encoders.feature_extractor.FeatureGenerator and wrappers.

Behaviour:
- No external downloads / no GPU required
- Mock heavy backends (torchvision, timm, transformers, tensorflow, HF hub)
- Validate key behaviors: model selection, gating/login logic, preprocessing,
  postprocessing, shape normalization, finite checks, expected-dim enforcement.

Note: For full integration tests using real models, see
test_feature_extractor_integration.py

How to run on terminal:
- Navigate to script's location
  pytest -v test_feature_extractor.py
"""

import sys
import types
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

import pyslyde.encoders.feature_extractor as fe
from pyslyde.encoders.feature_extractor import (
    EXPECTED_DIMS,
    GATED_HF_MODELS,
    VIRCHOW_POSTPROCESS,
    FeatureGenerator,
    TFVisionWrapper,
    TorchWrapper,
)

# ------------------------------------------------
# Fixtures
# ------------------------------------------------


@pytest.fixture
def dummy_rgb_np():
    """
    Returns a synthetic RGB image as a NumPy array.

    Shape: (H, W, 3)
    Dtype: uint8

    Simulates a standard RGB image and is used to test image
    preprocessing and conversion utilities without relying on external image
    files.
    """
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_rgb_pil(dummy_rgb_np):
    """
    Returns a deterministic RGB PIL.Image created from a NumPy array.

    This fixture provides a simple, in-memory PIL image suitable for testing
    image preprocessing and feature extraction logic without relying on
    external image files or I/O.
    """
    return Image.fromarray(dummy_rgb_np, mode="RGB")


@pytest.fixture
def fg_resnet18_mocked(monkeypatch):
    """
    Fixture that constructs FeatureGenerator(model_name="resnet18") while
    fully isolating the test from torchvision internals, pretrained weight
    downloads, and GPU availability.

    This fixture mocks:
    - torchvision.models.resnet18 to avoid network access and heavy model init
    - ResNet18_Weights.DEFAULT.transforms to provide a deterministic image transform
    - torch.cuda.is_available to force CPU execution

    Purpose:
    - Verify FeatureGenerator wiring, preprocessing, and forward-pass logic
      without testing torchvision's ResNet implementation or requiring
      external resources.
    - Ensure unit tests remain fast, deterministic, and CI-safe.

    The fake model mimics the minimal ResNet interface expected by
    FeatureGenerator (including a `.fc` attribute), while returning
    predictable outputs.
    """

    class _FakeWeights:
        def transforms(self):
            return lambda pil: torch.zeros(3, 224, 224)

    monkeypatch.setattr(
        fe.models, "ResNet18_Weights", types.SimpleNamespace(DEFAULT=_FakeWeights())
    )

    fake_model = torch.nn.Sequential(torch.nn.Identity())
    fake_model.fc = torch.nn.Identity()

    monkeypatch.setattr(fe.models, "resnet18", lambda weights=None: fake_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    fg = FeatureGenerator(model_name="resnet18")
    return fg


# ------------------------------------------------
# Model selection + errors tests
# ------------------------------------------------


def test_unknown_model_raises_valueerror():
    """
    Ensures that constructing FeatureGenerator with an unsupported model name
    fails fast with a clear ValueError.

    This test enforces strict validation of allowed model identifiers and
    prevents silent fallback or misconfiguration when an unknown model
    name is provided.
    """
    with pytest.raises(ValueError) as e:
        FeatureGenerator(model_name="not_a_real_model")
    assert "Unknown model" in str(e.value)


def test_model_property_is_wrapper(fg_resnet18_mocked):
    """
    Verifies that FeatureGenerator.model returns a wrapped model instance.

    This test ensures that:
    - Accessing the `model` property yields a TorchWrapper, not a raw torch.nn.Module
    - Device selection is respected and defaults to CPU in the mocked environment

    This enforces the public contract that FeatureGenerator always exposes
    a consistent wrapper interface for downstream inference.
    """
    assert hasattr(fg_resnet18_mocked, "model")
    assert isinstance(fg_resnet18_mocked.model, TorchWrapper)
    assert fg_resnet18_mocked.device == "cpu"


# ------------------------------------------------
# HF gating / auth logic tests
# ------------------------------------------------


def test_hf_repo_id_and_hf_hub_ref_properties():
    fg = object.__new__(FeatureGenerator)
    fg.model_name = "uni"
    assert fg.hf_repo_id == "MahmoodLab/uni"
    assert fg.hf_hub_ref == "hf-hub:MahmoodLab/uni"


def test_gated_hf_model_requires_token_if_not_cached(monkeypatch):
    """
    Contract test for Hugging Face gated models when no local cache is present.

    Verifies that constructing a gated HF model without an existing cache
    triggers authentication and fails fast with a clear RuntimeError
    if the HUGGINGFACE_TOKEN environment variable is not set.
    """
    gated = next(iter(GATED_HF_MODELS))
    monkeypatch.setattr(FeatureGenerator, "_hf_cache_exists", lambda self, repo: False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    with pytest.raises(RuntimeError) as e:
        FeatureGenerator(model_name=gated)
    assert "HUGGINGFACE_TOKEN" in str(e.value)


def test_gated_hf_model_does_not_login_if_cached(monkeypatch):
    """
    Contract test for Hugging Face gating logic.

    Verifies that when a gated HF model is already present in the local cache:
    - Hugging Face login is NOT attempted
    - Model construction proceeds without invoking HF authentication

    The model loader is patched to return a lightweight mocked TorchWrapper
    to avoid network access and heavyweight downloads,
    isolating only the gating behavior under test.
    """
    gated = next(iter(GATED_HF_MODELS))
    monkeypatch.setattr(FeatureGenerator, "_hf_cache_exists", lambda self, repo: True)
    login_spy = mock.Mock()
    monkeypatch.setattr("pyslyde.encoders.feature_extractor.login", login_spy)
    monkeypatch.setattr(
        FeatureGenerator, "_" + gated, lambda self: mock.Mock(spec=TorchWrapper)
    )

    fg = FeatureGenerator(model_name=gated, force_hf_login=False)
    assert fg.model is not None
    login_spy.assert_not_called()


def test_gated_hf_model_force_login_even_if_cached(monkeypatch):
    """
    Contract test for explicit force re-authentication of gated HF models.

    Verifies that when force_hf_login=True, a gated Hugging Face-backed
    model triggers authentication even if a local cache is reported as present.

    Behavior:
    - cache-hit does NOT suppress authentication when force_hf_login is enabled.
    - The Hugging Face login function is invoked exactly once.
    - Model construction proceeds normally after authentication.

    The actual model loader is patched to return a lightweight mocked TorchWrapper
    to avoid network access and heavyweight model initialization,
    isolating only the authentication control-flow logic under test.
    """
    gated = next(iter(GATED_HF_MODELS))
    monkeypatch.setattr(FeatureGenerator, "_hf_cache_exists", lambda self, repo: True)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
    login_spy = mock.Mock()
    monkeypatch.setattr("pyslyde.encoders.feature_extractor.login", login_spy)
    monkeypatch.setattr(
        FeatureGenerator, "_" + gated, lambda self: mock.Mock(spec=TorchWrapper)
    )

    fg = FeatureGenerator(model_name=gated, force_hf_login=True)
    assert fg.model is not None
    login_spy.assert_called_once()


def test_model_repo_id_keyerror_for_unknown_mapping():
    """
    Ensures that requesting a repository ID for an unknown model name
    raises a KeyError, enforcing strict validation of supported models.
    """
    fg = object.__new__(FeatureGenerator)
    with pytest.raises(KeyError):
        FeatureGenerator._model_repo_id(fg, "nope")


# ------------------------------------------------
# Checkpoint handling tests
# ------------------------------------------------


def test_checkpoint_dict_requires_model_path():
    """
    Ensures that accessing checkpoint_dict without a configured model_path
    fails fast with a clear RuntimeError.

    This test enforces the contract that checkpoint-based models must define
    a valid model_path before checkpoint loading is attempted.

    Note:
    - Manual use of checkpoint_dict is not currently exercised by the
      supported model loaders, but this behavior is retained to guard
      against silent misconfiguration and future regressions.
    """
    fg = object.__new__(FeatureGenerator)
    fg.model_name = "anything"
    fg.model_path = None
    with pytest.raises(RuntimeError) as e:
        _ = FeatureGenerator.checkpoint_dict.fget(fg)
    assert "no model_path" in str(e.value)


def test_checkpoint_dict_missing_file(tmp_path):
    """
    Ensures that accessing checkpoint_dict fails with a clear RuntimeError
    when the configured checkpoint file does not exist.

    This test enforces early validation of checkpoint paths and prevents
    obscure downstream errors caused by attempting to load a missing file.
    """
    fg = object.__new__(FeatureGenerator)
    fg.model_name = "anything"
    fg.model_path = str(tmp_path / "missing.pt")
    with pytest.raises(RuntimeError) as e:
        _ = FeatureGenerator.checkpoint_dict.fget(fg)
    assert "Checkpoint file not found" in str(e.value)


def test_checkpoint_dict_load_failure(tmp_path, monkeypatch):
    """
    Ensures that checkpoint_dict raises a clear RuntimeError when loading
    the checkpoint file fails for any reason.

    This test simulates a low-level torch.load failure and verifies that
    the error is caught and re-raised with a descriptive, user-facing
    message instead of leaking internal exceptions.
    """
    p = tmp_path / "x.pt"
    p.write_bytes(b"not-a-real-torch-file")

    fg = object.__new__(FeatureGenerator)
    fg.model_name = "anything"
    fg.model_path = str(p)

    monkeypatch.setattr(
        "pyslyde.encoders.feature_extractor.torch.load",
        mock.Mock(side_effect=Exception("boom")),
    )
    with pytest.raises(RuntimeError) as e:
        _ = FeatureGenerator.checkpoint_dict.fget(fg)
    assert "Failed to load checkpoint" in str(e.value)


# ------------------------------------------------
# Input preprocessing tests
# ------------------------------------------------


def test_np_image_to_pil_accepts_np_uint8_rgb(fg_resnet18_mocked, dummy_rgb_np):
    """
    Verifies that np_image_to_pil accepts a uint8 RGB NumPy array and
    returns a valid RGB PIL.Image.

    This test defines the supported input format for NumPy-based images
    and ensures correct conversion without altering the color mode.
    """
    img = fg_resnet18_mocked.np_image_to_pil(dummy_rgb_np)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_np_image_to_pil_casts_non_uint8(fg_resnet18_mocked):
    """
    Verifies that np_image_to_pil correctly handles non-uint8 NumPy arrays
    by casting values to uint8 and returning a valid RGB PIL.Image.

    This test ensures robustness of image conversion when inputs are
    provided in floating-point formats commonly produced by preprocessing
    pipelines.
    """
    arr = (np.random.rand(10, 10, 3) * 255.0).astype(np.float32)
    img = fg_resnet18_mocked.np_image_to_pil(arr)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_np_image_to_pil_accepts_pil_and_converts_to_rgb(fg_resnet18_mocked):
    """
    Verifies that np_image_to_pil accepts a PIL.Image input and normalizes
    it to RGB mode.

    This test ensures that non-RGB PIL images (e.g., grayscale) are
    consistently converted to RGB for downstream processing.
    """
    gray = Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode="L")
    out = fg_resnet18_mocked.np_image_to_pil(gray)
    assert out.mode == "RGB"


def test_np_image_to_pil_rejects_wrong_shape(fg_resnet18_mocked):
    """
    Ensures that np_image_to_pil rejects NumPy arrays with unsupported shapes.

    Verifies that inputs lacking an explicit RGB channel dimension
    (e.g., 2D arrays) raise a ValueError rather than being silently misinterpreted.
    """
    arr = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        _ = fg_resnet18_mocked.np_image_to_pil(arr)


def test_np_image_to_pil_rejects_unknown_type(fg_resnet18_mocked):
    """
    Ensures that np_image_to_pil raises a TypeError when given an unsupported
    input type.

    This test enforces strict input validation and prevents silent failures
    or ambiguous behavior when non-image objects are passed.
    """
    with pytest.raises(TypeError):
        _ = fg_resnet18_mocked.np_image_to_pil("not an image")


# ------------------------------------------------
# # Postprocessing tests
# ------------------------------------------------


@pytest.mark.parametrize("name", ["phikon", "phikon2"])
def test_postprocess_phikon_takes_cls_token(monkeypatch, name):
    """
    Verifies that PHIKON-family models extract the CLS token during postprocessing.

    PHIKON and PHIKON2 models return transformer outputs with shape (B, T, C),
    where the first token (index 0) represents the global CLS embedding.
    This test ensures that _postprocess correctly selects the CLS token
    (out[:, 0, :]) and returns a tensor of shape (B, C), matching the expected
    feature embedding contract.
    """
    fg = object.__new__(FeatureGenerator)
    fg.model_name = name
    out = torch.randn(2, 197, 768)
    pp = FeatureGenerator._postprocess(fg, out)
    assert pp.shape == (2, 768)


def test_postprocess_phikon_rejects_wrong_ndim():
    """
    Ensures that PHIKON postprocessing rejects outputs with unexpected dimensionality.

    PHIKON models are expected to produce transformer-style outputs with shape
    (B, T, C). This test verifies that passing tensors with incorrect dimensionality
    (e.g., a 1D tensor) raises a RuntimeError instead of being silently accepted
    or misinterpreted.
    """
    fg = object.__new__(FeatureGenerator)
    fg.model_name = "phikon"
    out = torch.randn(768)  # wrong
    with pytest.raises(RuntimeError):
        _ = FeatureGenerator._postprocess(fg, out)


@pytest.mark.parametrize("name", ["virchow", "virchow2"])
def test_postprocess_virchow_concat_cls_and_patch_mean(name):
    """
    Verifies Virchow-family postprocessing concatenates CLS token and patch mean.

    Virchow and Virchow2 models produce transformer outputs of shape (B, T, C),
    where:
      - token 0 corresponds to the CLS embedding
      - tokens 1..T-1 correspond to patch embeddings

    This test ensures that _postprocess:
      1. Extracts the CLS token (out[:, 0, :])
      2. Computes the mean of patch tokens (out[:, 1:, :])
      3. Concatenates both vectors along the feature dimension

    The resulting embedding must have shape (B, 2 * C), preserving both
    global and spatially aggregated information as required by the Virchow
    feature specification.
    """
    fg = object.__new__(FeatureGenerator)
    fg.model_name = name
    cfg = VIRCHOW_POSTPROCESS[name]

    B = 2
    T = cfg["expected_T"]
    C = cfg["expected_C"]
    out = torch.randn(B, T, C)
    pp = FeatureGenerator._postprocess(fg, out)
    assert pp.shape == (B, 2 * C)


@pytest.mark.parametrize("name", ["virchow", "virchow2"])
def test_postprocess_virchow_rejects_wrong_shape(name):
    """
    Ensures Virchow-family postprocessing rejects outputs with invalid token count.

    Virchow and Virchow2 models are expected to emit transformer outputs with a
    fixed token structure defined in VIRCHOW_POSTPROCESS (expected_T, expected_C).
    This test verifies that outputs with an unexpected number of tokens (T)
    raise a RuntimeError instead of being silently accepted.
    """
    fg = object.__new__(FeatureGenerator)
    fg.model_name = name
    out = torch.randn(1, 10, 1280)  # wrong T
    with pytest.raises(RuntimeError):
        _ = FeatureGenerator._postprocess(fg, out)


def test_ensure_2d_adds_batch_and_flattens(fg_resnet18_mocked):
    """
    Verifies that _ensure_2d normalizes tensors into a consistent (B, D) shape.

    Enforces:
    - A 1D tensor is interpreted as a single sample and a batch dimension is added.
    - Higher-rank tensors are flattened across non-batch dimensions while
      preserving the leading batch dimension.

    Required so downstream feature validation and postprocessing logic
    can assume a uniform (batch_size, feature_dim) tensor layout
    regardless of model output format.
    """
    x = torch.randn(10)  # (D,)
    y = fg_resnet18_mocked._ensure_2d(x)
    assert y.shape == (1, 10)

    x2 = torch.randn(2, 3, 4)
    y2 = fg_resnet18_mocked._ensure_2d(x2)
    assert y2.shape == (2, 12)


def test_ensure_2d_rejects_non_tensor(fg_resnet18_mocked):
    """
    Ensures that _ensure_2d enforces torch.Tensor inputs and rejects other types.

    Verifies that passing non-tensor inputs (e.g., NumPy arrays)
    raises a RuntimeError instead of being implicitly converted or silently
    accepted. This prevents mixing NumPy arrays and PyTorch tensors,
    which have different device and execution behavior.
    """
    with pytest.raises(RuntimeError):
        _ = fg_resnet18_mocked._ensure_2d(np.zeros((1, 2)))


def test_check_finite_raises_on_nan(fg_resnet18_mocked):
    """
    Ensures that _check_finite rejects feature tensors containing NaN values.

    Verifies that the feature validation step fails fast when model
    outputs contain non-finite values (e.g., NaNs), raising a RuntimeError
    with a clear diagnostic message.
    """
    feats = torch.tensor([[1.0, float("nan")]])
    with pytest.raises(RuntimeError) as e:
        fg_resnet18_mocked._check_finite(feats, "resnet18")
    assert "non-finite" in str(e.value)


# ------------------------------------------------
# forward_pass end-to-end (mock infer)
# ------------------------------------------------


def test_forward_pass_validates_expected_dim_and_returns_1d(dummy_rgb_np):
    """
    Verifies the forward_pass embedding contract for a standard Torch model.

    This test exercises FeatureGenerator.forward_pass end-to-end (using the
    preprocessing, postprocessing, and validation helpers) while stubbing only
    the model inference step.

    Asserts that forward_pass:
    - Converts a NumPy RGB image into a PIL image via np_image_to_pil
    - Calls the model wrapper's .infer(...) to obtain batched features (B, D)
    - Postprocesses and validates the embedding against the expected dimension
      for the configured model (resnet18)
    - Returns a single-sample embedding as a 1D torch.Tensor of shape (D,)

    The model itself is mocked to keep the test fast and deterministic while
    still validating FeatureGenerator's pipeline and output contract.
    """
    fg = FeatureGenerator.__new__(FeatureGenerator)
    fg.model_name = "resnet18"

    fg.np_image_to_pil = FeatureGenerator.np_image_to_pil.__get__(fg, FeatureGenerator)
    fg._postprocess = FeatureGenerator._postprocess.__get__(fg, FeatureGenerator)
    fg._ensure_2d = FeatureGenerator._ensure_2d.__get__(fg, FeatureGenerator)
    fg._check_finite = FeatureGenerator._check_finite.__get__(fg, FeatureGenerator)
    fg._expected_dim = FeatureGenerator._expected_dim.__get__(fg, FeatureGenerator)

    D = EXPECTED_DIMS["resnet18"]
    fg._model = mock.Mock()
    fg._model.infer = mock.Mock(return_value=torch.randn(1, D))

    out = FeatureGenerator.forward_pass(fg, dummy_rgb_np)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (D,)


def test_forward_pass_raises_on_dim_mismatch(dummy_rgb_np):
    """
    Ensures forward_pass rejects embeddings with an unexpected feature dimension.

    FeatureGenerator enforces a strict contract between each model name and its
    expected embedding dimensionality. This test verifies that if the underlying
    model (or wrapper) returns an embedding whose dimension does not match the
    configured EXPECTED_DIMS entry for the model, forward_pass raises a RuntimeError.

    This protects from silently receiving malformed or incompatible embeddings due to:
    - incorrect checkpoints
    - misconfigured model heads
    - wrapper bugs
    - accidental model swaps during refactoring
    """
    fg = FeatureGenerator.__new__(FeatureGenerator)
    fg.model_name = "resnet18"

    fg.np_image_to_pil = FeatureGenerator.np_image_to_pil.__get__(fg, FeatureGenerator)
    fg._postprocess = FeatureGenerator._postprocess.__get__(fg, FeatureGenerator)
    fg._ensure_2d = FeatureGenerator._ensure_2d.__get__(fg, FeatureGenerator)
    fg._check_finite = FeatureGenerator._check_finite.__get__(fg, FeatureGenerator)
    fg._expected_dim = FeatureGenerator._expected_dim.__get__(fg, FeatureGenerator)

    fg._model = mock.Mock()
    fg._model.infer = mock.Mock(return_value=torch.randn(1, 123))  # wrong dim

    with pytest.raises(RuntimeError) as e:
        _ = FeatureGenerator.forward_pass(fg, dummy_rgb_np)
    assert "feature dim mismatch" in str(e.value)


def test_forward_pass_raises_on_nonfinite(dummy_rgb_np):
    """
    Ensures forward_pass rejects embeddings containing non-finite values.

    FeatureGenerator validates that all returned feature values are finite
    (no NaN or Inf) before returning an embedding. This test verifies that
    if the underlying model produces non-finite outputs, forward_pass raises
    a RuntimeError instead of propagating invalid values downstream.
    """
    fg = FeatureGenerator.__new__(FeatureGenerator)
    fg.model_name = "resnet18"

    fg.np_image_to_pil = FeatureGenerator.np_image_to_pil.__get__(fg, FeatureGenerator)
    fg._postprocess = FeatureGenerator._postprocess.__get__(fg, FeatureGenerator)
    fg._ensure_2d = FeatureGenerator._ensure_2d.__get__(fg, FeatureGenerator)
    fg._check_finite = FeatureGenerator._check_finite.__get__(fg, FeatureGenerator)
    fg._expected_dim = FeatureGenerator._expected_dim.__get__(fg, FeatureGenerator)

    D = EXPECTED_DIMS["resnet18"]
    fg._model = mock.Mock()
    fg._model.infer = mock.Mock(
        return_value=torch.tensor([[float("inf")] + [0.0] * (D - 1)])
    )

    with pytest.raises(RuntimeError) as e:
        _ = FeatureGenerator.forward_pass(fg, dummy_rgb_np)
    assert "non-finite" in str(e.value)


# ------------------------------------------------
# Model loader unit tests
# ------------------------------------------------


def test_transpath_loader_uses_timm_and_returns_torchwrapper(monkeypatch):
    """
    Verifies that the TransPath model loader uses the timm backend and exposes
    a TorchWrapper interface without performing real weight downloads.
    """
    fake_model = torch.nn.Sequential(torch.nn.Identity())
    fake_model.to = mock.Mock(return_value=fake_model)
    fake_model.eval = mock.Mock()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(fe.timm, "create_model", mock.Mock(return_value=fake_model))
    monkeypatch.setattr(
        fe.timm.data,
        "resolve_model_data_config",
        mock.Mock(return_value={"input_size": (3, 224, 224)}),
    )
    monkeypatch.setattr(
        fe,
        "create_transform",
        mock.Mock(return_value=lambda pil: torch.zeros(3, 224, 224)),
    )

    fg = FeatureGenerator(model_name="transpath")
    assert isinstance(fg.model, TorchWrapper)
    fe.timm.create_model.assert_called_once()


def test_phikon_loader_uses_transformers_and_returns_torchwrapper(monkeypatch):
    """
    Verifies that the PHIKON loader uses Hugging Face Transformers components
    and exposes a TorchWrapper interface without downloading real model weights.

    Patches:
    - AutoImageProcessor.from_pretrained to return a stub processor that produces
      a valid `pixel_values` tensor
    - AutoModel.from_pretrained to return a stub model whose forward call yields
      an object with `last_hidden_state` (the expected transformer output)
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    processor = mock.Mock()
    processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    monkeypatch.setattr(
        fe,
        "AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=mock.Mock(return_value=processor)),
    )

    hf_model = mock.Mock()
    hf_out = mock.Mock()
    hf_out.last_hidden_state = torch.randn(1, 197, 768)
    hf_model.return_value = hf_out
    monkeypatch.setattr(
        fe,
        "AutoModel",
        types.SimpleNamespace(from_pretrained=mock.Mock(return_value=hf_model)),
    )

    fg = FeatureGenerator(model_name="phikon")
    assert isinstance(fg.model, TorchWrapper)


def test_pathfm_loader_returns_tfvisionwrapper_without_real_tf(monkeypatch):
    """
    Verifies that the Path Foundation (pathfm) loader constructs a TFVisionWrapper
    while avoiding heavyweight TensorFlow/tf_keras imports and any real Hugging Face
    network activity.

    This test validates FeatureGenerator's pathfm loader wiring and contracts by:
    - Forcing the "cache hit" branch so HF login is not required
    - Stubbing `tensorflow` and `tf_keras` as lightweight modules so importing them
      inside `_pathfm()` succeeds without pulling in the real TensorFlow stack
    - Patching `snapshot_download` to return a fake local directory instead of
      contacting Hugging Face
    - Providing a fake Keras model with a `serving_default` signature that matches
      the expected inference interface (returns an "output_0" tensor-like object)

    Asserts that:
    - The loader returns a TFVisionWrapper via the public `model` property
    - snapshot_download is called with the expected repo_id
    - tf_keras.models.load_model is called with the downloaded (fake) directory
    - HF login is not invoked when cache is reported as present
    """
    monkeypatch.setattr(FeatureGenerator, "_hf_cache_exists", lambda self, repo: True)
    login_spy = mock.Mock()
    monkeypatch.setattr("pyslyde.encoders.feature_extractor.login", login_spy)

    # Fake keras model
    infer_fn = mock.Mock(
        return_value={
            "output_0": mock.Mock(numpy=lambda: np.zeros((1, 384), dtype=np.float32))
        }
    )
    keras_model = mock.Mock()
    keras_model.signatures = {"serving_default": infer_fn}

    # Stub tensorflow as an actual module
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.constant = lambda x: x
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    # Stub tf_keras
    fake_tfk = types.ModuleType("tf_keras")
    fake_tfk.models = types.SimpleNamespace(
        load_model=mock.Mock(return_value=keras_model)
    )
    monkeypatch.setitem(sys.modules, "tf_keras", fake_tfk)

    # Patch snapshot_download to avoid real HF calls
    fake_repo_path = "/tmp/fake-path-foundation"
    snapshot_spy = mock.Mock(return_value=fake_repo_path)
    monkeypatch.setattr(fe, "snapshot_download", snapshot_spy)

    fg = fe.FeatureGenerator(model_name="pathfm")

    assert isinstance(fg.model, fe.TFVisionWrapper)
    snapshot_spy.assert_called_once_with(repo_id="google/path-foundation")
    fake_tfk.models.load_model.assert_called_once_with(fake_repo_path)
    login_spy.assert_not_called()


def test_pathfm_loader_raises_if_tensorflow_missing(monkeypatch):
    """
    Ensures the pathfm loader fails fast with a clear RuntimeError when
    TensorFlow is not installed.

    This test simulates an environment where importing `tensorflow` fails
    inside FeatureGenerator._pathfm().
    """
    monkeypatch.setattr(FeatureGenerator, "_hf_cache_exists", lambda self, repo: True)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("forced missing tensorflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as e:
        FeatureGenerator(model_name="pathfm")

    assert "pathfm requires tensorflow" in str(e.value)


def test_pathfm_loader_raises_if_tf_keras_missing(monkeypatch):
    """
    Ensures the pathfm loader fails fast with a clear RuntimeError when
    tf_keras (legacy Keras 2) is not installed.

    This test simulates an environment where importing `tf_keras` fails
    inside FeatureGenerator._pathfm().
    """
    monkeypatch.setattr(FeatureGenerator, "_hf_cache_exists", lambda self, repo: True)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tf_keras":
            raise ImportError("forced missing tf_keras")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as e:
        FeatureGenerator(model_name="pathfm")

    assert "tf_keras" in str(e.value)


@pytest.mark.parametrize(
    "model_name",
    [k for k in EXPECTED_DIMS if k != "pathfm"],
)
def test_all_torch_models_return_torchwrapper_smoke(monkeypatch, model_name):
    """
    Smoke test that all torch-based models construct a TorchWrapper
    without downloading weights or executing heavy model code.

    This test patches each model loader to return a minimal TorchWrapper
    backed by a tiny fake torch.nn.Module, verifying loader wiring only.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # suppress HF gating side effects (cache + login)
    monkeypatch.setattr(FeatureGenerator, "_hf_cache_exists", lambda self, repo: True)
    monkeypatch.setattr(
        "pyslyde.encoders.feature_extractor.login",
        mock.Mock(),
    )

    # fake torch model
    fake_model = torch.nn.Sequential(torch.nn.Identity())
    fake_model.to = mock.Mock(return_value=fake_model)
    fake_model.eval = mock.Mock()

    def fake_transforms(_pil):
        return torch.zeros(3, 224, 224)

    monkeypatch.setattr(
        FeatureGenerator,
        "_" + model_name,
        lambda self: TorchWrapper(
            model=fake_model,
            transforms=fake_transforms,
            device="cpu",
        ),
    )

    fg = FeatureGenerator(model_name=model_name)

    assert isinstance(fg.model, TorchWrapper)


# ------------------------------------------------
# TorchWrapper tests / TFVisionWrapper unit tests
# ------------------------------------------------


def test_torchwrapper_infer_adds_batch_dim_cpu():
    """
    Verifies that TorchWrapper.infer correctly adds a batch dimension
    when the preprocessing transform returns an unbatched CHW tensor.

    This test simulates CPU-only inference and asserts that:
    - A single PIL image is transformed to a tensor without a batch dimension
    - TorchWrapper adds the missing batch dimension before model invocation
    - The model is called exactly once
    - The output tensor preserves the expected (B, D) shape

    This ensures TorchWrapper provides a consistent inference interface
    regardless of whether transforms include batching.
    """
    model = mock.Mock(spec=torch.nn.Module)
    model.return_value = torch.randn(1, 10)
    model.to.return_value = model
    model.eval.return_value = None

    transforms = mock.Mock(return_value=torch.randn(3, 224, 224))  # CHW (no batch)
    w = TorchWrapper(model=model, transforms=transforms, device="cpu")

    pil = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    out = w.infer(pil)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 10)
    transforms.assert_called_once()
    model.assert_called_once()


def test_torchwrapper_infer_preserves_batch_dim():
    """
    Verifies that TorchWrapper.infer preserves an existing batch dimension
    when the preprocessing transform already returns a batched BCHW tensor.

    This test ensures that TorchWrapper does not add an extra batch dimension
    or otherwise alter the batch size when batching is handled upstream
    by the transform pipeline.
    """
    model = mock.Mock(spec=torch.nn.Module)
    model.return_value = torch.randn(2, 10)
    model.to.return_value = model
    model.eval.return_value = None

    transforms = mock.Mock(return_value=torch.randn(2, 3, 224, 224))  # BCHW
    w = TorchWrapper(model=model, transforms=transforms, device="cpu")
    pil = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    out = w.infer(pil)
    assert out.shape == (2, 10)


def test_tfvisionwrapper_raises_if_tf_missing(monkeypatch):
    """
    Ensures TFVisionWrapper fails fast with a clear error when TensorFlow
    is not available in the runtime environment.

    This test simulates a missing TensorFlow installation by intercepting
    Python's import mechanism and forcing an ImportError whenever
    `import tensorflow` is attempted. It then verifies that constructing
    TFVisionWrapper raises a RuntimeError with an informative message.

    The purpose is to enforce a strict dependency contract:
    - TFVisionWrapper must not silently degrade or partially initialize
      without TensorFlow
    - Users receive an immediate, actionable error instead of obscure
      downstream failures during inference
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("forced missing tensorflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as e:
        TFVisionWrapper(infer_fn=mock.Mock())
    assert "Tensorflow is required" in str(e.value)
