"""
feature_extractor.py currently supports extraction of only tile-based embeddings.
Extraction of slide-level embeddings is not yet supported.
"""

import os

import numpy as np
import timm
import torch
import torchvision.models as models
from huggingface_hub import hf_hub_download, login, snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModel

from pyslyde.encoders.ctran import ConvStem

GATED_HF_MODELS = {
    "uni",
    "uni2",
    "virchow",
    "virchow2",
    "gigapath",
    "hoptimus0",
    "hoptimus1",
    "pathfm",
}

EXPECTED_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "vgg16": 25088,
    "uni": 1024,
    "uni2": 1536,
    "virchow": 2560,
    "virchow2": 2560,
    "gigapath": 1536,
    "hoptimus0": 1536,
    "hoptimus1": 1536,
    "transpath": 768,
    "pathfm": 384,
    "phikon": 768,
    "phikon2": 1024,
}

VIRCHOW_POSTPROCESS = {
    "virchow": {"patch_start": 1, "expected_T": 257, "expected_C": 1280},
    "virchow2": {"patch_start": 5, "expected_T": 261, "expected_C": 1280},
}


class TorchWrapper:
    """
    Unified wrapper for PyTorch-based vision models.

    Handles input preprocessing, device placement, and inference-time
    execution to expose a consistent .infer(PIL.Image) -> torch.Tensor interface.
    """

    def __init__(self, model, transforms, device):
        self.model = model.to(device)
        self.transforms = transforms
        self.device = device
        self.model.eval()

    def infer(self, pil_img):
        x = self.transforms(pil_img)

        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        with torch.inference_mode():
            if self.device.startswith("cuda"):
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = self.model(x)
            else:
                out = self.model(x)

        return out.cpu()


class HFVisionWrapper(torch.nn.Module):
    """
    Wrapper for Hugging Face vision models to standardize their forward output.

    Adapts Hugging Face vision backbones so that a BCHW input tensor
    produces a raw token-level output (last_hidden_state), enabling
    consistent downstream postprocessing across all models.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def infer(self, pil_img):
        raise RuntimeError(
            "HFVisionWrapper must be wrapped by TorchWrapper for transforms + device handling."
        )

    def forward(self, x):
        out = self.hf_model(pixel_values=x)
        return out.last_hidden_state


class TFVisionWrapper:
    """
    Wrapper for TensorFlow/Keras vision models that exposes a PyTorch-like
    inference interface.

    Converts PIL images to TensorFlow tensors, runs inference via a TF
    serving signature, and returns embeddings as torch.Tensor for
    compatibility with the rest of the pipeline.
    """

    def __init__(self, infer_fn, image_size=(224, 224)):
        """
        Initialize the TensorFlow vision wrapper.

        Parameters:
        - infer_fn: TensorFlow serving function (e.g. model.signatures["serving_default"])
        - image_size: Target (H, W) resolution for input images.
        """
        self.infer_fn = infer_fn
        self.image_size = image_size

        try:
            import tensorflow as tf
        except ImportError as e:
            raise RuntimeError("Tensorflow is required but not found.") from e

        self.tf = tf

    def preprocess(self, img: Image.Image):
        """
        Preprocess a PIL image for TensorFlow inference.

        Converts the input to RGB, resizes to the configured image size,
        normalizes pixel values to [0, 1], and returns a batched
        TensorFlow tensor suitable for the model's serving signature.
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize(self.image_size[::-1], resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return self.tf.constant(arr)

    def infer(self, img: Image.Image):
        """
        Run inference on an image using the TensorFlow model.

        Applies preprocessing, invokes the TensorFlow serving signature,
        and converts the resulting embedding to a torch.Tensor for
        compatibility with the PyTorch-based pipeline.
        """
        x_tf = self.preprocess(img)
        out = self.infer_fn(x_tf)

        if "output_0" in out:
            emb = out["output_0"].numpy()
        elif len(out) == 1:
            emb = next(iter(out.values())).numpy()
        else:
            raise RuntimeError(f"Unexpected TF model outputs: {list(out.keys())}")

        return torch.from_numpy(emb)


class FeatureGenerator:
    """
    Factory and interface for extracting feature embeddings from vision models.

    Instantiates and manages model-specific feature extractors across
    different backends (PyTorch, Hugging Face, TensorFlow), and provides
    a unified forward_pass interface that returns validated, fixed-length
    embedding vectors.
    """

    def __init__(self, model_name, model_path=None):
        """
        Initialize a feature generator for the specified model.

        Parameters:
        - model_name: Identifier of the feature extraction backbone to use.
        - model_path: Optional path to a user-provided checkpoint for models
          that require external weights (reserved for future use).
        """
        self.model_path = model_path

        self._model = None
        self.transforms = None
        self._hf_logged_in = False

        self.model_name = model_name
        self.model = model_name

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @model.setter
    def model(self, value):
        m_name = "_" + value
        if not hasattr(self, m_name):
            supported = sorted(
                k[1:]
                for k in dir(self)
                if k.startswith("_") and callable(getattr(self, k))
            )
            raise ValueError(f"Unknown model '{value}'. Supported: {supported}")

        if value in GATED_HF_MODELS:
            repo = self._model_repo_id(value)
            if not self._hf_cache_exists(repo):
                self._hf_login()

        self._model = getattr(self, "_" + value)()

    @property
    def checkpoint_dict(self):
        """
        Provides ability to load model weights from a user-defined location,
        should future need arise.
        """
        if self.model_path is None:
            raise RuntimeError(
                f"Model '{self.model_name}' requires a pretrained checkpoint, "
                f"but no model_path was provided."
            )

        if not os.path.isfile(self.model_path):
            raise RuntimeError(
                f"Checkpoint file not found for model '{self.model_name}': "
                f"{self.model_path}"
            )

        try:
            return torch.load(self.model_path, map_location=torch.device("cpu"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint for model '{self.model_name}' "
                f"from '{self.model_path}'."
            ) from e

    def _hf_login(self):
        """
        Logs into HF using the HUGGINGFACE_TOKEN
        environment variable (set by the user).
        """
        if self._hf_logged_in:
            return

        token = os.getenv("HUGGINGFACE_TOKEN")
        if token is None:
            raise RuntimeError(
                "Environment variable HUGGINGFACE_TOKEN is required for HF models."
            )

        login(token)
        self._hf_logged_in = True

    def _hf_cache_exists(self, repo_id: str) -> bool:
        """
        Returns True if it can find a known file for the repo in the local HF cache.
        Checks for 'config.json' first, but fall back to a typical weight file.
        """
        for fname in (
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "model.safetensors",
        ):
            try:
                hf_hub_download(repo_id, filename=fname, local_files_only=True)
                print(f"Local cache exists for {self.model_name} at {repo_id}")
                return True
            except LocalEntryNotFoundError:
                continue
            except Exception:
                continue
        return False

    def _model_repo_id(self, name: str) -> str:
        """
        Maps internal model_name to HF repo id.

        Used for checking local cache availability or determining
        whether login is required for gated repositories.
        """
        repo_map = {
            "uni": "MahmoodLab/uni",
            "uni2": "MahmoodLab/UNI2-h",
            "virchow": "paige-ai/Virchow",
            "virchow2": "paige-ai/Virchow2",
            "gigapath": "prov-gigapath/prov-gigapath",
            "hoptimus0": "bioptimus/H-optimus-0",
            "hoptimus1": "bioptimus/H-optimus-1",
            "pathfm": "google/path-foundation",
            "phikon": "owkin/phikon",
            "phikon2": "owkin/phikon-v2",
            "transpath": "1aurent/swin_tiny_patch4_window7_224.CTransPath",
        }

        if name not in repo_map:
            raise KeyError(f"No Hugging Face repo mapping found for model '{name}'")

        return repo_map[name]

    def _resnet18(self):
        """
        Standard torchvision ResNet-18 feature extractor.
        """
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        transforms = weights.transforms()
        return TorchWrapper(model, transforms, self.device)

    def _resnet50(self):
        """
        Standard torchvision ResNet-50 feature extractor.
        """
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        transforms = weights.transforms()
        return TorchWrapper(model, transforms, self.device)

    def _vgg16(self):
        """
        Standard torchvision VGG16 feature extractor.
        """
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        weights = models.VGG16_Weights.DEFAULT
        transforms = weights.transforms()
        return TorchWrapper(model, transforms, self.device)

    def _transpath(self):
        """
        See https://huggingface.co/1aurent/swin_tiny_patch4_window7_224.CTransPath
        """
        model = timm.create_model(
            model_name="hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
            embed_layer=ConvStem,
            pretrained=True,
        )
        data_config = timm.data.resolve_model_data_config(model)
        transforms = create_transform(**data_config, is_training=False)
        return TorchWrapper(model, transforms, self.device)

    def _phikon(self):
        """
        Phikon (ViT-B/16) feature extractor.
        See https://huggingface.co/owkin/phikon
        """
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        model = AutoModel.from_pretrained("owkin/phikon")
        wrapper = HFVisionWrapper(model)
        transforms = self._hf_image_transform(processor)
        return TorchWrapper(wrapper, transforms, self.device)

    def _phikon2(self):
        """
        Phikon-v2 (ViT-L/16) feature extractor.
        See https://huggingface.co/owkin/phikon-v2
        """
        processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        wrapper = HFVisionWrapper(model)
        transforms = self._hf_image_transform(processor)
        return TorchWrapper(wrapper, transforms, self.device)

    def _uni(self):
        """
        See https://huggingface.co/MahmoodLab/UNI
        """
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        transforms = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return TorchWrapper(model, transforms, self.device)

    def _uni2(self):
        """
        See https://huggingface.co/MahmoodLab/UNI2-h
        """
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        transforms = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return TorchWrapper(model, transforms, self.device)

    def _virchow(self):
        """
        See https://huggingface.co/paige-ai/Virchow
        """
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        transforms = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return TorchWrapper(model, transforms, self.device)

    def _virchow2(self):
        """
        See https://huggingface.co/paige-ai/Virchow2
        """
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        transforms = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return TorchWrapper(model, transforms, self.device)

    def _gigapath(self):
        """
        See https://huggingface.co/prov-gigapath/prov-gigapath

        Note: For tile (not slide) encoding
        """
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        transforms = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return TorchWrapper(model, transforms, self.device)

    def _hoptimus0(self):
        """
        See https://huggingface.co/bioptimus/H-optimus-0
        """
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        return TorchWrapper(model, transforms, self.device)

    def _hoptimus1(self):
        """
        See https://huggingface.co/bioptimus/H-optimus-1
        """
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        return TorchWrapper(model, transforms, self.device)

    def _pathfm(self):
        """
        Google Path Foundation model (TensorFlow/Keras) from HF.
        See https://huggingface.co/google/path-foundation

        Notes:
        - This model runs in TensorFlow (not PyTorch).
        - Outputs are converted to torch.Tensor for consistency with the rest of the pipeline.
        - Model loading here uses tf_keras instead of from_pretrained_keras from legacy huggingface_hub,
          as it's no longer available in newer versions of huggingface_hub.

        Returns:
        - TFVisionWrapper: exposes .infer(pil_img) -> torch.Tensor embedding.
        """
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError as e:
            raise RuntimeError("pathfm requires tensorflow to be installed.") from e

        try:
            import tf_keras as tfk
        except ImportError as e:
            raise RuntimeError(
                "pathfm requires tf_keras (legacy Keras 2) to be installed."
            ) from e

        repo_path = snapshot_download(repo_id="google/path-foundation")
        model = tfk.models.load_model(repo_path)
        infer_fn = model.signatures["serving_default"]
        return TFVisionWrapper(infer_fn, image_size=(224, 224))

    def forward_pass(self, image_in):
        img = self.np_image_to_pil(image_in)
        feats = self.model.infer(img)
        feats = self._postprocess(feats)
        feats = self._ensure_2d(feats)
        self._check_finite(feats, self.model_name)
        exp = self._expected_dim()

        if feats.shape[1] != exp:
            raise RuntimeError(
                f"{self.model_name} feature dim mismatch: "
                f"expected {exp}, got {feats.shape[1]}"
            )
        return feats.squeeze(0)

    def np_image_to_pil(self, image_in):
        """
        Converts an input image to a PIL RGB Image.

        Accepts either:
        - a NumPy array of shape (H, W, 3) in RGB order, or
        - a PIL.Image.Image instance.

        Ensures:
        - uint8 pixel dtype (if NumPy input),
        - RGB color mode,
        - consistent PIL.Image.Image output.

        Raises:
        - TypeError for unsupported input types,
        - ValueError for invalid NumPy array shape.
        """
        if isinstance(image_in, np.ndarray):
            if image_in.ndim != 3 or image_in.shape[2] != 3:
                raise ValueError(
                    f"Expected HxWx3 RGB np.ndarray, got shape {image_in.shape}"
                )
            if image_in.dtype != np.uint8:
                image_in = image_in.astype(np.uint8)
            img = Image.fromarray(image_in)
        elif isinstance(image_in, Image.Image):
            img = image_in
        else:
            raise TypeError(f"Unsupported image type: {type(image_in)}")

        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _hf_image_transform(self, processor):
        """
        Wraps HF Image Processor so it behaves like a torchvision transform:
        PIL -> torch.Tensor (C,H,W)
        """

        def _t(pil_img):
            out = processor(images=pil_img, return_tensors="pt")
            return out["pixel_values"].squeeze(0)

        return _t

    def _postprocess(self, out: torch.Tensor) -> torch.Tensor:
        """
        Apply model-specific postprocessing to raw model outputs.

        Normalizes outputs into a consistent feature representation
        as required by the selected backbone.
        """
        name = self.model_name

        if name in {"phikon", "phikon2"}:
            if out.ndim != 3:
                raise RuntimeError(f"{name} expected (B,T,C), got {out.shape}")
            out = out[:, 0, :]

        if name in VIRCHOW_POSTPROCESS:
            cfg = VIRCHOW_POSTPROCESS[name]

            if out.ndim != 3:
                raise RuntimeError(f"{name} expected (B,T,C), got {out.shape}")

            B, T, C = out.shape
            exp_T, exp_C = cfg["expected_T"], cfg["expected_C"]
            if (exp_T is not None and T != exp_T) or (exp_C is not None and C != exp_C):
                raise RuntimeError(
                    f"{name} expected (B,{exp_T},{exp_C}), got {out.shape}"
                )

            class_token = out[:, 0, :]
            patch_mean = out[:, cfg["patch_start"] :, :].mean(dim=1)
            out = torch.cat([class_token, patch_mean], dim=-1)
        return out

    def _ensure_2d(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Ensures feature tensor has shape (B, D).

        Adds a batch dimension if needed and flattens remaining dimensions.
        Raises RuntimeError if the result cannot be represented as 2D.
        """
        if not torch.is_tensor(feats):
            raise RuntimeError(f"Expected torch.Tensor feats, got {type(feats)}")
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        feats = feats.reshape(feats.shape[0], -1)
        if feats.ndim != 2:
            raise RuntimeError(f"Expected feats to be 2D (B,D), got {feats.shape}")
        return feats

    def _check_finite(self, feats: torch.Tensor, name: str) -> None:
        """Validate that the feature tensor contains only finite values."""
        if not torch.isfinite(feats).all():
            raise RuntimeError(f"{name} produced non-finite features (NaN/Inf).")

    def _expected_dim(self) -> int:
        """Return the expected feature embedding dimension for the current model."""
        exp = EXPECTED_DIMS.get(self.model_name)
        if exp is not None:
            return exp
        raise RuntimeError(
            f"No expected dim configured for model '{self.model_name}' "
            f"in EXPECTED_DIMS."
        )
