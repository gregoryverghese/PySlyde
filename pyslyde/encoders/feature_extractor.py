"""
feature_extractor.py currently supports extraction of only tile-based embeddings.

Extraction of slide-level embeddings is not yet supported.

Access to gated Hugging Face models is supported via the HUGGINGFACE_TOKEN
environment variable.

Local model representations can be used instead of obtaining pretrained
weights from the model's default source over the internet, when available.
"""

import os

import numpy as np

try:
    import timm
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from huggingface_hub import hf_hub_download, login, snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked, to_2tuple
    from torchvision import transforms as T
    from transformers import AutoImageProcessor, AutoModel
except ImportError as e:
    raise ImportError(
        "Feature extraction requires the 'feature-extractor' optional "
        "dependency group. Install it with: pip install pyslyde[feature-extractor]"
    ) from e

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
    "ctranspath": 768,
    "pathfm": 384,
    "phikon": 768,
    "phikon2": 1024,
}

MODEL_PREPROCESS_CONFIG = {
    "gigapath": {
        "pipeline": ["resize", "center_crop", "to_tensor", "normalize"],
        "resize": 256,
        "center_crop": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "hoptimus0": {
        "pipeline": ["to_tensor", "normalize"],
        "mean": (0.707223, 0.578729, 0.703617),
        "std": (0.211883, 0.230117, 0.177517),
    },
    "hoptimus1": {
        "pipeline": ["to_tensor", "normalize"],
        "mean": (0.707223, 0.578729, 0.703617),
        "std": (0.211883, 0.230117, 0.177517),
    },
}

VIRCHOW_POSTPROCESS = {
    "virchow": {"patch_start": 1, "expected_T": 257, "expected_C": 1280},
    "virchow2": {"patch_start": 5, "expected_T": 261, "expected_C": 1280},
}


class ConvStem(nn.Module):
    """
    Patch embedding implementation used as a replacement for the
    default embedding layer in Swin Transformer models.

    This implementation is obtained from:
    https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        # flatten=True,
        **kwargs,
    ):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for layer in range(2):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        return x


MODEL_CONFIG = {
    "uni": {
        "backend": "timm",
        "source": "huggingface",
        "hf_repo_id": "MahmoodLab/uni",
        "requires_hf_auth": True,
        "architecture": "vit_large_patch16_224",
        "kwargs": {
            "img_size": 224,
            "patch_size": 16,
            "init_values": 1e-5,
            "num_classes": 0,
            "dynamic_img_size": True,
        },
    },
    "uni2": {
        "backend": "timm",
        "source": "huggingface",
        "hf_repo_id": "MahmoodLab/UNI2-h",
        "requires_hf_auth": True,
        "architecture": "vit_giant_patch14_224",
        "kwargs": {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": SwiGLUPacked,
            "act_layer": nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
    },
    "virchow": {
        "backend": "timm",
        "source": "huggingface",
        "hf_repo_id": "paige-ai/Virchow",
        "requires_hf_auth": True,
        "architecture": "vit_huge_patch14_224",
        "kwargs": {
            "img_size": 224,
            "init_values": 1e-5,
            "num_classes": 0,
            "mlp_ratio": 5.3375,
            "num_heads": 16,
            "embed_dim": 1280,
            "depth": 32,
            "global_pool": "",
            "mlp_layer": SwiGLUPacked,
            "act_layer": nn.SiLU,
            "dynamic_img_size": True,
        },
    },
    "virchow2": {
        "backend": "timm",
        "source": "huggingface",
        "hf_repo_id": "paige-ai/Virchow2",
        "requires_hf_auth": True,
        "architecture": "vit_huge_patch14_224",
        "kwargs": {
            "img_size": 224,
            "init_values": 1e-5,
            "num_classes": 0,
            "reg_tokens": 4,
            "mlp_ratio": 5.3375,
            "global_pool": "",
            "dynamic_img_size": True,
            "mlp_layer": SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        },
    },
    "gigapath": {
        "backend": "timm",
        "source": "huggingface",
        "hf_repo_id": "prov-gigapath/prov-gigapath",
        "requires_hf_auth": True,
        "architecture": "vit_giant_patch14_dinov2",
        "kwargs": {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 1536,
            "depth": 40,
            "num_heads": 24,
            "mlp_ratio": 5.33334,
            "num_classes": 0,
            "dynamic_img_size": True,
        },
    },
    "hoptimus0": {
        "backend": "timm",
        "source": "huggingface",
        "hf_repo_id": "bioptimus/H-optimus-0",
        "requires_hf_auth": True,
        "architecture": "vit_giant_patch14_reg4_dinov2",
        "kwargs": {
            "img_size": 224,
            "init_values": 1e-5,
            "global_pool": "token",
            "num_classes": 0,
            "dynamic_img_size": True,
        },
    },
    "hoptimus1": {
        "backend": "timm",
        "source": "huggingface",
        "hf_repo_id": "bioptimus/H-optimus-1",
        "requires_hf_auth": True,
        "architecture": "vit_giant_patch14_reg4_dinov2",
        "kwargs": {
            "img_size": 224,
            "init_values": 1e-5,
            "global_pool": "token",
            "num_classes": 0,
            "dynamic_img_size": True,
        },
    },
    "ctranspath": {
        "backend": "timm",
        "source": "huggingface",
        "hf_repo_id": "1aurent/swin_tiny_patch4_window7_224.CTransPath",
        "architecture": "swin_tiny_patch4_window7_224",
        "kwargs": {
            "embed_layer": ConvStem,
        },
    },
    "pathfm": {
        "backend": "tensorflow",
        "source": "huggingface",
        "hf_repo_id": "google/path-foundation",
        "requires_hf_auth": True,
    },
    "phikon": {
        "backend": "transformers",
        "source": "huggingface",
        "hf_repo_id": "owkin/phikon",
    },
    "phikon2": {
        "backend": "transformers",
        "source": "huggingface",
        "hf_repo_id": "owkin/phikon-v2",
    },
    "resnet18": {
        "backend": "torchvision",
        "source": "torchvision",
    },
    "resnet50": {
        "backend": "torchvision",
        "source": "torchvision",
    },
    "vgg16": {
        "backend": "torchvision",
        "source": "torchvision",
    },
}

GATED_HF_MODELS = {
    key for key, cfg in MODEL_CONFIG.items() if cfg.get("requires_hf_auth", False)
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


class HFVisionWrapper(nn.Module):
    """
    Wrapper for Hugging Face (HF) vision models to standardize their forward output.

    Adapts HF vision backbones so that a BCHW input tensor
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
        - infer_fn: TensorFlow serving function
        - image_size: Target (H, W) resolution for input images
        """
        self.infer_fn = infer_fn
        self.image_size = image_size

        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow model support requires the 'tensorflow' optional "
                "dependency group. Install it with: pip install pyslyde[tensorflow]"
            ) from e

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
            raise RuntimeError(f"Unexpected TF model outputs: {list(out.keys())}.")

        return torch.from_numpy(emb)


class FeatureGenerator:
    """
    Factory and interface for extracting feature embeddings from vision models.

    Instantiates and manages model-specific feature extractors across
    different backends (PyTorch, Hugging Face (HF), TensorFlow), and provides
    a unified forward_pass interface that returns validated, fixed-length
    embedding vectors.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        force_hf_login: bool = False,
    ):
        """
        Initialize a feature generator for the specified model.

        Parameters:
            model_name: Identifier of the feature extraction backbone to use.
            model_path: Optional path to a local pretrained model representation.
                If provided, the local representation is used instead of the
                model's defaultpretrained weight source (e.g., HF).
                The expected path type depends on the model, for instance:
                    - PyTorch/timm and torchvision models: checkpoint/weights file.
                    - Phikon and Phikon2: local HF directory containing the model,
                      configuration and preprocessing files.
                    - PathFM: local TensorFlow/Keras SavedModel directory.
            force_hf_login: If True, force a new HF login when HF is used as
                the model source.
        """
        self.model_path = model_path

        self._model = None
        self.transforms = None
        self._hf_logged_in = False
        self.force_hf_login = force_hf_login

        self.model_name = model_name
        self.model = model_name

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def hf_repo_id(self) -> str | None:
        """Return the HF repository ID for the current model."""
        return MODEL_CONFIG[self.model_name].get("hf_repo_id")

    @property
    def hf_hub_ref(self) -> str:
        """Return the timm-specific HF hub reference for the current model."""
        repo_id = self.hf_repo_id

        if repo_id is None:
            raise ValueError(
                f"Model '{self.model_name}' does not have a Hugging Face "
                "repository configured."
            )

        return f"hf-hub:{self.hf_repo_id}"

    @model.setter
    def model(self, value):
        m_name = "_" + value
        if not hasattr(self, m_name):
            supported = sorted(
                k[1:]
                for k in dir(self)
                if k.startswith("_") and callable(getattr(self, k))
            )
            raise ValueError(f"Unknown model '{value}'. Supported: {supported}.")

        cfg = MODEL_CONFIG[value]

        if self.model_path is None and cfg["source"] == "huggingface":
            self._prepare_hf_access(value)

        self._model = getattr(self, "_" + value)()

    def _hf_login(self, force=False):
        """
        Logs into HF using the HUGGINGFACE_TOKEN environment variable (set by the user).
        If force=True, re-runs login even if this instance already logged in.
        """
        if self._hf_logged_in and not force:
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
        Returns True if it can find a known file in the local HF cache.
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
                print(f"Local cache exists for {self.model_name} at {repo_id}.")
                return True
            except LocalEntryNotFoundError:
                continue
            except Exception:
                continue
        return False

    def _prepare_hf_access(self, model_name: str):
        """
        Prepare Hugging Face access for a gated model.

        No action is taken for models that do not require authentication.
        For gated models, Hugging Face authentication is forced when
        force_hf_login is enabled; otherwise, authentication is performed
        only when the model is not already available in the local Hugging Face
        cache.
        """
        if model_name not in GATED_HF_MODELS:
            return

        if self.force_hf_login:
            self._hf_login(force=True)
            return

        if not self._hf_cache_exists(self.hf_repo_id):
            self._hf_login()

    def _validate_model_path(self, expected_type: str) -> None:
        """
        Validate that model_path exists and has the expected type.

        Parameters
            expected_type : {"file", "dir"}
                Expected type of model_path:
                - "file": a local checkpoint/weights file.
                - "dir": a local model directory.
        """
        if self.model_path is None:
            raise ValueError(
                f"Cannot load local model for '{self.model_name}': "
                "no model_path was provided."
            )

        if expected_type == "file":
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"Expected model_path for '{self.model_name}' to point to a "
                    f"local checkpoint/weights file, but no file was found: "
                    f"{self.model_path}"
                )

        elif expected_type == "dir":
            if not os.path.isdir(self.model_path):
                raise NotADirectoryError(
                    f"Expected model_path for '{self.model_name}' to point to a "
                    f"local model directory, but no directory was found: "
                    f"{self.model_path}"
                )

        else:
            raise ValueError(
                f"Invalid expected_type '{expected_type}'. Expected 'file' or 'dir'."
            )

    def _load_weights(self):
        """
        Loads model weights from a user-defined location.

        Raises
            ValueError: If ``model_path`` was not provided.
            FileNotFoundError: If ``model_path`` does not point to a file.
            RuntimeError: If the weights cannot be loaded or the checkpoint
                has an unsupported format.
        """
        self._validate_model_path("file")

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint for model '{self.model_name}' "
                f"from '{self.model_path}'."
            ) from e

        if self.model_name == "ctranspath":
            try:
                return checkpoint["model"]
            except (TypeError, KeyError) as e:
                raise RuntimeError(
                    f"Invalid CTransPath checkpoint for '{self.model_name}': "
                    "expected a 'model' key containing the state_dict."
                ) from e

        return checkpoint

    def _load_local_timm_model(self):
        """
        Construct a timm model and load its weights from a local checkpoint.
        """
        cfg = MODEL_CONFIG[self.model_name]

        model = timm.create_model(
            cfg["architecture"],
            pretrained=False,
            **cfg["kwargs"],
        )

        model.load_state_dict(
            self._load_weights(),
            strict=True,
        )

        return model

    def _build_transforms(self, model_name: str):
        """
        Build a torchvision transform pipeline for models with explicit preprocessing configuration.

        Supported pipeline steps:
        - "resize"
        - "center_crop"
        - "to_tensor"
        - "normalize"
        """
        if model_name not in MODEL_PREPROCESS_CONFIG:
            raise KeyError(
                f"No preprocessing config found for '{model_name}'. "
                f"Add it to MODEL_PREPROCESS_CONFIG or use a different transform path."
            )

        cfg = MODEL_PREPROCESS_CONFIG[model_name]
        pipeline = cfg.get(
            "pipeline",
            ["resize", "center_crop", "to_tensor", "normalize"],
        )

        ops = []

        for step in pipeline:
            if step == "resize":
                size = cfg.get("resize")
                if size is not None:
                    ops.append(
                        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)
                    )

            elif step == "center_crop":
                size = cfg.get("center_crop")
                if size is not None:
                    ops.append(T.CenterCrop(size))

            elif step == "to_tensor":
                ops.append(T.ToTensor())

            elif step == "normalize":
                mean = cfg.get("mean")
                std = cfg.get("std")
                if mean is not None and std is not None:
                    ops.append(T.Normalize(mean=mean, std=std))

            else:
                raise ValueError(
                    f"Unknown preprocessing pipeline step '{step}' for '{model_name}'."
                )

        return T.Compose(ops)

    def _resnet18(self):
        """
        Standard torchvision ResNet-18 feature extractor.
        """
        weights = models.ResNet18_Weights.DEFAULT

        if self.model_path is not None:
            model = models.resnet18(weights=None)
            model.load_state_dict(self._load_weights(), strict=True)
        else:
            model = models.resnet18(weights=weights)

        model.fc = nn.Identity()
        transforms = weights.transforms()

        return TorchWrapper(model, transforms, self.device)

    def _resnet50(self):
        """
        Standard torchvision ResNet-50 feature extractor.
        """
        weights = models.ResNet50_Weights.DEFAULT

        if self.model_path is not None:
            model = models.resnet50(weights=None)
            model.load_state_dict(self._load_weights(), strict=True)
        else:
            model = models.resnet50(weights=weights)

        model.fc = nn.Identity()
        transforms = weights.transforms()

        return TorchWrapper(model, transforms, self.device)

    def _vgg16(self):
        """
        Standard torchvision VGG16 feature extractor.
        """
        weights = models.VGG16_Weights.DEFAULT

        if self.model_path is not None:
            model = models.vgg16(weights=None)
            model.load_state_dict(self._load_weights(), strict=True)
        else:
            model = models.vgg16(weights=weights)

        model.classifier = nn.Identity()
        transforms = weights.transforms()

        return TorchWrapper(model, transforms, self.device)

    def _ctranspath(self):
        """
        See unofficial: https://huggingface.co/1aurent/swin_tiny_patch4_window7_224.CTransPath
        and official: https://github.com/Xiyue-Wang/TransPath
        """
        if self.model_path is not None:
            model = self._load_local_timm_model()
        else:
            cfg = MODEL_CONFIG[self.model_name]
            model = timm.create_model(
                model_name=self.hf_hub_ref,
                pretrained=True,
                **cfg["kwargs"],
            )
        data_config = timm.data.resolve_model_data_config(model)
        transforms = create_transform(**data_config, is_training=False)
        return TorchWrapper(model, transforms, self.device)

    def _phikon(self):
        """
        Phikon (ViT-B/16) feature extractor.
        See https://huggingface.co/owkin/phikon
        """
        if self.model_path is not None:
            self._validate_model_path("dir")

            processor = AutoImageProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            model = AutoModel.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        else:
            processor = AutoImageProcessor.from_pretrained(self.hf_repo_id)
            model = AutoModel.from_pretrained(self.hf_repo_id)

        wrapper = HFVisionWrapper(model)
        transforms = self._hf_image_transform(processor)

        return TorchWrapper(wrapper, transforms, self.device)

    def _phikon2(self):
        """
        Phikon-v2 (ViT-L/16) feature extractor.
        See https://huggingface.co/owkin/phikon-v2
        """
        if self.model_path is not None:
            self._validate_model_path("dir")

            processor = AutoImageProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            model = AutoModel.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        else:
            processor = AutoImageProcessor.from_pretrained(self.hf_repo_id)
            model = AutoModel.from_pretrained(self.hf_repo_id)

        wrapper = HFVisionWrapper(model)
        transforms = self._hf_image_transform(processor)

        return TorchWrapper(wrapper, transforms, self.device)

    def _uni(self):
        """
        See https://huggingface.co/MahmoodLab/UNI
        """
        if self.model_path is not None:
            model = self._load_local_timm_model()
        else:
            model = timm.create_model(
                self.hf_hub_ref,
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
        if self.model_path is not None:
            model = self._load_local_timm_model()
        else:
            cfg = MODEL_CONFIG[self.model_name]
            model = timm.create_model(
                self.hf_hub_ref,
                pretrained=True,
                **cfg["kwargs"],
            )
        transforms = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return TorchWrapper(model, transforms, self.device)

    def _virchow(self):
        """
        See https://huggingface.co/paige-ai/Virchow
        """
        if self.model_path is not None:
            model = self._load_local_timm_model()
        else:
            model = timm.create_model(
                self.hf_hub_ref,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=nn.SiLU,
            )
        transforms = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return TorchWrapper(model, transforms, self.device)

    def _virchow2(self):
        """
        See https://huggingface.co/paige-ai/Virchow2
        """
        if self.model_path is not None:
            model = self._load_local_timm_model()
        else:
            model = timm.create_model(
                self.hf_hub_ref,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=nn.SiLU,
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
        if self.model_path is not None:
            model = self._load_local_timm_model()
        else:
            model = timm.create_model(
                self.hf_hub_ref,
                pretrained=True,
            )
        transforms = self._build_transforms("gigapath")
        return TorchWrapper(model, transforms, self.device)

    def _hoptimus0(self):
        """
        See https://huggingface.co/bioptimus/H-optimus-0
        """
        if self.model_path is not None:
            model = self._load_local_timm_model()
        else:
            model = timm.create_model(
                self.hf_hub_ref,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
            )
        transforms = self._build_transforms("hoptimus0")
        return TorchWrapper(model, transforms, self.device)

    def _hoptimus1(self):
        """
        See https://huggingface.co/bioptimus/H-optimus-1
        """
        if self.model_path is not None:
            model = self._load_local_timm_model()
        else:
            model = timm.create_model(
                self.hf_hub_ref,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
            )
        transforms = self._build_transforms("hoptimus1")
        return TorchWrapper(model, transforms, self.device)

    def _pathfm(self):
        """
        Google Path Foundation model (TensorFlow/Keras) from HF.
        See https://huggingface.co/google/path-foundation

        Note:
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
            raise ImportError(
                "pathfm requires the 'tensorflow' optional "
                "dependency group. Install it with: pip install pyslyde[tensorflow]"
            ) from e

        try:
            import tf_keras as tfk
        except ImportError as e:
            raise ImportError(
                "pathfm requires tf_keras (legacy Keras 2), part of the 'tensorflow' "
                "optional dependency group. Install it with: pip install pyslyde[tensorflow]"
            ) from e

        if self.model_path is not None:
            self._validate_model_path("dir")
            model_path = self.model_path
        else:
            model_path = snapshot_download(repo_id=self.hf_repo_id)
        try:
            model = tfk.models.load_model(model_path)
            infer_fn = model.signatures["serving_default"]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Path Foundation model for '{self.model_name}' "
                f"from '{model_path}'."
            ) from e

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
                f"expected {exp}, got {feats.shape[1]}."
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
                    f"Expected HxWx3 RGB np.ndarray, got shape {image_in.shape}."
                )
            if image_in.dtype != np.uint8:
                image_in = image_in.astype(np.uint8)
            img = Image.fromarray(image_in)
        elif isinstance(image_in, Image.Image):
            img = image_in
        else:
            raise TypeError(f"Unsupported image type: {type(image_in)}.")

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
                raise RuntimeError(f"{name} expected (B,T,C), got {out.shape}.")
            out = out[:, 0, :]

        if name in VIRCHOW_POSTPROCESS:
            cfg = VIRCHOW_POSTPROCESS[name]

            if out.ndim != 3:
                raise RuntimeError(f"{name} expected (B,T,C), got {out.shape}.")

            B, T, C = out.shape
            exp_T, exp_C = cfg["expected_T"], cfg["expected_C"]
            if (exp_T is not None and T != exp_T) or (exp_C is not None and C != exp_C):
                raise RuntimeError(
                    f"{name} expected (B,{exp_T},{exp_C}), got {out.shape}."
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
            raise RuntimeError(f"Expected torch.Tensor feats, got {type(feats)}.")
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        feats = feats.reshape(feats.shape[0], -1)
        if feats.ndim != 2:
            raise RuntimeError(f"Expected feats to be 2D (B,D), got {feats.shape}.")
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
            f"No expected embedding dim configured for model '{self.model_name}'."
        )
