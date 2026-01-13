"""
feature_extractor.py
"""

import os
import glob
import random
import argparse
import itertools
 
import pandas as pd
import sys

# print(sys.path)
import cv2
import timm
import torch
import torch.nn as nn
#import staintools
from PIL import Image
import numpy as np
import torchvision.models as models
from torchvision import transforms as T

from pyslyde.encoders.ctran import ctranspath
# from pyslyde.encoders.HistoSSLscaling.rl_benchmarks.models import iBOTViT 
# from pyslyde.encoders.HIPT.HIPT_4K.hipt_model_utils import eval_transforms
# from pyslyde.encoders.HIPT.HIPT_4K import vision_transformer as vits
# from pyslyde.encoders.HIPT.HIPT_4K.hipt_4k import HIPT_4K

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from huggingface_hub import login


HF_MODELS = {
    "uni",
    "uni2",
    "virchow2",
    "gigapath",
    "hoptimus0",
    "hoptimus1",
}

class FeatureGenerator():
    encoders= {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'vgg16': models.vgg16
              }
    def __init__(
            self,
            model_name,
            model_path,
            encoder_name='resnet18',
            contrastive=None):

        self.model_path = model_path
        self.encoder_name = encoder_name
        self.model = model_name
        self.model_name = model_name

        self.transforms = None
        self._model = None
        self._hf_logged_in = False


    @property
    def model(self):
        return self._model
    
    @property
    def device(self):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @model.setter
    def model(self, value):
        if value in HF_MODELS:
            self._hf_login()
        self._model = getattr(self, '_' + value)()
        

    @property
    def encoder(self):
        encoder = FeatureGenerator.encoders[self.encoder_name]
        return encoder
        

    @property
    def checkpoint_dict(self):
        print(f"Model: {self.model_path}")
        return torch.load(self.model_path, map_location=torch.device('cpu'))


    def _hf_login(self):
        """
        Logs into Hugging Face using the HUGGINGFACE_TOKEN 
        environment variable (set by the user).
        """
        if getattr(self, "_hf_logged_in", False):
            return

        token = os.getenv("HUGGINGFACE_TOKEN")
        if token is None:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN environment variable not set. "
                "Required for Hugging Face models."
            )

        login(token)
        self._hf_logged_in = True


    def _moco(self):
        state_dict = self.checkpoint_dict['state_dict']
        model=self.encoder()
        model.load_state_dict(state_dict,strict=False)
        model=torch.nn.Sequential(*list(model.children())[:-1])
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
        return model


    def _ciga(self):
        """
        See https://github.com/ozanciga/self-supervised-histopathology/blob/main/README.md
        """
        state_dict=self.checkpoint_dict['state_dict']
        for k in list(state_dict.keys()):
            k_new=k.replace('model.', '').replace('resnet.', '')
            state_dict[k_new] = state_dict.pop(k)

        model=self.encoder()
        model_dict=model.state_dict()
        state_dict={k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        transform = T.Compose(
            [T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
        self.transforms = transform
        return model.to(self.device)

    
    def _vgg16(self):
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        weights = models.VGG16_Weights.DEFAULT
        self.transforms = weights.transforms()
        return model.to(self.device)


    def _simclr(self):     
        for k in list(self.checkpoint_dict.keys()):
            if k.startswith('backbone'): 
                if not k.startswith('backbone.fc'):
                    self.checkpoint_dict[k[len('backbone.'):]] = self.checkpoint_dict[k]
            del self.checkpoint_dict[k]

        model=self.encoder()
        model.load_state_dict(self.checkpoint_dict,strict=False)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))    
        return model.to(self.device)


    def _transpath(self):
        model = ctranspath()
        model.head = nn.Identity()
        model.load_state_dict(self.checkpoint_dict['model'], strict=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean = mean, std = std)])
        self.transforms = transform
        return model.to(self.device)


    # def _hipt4k(self):
    #     model = HIPT_4K()
    #     model.eval()
    #     self.transforms = eval_transforms()
    #     return model


    # def _hipt256(self): 
    #     checkpoint_key = 'teacher'
    #     arch = 'vit_small'
    #     image_size=(256,256)
    #     model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    #     for p in model256.parameters():
    #         p.requires_grad = False
    #     state_dict = self.checkpoint_dict
    #     if checkpoint_key is not None and checkpoint_key in state_dict:
    #         print(f"Take key {checkpoint_key} in provided checkpoint dict")
    #         state_dict = state_dict[checkpoint_key]
        
    #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # remove `module.` prefix
    #     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}  # remove `backbone.` prefix induced by multicrop wrapper
    #     msg = model256.load_state_dict(state_dict, strict=False)
    #     model = model256

    #     self.transforms = T.Compose([
    #         T.Resize(image_size),
    #         T.ToTensor(),
    #         T.Normalize(
    #             [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    #     return model.to(self.device)


    # def _phikon(self):
    #     """
    #     See https://github.com/owkin/HistoSSLscaling/tree/main?tab=readme-ov-file#download
    #     """
    #     model = iBOTViT(
    #         architecture="vit_base_pancan", 
    #         encoder="teacher",
    #         weights_path=self.model_path  
    #     )
    #     self.transforms = model.transform
    #     return model.to(self.device)


    # def _dinobrca(self):
    #     arch = 'vit_small'
    #     image_size=(256,256)
    #     checkpoint_key = 'teacher'
        
    #     model = vits.__dict__[arch](patch_size=16, num_classes=0)
    #     for p in model.parameters():
    #         p.requires_grad = False
   
    #     transform = T.Compose([
    #         T.Resize(image_size),
    #         T.ToTensor(),
    #         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #     self.transforms = transform
    #     return model.to(self.device)


    def _uni(self):
        """
        See https://huggingface.co/MahmoodLab/UNI
        """
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni", 
            pretrained=True,
             init_values=1e-5, 
             dynamic_img_size=True
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.transforms = transform 
        return model.to(self.device)
  

    def _uni2(self):
        """
        See https://huggingface.co/MahmoodLab/UNI2-h
        """
        timm_kwargs = {
                    'img_size': 224, 
                    'patch_size': 14, 
                    'depth': 24,
                    'num_heads': 24,
                    'init_values': 1e-5, 
                    'embed_dim': 1536,
                    'mlp_ratio': 2.66667*2,
                    'num_classes': 0, 
                    'no_embed_class': True,
                    'mlp_layer': timm.layers.SwiGLUPacked, 
                    'act_layer': torch.nn.SiLU, 
                    'reg_tokens': 8, 
                    'dynamic_img_size': True
                }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.transforms = transform 
        return model.to(self.device)


    def _virchow2(self):
        """
        See https://huggingface.co/paige-ai/Virchow2 
        """
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", 
            pretrained=True, 
            mlp_layer=SwiGLUPacked, 
            act_layer=torch.nn.SiLU
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.transforms = transform 
        return model.to(self.device)
        

    def _gigapath(self):
        """
        See https://huggingface.co/prov-gigapath/prov-gigapath.
        """
        # this approach is for tile encoding. slide-level encoding is done differently
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.transforms = transform 
        return model.to(self.device)


    def _hoptimus0(self):
        """
        See https://huggingface.co/bioptimus/H-optimus-0
        """
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", 
            pretrained=True, 
            init_values=1e-5, 
            dynamic_img_size=False
        )
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        self.transforms = transform
        return model.to(self.device)

    
    def _hoptimus1(self):
        """
        See https://huggingface.co/bioptimus/H-optimus-1
        """
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", 
            pretrained=True, 
            init_values=1e-5, 
            dynamic_img_size=False
        )
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])        
        self.transforms = transform
        return model.to(self.device)   


    def forward_pass(self, image_in: np.ndarray) -> torch.Tensor:
        """
        Robust single-tile inference.

        Accepts:
        - np.ndarray HxWx3 RGB uint8 (your extract_tile output), or
        - PIL.Image.Image

        Returns:
        - torch.Tensor (D,) feature vector
        """    
        self.model.eval()

        if self.transforms is None:
            raise RuntimeError(
                f"No transforms set for model '{self.model_name}'. "
                "Set self.transforms in the model constructor."
            )

        if isinstance(image_in, np.ndarray):
            if image_in.ndim != 3 or image_in.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 RGB np.ndarray, got shape {image_in.shape}")
            if image_in.dtype != np.uint8:
                image_in = image_in.astype(np.uint8)
            image = Image.fromarray(image_in)
        elif isinstance(image_in, Image.Image):
            image = image_in
        else:
            raise TypeError(f"Unsupported image type: {type(image_in)}")

        if image.mode != "RGB":
            image = image.convert("RGB")

        x = self.transforms(image)
        if not torch.is_tensor(x):
            raise RuntimeError(f"Transforms must return a torch.Tensor, got {type(x)}")
        if x.ndim != 3:
            raise RuntimeError(f"Expected transformed tensor (C,H,W), got shape {tuple(x.shape)}")
        if x.shape[0] != 3:
            raise RuntimeError(f"Expected 3-channel tensor after transforms, got C={x.shape[0]}")

        x = x.unsqueeze(0).to(self.device, non_blocking=True)
        self.model = self.model.to(self.device)
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        with torch.inference_mode():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(x)
            else:
                out = self.model(x)

            # Confirm feature embedding size
            if self.model_name == "virchow2":
                if out.ndim != 3:
                    raise RuntimeError(f"Virchow2 expected token output (B,T,C), got {out.shape}")

                B, T, C = out.shape
                if T != 261 or C != 1280:
                    raise RuntimeError(
                        f"Virchow2 unexpected token shape: expected (B,261,1280), got {out.shape}. "
                        "Check transforms/input size and model call path."
                    )

                class_token = out[:, 0]      # (B,1280)
                patch_tokens = out[:, 5:]    # (B,256,1280)
                feats = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # (B,2560)

                if feats.shape != (B, 2560):
                    raise RuntimeError(
                        f"Virchow2 embedding shape mismatch: expected ({B},2560), got {feats.shape}"
                    )

            elif self.model_name in ["hoptimus0", "hoptimus1"]:
                feats = out
                feats = feats.reshape(feats.shape[0], -1)
                if feats.shape[1] != 1536:
                    raise RuntimeError(
                        f"Expected (B,1536) for {self.model_name}, got {feats.shape}"
                    )
            else:
                feats = out

            feats = feats.reshape(feats.shape[0], -1)

        return feats.squeeze(0)
