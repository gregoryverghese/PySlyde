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
#from pyslyde.encoders.HistoSSLscaling.rl_benchmarks.models import iBOTViT 
#from HIPT_4K.hipt_model_utils import get_vit256, get_vit4k
#from pyslyde.encoders.HIPT.HIPT_4K.hipt_model_utils import eval_transforms
#from pyslyde.encoders.HIPT.HIPT_4K import vision_transformer as vits
#from pyslyde.encoders.HIPT.HIPT_4K.hipt_4k import HIPT_4K
#from lmdb_data import LMDBRead, LMDBWrite

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from huggingface_hub import login


class FeatureGenerator():
    encoders= {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50
              }
    def __init__(
            self,
            model_name,
            model_path,
            encoder_name='resnet18',
            contrastive=None):

        self.model_path=model_path
        self.encoder_name=encoder_name
        self.model = model_name
        self.model_name = model_name
        #self.transforms = None
        #self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        #self._model = None
        #print(self.device) 

    @property
    def model(self):
        return self._model
    
    @property
    def device(self):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @model.setter
    def model(self,value):
        self._model=getattr(self,'_'+value)()
        

    @property
    def encoder(self):
        encoder=FeatureGenerator.encoders[self.encoder_name]
        return encoder
        

    @property
    def checkpoint_dict(self):
        print(f"Model: {self.model_path}")
        return torch.load(self.model_path,map_location=torch.device('cpu'))


    #Load pretrained MoCO model from
    def _moco(self):
        state_dict=self.checkpoint_dict['state_dict']
        model=self.encoder()
        model.load_state_dict(state_dict,strict=False)
        #remove final linear layer
        model=torch.nn.Sequential(*list(model.children())[:-1])
        return model

    
    #Load pretrained simclr model from 
    #https://github.com/ozanciga/self-supervised-histopathology/blob/main/README.md
    def _ciga(self):
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
        transform=T.Compose(
            [T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

        self.transforms = transform
        return model.to(self.device)


    def _vgg16(self):
        net=models.vgg16(pretrained=True)
        model=torch.nn.Sequential(*(list(net.children())[:-1]))
        return model
        

    def _simclr(self):
        for k in list(checkpoint_dict.keys()):
            if k.startswith('backbone'): 
                if not k.startswith('backbone.fc'):
                    checkpoint_dict[k[len(layer_name):]] = checkpoint_dict[k]
            del checkpoint_dict[k]

        model=self.encoder()
        model.load_state_dict(checkpoint_dict,strict=False)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))    
        return model


    def _hipt4k(self):
        model = HIPT_4K()
        model.eval()
        self.transforms = eval_transforms()
        return model


    def _hipt256(self):
        
        checkpoint_key = 'teacher'
        arch = 'vit_small'
        image_size=(256,256)
        model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
        for p in model256.parameters():
            p.requires_grad = False
        state_dict = self.checkpoint_dict
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        model = model256

        self.transforms = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        return model.to(self.device)


    #def _phikon(self):
        """
        See https://github.com/owkin/HistoSSLscaling/tree/main?tab=readme-ov-file#download
        """
        #model = iBOTViT(
            #architecture="vit_base_pancan", 
            #encoder="teacher",
            #weights_path=self.model_path  
        #)
        #self.transforms = model.transform
        #print(self.transforms)
        #return model.to(self.device)


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


    def _dinobrca(self):
        arch = 'vit_small'
        image_size=(256,256)
        checkpoint_key = 'teacher'
        
        model = vits.__dict__[arch](patch_size=16, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
   
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transforms = transform
        return model.to(self.device)

    def _uni(self):
        """
        See https://github.com/mahmoodlab/UNI
        """
        model = timm.create_model(
                "vit_large_patch16_224", img_size=224,
            init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(self.checkpoint_dict, strict=True)
        
        transform = T.Compose(
            [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    
        self.transforms = transform 
        return model.to(self.device)

    def _virchow2(self):
        """
        see https://huggingface.co/paige-ai/Virchow2 
        """

        #login(os.getenv('HUGGINGFACE_TOKEN'))  # HUGGINGFACE_TOKEN is an environment variable
        login(os.getenv('HUGGINGFACE_TOKEN'), add_to_git_credential=True)  # To save token to your Git credentials

        # need to specify MLP layer and activation function for proper init
        model = timm.create_model("hf-hub:paige-ai/Virchow2", 
                                pretrained=True, 
                                mlp_layer=SwiGLUPacked, 
                                act_layer=torch.nn.SiLU
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
        self.transforms = transform 
        return model.to(self.device)
        

    def _gigapath(self):
        """
        see https://huggingface.co/prov-gigapath/prov-gigapath
        Size of tile embedding output by the model is 1536.
        """
        
        #login(os.getenv('HUGGINGFACE_TOKEN'))  # HUGGINGFACE_TOKEN is an environment variable
        login(os.getenv('HUGGINGFACE_TOKEN'), add_to_git_credential=True)  # To save token to your Git credentials

        # this approach is for tile encoding. slide-level encoding is done a different way
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


    def forward_pass(self, image):
        self.model.eval()
        #print('device cuda', next(self.model.parameters()).is_cuda)
        image = Image.fromarray(image)
        image = self.transforms(image)    
        image = image.to(self.device)
        image = torch.unsqueeze(image,0)

        """
        with torch.no_grad():
            features = self.model(image)
        """
        
        if self.model_name == "virchow2":
            if torch.cuda.is_available():
                # recommended by developer for use when on GPU
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(image)
            else:
                with torch.no_grad():
                    output = self.model(image)
            
            class_token = output[:, 0]    # size: 1 x 1280
            patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
            features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        
        else:
            with torch.no_grad():
                features = self.model(image)

        features = torch.squeeze(features)
        #print(f"Features successfuly extracted by {self.model_name}")

        return features


