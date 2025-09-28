import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel

from dinov2.models import vision_transformer as vits
from dinov2.models import vision_transformer_inchannel4 as vits_inchannel4
from dinov2.models import vision_transformer_inchannel5 as vits_inchannel5
from Seco.models.moco2_module import MocoV2
from copy import deepcopy

import os

#local_rank = int(os.environ["LOCAL_RANK"])



class FrozenSSLEmbedder_vits14_patch(nn.Module):
    def __init__(self, path="/ceph-data/drm/code/dinov2-main", device=torch.device('cuda'), feat_dim = 384):
        super().__init__()
        self.model = vits.vit_small(img_size=518, patch_size=14, init_values=1, num_register_tokens=4, block_chunks=0)
        self.model.load_state_dict(torch.load('/ceph-data/drm/code/dinov2-main/pretrained_models/dinov2_vits14_reg4_pretrain.pth'), strict=True)

    def forward(self, image):
        features_dict = self.model.forward_features(image)
        features = features_dict['x_norm_patchtokens']

        return features

class FrozenSSLEmbedder_vits14_vector(nn.Module):
    def __init__(self, path="/ceph-data/drm/code/dinov2-main/", device=torch.device('cuda'), feat_dim = 384):
        super().__init__()
        # dinov2_vits14, feat_dim = 384; dinov2_vitb14, feat_dim = 768; dinov2_vitl14, feat_dim = 1024
        self.model = vits.vit_small(img_size=518, patch_size=14, init_values=1, num_register_tokens=4, block_chunks=0)

        self.model.load_state_dict(torch.load('/ceph-data/drm/code/dinov2-main/pretrained_models/dinov2_vits14_reg4_pretrain.pth'), strict=True)

    def forward(self, image):
        features_dict = self.model.forward_features(image)
        features = features_dict['x_norm_clstoken']
        features = features.unsqueeze(1)

        return features

class VAE_SSLEmbedder_vits14_vector(nn.Module):
    #def __init__(self, path="/ceph-data/drm/code/dinov2-main/", device=torch.device(f"cuda:{local_rank}"), feat_dim = 384):
    def __init__(self, path="/ceph-data/drm/code/dinov2-main/", device=torch.device('cuda'), feat_dim = 384):
        super().__init__()
        # dinov2_vits14, feat_dim = 384; dinov2_vitb14, feat_dim = 768; dinov2_vitl14, feat_dim = 1024
        self.model = vits_inchannel4.vit_small_inchannel4(img_size=70, patch_size=14, init_values=1, num_register_tokens=4, block_chunks=0)

        self.model.load_state_dict(torch.load('/ceph-data/drm/code/dinov2-main/pretrained_models/dinov2_vits14_reg4_pretrain.pth'), strict=False)

    def forward(self, z_image):
        B,C,H,W = z_image.size()
        z_image =nn.functional.interpolate(z_image, size=(70,70), mode='bilinear')
        features_dict = self.model.forward_features(z_image)
        features = features_dict['x_norm_clstoken']
        features = features.unsqueeze(1)

        return features


class SSLEmbedder_vits14_vector_with_label(nn.Module):
    def __init__(self, path="/ceph-data/drm/code/dinov2-main/", device=torch.device('cuda'), feat_dim = 384):
        super().__init__()
        # dinov2_vits14, feat_dim = 384; dinov2_vitb14, feat_dim = 768; dinov2_vitl14, feat_dim = 1024
        self.model = vits_inchannel4.vit_small_inchannel4(img_size=518, patch_size=14, init_values=1, num_register_tokens=4, block_chunks=0)

        self.model.load_state_dict(torch.load('/ceph-data/drm/code/dinov2-main/pretrained_models/dinov2_vits14_reg4_pretrain.pth'), strict=False)

    def forward(self, image, label):
        input_ssl = torch.cat((image,label), dim=1)
        features_dict = self.model.forward_features(input_ssl)
        features = features_dict['x_norm_clstoken']
        features = features.unsqueeze(1)

        return features


class VAE_SSLEmbedder_vits14_vector_with_label(nn.Module):
    def __init__(self, path="/ceph-data/drm/code/dinov2-main/", device=torch.device('cuda'), feat_dim = 384):
        super().__init__()
        # dinov2_vits14, feat_dim = 384; dinov2_vitb14, feat_dim = 768; dinov2_vitl14, feat_dim = 1024
        self.model = vits_inchannel5.vit_small_inchannel5(img_size=70, patch_size=14, init_values=1, num_register_tokens=4, block_chunks=0)

        self.model.load_state_dict(torch.load('/ceph-data/drm/code/dinov2-main/pretrained_models/dinov2_vits14_reg4_pretrain.pth'), strict=False)

    def forward(self, z_image, label):
        B,C,H,W = z_image.size()
        z_image =nn.functional.interpolate(z_image, size=(70,70), mode='bilinear')
        label = nn.functional.interpolate(label, size=(70,70), mode='nearest')
        input_ssl = torch.cat((z_image,label), dim=1)
        features_dict = self.model.forward_features(input_ssl)
        features = features_dict['x_norm_clstoken']
        features = features.unsqueeze(1)

        return features

class FrozenSSLEmbedder_Seco_resnet50(nn.Module):
    def __init__(self, path="/ceph-data/drm/code/NIPS2024-FreestyleNet-SSL_v0/Seco/pretrained_model/seco_resnet50_1m.ckpt"):
        super().__init__()
        self.model = MocoV2.load_from_checkpoint(path)
        self.backbone = deepcopy(self.model.encoder_q)

    def forward(self, image):
        features = self.backbone(image)
        features = features.unsqueeze(1)


        return features



