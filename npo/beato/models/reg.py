import torch
from torch import nn
from torchvision import models
from torchvision.models import list_models as tv_lm
from timm import list_models as timm_lm
from timm import create_model
from functools import partial
from npo.beato.models import basic as NBasic


def extract_backbone(backbone='resnet18', **kwargs):    
    kwargs['norm_layer'] = partial(nn.BatchNorm2d, momentum=.5) if 'vit' not in backbone else partial(nn.LayerNorm, eps=1e-5)
    if backbone in tv_lm():
        backbone: nn.Module = eval(f'models.{backbone}')(**kwargs)
    else:
        backbone: nn.Module = create_model(backbone, **kwargs)

    flag = kwargs.get('last_layer', None)

    last_layer = getattr(backbone, 'fc', None)

    if last_layer is None:
        last_layer = getattr(backbone, 'head', None)
        backbone.head = nn.Linear(1024, 1) if flag is None else flag
    else:
        backbone.fc = nn.Linear(1024, 1) if flag is None else flag
    return backbone


class RegModel(nn.Module):
    def __init__(self, backbone='vit_small_patch16_224', **kwargs):
        super(RegModel, self).__init__()
        self.mm_conv = nn.Conv2d(1, 3, 1, 1, bias=False)
        self.backbone = extract_backbone(backbone, **kwargs)

    def forward(self, x):
        x = self.mm_conv(x)
        x = self.backbone(x)
        return x


class PromptRegModel(nn.Module):
    def __int__(self, reg_param, **kwargs):
        super(PromptRegModel, self).__init__()
        self.backbone = extract_backbone(reg_param, **kwargs)
        
        
# def 