#!/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import transformers
import numpy as np
from src.model.atten_unet import AttentionUNet
from util.data_util import get_class_mapping


class Gun(nn.Module):
    def __init__(self,
                 embedder,
                 encoder,
                 super_mode: str = 'before',
                 unet_down_channel: int = 256):
        super(Gun, self).__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.class_mapping = get_class_mapping(super_mode=super_mode)
        self.segmentation_net = AttentionUNet(input_channels=self.attn_channel,
                                              class_number=len(self.class_mapping.keys()),
                                              down_channel=unet_down_channel)
        
