#!/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, AlbertModel
import numpy as np
from typing import Dict, List
from torch.nn import ModuleDict
from src.model.atten_unet import AttentionUNet
from util.data_util import get_class_mapping
from util.layers import ElementWiseMatrixAttention, DotProductMatrixAttention, CosineMatrixAttention, BilinearMatrixAttention, \
    LinearMatrixAttention, InputVariationalDropout


class Gun(nn.Module):
    def __init__(self,
                 embedder: str = 'bert',
                 encoder: str = 'lstm',
                 super_mode: str = 'before',
                 unet_down_channel: int = 256,
                 inp_drop_rate: float = 0.2,
                 out_drop_rate: float = 0.2,
                 loss_weights: List = (0.2, 0.4, 0.4)
                 ):
        super(Gun, self).__init__()
        # input dropout
        if inp_drop_rate > 0:
            self.var_inp_dropout = InputVariationalDropout(p=inp_drop_rate)
        else:
            self.var_inp_dropout = lambda x: x
        # output dropout
        if out_drop_rate > 0:
            self.var_out_dropout = InputVariationalDropout(p=out_drop_rate)
        else:
            self.var_out_dropout = lambda x: x

        if embedder == 'bert':
            self.embedder = BertModel.from_pretrained('bert-base-chinese')
        elif embedder == 'albert':
            self.embedder = AlbertModel.from_pretrained('bert-base-chinese')
        elif embedder == 'roberta':
            self.embedder = AlbertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        else:
            raise NameError("Name {} is not in bert model list.".format(embedder))

        if encoder == 'rnn':
            raise NameError("Pls use lstm as text encoder")
        elif encoder == 'lstm':
            self.encoder = nn.LSTM(
                input_size=768,
                hidden_size=200,
                num_layers=1,
                dropout=inp_drop_rate,
                batch_first=True
            )
        elif encoder == 'bilstm':
            self.encoder = nn.LSTM(
                input_size=768,
                hidden_size=200,
                num_layers=1,
                dropout=inp_drop_rate,
                batch_first=True,
                bidirectional=True
            )
        else:
            raise NameError("Name {} is not in rnn model list.".format(embedder))

        # why use these type of attention calculation
        nni_choices = ['ele', 'dot', 'cos', 'emb_dot', 'emb_cos', 'linear', 'bilinear']
        self.segment_choices = ['ele', 'dot', 'cos', 'linear']

        # if expand bi-direction, we will regard forward/backward as two channels
        self.expand_bidir = False

        self.similar_function = ModuleDict({
            'ele': ElementWiseMatrixAttention(),
            'dot': DotProductMatrixAttention(),
            'cos': CosineMatrixAttention(),
            'emb_dot': DotProductMatrixAttention(),
            'emb_cos': CosineMatrixAttention(),
            'bilinear': BilinearMatrixAttention(matrix_1_dim=self.output_size, matrix_2_dim=self.output_size),
            'linear': LinearMatrixAttention(tensor_1_dim=self.output_size, tensor_2_dim=self.output_size)
        })

        self.attn_channel = 0
        for choice in self.segment_choices:
            if choice == 'ele':
                self.attn_channel += self.output_size
            elif choice in ['dot', 'cos', 'emb_dot', 'emb_cos', 'bilinear', 'linear']:
                if self.expand_bidir:
                    self.attn_channel += 2
                else:
                    self.attn_channel += 1

        self.class_mapping: Dict[str, int] = get_class_mapping(super_mode=super_mode)
        self.segmentation_net = AttentionUNet(input_channels=self.attn_channel,
                                              class_number=len(self.class_mapping.keys()),
                                              down_channel=unet_down_channel)

        self.weight_tensor = torch.tensor([loss_weights[0], loss_weights[1], 1 - loss_weights[0] - loss_weights[1]])
        self.loss = nn.CrossEntropyLoss(ignore_index=-1,
                                        weight=self.weight_tensor)

        self.min_width = 8
        self.min_height = 8

    def forward(self, matrix_map,
                ):
        pass
