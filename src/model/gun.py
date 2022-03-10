#!/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, AlbertModel
from typing import Dict, List
from torch.nn import ModuleDict
from src.model.atten_unet import AttentionUNet
from util.data_util import get_class_mapping
from src.model.layers import ElementWiseMatrixAttention, DotProductMatrixAttention, CosineMatrixAttention, \
    BilinearMatrixAttention, \
    LinearMatrixAttention, InputVariationalDropout, get_text_field_mask
from torch.nn.utils.rnn import pad_sequence


class Gun(nn.Module):
    def __init__(self,
                 embedder: str = 'bert',
                 encoder: str = 'bilstm',
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
            self.output_size = 200
        elif encoder == 'bilstm':
            self.encoder = nn.LSTM(
                input_size=768,
                hidden_size=200,
                num_layers=1,
                dropout=inp_drop_rate,
                batch_first=True,
                bidirectional=True
            )
            self.output_size = 400
        else:
            raise NameError("Name {} is not in rnn model list.".format(embedder))

        # why use these type of attention calculation
        # nni_choices = ['ele', 'dot', 'cos', 'emb_dot', 'emb_cos', 'linear', 'bilinear']
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

        # weights of diff classes
        self.weight_tensor = torch.tensor([loss_weights[0], loss_weights[1], 1 - loss_weights[0] - loss_weights[1]])
        self.loss = nn.CrossEntropyLoss(ignore_index=-1,
                                        weight=self.weight_tensor)

        self.min_width = 8
        self.min_height = 8

    def forward(self, matrix_map: torch.Tensor,
                context_str: List[str],
                cur_str: List[str],
                restate_str: List[str],
                context_tokens: Dict[str, torch.Tensor] = None,
                cur_tokens: Dict[str, torch.Tensor] = None,
                joint_tokens: Dict[str, torch.Tensor] = None,
                joint_border: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        attn_features = []

        # no joint encoding
        if context_tokens is not None:
            if 'bert-type-ids' in context_tokens:
                # fmod to avoid out of index
                context_tokens['bert-type-ids'] = torch.fmod(context_tokens['bert-type-ids'], 2)

            context_embedding = self.word_embedder(context_tokens)
            cur_embedding = self.word_embedder(cur_tokens)

            batch_size, context_len, _ = context_embedding.size()
            joint_embedding = torch.cat([context_embedding, cur_embedding], dim=1)

            # add variational dropout
            joint_embedding = self.var_inp_dropout(joint_embedding)
            context_embedding, cur_embedding = joint_embedding[:, :context_len, :], joint_embedding[:, context_len:, :]

            # get context-sensitive representations
            context_mask = get_text_field_mask(context_tokens)
            context_repr = self.text_encoder(context_embedding, context_mask)

            # get current representation
            cur_mask = get_text_field_mask(cur_tokens)
            cur_repr = self.text_encoder(cur_embedding, cur_mask)

            context_repr = self.var_out_dropout(context_repr)
            cur_repr = self.var_out_dropout(cur_repr)
        else:
            if 'bert-type-ids' in joint_tokens:
                # fmod to avoid out of index
                joint_tokens['bert-type-ids'] = torch.fmod(joint_tokens['bert-type-ids'], 2)

            joint_embedding = self.word_embedder(joint_tokens)

            joint_embedding = self.var_inp_dropout(joint_embedding)

            joint_mask = get_text_field_mask(joint_tokens)

            joint_repr = self.text_encoder(joint_embedding, joint_mask)
            joint_repr = self.var_out_dropout(joint_repr)
            # split repr into context_repr and cur_repr
            batch_size, _ = joint_border.shape
            joint_border = joint_border.view(batch_size)

            context_reprs = []
            context_embeddings = []
            cur_reprs = []
            cur_embeddings = []
            for i in range(batch_size):
                context_embeddings.append(joint_embedding[i, :joint_border[i]])
                context_reprs.append(joint_repr[i, :joint_border[i]])
                cur_embeddings.append(joint_embedding[i, joint_border[i]:])
                cur_reprs.append(joint_repr[i, joint_border[i]:])

            context_repr = pad_sequence(context_reprs, batch_first=True)
            cur_repr = pad_sequence(cur_reprs, batch_first=True)
            context_embedding = pad_sequence(context_embeddings, batch_first=True)
            cur_embedding = pad_sequence(cur_embeddings, batch_first=True)

        # padding feature map matrix to satisfy the minimum height/width of UNet model
        if context_repr.shape[1] < self.min_height:
            _, cur_height, hidden_size = context_repr.shape
            out_tensor = context_repr.data.new(batch_size, self.min_height, hidden_size).fill_(0)
            out_tensor[:, :cur_height, :] = context_repr
            context_repr = out_tensor

        if cur_repr.shape[1] < self.min_width:
            _, cur_width, hidden_size = cur_repr.shape
            out_tensor = cur_repr.data.new(batch_size, self.min_width, hidden_size).fill_(0)
            out_tensor[:, :cur_width, :] = cur_repr
            cur_repr = out_tensor

        context_forward, context_backward = context_repr[:, :, :self.hidden_size], context_repr[:, :, self.hidden_size:]
        cur_forward, cur_backward = cur_repr[:, :, :self.hidden_size], cur_repr[:, :, self.hidden_size:]

        for choice in self.segment_choices:
            if choice == 'ele':
                attn_features.append(self.similar_function[choice](context_repr,
                                                                   cur_repr))
            elif 'emb' in choice:
                attn_features.append(self.similar_function[choice](context_embedding,
                                                                   cur_embedding).unsqueeze(dim=1))
            else:
                if self.expand_bidir:
                    attn_features.append(self.similar_function[choice](context_forward,
                                                                       cur_forward).unsqueeze(dim=1))
                    attn_features.append(self.similar_function[choice](context_backward,
                                                                       cur_backward).unsqueeze(dim=1))
                else:
                    attn_features.append(self.similar_function[choice](context_repr,
                                                                       cur_repr).unsqueeze(dim=1))

        attn_input = torch.cat(attn_features, dim=1)

        # here we assume the attn_input as batch_size x channel x width x height
        attn_map = self.segmentation_net(attn_input)

        # attn_map: batch_size x width x height x class
        batch_size, width, height, class_size = attn_map.size()

        # if the current length and height is not equal to matrix-map
        if width != matrix_map.shape[1] or height != matrix_map.shape[2]:
            out_tensor = matrix_map.data.new(batch_size, width, height).fill_(-1)
            out_tensor[:, :matrix_map.shape[1], :matrix_map.shape[2]] = matrix_map
            matrix_map = out_tensor

        attn_mask = (matrix_map != -1).long()
        attn_map_flatten = attn_map.view(batch_size * width * height, class_size)
        matrix_map_flatten = matrix_map.view(batch_size * width * height).long()

        # cross entropy loss
        loss_val = self.loss(attn_map_flatten, matrix_map_flatten)
        outputs = {'loss': loss_val}

        return outputs


def main():
    model = Gun()
    print(model)

if __name__ == '__main__':
    main()
