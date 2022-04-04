
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import VANFPN
# from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
import maskf4

BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
relu_inplace = True
ALIGN_CORNERS = True
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)




class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):

        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        proxy = proxy.reshape(batch_size,256,6,1)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map

        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            # nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class HighResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        global ALIGN_CORNERS
        # extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()
        ALIGN_CORNERS =False
            # config.MODEL.ALIGN_CORNERS
        num_classes = 6
        # self.gather_head = SpatialGather_Module(num_classes)
        self.aux_head = nn.Sequential(   ##去掉
            nn.Conv2d(256, 256,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(256),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(256, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.ocr_distri_head = SpatialOCR_Module(in_channels=256,
                                                 key_channels=256,
                                                 out_channels=256,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Sequential(nn.Conv2d(
            256, 6, kernel_size=1, stride=1, padding=0, bias=True),
            ModuleHelper.BNReLU(num_features=6))
        self.head = nn.Sequential(nn.Conv2d(
            1024, 256, kernel_size=1, stride=1, padding=0, bias=True),
            ModuleHelper.BNReLU(num_features=256))

        last_inp_channels = 256
        # last_inp_channel = 256
        ocr_mid_channels = 512  #512
        ocr_key_channels = 256   #256

        hidden_dim = 256
        self.level_embed = nn.Embedding(3, hidden_dim)

        self.query_feat = nn.Embedding(num_classes, hidden_dim)
        self.query_embed = nn.Embedding(num_classes, hidden_dim)
        self.pe_layer = maskf4.PositionEmbeddingSine(128, normalize=True)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.class_embed = nn.Linear(256, num_classes)  ##去掉
        self.mask_embed = maskf4.MLP(256, 256, 6, 3)   ##去掉

        self.num_heads = 8
        self.apply(self.init_weights)

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()


        for _ in range(3):
            self.transformer_self_attention_layers.append(
                maskf4.SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                maskf4.CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                maskf4.FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=2048,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        # decoder_output = decoder_output.transpose(0, 1)
        decoder_output = decoder_output.transpose(0, 1)
        # outputs_class = self.class_embed(decoder_output)
        # mask_embed = self.mask_embed(decoder_output)
        # outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)#bchw

        # outputs_mask = torch.einsum("bqf,bfhw->bqhw", decoder_output, mask_features)

        outputs_mask = self.ocr_distri_head(mask_features,decoder_output)
        outputs_mask = self.cls_head(outputs_mask)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]

        # attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)

        outputs_mask = F.interpolate(outputs_mask, size=(512, 512), mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = 0
        # attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
        #                                                                                                  1) < 0.5).bool()
        # attn_mask = attn_mask.detach()

        return  outputs_mask, attn_mask


    def forward(self, x):
        x0 = x[3]
        x1 = x[2]
        x2 = x[1]
        x3 = x[0]
        out_aux_seg = []
        x0_h, x0_w = 128, 128
        x11 = F.interpolate(x1, size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x22 = F.interpolate(x2, size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x33 = F.interpolate(x3, size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)

        x0 = torch.cat([x0, x11, x22, x33], 1)
        x0 = self.head(x0)



        bs = x[0].shape[0]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        pos = []
        size_list = []

        src = []
        for i in range(3):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append((x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        predictions_class = []
        predictions_mask = []
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        # auxout = self.aux_head(x0)
        # output = self.gather_head(x0,auxout)
        # output = output.reshape(6,4,256)
        outputs_mask, attn_mask = self.forward_prediction_heads(output, x0, attn_mask_target_size=size_list[0])
        predictions_mask.append(outputs_mask)
        for i in range(3):
            level_index = (i % 3)
            # attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                # memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            # output = self.decoder_norm[i](output)
            # if i == 2:

            outputs_mask, attn_mask = self.forward_prediction_heads(output, x0, attn_mask_target_size=size_list[(i + 1) % 3])
            # predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # assert len(predictions_class) == 10 + 1


        return predictions_mask


    def init_weights(self, pretrained='', ):
        logger.info('=> init weights from normal distribution')
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                # print('skipped', name)
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
