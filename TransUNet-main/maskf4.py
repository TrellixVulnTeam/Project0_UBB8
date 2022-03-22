import logging
import torch
import copy
import math
from osgeo import gdal
import fvcore.nn.weight_init as weight_init
import torch.distributed as dist
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Set
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch import conv2d as Conv2d
from torch.nn import functional as F
import torchvision
from Imagelist import ImageList
import itertools
from torch import Tensor
import numpy as np
import VAN


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def dice_loss(inputs, targets, num_masks):
    inputs = inputs.sigmoid()
    # inputs = inputs.relu()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    # prob = inputs.relu()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        # empty_weight[-2] = 0.8
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([torch.arange(0, 6).cuda()[J] for (_, J) in indices])
        target_classes_o = torch.cat([targets.unique()[J] for t, (_, J) in zip(targets,indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):

        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [torch.nn.functional.one_hot(t, num_classes=6).permute(2, 0, 1).type(torch.float).index_select(0, t.unique()) for t in targets]
        # gt = targets

        # masks = torch.nn.functional.one_hot(gt, num_classes=6).permute(0, 3, 1, 2)#[:, :5, :, :]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]


        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(targets.unique()) for t in targets)
        # num_masks = 6
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_masks = torch.clamp(num_masks, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def batch_dice_loss(inputs, targets):
    inputs = inputs.sigmoid()
    # inputs = inputs.relu()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    hw = inputs.shape[1]
    prob = inputs.sigmoid()
    # prob = inputs.relu()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)
    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )
    return loss / hw


class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):

        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):

        bs, num_queries = outputs["pred_logits"].shape[:2]
        gt = targets
        targets = torch.nn.functional.one_hot(gt, num_classes=6)#[:, :, :, :5]
        # masks = [v["masks"] for v in targets]
        # h_max = max([m.shape[1] for m in masks])
        # w_max = max([m.shape[2] for m in masks])
        indices = []
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # tgt_ids = targets[b]["labels"]
            # tgt_ids = torch.arange(0, 6)
            tgt_ids = gt[b].unique()
            # tgt_mask = targets[b]["masks"].to(out_mask)
            tgt_mask = torch.nn.functional.one_hot(gt[b], num_classes=6).permute(2, 0, 1).type(torch.float)
            tgt_mask = tgt_mask.index_select(0, tgt_ids)
            # tgt_mask = targets.permute(0, 3, 1, 2).type(torch.float)[b]
            cost_class = -out_prob[:, tgt_ids]
            # tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")
            tgt_mask = tgt_mask[:, None]
            out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W]
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

            cost_dice = batch_dice_loss(out_mask, tgt_mask)
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):

        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class MaskFormer(nn.Module):

    def __init__(self, backbone,):
        super().__init__()
        self.backbone = backbone.cuda()
        # model = efficientnetv2.efficientnetv2_s().cuda()
        # self.feature = model.blocks
        # self.stem = model.stem
        self.fpn = VAN.FPN().cuda()
        self.tran = nn.Conv2d(
            20,
            20,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        N_steps = 128   # hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        # transformer = Transformer(
        #     d_model=256,
        #     dropout=0.1,
        #     nhead=8,
        #     dim_feedforward=2048,
        #     num_encoder_layers=0,   # 0
        #     num_decoder_layers=6,
        #     normalize_before=False,
        #     return_intermediate_dec=True,
        # )
        # self.transformer = transformer
        # self.input_proj = nn.Sequential()  # Conv2d(256, 256, kernel_size=1)
        # weight_init.c2_xavier_fill(self.input_proj)
        num_queries = 20
        hidden_dim = 256
        self.num_heads = 8
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(256, 6+1)
        self.mask_embed = MLP(256, 256, 20, 3)  # 第三个参数：cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
        self.aux_loss = True  # deep_supervision
        self.mask_classification = True
        self.sem_seg_postprocess_before_inference = False

        self.hw = [32, 64, 160, 512]
        self.input_proj = nn.ModuleList()
        for _ in range(3):
            self.input_proj.append(nn.Conv2d(self.hw[_], hidden_dim, kernel_size=1))
            weight_init.c2_xavier_fill(self.input_proj[-1])

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.size = [32, 64, 160, 256]

        for _ in range(10):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=2048,
                    dropout=0.0,
                    normalize_before=False,
                )
            )


        # self.decoder_norm = nn.LayerNorm(256)
        #     self.query_feat.append(nn.Embedding(num_queries, self.size[level_index]))
        #     self.query_embed.append(nn.Embedding(num_queries, self.size[level_index]))
        #
        #     self.level_embed.append(nn.Embedding(3, self.hw[level_index]))

        self.level_embed = nn.Embedding(3, hidden_dim)
        weight_dict = {"loss_ce": 1, "loss_mask": 20, "loss_dice": 1}
        aux_weight_dict = {}
        for i in range(11 - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=20,
            cost_dice=1,
        )
        losses = ["labels", "masks"]
        criterion = SetCriterion(6, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses,)
        self.criterion = criterion


    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        # decoder_output = decoder_output.transpose(0, 1)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def forward(self, images, targets):

        # x0 = self.stem(images)
        # x1 = self.feature(x0)
        x1 = self.backbone(images)
        # mask_features = self.backbone(images)
        mask_features = self.fpn(x1)
        mask_features = self.tran(mask_features)  # 第二纬度c可变 只要和mlp配合好就行
        # trande = TransformerDecoder(img_feature, 256)
        pos = self.pe_layer(x1[3])
        src = x1[3]
        # pos = self.pe_layer(x1)
        # src = x1
        mask = None
        src = []
        pos = []
        size_list = []

        for i in range(3):
            size_list.append(x1[i].shape[-2:])
            pos.append(self.pe_layer(x1[i], None).flatten(2))
            src.append(self.input_proj[i](x1[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(10):
            level_index = 2 - (i % 3)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
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
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(i + 1) % 3])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == 10 + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        # hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)
        # hs, memory = self.transformer(src, mask, self.query_embed.weight, pos)



        # outputs_class = self.class_embed(hs)
        # out = {"pred_logits": outputs_class[-1]}
        # if self.aux_loss:
        #     # [l, bs, queries, embed]
        #     mask_embed = self.mask_embed(hs)
        #     outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)#[:, :5, :, :])
        #     out["pred_masks"] = outputs_seg_masks[-1]
        #     out["aux_outputs"] = self._set_aux_loss(
        #         outputs_class if self.mask_classification else None, outputs_seg_masks
        #     )
        # else:
        #     # FIXME h_boxes takes the last one computed, keep this in mind
        #     # [bs, queries, embed]
        #     mask_embed = self.mask_embed(hs[-1])
        #     outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        #     out["pred_masks"] = outputs_seg_masks

        if self.training:
            # targets = targets
            losses = self.criterion(out,  targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = out["pred_logits"]
            mask_pred_results = out["pred_masks"]
            # upsample masks
            # mask_pred_results = F.interpolate(
            #     mask_pred_results,
            #     size=(images.shape[-2], images.shape[-1]),
            #     mode="bilinear",
            #     align_corners=False,
            # )
            processed_result = []
            processed_results = []
            image_sizes = [(im.shape[-2], im.shape[-1]) for im in images]
            for mask_cls_result, mask_pred_result, image_size in zip(
                    mask_cls_results, mask_pred_results, image_sizes
            ):
                # semantic segmentation inference
                r = self.semantic_inference(mask_cls_result, mask_pred_result)
                # if not self.sem_seg_postprocess_before_inference:
                #     r = sem_seg_postprocess(r, image_size, height, width)
                # r = r[None]
                processed_result.append(r)

                # processed_results.append({"sem_seg": r})
            pro_re = torch.stack(processed_result, 0)

            return pro_re


def maybe_add_full_model_gradient_clipping(optim):

    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, 5)
                super().step(closure=closure)
    return FullModelGradientClippingOptimizer


norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )


def build_optimizer(net):
    params = []
    hyperparams = {}
    hyperparams["lr"] = 0.0001
    hyperparams["weight_decay"] = 0.0

    defaults = {}
    defaults["lr"] = 0.0005  #0.0001
    defaults["weight_decay"] = 0.0001 #0.0001
    memo: Set[torch.nn.parameter.Parameter] = set()

    for module_name, module in net.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            # if "backbone.blocks" in module_name or "backbone.head" in module_name or "backbone.stem" in module_name:
            if "backbone" in module_name :
                print(module_name)
                hyperparams["lr"] = 7e-4
            if isinstance(module, norm_module_types):
                print('norm:%s' % module)
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, torch.nn.Embedding):
                print('embedding:%s' % module)
                hyperparams["weight_decay"] = 0.0
            params.append({"params": [value], **hyperparams})
    optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, 0.0007,)
    # optimizer = torch.optim.AdamW(params,0.0001)
    return optimizer


def split_train_val(image_paths, label_paths, val_index=0):
        # 分隔训练集和验证集
        train_image_paths, train_label_paths, val_image_paths, val_label_paths = [], [], [], []
        for i in range(len(image_paths)):
            # 训练验证4:1,即每5个数据的第val_index个数据为验证集
            if i % 5 == val_index:
                val_image_paths.append(image_paths[i])
                val_label_paths.append(label_paths[i])
            else:
                train_image_paths.append(image_paths[i])
                train_label_paths.append(label_paths[i])
        print("Number of train images: ", len(train_image_paths))
        print("Number of val images: ", len(val_image_paths))
        return train_image_paths, train_label_paths, val_image_paths, val_label_paths


if __name__ == '__main__':
    import efficientnetv2
    from trainer import TUDataset, f_score
    from torch.utils.data import DataLoader
    import glob
    from tqdm import *
    import os
    from mmseg.utils import get_root_logger
    from mmcv.runner import load_checkpoint
    from torchsummary import summary
    # setup_seed(48)
    # backbone = efficientnetv2.efficientnetv2_s()
    backbone = VAN.van_tiny()
    # logger = get_root_logger()
    # load_checkpoint(backbone, r'E:\data\weight\van_tiny_754.pth.tar', map_location='cpu', strict=False, logger=logger)
    lr = 8e-5
    net = MaskFormer(backbone=backbone).cuda()
    # optimizer = build_optimizer(net)
    # summary(net, (3, 512, 512))

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0001)
    # data_batch = MaskFormerSemanticDatasetMapper()(a, gt)
    image_paths = glob.glob(r'D:\train\*tif')
    label_paths =glob.glob(r'D:\labelgray\*tif')
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(image_paths,
                                                                                             label_paths,
                                                                                             0)
    batch_size = 2
    image_path = r'D:\train'
    train_num = len(os.listdir(image_path))
    val_num = math.ceil(len(os.listdir(image_path)) // 5)
    epoch_step = train_num // batch_size
    epoch_step_val = val_num // batch_size
    epoch = 140
    db_train = TUDataset(train_image_paths, train_label_paths, mode='train')
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    db_train_val = TUDataset(val_image_paths, val_label_paths, mode='val')
    valloader = DataLoader(db_train_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2,)
    a = torch.load(r'E:\data\weight\van_tiny_754.pth.tar')['state_dict']
    # a = torch.load(r'E:\data\weight\van_small_811.pth.tar')['state_dict']
    model2_dict = net.state_dict()
    state_dict = {k: v for k, v in a.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    net.load_state_dict(model2_dict)
    for epoch_num in range(epoch):

        net.train()
        loss0 = 0
        trep = epoch_step - epoch_step_val
        with tqdm(total=trep, desc=f'Epoch {epoch_num + 1}/{epoch}', postfix=dict, mininterval=0.3, colour='cyan') as pbar:
            for i, sampled in enumerate(trainloader):
                image_batch, label_batch = sampled
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()
                out = net(image_batch, label_batch)
                optimizer.zero_grad()
                # loss = torch.mean(out.values())
                loss = sum(out.values())/33
                loss.backward()
                optimizer.step()
                loss0 += loss.item()
                pbar.set_postfix(**{'loss': loss0 / (i + 1), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
                pbar.update(1)
        net.eval()
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch_num + 1}/{epoch}', postfix=dict, mininterval=0.3, colour='cyan') as pbar:
            miou = 0
            acc = 0
            oa = 0
            f1 = 0
            giou = 0
            for i, sampled in enumerate(valloader):
                image_batch, label_batch = sampled
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()

                with torch.no_grad():
                    out = net(image_batch, label_batch)
                    vmiou, vacc, voa, vf1, vgiou = f_score(out, label_batch)
                    miou += vmiou
                    acc += vacc
                    oa += voa
                    f1 += vf1
                    giou += vgiou
                pbar.set_postfix(**{'miou': miou / (i + 1), 'acc': acc/(i+1), 'oa': oa/(i+1), 'f1': f1/(i+1), 'giou': giou/(i+1), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
                pbar.update(1)
        lr_scheduler.step()


    # lossce = out['loss_ce']
    # lossmask = out['loss_mask']
    # lossdice = out['loss_dice']
    # lossce.backward(retain_graph=True)
    # optimizer.zero_grad()
    # lossmask.backward(retain_graph=True)
    # lossdice.backward(retain_graph=True)
    # print(net.feature[0].project_conv.conv.weight.grad[0,0])



    # print(net.feature[0].project_conv.conv.weight.grad[0,0])

