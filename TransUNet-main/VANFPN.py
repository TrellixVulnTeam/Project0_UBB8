import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from maskf1 import PositionEmbeddingSine, MLP, Transformer
import math
import OCR


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x, H, W):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x), H, W))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class FPN(nn.Module):
    def __init__(self, d32,d16,d8,d4):
        super(FPN, self).__init__()
        # self.toplayer = nn.Conv2d(304, 256, kernel_size=1, stride=1, padding=0)

        # self.latlayer1 = nn.Conv2d(160, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(32, 256, kernel_size=1, stride=1, padding=0)

        # self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer1 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

        self.toplayer = nn.Conv2d(d32, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(d16, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(d8, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(d4, 256, kernel_size=1, stride=1, padding=0)
        # self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(128, 20, kernel_size=1, stride=1, padding=0)

        # self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)



    def _upsample(self, x, h, w):
         return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self,x):
        p5 = x[3]
        x16 = x[2]
        x8 = x[1]
        x4 = x[0]
        p5 = self.toplayer(p5)
        out = []
        p4 = self._upsample_add(p5, self.latlayer1(x16))
        p3 = self._upsample_add(p4, self.latlayer2(x8))
        p2 = self._upsample_add(p3, self.latlayer3(x4))
        # p4 = self.smooth1(p4)
        # p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)
        _, _, h, w = p2.size()

        s5 = F.relu(self.gn2(self.conv2(p5)))
        out.append(s5)
        # s5 = self._upsample(s5, h, w)
        # s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)

        # s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        s4 = F.relu(self.gn2(self.conv2(p4)))
        out.append(s4)
        s4 = self._upsample(s4, h, w)
        # s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # s3 = F.relu(self.gn2(self.conv2(p3)))
        out.append(p3)
        # s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        out.append(p2)
        # s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        # print(self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w).size())
        # end = self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)
        # end = self._upsample(s2 + s3 + s4 + s5, 4 * h, 4 * w)
        return out


class VAN(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=6, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        # self.head = FPN()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        # logger = get_root_logger()
        # load_checkpoint(self, r'E:\data\weight\van_tiny_754.pth.tar', map_location='cpu', strict=False, logger=logger)
        self.apply(self._init_weights)
        # self.ocr = OCR.HighResolutionNet()

        # self.toplayer = nn.Conv2d(304, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer1 = nn.Conv2d(160, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(32, 256, kernel_size=1, stride=1, padding=0)
        #tiny

        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        #small

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 6, kernel_size=1, stride=1, padding=0)

        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)
        num_queries = 6
        # self.pe_layer = PositionEmbeddingSine(128, normalize=True)
        # self.query_embed = nn.Embedding(num_queries, 256)
        # transformer = Transformer(
        #     d_model=256,
        #     dropout=0.1,
        #     nhead=8,
        #     dim_feedforward=2048,
        #     num_encoder_layers=0,  # 0
        #     num_decoder_layers=6,
        #     normalize_before=False,
        #     return_intermediate_dec=True,
        # )
        # self.transformer = transformer
        # self.mask_embed = MLP(256, 256, 18, 3)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def init_weights(self, pretrained=r'E:\data\weight\van_tiny_754.pth.tar'):
        # if isinstance(pretrained, str):


    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        # outs = self.head(outs)
        p5 = outs[3]
        x16 = outs[2]
        x8 = outs[1]
        x4 = outs[0]
        p5 = self.toplayer(p5)
        # pos = self.pe_layer(p5)
        # hs = self.transformer(p5, None, self.query_embed.weight, pos)
        # emb = self.mask_embed(hs)
        p4 = self._upsample_add(p5, self.latlayer1(x16))
        p3 = self._upsample_add(p4, self.latlayer2(x8))
        p2 = self._upsample_add(p3, self.latlayer3(x4))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        _, _, h, w = p2.size()
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)
        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        # print(self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w).size())
        deco = self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)
        # end = torch.einsum("lbqc,bchw->lbqhw", emb, deco)[-1].view(4, 6, 512, 512)
        # end = self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)
        return deco
        # end = self.ocr(outs)
        # return end
        # return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@BACKBONES.register_module()
class van_tiny(VAN):
    def __init__(self, **kwargs):
        super(van_tiny, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class van_small(VAN):
    def __init__(self, **kwargs):
        super(van_small, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class van_base(VAN):
    def __init__(self, **kwargs):
        super(van_base, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class van_large(VAN):
    def __init__(self, **kwargs):
        super(van_large, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 5, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

if __name__ == '__main__':
    net = van_tiny().cuda()
    # from torchsummary import summary
    # summary(net, (3,512,512))
    a = torch.randn(2, 3, 512, 512).cuda()
    b = net(a)
    print(b)