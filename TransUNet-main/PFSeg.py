import torch.nn as nn
import torch
import torch.nn.functional as F

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """

    return nn.BatchNorm2d(in_channels)

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


class PointMatcher(nn.Module):
    """
        Simple Point Matcher
    """
    def __init__(self, dim, kernel_size=3):
        super(PointMatcher, self).__init__()
        self.match_conv = nn.Conv2d(dim*2, 1, kernel_size, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_high, x_low = x
        x_low = F.upsample(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        certainty = self.match_conv(torch.cat([x_high, x_low], dim=1))
        return self.sigmoid(certainty)


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


class PointFlowModuleWithMaxAvgpool(nn.Module):
    def __init__(self, in_planes,  dim=64, maxpool_size=8, avgpool_size=8, matcher_kernel_size=3,
                  edge_points=64):
        super(PointFlowModuleWithMaxAvgpool, self).__init__()
        self.dim = dim
        self.point_matcher = PointMatcher(dim, matcher_kernel_size)
        self.down_h = nn.Conv2d(in_planes, dim, 1)
        self.down_l = nn.Conv2d(in_planes, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool_size = maxpool_size
        self.avgpool_size = avgpool_size
        self.edge_points = edge_points
        self.max_pool = nn.AdaptiveMaxPool2d((maxpool_size, maxpool_size), return_indices=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))
        self.edge_final = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=1, bias=False),
            Norm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        x_high, x_low = x
        stride_ratio = x_low.shape[2] / x_high.shape[2]
        x_high_embed = self.down_h(x_high)
        x_low_embed = self.down_l(x_low)
        N, C, H, W = x_low.shape
        N_h, C_h, H_h, W_h = x_high.shape

        certainty_map = self.point_matcher([x_high_embed, x_low_embed])
        avgpool_grid = self.avg_pool(certainty_map)
        _, _, map_h, map_w = certainty_map.size()
        avgpool_grid = F.interpolate(avgpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)

        # edge part
        x_high_edge = x_high - x_high * avgpool_grid
        edge_pred = self.edge_final(x_high_edge)
        point_indices, point_coords = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.edge_points)
        sample_x = point_indices % W_h * stride_ratio
        sample_y = point_indices // W_h * stride_ratio
        low_edge_indices = sample_x + sample_y * W
        low_edge_indices = low_edge_indices.unsqueeze(1).expand(-1, C, -1).long()
        high_edge_feat = point_sample(x_high, point_coords)
        low_edge_feat = point_sample(x_low, point_coords)
        affinity_edge = torch.bmm(high_edge_feat.transpose(2, 1), low_edge_feat).transpose(2, 1)
        affinity = self.softmax(affinity_edge)
        high_edge_feat = torch.bmm(affinity, high_edge_feat.transpose(2, 1)).transpose(2, 1)
        fusion_edge_feat = high_edge_feat + low_edge_feat

        # residual part
        maxpool_grid, maxpool_indices = self.max_pool(certainty_map)
        maxpool_indices = maxpool_indices.expand(-1, C, -1, -1)
        maxpool_grid = F.interpolate(maxpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)
        x_indices = maxpool_indices % W_h * stride_ratio
        y_indices = maxpool_indices // W_h * stride_ratio
        low_indices = x_indices + y_indices * W
        low_indices = low_indices.long()
        x_high = x_high + maxpool_grid*x_high
        flattened_high = x_high.flatten(start_dim=2)
        high_features = flattened_high.gather(dim=2, index=maxpool_indices.flatten(start_dim=2)).view_as(maxpool_indices)
        flattened_low = x_low.flatten(start_dim=2)
        low_features = flattened_low.gather(dim=2, index=low_indices.flatten(start_dim=2)).view_as(low_indices)
        feat_n, feat_c, feat_h, feat_w = high_features.shape
        high_features = high_features.view(feat_n, -1, feat_h*feat_w)
        low_features = low_features.view(feat_n, -1, feat_h*feat_w)
        affinity = torch.bmm(high_features.transpose(2, 1), low_features).transpose(2, 1)
        affinity = self.softmax(affinity)  # b, n, n
        high_features = torch.bmm(affinity, high_features.transpose(2, 1)).transpose(2, 1)
        fusion_feature = high_features + low_features
        mp_b, mp_c, mp_h, mp_w = low_indices.shape
        low_indices = low_indices.view(mp_b, mp_c, -1)

        final_features = x_low.reshape(N, C, H*W).scatter(2, low_edge_indices, fusion_edge_feat)
        final_features = final_features.scatter(2, low_indices, fusion_feature).view(N, C, H, W)

        return final_features, edge_pred


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True)
                  for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class UperNetAlignHeadMaxAvgpool(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[32, 64, 160, 2048], fpn_dim=256,
                 fpn_dsn=False, reduce_dim=64, ignore_background=False, max_pool_size=8,
                 avgpool_size=8, edge_points=32):
        super(UperNetAlignHeadMaxAvgpool, self).__init__()

        self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if ignore_background:
                self.fpn_out_align.append(
                    PointFlowModuleWithMaxAvgpool(fpn_dim, dim=reduce_dim, maxpool_size=max_pool_size,
                                                  avgpool_size=avgpool_size, edge_points=edge_points))
            else:
                self.fpn_out_align.append(
                    PointFlowModuleWithMaxAvgpool(fpn_dim, dim=reduce_dim, maxpool_size=max_pool_size,
                                                  avgpool_size=avgpool_size, edge_points=edge_points))

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out

        fpn_feature_list = [f]
        edge_preds = []
        out = []

        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f, edge_pred = self.fpn_out_align[i]([f, conv_x])
            f = conv_x + f
            edge_preds.append(edge_pred)
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, edge_preds


class AlignNetResNetMaxAvgpool(nn.Module):
    def __init__(self, num_classes, trunk='resnet-101', criterion=None, variant='D', skip='m1', skip_num=48,
                 fpn_dsn=False, inplanes=128, reduce_dim=64, ignore_background=False,
                 max_pool_size=8, avgpool_size=8, edge_points=32):
        super(AlignNetResNetMaxAvgpool, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn

        # if trunk == trunk == 'resnet-50-deep':
        #     resnet = Resnet_Deep.resnet50()
        # elif trunk == 'resnet-101-deep':
        #     resnet = Resnet_Deep.resnet101()
        # else:
        #     raise ValueError("Not a valid network arch")
        #
        # resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        # self.layer0 = resnet.layer0
        # self.layer1, self.layer2, self.layer3, self.layer4 = \
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        # del resnet

        # if self.variant == 'D':
        #     for n, m in self.layer3.named_modules():
        #         if 'conv2' in n:
        #             m.dilation, m.padding = (2, 2), (2, 2)
        #         elif 'downsample.0' in n:
        #             m.stride = (2, 2)
        #     for n, m in self.layer4.named_modules():
        #         if 'conv2' in n:
        #             m.dilation, m.padding = (4, 4), (4, 4)
        #         elif 'downsample.0' in n:
        #             m.stride = (2, 2)
        # else:
        #     print("Not using Dilation ")

        inplane_head = 256
        self.head = UperNetAlignHeadMaxAvgpool(inplane_head, num_class=num_classes,
                                               fpn_dsn=fpn_dsn, reduce_dim=reduce_dim,
                                               ignore_background=ignore_background, max_pool_size=max_pool_size,
                                               avgpool_size=avgpool_size, edge_points=edge_points)

    def forward(self, x, gts=None):
        # x_size = x.size()
        # x0 = self.layer0(x)
        out = []
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # x4 = self.layer4(x3)
        x = self.head([x[0], x[1], x[2], x[3]])
        main_out = Upsample(x[0], [512,512])
        edge_preds = [Upsample(edge_pred, [512,512]) for edge_pred in x[1]]
        # if self.training:
        #     if not self.fpn_dsn:
        #         return self.criterion([main_out, edge_preds], gts)
        #     return self.criterion(x, gts)

        return main_out
