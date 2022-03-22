import torch.nn as nn
import torch
from torchvision import models
from torchsummary import summary


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch * BasicBlock.expansion, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != BasicBlock.expansion * out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch * BottleNeck.expansion)

        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch * BottleNeck.expansion)

            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())



class ResNet(nn.Module):
    def __init__(self, in_ch, out_ch, block, num_block,downsample_factor):
        super().__init__()
        self.in_ch = 64
        if downsample_factor == 8:
            s = [1, 2, 1, 1]

        elif downsample_factor == 32:
            s = [1, 2, 2, 2]
            
        elif downsample_factor == 16:
            s = [1, 2, 2, 1]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)

        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv2_x = self._make_layer(block, 64, num_block[0],  s[0])
        self.conv3_x = self._make_layer(block, 128, num_block[1], s[1])
        self.conv4_x = self._make_layer(block, 256, num_block[2], s[2])
        self.conv5_x = self._make_layer(block, 512, num_block[3], s[3])
        self.reduce = _ConvBnReLU(256, 48, 1, 1, 0, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.doubleconv = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, 3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #         nn.ReLU(inplace=True)
        # )
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.dconv_last = nn.Sequential(
            nn.Conv2d(320, 64, 3, padding=1),
            # nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_ch, 1)
        )

    def _make_layer(self, block, out_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_ch, out_ch, stride))
            self.in_ch = out_ch * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x)
        mp1 = self.maxpool(conv1)
        conv2 = self.conv2_x(mp1)
        _conv2 = self.reduce(conv2)
        conv3 = self.conv3_x(conv2)
        conv4 = self.conv4_x(conv3)
        bottle = self.conv5_x(conv4)
        # x = self.upsample(bottle)
        # x = bottle
        # x = torch.cat([x, conv4], dim=1)
        # x = double_conv(x.size(1), conv4.size(1)).cuda()(x)
        # # x = self.dconv_up3(x)
        # x = self.upsample(x)
        # x = torch.cat([x, conv3], dim=1)
        #
        # x = double_conv(x.size(1), conv3.size(1)).cuda()(x)
        # # x = self.dconv_up2(x)
        # x = self.upsample(x)
        # x = torch.cat([x, conv2], dim=1)
        #
        # x = double_conv(x.size(1), conv2.size(1)).cuda()(x)
        # # x = self.dconv_up1(x)
        # x = self.upsample(x)
        # x = torch.cat([x, conv1], dim=1)

        # out = self.dconv_last(x)

        return bottle, _conv2

    def load_pretrained_weights(self):

        model_dict = self.state_dict()
        resnet34_weights = models.resnet34(True).state_dict()
        count_res = 0
        count_my = 0

        reskeys = list(resnet34_weights.keys())
        mykeys = list(model_dict.keys())

        # print(models.resnet34())

        corresp_map = []
        while (True):  # 后缀相同的放入list
            reskey = reskeys[count_res]
            mykey = mykeys[count_my]

            if "fc" in reskey:
                break

            while reskey.split(".")[-1] not in mykey:
                count_my += 1
                mykey = mykeys[count_my]

            corresp_map.append([reskey, mykey])
            count_res += 1
            count_my += 1

        for k_res, k_my in corresp_map:
            model_dict[k_my] = resnet34_weights[k_res]

        try:
            self.load_state_dict(model_dict)
            print("Loaded resnet34 weights in mynet !")
        except:
            print("Error resnet34 weights in mynet !")
            raise

    def load_pretrained_weights101(self):

        model_dict = self.state_dict()
        resnet50_weights = models.resnet101(True).state_dict()
        count_res = 0
        count_my = 0

        reskeys = list(resnet50_weights.keys())
        mykeys = list(model_dict.keys())

        # print(models.resnet34())

        corresp_map = []
        while (True):  # 后缀相同的放入list
            reskey = reskeys[count_res]
            mykey = mykeys[count_my]

            if "fc" in reskey:
                break

            while reskey.split(".")[-1] not in mykey:
                count_my += 1
                mykey = mykeys[count_my]

            corresp_map.append([reskey, mykey])
            count_res += 1
            count_my += 1

        for k_res, k_my in corresp_map:
            model_dict[k_my] = resnet50_weights[k_res]
        torch.save(model_dict, '/model.pth')
        try:
            self.load_state_dict(model_dict)
            print("Loaded resnet101 weights in mynet !")
        except:
            print("Error resnet101 weights in mynet !")
            raise

    def load_pretrained_weights50(self):

        model_dict = self.state_dict()
        resnet50_weights = models.resnet50(True).state_dict()
        count_res = 0
        count_my = 0

        reskeys = list(resnet50_weights.keys())
        mykeys = list(model_dict.keys())

        # print(models.resnet34())

        corresp_map = []
        while (True):  # 后缀相同的放入list
            reskey = reskeys[count_res]
            mykey = mykeys[count_my]

            if "fc" in reskey:
                break

            while reskey.split(".")[-1] not in mykey:
                count_my += 1
                mykey = mykeys[count_my]

            corresp_map.append([reskey, mykey])
            count_res += 1
            count_my += 1

        for k_res, k_my in corresp_map:
            model_dict[k_my] = resnet50_weights[k_res]
        torch.save(model_dict, '/model.pth')
        try:
            self.load_state_dict(model_dict)
            print("Loaded resnet50 weights in mynet !")
        except:
            print("Error resnet50 weights in mynet !")
            raise


def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34(in_channel, out_channel, pretrain=True):
    """ return a ResNet 34 object
    """
    model = ResNet(in_channel, out_channel, BasicBlock, [3, 4, 6, 3])
    if pretrain:
        model.load_pretrained_weights()
    return model


def resnet50(downsample_factor, pretrain=True):
    """ return a ResNet 50 object
    """
    model = ResNet(3, 6, BottleNeck,  [3, 4, 6, 3], downsample_factor).cuda()
    summary(model, ( 3, 512, 512))
    if pretrain:
        model.load_pretrained_weights50()
    return model


def resnet101(in_channel, out_channel, pretrain=True):
    """ return a ResNet 101 object
    """
    model = ResNet(in_channel, out_channel, BottleNeck, [3, 4, 23, 3]).cuda()
    if pretrain:
        model.load_pretrained_weights101()
    return model


def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


if __name__ == '__main__':
    net = resnet50(3, 4, True)
    print(net)
    summary(net, (3,512,512))
    x = torch.rand((1, 3, 512, 512))
    # print(net.forward(x).shape)
