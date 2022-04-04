import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

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
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 6, kernel_size=1, stride=1, padding=0)

        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)



    def _upsample(self, x, h, w):
         return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self,x):
        p5 = x[3]#1/32
        x16 = x[2]
        x8 = x[1]
        x4 = x[0]
        p5 = self.toplayer(p5)
        out = []
        p4 = self._upsample_add(p5, self.latlayer1(x16))
        p3 = self._upsample_add(p4, self.latlayer2(x8))
        p2 = self._upsample_add(p3, self.latlayer3(x4))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        _, _, h, w = p2.size()

        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # out.append(s5)
        # s5 = self._upsample(s5, h, w)
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)

        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)

        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))

        end = self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)
        # end = self._upsample(s2 + s3 + s4 + s5, 4 * h, 4 * w)
        return end