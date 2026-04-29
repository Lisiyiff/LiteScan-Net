# models/comparison/STANet_Source/stanet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone


class STANet(nn.Module):
    """
    STANet (Spatial-Temporal Attention Network) 纯净版封装
    """

    def __init__(self, backbone_type='resnet18', in_c=3, f_c=64, ds=1, mode='BAM'):
        super(STANet, self).__init__()

        # 1. 特征提取器
        self.netF = backbone.define_F(in_c=in_c, f_c=f_c, type='mynet3')

        # 2. 时空注意力模块
        self.netA = backbone.CDSA(in_c=f_c, ds=ds, mode=mode)

    def forward(self, img1, img2):
        # 1. 提取特征
        feat1 = self.netF(img1)
        feat2 = self.netF(img2)

        # 2. 应用注意力机制
        feat1, feat2 = self.netA(feat1, feat2)

        # 3. 计算欧氏距离 [核心修复]
        # F.pairwise_distance 在高维张量上行为不符合预期
        # 我们显式计算 dim=1 (Channel) 的 L2 范数
        dist = torch.norm(feat1 - feat2, dim=1, keepdim=True)  # [B, 1, H/4, W/4]

        # 4. 上采样回原图尺寸
        dist = F.interpolate(dist, size=img1.shape[2:], mode='bilinear', align_corners=True)

        # 5. 去掉 Channel 维度，返回 [B, H, W]
        return dist.squeeze(1)