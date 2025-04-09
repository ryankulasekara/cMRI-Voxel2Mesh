import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from model.mesh_utils import UNetLayer


class VoxelDecoder(nn.Module):
    def __init__(self, config):
        super(VoxelDecoder, self).__init__()

        self.config = config
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True) if config.ndims == 3 else (
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))

        # create upsampling layers (unet decoder)
        up_layers = []
        for i in range(config.steps, 0, -1):
            up_layers.append(
                UNetLayer(
                    config.first_layer_channels * 2 ** i,
                    config.first_layer_channels * 2 ** (i - 1),
                    config.ndims
                )
            )

        self.up_layers = nn.Sequential(*up_layers)

        # layer normalization before heads
        self.norm = nn.InstanceNorm3d(config.voxel_feature_dim)

        # output both segmentation and features
        self.seg_head = nn.Conv3d(config.voxel_feature_dim, 1, kernel_size=1)
        self.feature_head = nn.Conv3d(config.voxel_feature_dim, config.voxel_feature_dim,
                                      kernel_size=1)

        # initialize heads with smaller weights
        nn.init.kaiming_normal_(self.seg_head.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.kaiming_normal_(self.feature_head.weight, mode='fan_in')
        self.seg_head.bias.data.zero_()
        self.feature_head.bias.data.zero_()


    def forward(self, down_outputs):
        x = down_outputs[-1]

        for i, unet_layer in enumerate(self.up_layers):
            # upsampling
            x = self.upsample(x)
            x = unet_layer(x)

            # gradient clipping for stability - dealing with NaNs popping up in this layer...?
            x = torch.clamp(x, -1e3, 1e3)
            if i % 2 == 0:
                x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

        # normalization for stability
        x = self.norm(x)
        x = torch.clamp(x, -50, 50)

        return {
            'segmentation': torch.sigmoid(self.seg_head(x)),  # [B, 1, D, H, W] - sigmoid scales from 0-1 (confidence)
            'features': F.tanhshrink(self.feature_head(x))  # [B, 16, D, H, W]
        }
