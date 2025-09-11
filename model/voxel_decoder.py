# voxel_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mesh_utils import UNetLayer

class VoxelDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='trilinear' if config.ndims == 3 else 'bilinear',
            align_corners=False,
        )

        ups = []
        for i in range(config.steps, 0, -1):
            ups.append(UNetLayer(
                config.first_layer_channels * 2 ** i,
                config.first_layer_channels * 2 ** (i - 1),
                config.ndims
            ))
        self.up_layers = nn.Sequential(*ups)

        # input norm expects channels = last encoder channels
        self.input_norm = nn.GroupNorm(
            num_groups=4,
            num_channels=config.first_layer_channels * 2 ** config.steps,
            eps=1e-5
        )

        self.seg_head = nn.Conv3d(config.voxel_feature_dim, 1, kernel_size=1, bias=True)
        self.feature_head = nn.Conv3d(config.voxel_feature_dim, config.voxel_feature_dim, kernel_size=1)

        nn.init.kaiming_normal_(self.seg_head.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.feature_head.weight, mode='fan_in', nonlinearity='linear')
        self.seg_head.bias.data.zero_()
        self.feature_head.bias.data.zero_()

    def forward(self, down_outputs):
        with torch.autocast(device_type='cuda', enabled=False):
            x = down_outputs[-1].float()

            x = self.input_norm(x)
            x = torch.clamp(x, -1e2, 1e2)

            for unet_layer in self.up_layers:
                x = self.upsample(x.float()).float()
                x = unet_layer(x.float())
                x = torch.clamp(x, -1e2, 1e2)

            # heads
            features = torch.tanh(self.feature_head(x.float()))
            features = torch.clamp(features, -10, 10)

            # non-normalized logits
            seg_logits = self.seg_head(x.float())
            seg_logits = torch.clamp(seg_logits, -50, 50)

            # do sigmoid to get probabilities/confidences at each voxel
            seg = torch.sigmoid(seg_logits)

            return {
                "segmentation": seg,
                "features": features
            }

