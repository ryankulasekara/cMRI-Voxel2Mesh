import torch
import torch.nn as nn

from model.mesh_utils import UNetLayer


class VoxelDecoder(nn.Module):
    def __init__(self, config):
        super(VoxelDecoder, self).__init__()

        self.config = config
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear',
                                    align_corners=True) if config.ndims == 3 else nn.Upsample(scale_factor=2,
                                                                                              mode='bilinear',
                                                                                              align_corners=True)

        # create a series of upsampling layers (U-Net decoder)
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

    def forward(self, down_outputs):
        x = down_outputs[-1]

        for i, unet_layer in enumerate(self.up_layers):
            # upsampling
            x = self.upsample(x)
            x = unet_layer(x)

        return x