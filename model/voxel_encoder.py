import torch
import torch.nn as nn

from model.mesh_utils import UNetLayer


class VoxelEncoder(nn.Module):
    def __init__(self, config):
        super(VoxelEncoder, self).__init__()

        self.config = config
        self.max_pool = nn.MaxPool3d(2) if config.ndims == 3 else nn.MaxPool2d(2)

        # create downsampling layers (unet encoder)
        down_layers = [UNetLayer(config.num_input_channels, config.first_layer_channels, config.ndims)]
        for i in range(1, config.steps + 1):
            down_layers.append(
                UNetLayer(
                    config.first_layer_channels * 2 ** (i - 1),
                    config.first_layer_channels * 2 ** i,
                    config.ndims
                )
            )

        self.down_layers = nn.Sequential(*down_layers)

    def forward(self, x):
        down_outputs = []  # skip connections
        x = self.down_layers[0](x)
        down_outputs.append(x)

        for unet_layer in self.down_layers[1:]:
            # max pool for downsampling
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)

        return down_outputs