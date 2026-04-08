import torch
import torch.nn as nn
from model.mesh_utils import UNetLayer

class VoxelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_pool = nn.MaxPool3d(2) if config.ndims == 3 else nn.MaxPool2d(2)

        # down layers for unet... just convs
        downs = [UNetLayer(config.num_input_channels, config.first_layer_channels, config.ndims)]
        for i in range(1, config.steps + 1):
            downs.append(UNetLayer(
                config.first_layer_channels * 2 ** (i - 1),
                config.first_layer_channels * 2 ** i,
                config.ndims
            ))
        self.down_layers = nn.Sequential(*downs)

    def forward(self, x):
        with torch.autocast(device_type='cuda', enabled=False):
            x = x.float()
            down_outputs = []
            for i, unet_layer in enumerate(self.down_layers):
                if i > 0:
                    x = self.max_pool(x.float()).float()
                x = unet_layer(x.float())

                # keep values in a sane range... was having issues with some values blowing up
                x = torch.clamp(x, -1e2, 1e2)
                down_outputs.append(x)
            return down_outputs
