import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from itertools import combinations


class UNetLayer(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, ndims, batch_norm=True):
        super(UNetLayer, self).__init__()

        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        batch_norm_op = nn.BatchNorm2d if ndims == 2 else nn.BatchNorm3d

        layers = [
            conv_op(num_channels_in, num_channels_out, kernel_size=3, padding=1),
            batch_norm_op(num_channels_out) if batch_norm else nn.Identity(),
            nn.ReLU(),
            conv_op(num_channels_out, num_channels_out, kernel_size=3, padding=1),
            batch_norm_op(num_channels_out) if batch_norm else nn.Identity(),
            nn.ReLU()
        ]

        self.unet_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.unet_layer(x)
