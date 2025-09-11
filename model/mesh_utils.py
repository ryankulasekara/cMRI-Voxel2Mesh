import torch.nn as nn
import torch


# class UNetLayer(nn.Module):
#     def __init__(self, num_channels_in, num_channels_out, ndims, batch_norm=True):
#         super(UNetLayer, self).__init__()
#         """
#         U-Net layer used in voxel encoder & voxel decoder
#         Got the bulk of this code from the voxel2mesh repo
#         """
#
#
#         conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
#         batch_norm_op = nn.InstanceNorm2d if ndims == 2 else nn.InstanceNorm3d
#
#         layers = [
#             conv_op(num_channels_in, num_channels_out, kernel_size=3, padding=1),
#             batch_norm_op(num_channels_out) if batch_norm else nn.Identity(),
#             nn.ReLU(),
#             conv_op(num_channels_out, num_channels_out, kernel_size=3, padding=1),
#             batch_norm_op(num_channels_out) if batch_norm else nn.Identity(),
#             nn.ReLU()
#         ]
#
#         self.unet_layer = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.unet_layer(x)

class UNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ndims=3, norm="instance"):
        super().__init__()

        if ndims == 3:
            Conv = nn.Conv3d
            Norm = nn.InstanceNorm3d if norm == "instance" else nn.GroupNorm
        else:
            Conv = nn.Conv2d
            Norm = nn.InstanceNorm2d if norm == "instance" else nn.GroupNorm

        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1)

        if norm == "group":
            self.norm1 = Norm(num_groups=8, num_channels=out_channels)
            self.norm2 = Norm(num_groups=8, num_channels=out_channels)
        else:  # instance norm
            self.norm1 = Norm(out_channels)
            self.norm2 = Norm(out_channels)

        self.activation = nn.ReLU(inplace=True)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        return x