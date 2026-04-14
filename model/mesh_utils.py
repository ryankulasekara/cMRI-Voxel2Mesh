import torch.nn as nn
import torch


class UNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ndims=3):
        super().__init__()

        if ndims == 3:
            Conv = nn.Conv3d
            Norm = nn.InstanceNorm3d
        else:
            Conv = nn.Conv2d
            Norm = nn.InstanceNorm2d

        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = Norm(out_channels)
        self.norm2 = Norm(out_channels)

        self.activation = nn.ReLU(inplace=True)

        # initialize weights
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


def normalize_points(points, eps=1e-6):
    """
    Normalize a mesh so each dimension is within [-1,1]

    :param points: array of points for each mesh (B, N, 3)
    :param eps: prevents divide by zero
    :return: normalized mesh points
    """

    min_val, _ = points.min(dim=1, keepdim=True)
    max_val, _ = points.max(dim=1, keepdim=True)
    center = (min_val + max_val) / 2
    scale = (max_val - min_val).max(dim=2, keepdim=True)[0] + eps

    return (points - center) / scale
