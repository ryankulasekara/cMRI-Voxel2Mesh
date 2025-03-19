import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelEncoder(nn.Module):
    def __init__(self):
        super(VoxelEncoder, self).__init__()

        # voxel encoder does downsampling, so do 3D conv blocks to reduce spatial resolution
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

    def conv_block(self, in_channels, out_channels):
        """
        This is the convolution block that will downsample/reduce spatial resolution in input volume
        Does 3D convolution w/ ReLU & MaxPooling
        """

        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        """
        Forward propagation through voxel encoder block
        """
        skip_connections = []

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # enc4 is the output of the encoder, skip connections go to decoder block
        return enc4, [enc1, enc2, enc3]


