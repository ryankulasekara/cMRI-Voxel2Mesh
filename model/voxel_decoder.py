import torch
import torch.nn as nn
import torch.nn.functional as F

# class VoxelDecoder(nn.Module):
#     def __init__(self):
#         super(VoxelDecoder, self).__init__()
#
#         # voxel decoder does upsampling, so need to do 3D upconvolutions to gain back spatial resolution
#         self.upc1 = self.upconv_block(256, 128)
#         self.upc2 = self.upconv_block(128, 64)
#         self.upc3 = self.upconv_block(64, 32)
#
#         # this is the final output, should match input size to voxel encoder block
#         self.upc4 = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
#
#     def upconv_block(self, in_channels, out_channels):
#         """
#         This is the convolution block that will upsample to increase spatial resolution
#         """
#
#         return nn.Sequential(
#             nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU()
#         )
#
#     def forward(self, x, skip_connections):
#         """
#         Forward propagation through voxel decoder block
#         """
#
#         # do upconvolutions & concatenate with skip connections from voxel encoder
#         print(f"Input shape: {x.shape}")
#
#         upc1 = self.upc1(x)
#         print(f"After upc1: {upc1.shape}")
#
#         print(f"skip_connections[2] shape: {skip_connections[2].shape}")
#         upc1 = torch.cat((upc1, skip_connections[2]), dim=1)
#         print(f"After concat upc1 & skip_connections[2]: {upc1.shape}")
#
#         upc2 = self.upc2(upc1)
#         print(f"After upc2: {upc2.shape}")
#
#         print(f"skip_connections[1] shape: {skip_connections[1].shape}")
#         upc2 = torch.cat((upc2, skip_connections[1]), dim=1)
#         print(f"After concat upc2 & skip_connections[1]: {upc2.shape}")
#
#         upc3 = self.upc3(upc2)
#         print(f"After upc3: {upc3.shape}")
#
#         print(f"skip_connections[0] shape: {skip_connections[0].shape}")
#         upc3 = torch.cat((upc3, skip_connections[0]), dim=1)
#         print(f"After concat upc3 & skip_connections[0]: {upc3.shape}")
#
#         out = self.upc4(upc3)
#         print(f"Output shape: {out.shape}")
#
#         return out

class VoxelDecoder(nn.Module):
    def __init__(self):
        super(VoxelDecoder, self).__init__()

        # Define the upsampling and convolution layers
        self.upc1 = self.upconv_block(256, 128)
        self.upc2 = self.upconv_block(256, 64)  # Change from 128 to 256
        self.upc3 = self.upconv_block(128, 32)
        self.upc4 = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)  # Final output layer

        self.reduce_channels = nn.Conv3d(64, 32, kernel_size=1)

    def upconv_block(self, in_channels, out_channels):
        """
        This block does the upsampling followed by a 3D convolution to reduce channels.
        """
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )

    def forward(self, x, skip_connections):
        """
        Forward propagation through voxel decoder block.
        """
        # Do upconvolutions & concatenate with skip connections from voxel encoder
        print(f"Input shape: {x.shape}")

        upc1 = self.upc1(x)
        print(f"After upc1: {upc1.shape}")

        # Resize skip_connection[2] to match upc1 shape if needed
        skip_connection_2_resized = F.interpolate(skip_connections[2], size=upc1.shape[2:], mode='trilinear', align_corners=False)
        print(f"skip_connections[2] resized shape: {skip_connection_2_resized.shape}")
        upc1 = torch.cat((upc1, skip_connection_2_resized), dim=1)
        print(f"After concat upc1 & skip_connections[2]: {upc1.shape}")

        upc2 = self.upc2(upc1)
        print(f"After upc2: {upc2.shape}")

        # Resize skip_connection[1] to match upc2 shape if needed
        skip_connection_1_resized = F.interpolate(skip_connections[1], size=upc2.shape[2:], mode='trilinear', align_corners=False)
        print(f"skip_connections[1] resized shape: {skip_connection_1_resized.shape}")
        upc2 = torch.cat((upc2, skip_connection_1_resized), dim=1)
        print(f"After concat upc2 & skip_connections[1]: {upc2.shape}")

        upc3 = self.upc3(upc2)
        print(f"After upc3: {upc3.shape}")

        # Resize skip_connection[0] to match upc3 shape if needed
        skip_connection_0_resized = F.interpolate(skip_connections[0], size=upc3.shape[2:], mode='trilinear', align_corners=False)
        print(f"skip_connections[0] resized shape: {skip_connection_0_resized.shape}")
        upc3 = torch.cat((upc3, skip_connection_0_resized), dim=1)
        upc3 = self.reduce_channels(upc3)
        print(f"After concat upc3 & skip_connections[0]: {upc3.shape}")

        out = self.upc4(upc3)
        print(f"Output shape: {out.shape}")

        return out