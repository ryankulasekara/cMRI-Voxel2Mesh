import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mesh_utils import UNetLayer

class VoxelDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # define the upsample block for decoder
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='trilinear' if config.ndims == 3 else 'bilinear',
            align_corners=False,
        )

        # up layers for unet - upsampling & concatenation from skip connections
        self.up_layers = nn.ModuleList()
        for i in range(config.steps, 0, -1):
            # decoder channels
            dec_ch = config.first_layer_channels * (2 ** i)

            # skip connection channels from encoder at same level
            skip_ch = config.first_layer_channels * (2 ** (i - 1))

            # output after unet layer
            out_ch = config.first_layer_channels * (2 ** (i - 1))

            # append after concatenation
            self.up_layers.append(UNetLayer(dec_ch + skip_ch, out_ch, config.ndims))

        # input norm expects channels = last encoder channels
        self.input_norm = nn.GroupNorm(
            num_groups=4,
            num_channels=config.first_layer_channels * 2 ** config.steps,
            eps=1e-5
        )

        # heads
        self.seg_head = nn.Conv3d(config.voxel_feature_dim, config.num_classes, kernel_size=1, bias=True)
        self.feature_head = nn.Conv3d(config.voxel_feature_dim, config.voxel_feature_dim, kernel_size=1)

        # feature heads for each class/chamber
        self.class_feature_heads = nn.ModuleList([
            nn.Conv3d(config.voxel_feature_dim, config.voxel_feature_dim, kernel_size=1)
            for _ in range(config.num_classes)
        ])

        nn.init.kaiming_normal_(self.seg_head.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.feature_head.weight, mode='fan_in', nonlinearity='linear')
        self.seg_head.bias.data.zero_()
        self.feature_head.bias.data.zero_()

    def forward(self, down_outputs):
        with torch.autocast(device_type='cuda', enabled=False):
            # start w last one
            x = down_outputs[-1].float()
            x = self.input_norm(x)
            x = torch.clamp(x, -1e2, 1e2)

            # do upsampling and skip connections
            for i, unet_layer in enumerate(self.up_layers):
                x = self.upsample(x.float()).float()

                # get skip connection from same index in encoder module
                skip_idx = -(i + 2)
                if abs(skip_idx) <= len(down_outputs):
                    skip = down_outputs[skip_idx].float()

                    # make sure dims match
                    if skip.shape[2:] != x.shape[2:]:
                        skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)

                    # concatenate skip connection
                    x = torch.cat([x, skip], dim=1)

                # apply unet layer now
                x = unet_layer(x.float())
                x = torch.clamp(x, -1e2, 1e2)

            # last conv to match feature dims
            if x.shape[1] != self.config.voxel_feature_dim:
                if not hasattr(self, 'final_conv'):
                    self.final_conv = nn.Conv3d(x.shape[1], self.config.voxel_feature_dim, kernel_size=1).to(x.device)
                x = self.final_conv(x)

            # main features
            features = torch.tanh(self.feature_head(x.float()))
            features = torch.clamp(features, -10, 10)

            # class specific features
            class_features = []
            for head in self.class_feature_heads:
                class_feat = torch.tanh(head(features))
                class_features.append(class_feat)

            # segmentation predictions
            seg_logits = self.seg_head(x.float())
            seg_logits = torch.clamp(seg_logits, -50, 50)
            seg = torch.tanh(seg_logits)

            return {
                "segmentation": seg,
                "features": features,
                "class_features": class_features
            }