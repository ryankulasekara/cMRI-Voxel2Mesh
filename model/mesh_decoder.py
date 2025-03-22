import torch
import torch.nn as nn
from model.template_mesh import TemplateMesh


class MeshDecoder(nn.Module):
    def __init__(self, config):
        super(MeshDecoder, self).__init__()

        self.config = config
        self.template_mesh = TemplateMesh()  # Load the template mesh

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 3, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, voxel_output):
        batch_size = voxel_output.shape[0]

        # expand template vertices for batch processing
        vertices = self.template_mesh.get_vertices().to(voxel_output.device)  # Shape: (num_points, 3)
        vertices = vertices.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (B, num_points, 3)
        vertices = vertices.permute(0, 2, 1)  # Shape: (B, 3, num_points)

        # predict displacement
        x = self.relu(self.conv1(vertices))
        x = self.relu(self.conv2(x))
        displacement = self.conv3(x)

        # deform mesh based on predicted displacement
        new_vertices = vertices + displacement
        return new_vertices.permute(0, 2, 1)  # shape: (B, num_points, 3)
