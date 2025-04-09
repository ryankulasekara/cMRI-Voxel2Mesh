import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pyvista as pv
import numpy as np

from model.template_mesh import TemplateMesh

class MeshDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.template_mesh = TemplateMesh()

        # not sure what to call this so I named it vertex encoder...
        # combines x,y,z coordinates of voxels w/ the features extracted at each voxel
        self.vertex_encoder = nn.Linear(3 + config.voxel_feature_dim, 128)

        # these are the layers of the graph convolutional network (GCN)
        self.conv1 = GCNConv(128, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 256)
        self.conv4 = GCNConv(256, 128)

        # takes the output of the gcn and transforms it into displacements
        # the tanh at the end scales the displacements from [-1,1]
        self.displacement_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()
        )

    def forward(self, voxel_output):
        # get the features at each voxel
        features = voxel_output['features']  # [B, 16, D, H, W]

        # load the template mesh
        vertices = self.template_mesh.get_vertices().to(features.device)  # [N, 3]
        faces = self.template_mesh.get_faces().to(features.device)  # [F, 3]

        # get the batch size
        B = features.shape[0]

        # sample the voxel features at the vertices to get features at each vertex
        vertex_features = self.sample_voxel_features(voxel_output, vertices)  # [B, N, C]

        # get into correct shape for the gcn to handle
        edge_index = self.faces_to_edges(faces)  # [2, E]
        vertex_coords = vertices.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([
            torch.clamp(vertex_coords, -10, 10),  # the clamps here make sure there aren't extremes
            torch.clamp(vertex_features, -5, 5)   # was getting bad instability before adding these
        ], dim=-1)
        x = self.vertex_encoder(x)

        # gcn layers
        displacements = []
        for i in range(B):
            xi = x[i]
            xi = F.relu(self.conv1(xi, edge_index))
            xi = F.relu(self.conv2(xi, edge_index))
            xi = F.relu(self.conv3(xi, edge_index))
            xi = F.relu(self.conv4(xi, edge_index))
            displacements.append(self.displacement_head(xi))

        # apply the displacements on the template mesh - clamp displacements to make sure nothing too crazy is applied
        deformed_vertices = vertices + torch.clamp(torch.stack(displacements), -0.25, 0.25)  # [B, N, 3]
        return deformed_vertices

    def faces_to_edges(self, faces):
        # we need to convert the faces into edges (like in original voxel2mesh code)
        edges = torch.cat([faces[:, :2], faces[:, 1:], faces[:, [0, 2]]], dim=0)
        return edges.unique(dim=0).t().contiguous()  # [2, E]

    def sample_voxel_features(self, voxel_output, vertices):
        # sample features at each vertex
        # TODO replace w/ neighborhood sampling like in original voxel2mesh code?

        # get dimensions of features [batch size, channels, dim, height, width]
        B, C, D, H, W = voxel_output['features'].shape

        # get number of vertices
        N = vertices.shape[0]

        # make sure vertices have shape [B, N, 3]
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0).expand(B, -1, -1)

        # normalize vertices to [-1, 1] - assumes orig. vertices are [0,1]
        vertices_norm = (vertices - 0.5) * 2

        # reshape to [B, N, 1, 1, 3]
        grid = vertices_norm.unsqueeze(2).unsqueeze(2)  # Correct reshaping

        # sample features w/ pytorch builtin function
        features = F.grid_sample(
            voxel_output['features'],
            grid,
            align_corners=True,
            mode='bilinear',
            padding_mode='border'
        )

        return features.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B, N, C]

