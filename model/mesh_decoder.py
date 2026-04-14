import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pyvista as pv
import numpy as np

from model.mesh_utils import normalize_points
from model.template_mesh import TemplateMesh
from data import map_coordinates

class MeshDecoder(nn.Module):
    def __init__(self, config, chamber):
        super().__init__()
        self.config = config
        self.chamber = chamber
        self.template_mesh = TemplateMesh()

        # vertex encoder... maybe change this in the future?
        # seems like the easiest way to correlate features & coordinates
        self.vertex_encoder = nn.Sequential(
            nn.Linear(3 + config.voxel_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # gcn architecture
        self.conv1 = GCNConv(128, 256)
        self.conv2 = GCNConv(256, 512)
        self.conv3 = GCNConv(512, 1024)
        self.conv4 = GCNConv(1024, 512)
        self.conv5 = GCNConv(512, 256)
        self.conv6 = GCNConv(256, 128)

        # batch norm
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

        # displacement head
        self.displacement_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()
        )

        # residual connection for template mesh
        self.residual_factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, voxel_output):
        # get feature vector from voxel decoder
        features = voxel_output['features']

        # get template mesh
        vertices = self.template_mesh.get_vertices().to(features.device)
        faces = self.template_mesh.get_faces().to(features.device)
        B = features.shape[0]

        # sample features w proper coordinate mapping
        vertex_features = self.sample_voxel_features(voxel_output, vertices)
        edge_index = self.faces_to_edges(faces)
        vertex_coords = vertices.unsqueeze(0).expand(B, -1, -1)

        # concatenate coordinates and features into single feature vector
        x = torch.cat([vertex_coords, vertex_features], dim=-1)
        x = self.vertex_encoder(x)

        # GCN w residual connections
        displacements = []
        for i in range(B):
            xi = x[i]

            # GCN layers
            xi1 = F.relu(self.conv1(xi, edge_index))
            xi2 = F.relu(self.conv2(xi1, edge_index))
            xi3 = F.relu(self.conv3(xi2, edge_index))
            xi4 = F.relu(self.conv4(xi3, edge_index))
            xi5 = F.relu(self.conv5(xi4, edge_index))
            xi6 = F.relu(self.conv6(xi5, edge_index))

            displacements.append(self.displacement_head(xi6))

        # get displacements... if nan, change to 0 for stability
        displacements = torch.stack(displacements)
        displacements = torch.nan_to_num(displacements, nan=0.0)

        # apply displacements to template mesh vertices
        deformed_vertices = vertices + torch.clamp(displacements, -2.5, 2.5)

        return deformed_vertices

    def sample_voxel_features(self, voxel_output, vertices):
        """
        Sample features from volume to the same [-1,1] coordinate system that vertices are in

        :param voxel_output: output from voxel decoder
        :param vertices: template mesh vertices
        """
        B, C, D, H, W = voxel_output['features'].shape

        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0).expand(B, -1, -1)

        # make sure template mesh vertices are in [-1,1] space
        min_v = vertices.min(dim=0, keepdim=True)[0]
        max_v = vertices.max(dim=0, keepdim=True)[0]
        vertices_norm = 2 * (vertices - min_v) / (max_v - min_v + 1e-6) - 1

        # this logic decides scale and location for template mesh based on the chamber
        if self.chamber == 0:  # LV - want this to be larger, located central/right side of image
            vertices_norm /= 1.5
            # vertices[:,1] += 0.2

        elif self.chamber == 1:  # RV - want this to be larger, located central/left side of image
            vertices_norm /= 1.5
            # vertices[:,1] -= 0.2

        elif self.chamber == 2:  # AORTA - smaller, upper, central
            vertices_norm /= 2.5
            vertices_norm[:, 2] += 0.2

        elif self.chamber == 3:  # PULMONARY TRUNK - smaller, upper, north/central
            vertices_norm /= 2.5
            # vertices[:,0] -= 0.2
            vertices_norm[:, 2] += 0.2

        elif self.chamber == 4:  # LA - bigger, upper, right
            vertices_norm /= 2
            # vertices[:,1] += 0.2
            vertices_norm[:, 2] += 0.2

        elif self.chamber == 5:  # RA - bigger, upper, left
            vertices_norm /= 2
            # vertices[:,1] -= 0.2
            vertices_norm[:, 2] += 0.2

        # vertices are already in [-1,1] space, so we can use them directly
        # just reshape to get dims right for grid_sample
        grid = vertices_norm.unsqueeze(2).unsqueeze(2)

        # TODO: this grid sample function is the easiest way I could find to sample the voxel features to the vertex
        # TODO: space... look into 'learned neighborhood sampling' mentioned in paper
        features = F.grid_sample(
            voxel_output['features'],
            grid,
            align_corners=True,
            mode='bilinear',
            padding_mode='border'
        )

        return features.squeeze(-1).squeeze(-1).permute(0, 2, 1)


    def faces_to_edges(self, faces):
        # we need to convert the faces into edges
        edges = torch.cat([faces[:, :2], faces[:, 1:], faces[:, [0, 2]]], dim=0)
        return edges.unique(dim=0).t().contiguous()
