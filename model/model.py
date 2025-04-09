import torch
import torch.nn as nn
import pyvista as pv
import numpy as np
import torch.nn.functional as F

from config import NRRD_DIMENSIONS
from model.voxel_encoder import VoxelEncoder
from model.voxel_decoder import VoxelDecoder
from model.mesh_decoder import MeshDecoder
from model.losses import chamfer_distance, mesh_edge_loss, laplacian_smoothing, cross_entropy_loss
from model.template_mesh import TemplateMesh


class Voxel2Mesh(nn.Module):
    def __init__(self, config):
        super(Voxel2Mesh, self).__init__()
        self.config = config
        self.template_mesh = TemplateMesh()

        # initialize each component of the network
        self.voxel_encoder = VoxelEncoder(config)
        self.voxel_decoder = VoxelDecoder(config)
        self.mesh_decoder = MeshDecoder(config)

    def forward(self, data):
        voxel_features = self.voxel_encoder(data['x'])
        voxel_output = self.voxel_decoder(voxel_features)
        mesh_output = self.mesh_decoder(voxel_output)

        return {
            'mesh': mesh_output,
            'segmentation': voxel_output['segmentation']
        }

    def loss(self, data):
        """
        Computes the total loss for Voxel2Mesh using weighted combination of loss functions.
        These are the two main ones from the paper:
            1. Cross-Entropy Loss: computed between predicted voxels & labels
            2. Chamfer Loss: computed between predicted mesh & marching cubes from labels
        """

        # get predictions from model
        # pred is a dict:
        # pred['segmentation'] is the segmentation output from voxel decoder
        # pred['mesh'] is the deformed mesh output from the mesh decoder
        pred = self.forward(data)

        # cross-entropy loss
        ce_loss = cross_entropy_loss(
            pred['segmentation'].squeeze(1),
            data['y_voxels'].float()
        )

        # mesh losses
        faces = self.template_mesh.get_faces().to(data['x'].device)
        chamfer_loss = chamfer_distance(pred['mesh'], data['surface_points'])
        edge_loss = mesh_edge_loss(pred['mesh'], faces)
        laplacian_loss = laplacian_smoothing(pred['mesh'], faces)

        # # debugging code
        # faces_pyvista = []
        # for face in self.template_mesh.get_faces():
        #     faces_pyvista.append([3, *face])
        # faces_pyvista = np.array(faces_pyvista).flatten()
        # pv_mesh = pv.PolyData(data['surface_points'].cpu().detach().numpy().squeeze(0))
        # pv_mesh.plot()
        #
        # pv_mesh = pv.PolyData(pred['mesh'].cpu().detach().numpy().squeeze(0))
        # pv_mesh.plot()

        # weighted sum of losses
        total_loss = (
                2.0 * chamfer_loss +
                1.2 * ce_loss +
                0.3 * laplacian_loss +
                1.5 * edge_loss
        )

        log = {
            "loss": total_loss.item(),
            "chamfer_loss": chamfer_loss.item(),
            "ce_loss": ce_loss.item(),
            "edge_loss": edge_loss.item(),
            "laplacian_loss": laplacian_loss.item()
        }

        return total_loss, log