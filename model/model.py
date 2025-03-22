import torch
import torch.nn as nn
import torch.nn.functional as F

from model.voxel_encoder import VoxelEncoder
from model.voxel_decoder import VoxelDecoder
from model.mesh_decoder import MeshDecoder
from model.losses import chamfer_distance, mesh_edge_loss, laplacian_smoothing, cross_entropy_loss
from model.template_mesh import TemplateMesh


class Voxel2Mesh(nn.Module):
    def __init__(self, config):
        super(Voxel2Mesh, self).__init__()

        self.config = config
        self.voxel_encoder = VoxelEncoder(config)
        self.voxel_decoder = VoxelDecoder(config)
        self.mesh_decoder = MeshDecoder(config)
        self.template_mesh = TemplateMesh()

    def forward(self, data):
        voxel_features = self.voxel_encoder(data['x'])
        voxel_output = self.voxel_decoder(voxel_features)
        mesh_output = self.mesh_decoder(voxel_output)  # No need to pass template vertices

        return mesh_output

    def loss(self, data):
        """
        Computes the total loss for Voxel2Mesh.
        """
        pred = self.forward(data)

        ce_loss = torch.tensor(0.0, device=data['x'].device)
        chamfer_loss = torch.tensor(0.0, device=data['x'].device)
        edge_loss = torch.tensor(0.0, device=data['x'].device)
        laplacian_loss = torch.tensor(0.0, device=data['x'].device)

        # get the original faces from the template mesh
        faces = self.template_mesh.get_faces().to(data['x'].device)  # Shape: (num_faces, 3)

        for c in range(self.config.num_classes - 1):
            target_points = data['surface_points'][c]

            # deformed vertices from the prediction
            vertices = pred[c][:, :3]  # Assuming pred[c] gives 3D points
            voxel_features = self.voxel_encoder(data['x'])  # Use voxel features from the encoder

            # ensure proper shape for chamfer_distance
            vertices = vertices.unsqueeze(0)  # Add batch dimension
            target_points = target_points.unsqueeze(0)  # Add batch dimension

            # loss calculations
            chamfer_loss += chamfer_distance(vertices, target_points)
            edge_loss += mesh_edge_loss(vertices, faces)  # Use original faces
            laplacian_loss += laplacian_smoothing(vertices, faces)

            # pass voxel features to voxel decoder to get pred_voxels
            pred_voxels = self.voxel_decoder(voxel_features)  # Get voxel predictions

            # add channel dimension to y_voxels before resizing
            # assume the target has shape (1, 100, 100, 22)
            target_voxels = data['y_voxels'].unsqueeze(1).float()  # Convert to float

            # resize target_voxels to match the output shape from the model
            target_voxels_resized = F.interpolate(
                target_voxels, size=(96, 96, 16), mode='trilinear', align_corners=False
            ).to(data['x'].device)

            # remove the extra channel dimension from target_voxels_resized
            target_voxels_resized = target_voxels_resized.squeeze(1)  # Remove the singleton channel dimension

            # convert target_voxels_resized to Long type
            target_voxels_resized = target_voxels_resized.long()  # Convert to Long type

            # ensure pred_voxels has the correct shape for cross-entropy loss
            ce_loss += cross_entropy_loss(pred_voxels, target_voxels_resized)

        # weighted sum of losses
        total_loss = (
                1.0 * chamfer_loss +
                1.0 * ce_loss +
                0.1 * laplacian_loss +
                1.0 * edge_loss
        )

        log = {
            "loss": total_loss.item(),
            "chamfer_loss": chamfer_loss.item(),
            "ce_loss": ce_loss.item(),
            "edge_loss": edge_loss.item(),
            "laplacian_loss": laplacian_loss.item()
        }

        return total_loss, log