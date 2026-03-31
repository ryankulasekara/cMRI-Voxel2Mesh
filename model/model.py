import torch
import torch.nn as nn
import pyvista as pv
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import NRRD_DIMENSIONS
from data import extract_surface_points
from model.voxel_encoder import VoxelEncoder
from model.voxel_decoder import VoxelDecoder
from model.mesh_decoder import MeshDecoder
from model.losses import chamfer_distance, mesh_edge_loss, laplacian_smoothing, cross_entropy_loss, normal_consistency_loss, dice_loss
from model.template_mesh import TemplateMesh


class Voxel2Mesh(nn.Module):
    def __init__(self, config):
        super(Voxel2Mesh, self).__init__()
        self.config = config
        self.template_mesh = TemplateMesh()

        # initialize each component of the network
        self.voxel_encoder = VoxelEncoder(config)
        self.voxel_decoder = VoxelDecoder(config)
        self.mesh_decoders = nn.ModuleList([MeshDecoder(config, c) for c in range(config.num_mesh_classes)])

        # store which classes have meshes (first num_mesh_classes)
        self.mesh_classes = list(range(config.num_mesh_classes))
        self.non_mesh_classes = list(range(config.num_mesh_classes, config.num_classes))

    def forward(self, data):
        voxel_features = self.voxel_encoder(data['x'])
        voxel_output = self.voxel_decoder(voxel_features)
        mesh_outputs = []

        for c in self.mesh_classes:
            # Use class-specific features for each mesh decoder
            class_specific_output = {
                'features': voxel_output['class_features'][c],
                'segmentation': voxel_output['segmentation']
            }
            mesh_outputs.append(self.mesh_decoders[c](class_specific_output))

        visualize_slice(
            input_vol=data['x'],
            label_vol=data['y_voxels'],
            pred_vol=voxel_output['segmentation'],
            slice_idx=15
        )

        visualize_slice(
            input_vol=data['x'],
            label_vol=data['y_voxels'],
            pred_vol=voxel_output['segmentation'],
            slice_idx=22
        )

        # debugging
        # faces = self.template_mesh.get_faces().cpu().numpy()
        # faces_pyvista = []
        # for face in faces:
        #     faces_pyvista.append([3, *face])
        # faces_pyvista = np.array(faces_pyvista).flatten()
        # visualize_meshes((extract_surface_points(voxel_output["segmentation"][:,0].cpu().detach().numpy()),
        #                   extract_surface_points(voxel_output["segmentation"][:,1].cpu().detach().numpy())),
        #                   faces=faces_pyvista)


        return {
            'meshes': mesh_outputs,
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
        faces = self.template_mesh.get_faces().to(data['x'].device)

        # cross-entropy loss
        ce_loss = cross_entropy_loss(
            pred['segmentation'],
            data['y_voxels'].float()
        ) / self.config.num_classes

        dsc_loss = dice_loss(
            pred['segmentation'],
            data['y_voxels'].float()
        )

        # mesh losses
        chamfer_loss_total, edge_loss_total, lap_loss_total, normal_loss_total = 0, 0, 0, 0

        for c, mesh_idx in enumerate(self.mesh_classes):
            # extract surface from ground truth for class c
            mesh_pred = pred["meshes"][mesh_idx]

            # extract target mesh for this chamber
            target_mesh = extract_surface_points(pred["segmentation"][:, c].cpu().detach().numpy())

            chamfer_loss = chamfer_distance(mesh_pred, target_mesh)
            edge_loss = mesh_edge_loss(mesh_pred, faces)
            lap_loss = laplacian_smoothing(mesh_pred, faces)
            normal_loss = normal_consistency_loss(mesh_pred, faces)

            chamfer_loss_total += chamfer_loss
            edge_loss_total += edge_loss
            lap_loss_total += lap_loss
            normal_loss_total += normal_loss

        total_loss = (
                0.5 * ce_loss +
                0.75 * dsc_loss +
                1.2 * chamfer_loss_total / self.config.num_mesh_classes +
                0.1 * edge_loss_total / self.config.num_mesh_classes +
                1.5 * lap_loss_total / self.config.num_mesh_classes +
                1.5 * normal_loss_total / self.config.num_mesh_classes
        )

        log = {
            "ce_loss": ce_loss.item(),
            "dice_loss": dsc_loss.item(),
            "chamfer_loss": chamfer_loss_total.item(),
            "edge_loss": edge_loss_total.item(),
            "laplacian_loss": lap_loss_total.item(),
            "normal_loss": normal_loss_total.item()
        }
        return total_loss, log


def visualize_slice(input_vol, label_vol, pred_vol, slice_idx=None, alpha=0.4):
    """
    Visualize slice from MRI volume... displays ground truth labels & predicted segmentation

    :param input_vol: MRI image volume
    :param label_vol: ground truth labels
    :param pred_vol: predicted labels from segmentation
    :param slice_idx: slice index to display
    :param alpha: opacity
    """

    # convert to np
    if isinstance(input_vol, torch.Tensor):
        input_vol = input_vol.detach().cpu().numpy()
    if isinstance(label_vol, torch.Tensor):
        label_vol = label_vol.detach().cpu().numpy()
    if isinstance(pred_vol, torch.Tensor):
        pred_vol = pred_vol.detach().cpu().numpy()

    b, _, z, y, x = input_vol.shape
    input_vol = np.squeeze(input_vol)
    label_vol = np.squeeze(label_vol)
    pred_vol = np.squeeze(pred_vol)

    # if slice not specified, pick middle slice to display
    if slice_idx is None:
        slice_idx = z // 2

    input_slice = input_vol[:, :, slice_idx]
    label_slice = label_vol[:, :, :, slice_idx]
    pred_slice = pred_vol[:, :, :, slice_idx]

    label_mask = np.zeros(label_slice.shape[1:], dtype=np.int32)
    label_mask[label_slice[0] > 0.5] = 1  # LV
    label_mask[label_slice[1] > 0.5] = 2  # RV
    label_mask[label_slice[2] > 0.5] = 3  # FAT

    pred_mask = np.zeros(pred_slice.shape[1:], dtype=np.int32)
    pred_mask[pred_slice[0] > 0.0] = 1  # LV
    pred_mask[pred_slice[1] > 0.0] = 2  # RV
    pred_mask[pred_slice[2] > 0.0] = 3  # FAT

    # colormap
    class_colors = {
        0: (0.0, 0.0, 0.0),   # background
        1: (0.75, 0.0, 0.75),   # LV
        2: (0.0, 0.0, 1.0),   # RV
        3: (0.8, 0.8, 0.0),   # FAT
    }

    # overlay colormap onto slice
    def overlay_segmentation(mask, colors, alpha):
        overlay = np.zeros((*mask.shape, 4))
        for label, color in colors.items():
            if label == 0:
                continue
            m = mask == label
            overlay[m, :3] = color
            overlay[m, 3] = alpha
        return overlay

    label_overlay = overlay_segmentation(label_mask, class_colors, alpha)
    pred_overlay = overlay_segmentation(pred_mask, class_colors, alpha)

    # plot using matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    ax[0].imshow(input_slice, cmap="gray")
    ax[0].imshow(label_overlay)
    ax[0].set_title("Ground Truth Labels")

    ax[1].imshow(input_slice, cmap="gray")
    ax[1].imshow(pred_overlay)
    ax[1].set_title("Predicted Segmentation")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_meshes(meshes, faces, titles=None):
    """
    Helper function to visualize 3D meshes

    :param meshes: PV meshes
    :param faces: from template mesh
    :param titles:
    :return:
    """

    # pv.set_jupyter_backend('static')

    plotter = pv.Plotter()

    for i, mesh_vertices in enumerate(meshes):

        # Handle tensor conversion and batch dimension
        if isinstance(mesh_vertices, torch.Tensor):
            mesh_vertices = mesh_vertices.detach().cpu().numpy()

        # Remove batch dimension if present [B, N, 3] -> [N, 3]
        if mesh_vertices.ndim == 3:
            mesh_vertices = mesh_vertices[0]  # Take first batch element

        # Create simple point cloud for visualization
        cloud = pv.PolyData(mesh_vertices, faces)

        # Add faces if you have them (optional)
        # For now, just visualize as point cloud
        plotter.add_mesh(cloud, color=['purple', 'blue', 'pink', 'green','orange', 'cyan'][i])

    plotter.show()

