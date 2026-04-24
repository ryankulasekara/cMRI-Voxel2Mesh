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
            # use class-specific features for each pass thru mesh decoder
            class_specific_output = {
                'features': voxel_output['class_features'][c],
                'segmentation': voxel_output['segmentation']
            }
            mesh_outputs.append(self.mesh_decoders[c](class_specific_output))


        # the slices to pop up when running model, just useful to see if it's converging or not
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
            slice_idx=24
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

    def loss(self, data, loss_weights=None, epoch=0):
        """
        Computes the total loss w/ progressive weighting

        :param data: input data dict
        :param loss_weights: dict of weights for loss function
        :param epoch: current epoch
        """

        # get predictions
        pred = self.forward(data)
        faces = self.template_mesh.get_faces().to(data['x'].device)

        # default weights if not provided
        if loss_weights is None:
            loss_weights = {
                'ce_weight': 0.5,
                'dice_weight': 0.75,
                'chamfer_weight': 1.2,
                'edge_weight': 0.1,
                'lap_weight': 1.5,
                'normal_weight': 1.5
            }

        # always compute segmentation losses
        ce_loss = cross_entropy_loss(
            pred['segmentation'],
            data['y_voxels'].float()
        ) / self.config.num_classes

        dsc_loss = dice_loss(
            pred['segmentation'],
            data['y_voxels'].float()
        )

        # initialize mesh losses
        chamfer_loss_total = 0
        edge_loss_total = 0
        lap_loss_total = 0
        normal_loss_total = 0

        # only compute mesh losses if weights are non-zero.. just saves time for warmup epochs
        if loss_weights['chamfer_weight'] > 0 or loss_weights['edge_weight'] > 0:
            for c, mesh_pred in enumerate(pred["meshes"]):
                # get target mesh
                target_mesh = extract_surface_points(
                    pred["segmentation"][:, c].cpu().detach().numpy()
                )
                target_mesh = target_mesh.to(data['x'].device)

                # compute mesh losses if weights > 0
                if loss_weights['chamfer_weight'] > 0:
                    chamfer_loss = chamfer_distance(mesh_pred, target_mesh)
                    chamfer_loss_total += chamfer_loss

                if loss_weights['edge_weight'] > 0:
                    edge_loss = mesh_edge_loss(mesh_pred, faces)
                    edge_loss_total += edge_loss

                if loss_weights['lap_weight'] > 0:
                    lap_loss = laplacian_smoothing(mesh_pred, faces)
                    lap_loss_total += lap_loss

                if loss_weights['normal_weight'] > 0:
                    normal_loss = normal_consistency_loss(mesh_pred, faces)
                    normal_loss_total += normal_loss

            # average over num of mesh classes
            num_mesh = len(pred["meshes"])
            chamfer_loss_total = chamfer_loss_total / num_mesh
            edge_loss_total = edge_loss_total / num_mesh
            lap_loss_total = lap_loss_total / num_mesh
            normal_loss_total = normal_loss_total / num_mesh

        # apply weights
        total_loss = (
                loss_weights['ce_weight'] * ce_loss +
                loss_weights['dice_weight'] * dsc_loss +
                loss_weights['chamfer_weight'] * chamfer_loss_total +
                loss_weights['edge_weight'] * edge_loss_total +
                loss_weights['lap_weight'] * lap_loss_total +
                loss_weights['normal_weight'] * normal_loss_total
        )

        if loss_weights['chamfer_weight'] == 0:
            log = {
                "ce_loss": ce_loss.item(),
                "dice_loss": dsc_loss.item(),
                "chamfer_loss": chamfer_loss_total,
                "edge_loss": edge_loss_total,
                "laplacian_loss": lap_loss_total,
                "normal_loss": normal_loss_total,
                "epoch": epoch
            }
        else:
            log = {
                "ce_loss": ce_loss.item(),
                "dice_loss": dsc_loss.item(),
                "chamfer_loss": chamfer_loss_total.item(),
                "edge_loss": edge_loss_total.item(),
                "laplacian_loss": lap_loss_total.item(),
                "normal_loss": normal_loss_total.item(),
                "epoch": epoch
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

    # if slice not specified, pick middle slice to display
    if slice_idx is None:
        slice_idx = z // 2

    for batch in range(0,b):
        input_slice = np.squeeze(input_vol[batch, :, :, :, slice_idx])
        label_slice = np.squeeze(label_vol[batch, :, :, :, slice_idx])
        pred_slice = np.squeeze(pred_vol[batch, :, :, :, slice_idx])

        label_mask = np.zeros(label_slice.shape[1:], dtype=np.int32)
        label_mask[label_slice[0] > 0.5] = 1  # LV
        label_mask[label_slice[1] > 0.5] = 2  # RV
        label_mask[label_slice[2] > 0.5] = 3  # AORTA
        label_mask[label_slice[3] > 0.5] = 4  # PT
        label_mask[label_slice[4] > 0.5] = 5  # LA
        label_mask[label_slice[5] > 0.5] = 6  # RA
        label_mask[label_slice[6] > 0.5] = 7  # FAT

        if pred_slice.ndim == 3:
            # Get class with maximum probability
            max_probs = np.max(pred_slice, axis=0)
            pred_mask = np.argmax(pred_slice, axis=0) + 1  # +1 because 0 is background
            # only assign class if max probability > 0.0
            pred_mask[max_probs <= 0.0] = 0
        else:
            pred_mask = np.zeros(pred_slice.shape[1:], dtype=np.int32)
            # get class with highest probability for each pixel
            for h in range(pred_slice.shape[1]):
                for w in range(pred_slice.shape[2]):
                    class_probs = pred_slice[:, h, w]
                    max_class = np.argmax(class_probs)
                    if class_probs[max_class] > 0.0:
                        pred_mask[h, w] = max_class + 1
                    else:
                        pred_mask[h, w] = 0

        # colormap
        class_colors = {
            0: (0.0, 0.0, 0.0),   # background
            1: (0.6, 0.0, 0.8),   # LV
            2: (0.0, 0.0, 1.0),   # RV
            3: (1.0, 0.0, 1.0),   # AORTA
            4: (0.01, 0.75, 0.0),   # PT
            5: (1.0, 0.46, 0.09),  # LA
            6: (0.15, 0.63, 0.68), # RA
            7: (0.8, 0.8, 0.0),   # FAT
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

    plotter = pv.Plotter()

    for i, mesh_vertices in enumerate(meshes):

        # convert to numpy
        if isinstance(mesh_vertices, torch.Tensor):
            mesh_vertices = mesh_vertices.detach().cpu().numpy()

        # get rid of batch dim
        if mesh_vertices.ndim == 3:
            mesh_vertices = mesh_vertices[0]

        # create meshes
        cloud = pv.PolyData(mesh_vertices, faces)
        plotter.add_mesh(cloud, color=['purple', 'blue', 'pink', 'green','orange', 'cyan'][i])

    plotter.show()


def get_loss_weights(epoch, total_epochs, warmup_epochs=50):
    """
    Progressive loss weighting strategy
    - 1st phase (epoch < warmup_epochs): only segmentation losses
    - 2nd phase: gradually introduce mesh losses
    """
    if epoch < warmup_epochs:
        # Phase 1: Only segmentation losses
        return {
            'ce_weight': 2.5,
            'dice_weight': 0.0,
            'chamfer_weight': 0.0,
            'edge_weight': 0.0,
            'lap_weight': 0.0,
            'normal_weight': 0.0
        }
    else:
        # Phase 2: Gradually increase mesh losses
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        ramp = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))

        return {
            'ce_weight': 2.5,
            'dice_weight': 0.0,
            'chamfer_weight': 1.0 * ramp,
            'edge_weight': 0.1 * ramp,
            'lap_weight': 2.0 * ramp,
            'normal_weight': 2.0 * ramp
        }