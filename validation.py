import torch
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from config import *
from data import load_images, load_labels, map_coordinates
from model.model import Voxel2Mesh
from model.template_mesh import TemplateMesh


def visualize_segmentation_slices(image, label, prediction, slice_indices):
    """
    Visualize slice from MRI volume... displays actual label & predicted segmentation for each slice

    :param image: MRI image volume
    :param label: ground truth labels
    :param prediction: predicted labels
    :param slice_indices: slice indices to display from volume
    """
    # move to CPU and convert to numpy
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy().squeeze()
    if torch.is_tensor(label):
        label = label.detach().cpu().numpy().squeeze()
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy().squeeze()

    # colormap
    class_colors = {
        0: (0.0, 0.0, 0.0),  # background
        1: (0.75, 0.0, 0.75),  # LV
        2: (0.0, 0.0, 1.0),  # RV
        3: (0.8, 0.8, 0.0),  # FAT (if present)
    }

    def overlay_segmentation(mask, colors, alpha=0.5):
        overlay = np.zeros((*mask.shape, 4))
        for label_val, color in colors.items():
            if label_val == 0:
                continue
            m = mask == label_val
            if m.any():
                overlay[m, :3] = color
                overlay[m, 3] = alpha
        return overlay

    fig, axes = plt.subplots(len(slice_indices), 3, figsize=(15, 5 * len(slice_indices)))

    for idx, slice_idx in enumerate(slice_indices):
        # get slice to display
        image_slice = image[:, :, slice_idx] if image.ndim == 3 else image[slice_idx, :, :]

        # create masks for ground truth & prediction
        label_mask = np.zeros(image_slice.shape, dtype=np.int32)
        pred_mask = np.zeros(image_slice.shape, dtype=np.int32)

        # get masks for each class
        num_classes = label.shape[0] if label.ndim == 4 else label.shape[-1]
        for c in range(num_classes):
            if label.ndim == 4:
                label_slice = label[c, :, :, slice_idx]
            else:
                label_slice = label[:, :, slice_idx, c] if label.ndim == 4 else label[:, :, c, slice_idx]

            pred_slice = prediction[c, :, :, slice_idx] if prediction.ndim == 4 else prediction[:, :, slice_idx, c]

            label_mask[label_slice > 0.5] = c + 1
            pred_mask[pred_slice > 0.0] = c + 1

        # create overlays
        label_overlay = overlay_segmentation(label_mask, class_colors, alpha=0.6)
        pred_overlay = overlay_segmentation(pred_mask, class_colors, alpha=0.6)

        # plot w/ matplotlib
        axes[idx, 0].imshow(image_slice, cmap='gray')
        axes[idx, 0].set_title(f'MRI Slice {slice_idx}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(image_slice, cmap='gray')
        axes[idx, 1].imshow(label_overlay)
        axes[idx, 1].set_title(f'Ground Truth - Slice {slice_idx}')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(image_slice, cmap='gray')
        axes[idx, 2].imshow(pred_overlay)
        axes[idx, 2].set_title(f'Prediction - Slice {slice_idx}')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()
    return fig


def create_marching_cubes_mesh(segmentation, class_idx, threshold=0.5):
    """
    Create marching cubes mesh for a specific class from segmentation volume

    :param segmentation: segmentation from MRI volume
    :param class_idx: class from segmentation
    :param threshold: threshold for marching cubes
    """
    import skimage.measure

    if torch.is_tensor(segmentation):
        segmentation = segmentation.detach().cpu().numpy()

    # get the specific class
    if segmentation.ndim == 4:
        class_seg = segmentation[0, class_idx]
    elif segmentation.ndim == 5:
        class_seg = segmentation[0, class_idx]
    else:
        class_seg = segmentation[class_idx]

    # get segmentation values higher than chosen threshold
    binary = (class_seg > threshold).astype(np.uint8)

    if binary.sum() == 0:
        return None, None

    # create marching cubes mesh
    verts, faces, _, _ = skimage.measure.marching_cubes(binary, level=0.5)

    # map to normalized coordinates to match MRI dimensions
    verts_normalized = map_coordinates(verts)

    return verts_normalized, faces


def visualize_meshes_combined(predicted_meshes, segmentation, template_faces):
        """
        Visualize all meshes in a single PyVista window... including marching cubes representation of fat

        :param predicted_meshes: list of meshes to show from model output
        :param segmentation: fat segmentation volume
        :param template_faces: faces from template mesh
        """
        plotter = pv.Plotter()

        # colormap
        mesh_colors = {
            0: 'purple',  # LV
            1: 'blue',  # RV
        }

        # meshes from model output
        for i, mesh_vertices in enumerate(predicted_meshes):
            if torch.is_tensor(mesh_vertices):
                mesh_vertices = mesh_vertices.detach().cpu().numpy()

            # get rid of batch dimension if it's still there
            if mesh_vertices.ndim == 3:
                mesh_vertices = mesh_vertices[0]

            # create pv mesh
            mesh = pv.PolyData(mesh_vertices, template_faces)
            color = mesh_colors.get(i, 'gray')

            plotter.add_mesh(mesh, color=color, opacity=1.0, label=f'Chamber {i}', show_edges=False)

        # add marching cubes from fat segmentation labels
        fat_class_idx = 2
        if segmentation.shape[1] > fat_class_idx:
            fat_verts, fat_faces = create_marching_cubes_mesh(segmentation, fat_class_idx)

            if fat_verts is not None and len(fat_verts) > 0:

                # create pv mesh from fat_verts
                if fat_faces is not None:

                    # convert faces to pv format
                    faces_pv = []
                    for face in fat_faces:
                        faces_pv.append([3, *face])
                    faces_pv = np.array(faces_pv).flatten()
                    fat_mesh = pv.PolyData(fat_verts, faces_pv)

                else:
                    fat_mesh = pv.PolyData(fat_verts)

                plotter.add_mesh(fat_mesh, color='yellow', opacity=0.4, label='Fat (Marching Cubes)', show_edges=False)

        # plot
        plotter.add_axes()
        plotter.add_legend()
        plotter.show()
        return plotter


def main():
    # make sure to use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize template mesh
    template = TemplateMesh()
    template_faces = template.get_faces().cpu().numpy()

    # make sure faces are readable by pyvista
    faces_pyvista = []
    for face in template_faces:
        faces_pyvista.append([3, *face])
    faces_pyvista = np.array(faces_pyvista).flatten()

    # load trained model
    config.batch_size = 1
    model = Voxel2Mesh(config).to(device)
    model.load_state_dict(torch.load("voxel2mesh_model.pth", map_location=device))
    model.eval()

    # load test images
    print("Loading test images...")
    test_images, headers = load_images(TEST_IMAGES)
    test_labels = load_labels(TEST_LABELS, headers)

    val_images_tensor = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2, 4)
    val_labels_tensor = torch.tensor(test_labels, dtype=torch.long).permute(0, 2, 3, 1, 4)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("\nProcessing validation samples...")
    sample_count = 0

    with torch.no_grad():
        for images, labels in val_loader:
            if sample_count >= len(test_images):
                break

            images = images.to(device)
            labels = labels.to(device)

            # run model on val data
            output = model({'x': images, 'y_voxels': labels})

            # just for reference... what % of voxels are each class
            pred_seg = output['segmentation']
            print(f"\nSample {sample_count + 1}:")
            for c in range(min(pred_seg.shape[1], 7)):
                pred_count = (pred_seg[0, c] > 0.5).sum().item()
                total_voxels = pred_seg[0, c].numel()
                print(f"  Class {c}: {pred_count}/{total_voxels} voxels predicted ({100 * pred_count / total_voxels:.2f}%)")

            # visualize slices
            visualize_segmentation_slices(
                images[0], labels[0], pred_seg[0],
                slice_indices=(15,22),
            )

            # visualize meshes
            visualize_meshes_combined(
                output['meshes'],
                pred_seg,
                faces_pyvista,
            )

            sample_count += 1


if __name__ == "__main__":
    main()