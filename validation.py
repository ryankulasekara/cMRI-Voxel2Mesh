# import numpy as np
# import pyvista as pv
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, TensorDataset
#
# from config import *
# from data import load_images, load_labels, map_coordinates
# from model.model import Voxel2Mesh
# from model.template_mesh import TemplateMesh
#
#
# def visualize_segmentation_slices(image, label, prediction, slice_indices):
#     """
#     Visualize slice from MRI volume... displays actual label & predicted segmentation for each slice
#
#     :param image: MRI image volume
#     :param label: ground truth labels
#     :param prediction: predicted labels
#     :param slice_indices: slice indices to display from volume
#     """
#     # move to CPU and convert to numpy
#     if torch.is_tensor(image):
#         image = image.detach().cpu().numpy().squeeze()
#     if torch.is_tensor(label):
#         label = label.detach().cpu().numpy().squeeze()
#     if torch.is_tensor(prediction):
#         prediction = prediction.detach().cpu().numpy().squeeze()
#
#     # colormap
#     class_colors = {
#         0: (0.0, 0.0, 0.0),   # background
#         1: (0.75, 0.0, 0.75),   # LV
#         2: (0.0, 0.0, 1.0),   # RV
#         3: (1.0, 0.0, 1.0),   # AORTA
#         4: (0.01, 0.75, 0.0),   # PT
#         5: (1.0, 0.46, 0.09),  # LA
#         6: (0.15, 0.63, 0.68), # RA
#         7: (0.8, 0.8, 0.0),   # FAT
#     }
#
#     def overlay_segmentation(mask, colors, alpha=0.5):
#         overlay = np.zeros((*mask.shape, 4))
#         for label_val, color in colors.items():
#             if label_val == 0:
#                 continue
#             m = mask == label_val
#             if m.any():
#                 overlay[m, :3] = color
#                 overlay[m, 3] = alpha
#         return overlay
#
#     fig, axes = plt.subplots(len(slice_indices), 3, figsize=(15, 5 * len(slice_indices)))
#
#     for idx, slice_idx in enumerate(slice_indices):
#         # get slice to display
#         image_slice = image[:, :, slice_idx] if image.ndim == 3 else image[slice_idx, :, :]
#
#         # create masks for ground truth & prediction
#         label_mask = np.zeros(image_slice.shape, dtype=np.int32)
#         pred_mask = np.zeros(image_slice.shape, dtype=np.int32)
#
#         # get masks for each class
#         num_classes = label.shape[0] if label.ndim == 4 else label.shape[-1]
#         for c in range(num_classes):
#             if label.ndim == 4:
#                 label_slice = label[c, :, :, slice_idx]
#             else:
#                 label_slice = label[:, :, slice_idx, c] if label.ndim == 4 else label[:, :, c, slice_idx]
#
#             pred_slice = prediction[c, :, :, slice_idx] if prediction.ndim == 4 else prediction[:, :, slice_idx, c]
#
#             label_mask[label_slice > 0.5] = c + 1
#             pred_mask[pred_slice > 0.0] = c + 1
#
#         # create overlays
#         label_overlay = overlay_segmentation(label_mask, class_colors, alpha=0.6)
#         pred_overlay = overlay_segmentation(pred_mask, class_colors, alpha=0.6)
#
#         # plot w/ matplotlib
#         axes[idx, 0].imshow(image_slice, cmap='gray')
#         axes[idx, 0].set_title(f'MRI Slice {slice_idx}')
#         axes[idx, 0].axis('off')
#
#         axes[idx, 1].imshow(image_slice, cmap='gray')
#         axes[idx, 1].imshow(label_overlay)
#         axes[idx, 1].set_title(f'Ground Truth - Slice {slice_idx}')
#         axes[idx, 1].axis('off')
#
#         axes[idx, 2].imshow(image_slice, cmap='gray')
#         axes[idx, 2].imshow(pred_overlay)
#         axes[idx, 2].set_title(f'Prediction - Slice {slice_idx}')
#         axes[idx, 2].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#     return fig
#
#
# def create_marching_cubes_mesh(segmentation, class_idx, threshold=0.0):
#     """
#     Create marching cubes mesh for a specific class from segmentation volume
#
#     :param segmentation: segmentation from MRI volume
#     :param class_idx: class from segmentation
#     :param threshold: threshold for marching cubes
#     """
#     import skimage.measure
#
#     if torch.is_tensor(segmentation):
#         segmentation = segmentation.detach().cpu().numpy()
#
#     # get the specific class
#     if segmentation.ndim == 4:
#         class_seg = segmentation[0, class_idx]
#     elif segmentation.ndim == 5:
#         class_seg = segmentation[0, class_idx]
#     else:
#         class_seg = segmentation[class_idx]
#
#     # get segmentation values higher than chosen threshold
#     binary = (class_seg > threshold).astype(np.uint8)
#
#     if binary.sum() == 0:
#         return None, None
#
#     # create marching cubes mesh
#     verts, faces, _, _ = skimage.measure.marching_cubes(binary, level=0.5)
#
#     # map to normalized coordinates to match MRI dimensions
#     verts_normalized = map_coordinates(verts)
#
#     return verts_normalized, faces
#
#
# def visualize_meshes_combined(predicted_meshes, segmentation, template_faces):
#         """
#         Visualize all meshes in a single PyVista window... including marching cubes representation of fat
#
#         :param predicted_meshes: list of meshes to show from model output
#         :param segmentation: fat segmentation volume
#         :param template_faces: faces from template mesh
#         """
#         plotter = pv.Plotter()
#
#         # colormap
#         mesh_colors = {
#             0: 'purple',  # LV
#             1: 'blue',  # RV
#             2: 'pink',
#             3: 'green',
#             4: 'orange',
#             5: 'cyan'
#         }
#
#         # meshes from model output
#         for i, mesh_vertices in enumerate(predicted_meshes):
#             if torch.is_tensor(mesh_vertices):
#                 mesh_vertices = mesh_vertices.detach().cpu().numpy()
#
#             # get rid of batch dimension if it's still there
#             if mesh_vertices.ndim == 3:
#                 mesh_vertices = mesh_vertices[0]
#
#             # create pv mesh
#             mesh = pv.PolyData(mesh_vertices, template_faces)
#             color = mesh_colors.get(i, 'gray')
#
#             plotter.add_mesh(mesh, color=color, opacity=1.0, label=f'Chamber {i}', show_edges=False)
#
#         # add marching cubes from fat segmentation labels
#         fat_class_idx = 6
#         if segmentation.shape[1] > fat_class_idx:
#             fat_verts, fat_faces = create_marching_cubes_mesh(segmentation, fat_class_idx)
#
#             if fat_verts is not None and len(fat_verts) > 0:
#
#                 # create pv mesh from fat_verts
#                 if fat_faces is not None:
#
#                     # convert faces to pv format
#                     faces_pv = []
#                     for face in fat_faces:
#                         faces_pv.append([3, *face])
#                     faces_pv = np.array(faces_pv).flatten()
#                     fat_mesh = pv.PolyData(fat_verts, faces_pv)
#
#                 else:
#                     fat_mesh = pv.PolyData(fat_verts)
#
#                 plotter.add_mesh(fat_mesh, color='yellow', opacity=0.5, label='Fat (Marching Cubes)', show_edges=False)
#
#         # plot
#         plotter.add_axes()
#         plotter.add_legend()
#         plotter.show()
#         return plotter
#
#
# def main():
#     # make sure to use gpu
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # initialize template mesh
#     template = TemplateMesh()
#     template_faces = template.get_faces().cpu().numpy()
#
#     # make sure faces are readable by pyvista
#     faces_pyvista = []
#     for face in template_faces:
#         faces_pyvista.append([3, *face])
#     faces_pyvista = np.array(faces_pyvista).flatten()
#
#     # load trained model
#     config.batch_size = 1
#     model = Voxel2Mesh(config).to(device)
#     model.load_state_dict(torch.load("voxel2mesh_model.pth", map_location=device))
#     model.eval()
#
#     # load test images
#     print("Loading test images...")
#     test_images, headers = load_images(TEST_IMAGES)
#     test_labels = load_labels(TEST_LABELS, headers)
#
#     val_images_tensor = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2, 4)
#     val_labels_tensor = torch.tensor(test_labels, dtype=torch.long).permute(0, 2, 3, 1, 4)
#     val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
#
#     print("\nProcessing validation samples...")
#     sample_count = 0
#
#     with torch.no_grad():
#         for images, labels in val_loader:
#             if sample_count >= len(test_images):
#                 break
#
#             images = images.to(device)
#             labels = labels.to(device)
#
#             # run model on val data
#             output = model({'x': images, 'y_voxels': labels})
#
#             # just for reference... what % of voxels are each class
#             pred_seg = output['segmentation']
#             print(f"\nSample {sample_count + 1}:")
#             for c in range(min(pred_seg.shape[1], 7)):
#                 pred_count = (pred_seg[0, c] > 0.5).sum().item()
#                 total_voxels = pred_seg[0, c].numel()
#                 print(f"  Class {c}: {pred_count}/{total_voxels} voxels predicted ({100 * pred_count / total_voxels:.2f}%)")
#
#             # visualize slices
#             visualize_segmentation_slices(
#                 images[0], labels[0], pred_seg[0],
#                 slice_indices=(15,22),
#             )
#
#             # visualize meshes
#             visualize_meshes_combined(
#                 output['meshes'],
#                 pred_seg,
#                 faces_pyvista,
#             )
#
#             sample_count += 1
#
#
# if __name__ == "__main__":
#     main()

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from config import *
from data import load_images, load_labels, map_coordinates
from model.model import Voxel2Mesh
from model.template_mesh import TemplateMesh


def compute_dsc(pred_mask, gt_mask, smooth=1e-6):
    """
    Compute Dice Similarity Coefficient between prediction and ground truth

    :param pred_mask: binary prediction mask from model (numpy array)
    :param gt_mask: binary ground truth mask (numpy array)
    :param smooth: smoothing factor... was getting div by zero before
    :return: DSC value
    """
    # flatten masks
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    # compute dsc
    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum()
    dsc = (2. * intersection + smooth) / (union + smooth)

    return dsc


def compute_chamber_dsc(pred_seg, gt_seg, class_idx, threshold=0.0):
    """
    Compute DSC for a specific chamber/class

    :param pred_seg: predicted segmentation volume
    :param gt_seg: ground truth segmentation volume
    :param class_idx: index of the class/chamber
    :param threshold: threshold for prediction mask... I'm using tanh so 0.0
    :return: DSC value for the specified class
    """

    # get binary masks for class
    pred_mask = (pred_seg[class_idx] > threshold).astype(np.uint8)
    gt_mask = (gt_seg[class_idx] > 0.5).astype(np.uint8)

    # compute DSC
    dsc = compute_dsc(pred_mask, gt_mask)

    return dsc


def compute_all_chamber_dsc(pred_seg, gt_seg, num_classes=6, threshold=0.0):
    """
    Compute DSC for all chambers/classes

    :param pred_seg: predicted segmentation volume
    :param gt_seg: ground truth segmentation volume
    :param num_classes: total number of classes
    :param threshold: threshold for prediction mask... I'm using tanh so 0.0
    :return: dict of DSC values per class, average DSC, and class names
    """

    # dict for classes
    class_names = {
        0: 'LV',
        1: 'RV',
        2: 'AORTA',
        3: 'PT',
        4: 'LA',
        5: 'RA',
        6: 'FAT'
    }

    dsc_scores = {}
    valid_classes = []

    for c in range(0, num_classes):
        dsc = compute_chamber_dsc(pred_seg, gt_seg, c, threshold)
        dsc_scores[c] = dsc
        valid_classes.append(dsc)

    # compute average DSC (excluding background)
    avg_dsc = np.mean(valid_classes) if valid_classes else 0.0

    return dsc_scores, avg_dsc, class_names


def print_dsc_results(dsc_scores, avg_dsc, class_names):
    """
    Print DSC results in a formatted way

    :param dsc_scores: dictionary of DSC scores per class
    :param avg_dsc: average DSC across all classes
    :param class_names: dict of classes
    """
    print("\n" + "=" * 50)
    print("DICE SIMILARITY COEFFICIENT RESULTS")
    print("=" * 50)

    for c, dsc in dsc_scores.items():
        class_name = class_names.get(c, f'Class {c}')
        print(f"  {class_name:12}: {dsc:.4f}")

    print("-" * 50)
    print(f"  {'Average':12}: {avg_dsc:.4f}")
    print("=" * 50 + "\n")


def visualize_segmentation_slices(image, label, prediction, slice_indices, dsc_scores=None):
    """
    Visualize slice from MRI volume... displays actual label & predicted segmentation for each slice
    Optionally displays DSC scores in the plot title

    :param image: MRI image volume
    :param label: ground truth labels
    :param prediction: predicted labels
    :param slice_indices: slice indices to display from volume
    :param dsc_scores: optional DSC scores to display in title
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
        3: (1.0, 0.0, 1.0),  # AORTA
        4: (0.01, 0.75, 0.0),  # PT
        5: (1.0, 0.46, 0.09),  # LA
        6: (0.15, 0.63, 0.68),  # RA
        7: (0.8, 0.8, 0.0),  # FAT
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

    # add DSC scores to overall figure title if provided
    if dsc_scores is not None:
        class_names = {0: 'LV', 1: 'RV', 2: 'AORTA', 3: 'PT', 4: 'LA', 5: 'RA', 6: 'FAT'}
        dsc_text = "DSC: " + ", ".join([f"{class_names[c]}={dsc:.3f}" for c, dsc in dsc_scores.items()])
        fig.suptitle(dsc_text, fontsize=12, y=1.02)

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


def create_marching_cubes_mesh(segmentation, class_idx, threshold=0.0):
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
        2: 'pink',
        3: 'green',
        4: 'orange',
        5: 'cyan'
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
    fat_class_idx = 6
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

            plotter.add_mesh(fat_mesh, color='yellow', opacity=0.5, label='Fat (Marching Cubes)', show_edges=False)

    # plot
    plotter.add_axes()
    plotter.add_legend()
    plotter.show()
    return plotter


def compute_and_save_summary_statistics(all_dsc_results, save_path=None):
    """
    Compute & save stats

    :param all_dsc_results: list of dictionaries containing DSC results per sample
    :param save_path: optional path to save summary to file
    """
    if not all_dsc_results:
        return

    # extract class names from first sample
    class_names = {0: 'LV', 1: 'RV', 2: 'AORTA', 3: 'PT', 4: 'LA', 5: 'RA', 6: 'FAT'}
    class_dsc_values = {c: [] for c in class_names.keys()}
    avg_dsc_values = []

    # get all DSC values
    for sample_results in all_dsc_results:
        dsc_scores, avg_dsc, _ = sample_results
        for c, dsc in dsc_scores.items():
            class_dsc_values[c].append(dsc)
        avg_dsc_values.append(avg_dsc)

    # compute stats
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS ACROSS ALL VALIDATION SAMPLES")
    print("=" * 60)
    summary = {}
    for c in class_names.keys():
        if class_dsc_values[c]:
            mean_dsc = np.mean(class_dsc_values[c])
            std_dsc = np.std(class_dsc_values[c])
            min_dsc = np.min(class_dsc_values[c])
            max_dsc = np.max(class_dsc_values[c])

            summary[class_names[c]] = {
                'mean': mean_dsc,
                'std': std_dsc,
                'min': min_dsc,
                'max': max_dsc
            }

            print(f"\n{class_names[c]}:")
            print(f"  Mean DSC: {mean_dsc:.4f} ± {std_dsc:.4f}")
            print(f"  Range: [{min_dsc:.4f}, {max_dsc:.4f}]")

    # overall average
    overall_mean = np.mean(avg_dsc_values)
    overall_std = np.std(avg_dsc_values)
    print(f"\n{'OVERALL':12}:")
    print(f"  Mean DSC: {overall_mean:.4f} ± {overall_std:.4f}")
    print("=" * 60 + "\n")

    # save to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Dice Similarity Coefficient Summary\n")
            f.write("=" * 50 + "\n\n")

            for class_name, stats in summary.items():
                f.write(f"{class_name}:\n")
                f.write(f"  Mean: {stats['mean']:.4f}\n")
                f.write(f"  Std: {stats['std']:.4f}\n")
                f.write(f"  Min: {stats['min']:.4f}\n")
                f.write(f"  Max: {stats['max']:.4f}\n\n")

            f.write(f"Overall Average:\n")
            f.write(f"  Mean: {overall_mean:.4f}\n")
            f.write(f"  Std: {overall_std:.4f}\n")

        print(f"Summary saved to {save_path}")

    return summary


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
    all_dsc_results = []

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
                print(
                    f"  Class {c}: {pred_count}/{total_voxels} voxels predicted ({100 * pred_count / total_voxels:.2f}%)")


            # convert to numpy for DSC computation
            pred_seg_np = pred_seg[0].cpu().numpy()
            labels_np = labels[0].cpu().numpy()

            # compute dsc
            dsc_scores, avg_dsc, class_names = compute_all_chamber_dsc(
                pred_seg_np, labels_np,
                num_classes=min(pred_seg.shape[1], 8),
                threshold=0.0
            )

            # store results & print
            all_dsc_results.append((dsc_scores, avg_dsc, class_names))
            print_dsc_results(dsc_scores, avg_dsc, class_names)

            # visualize slices with DSC scores in title
            visualize_segmentation_slices(
                images[0], labels[0], pred_seg[0],
                slice_indices=(13, 15, 22, 24),
                dsc_scores=dsc_scores
            )

            # visualize meshes
            visualize_meshes_combined(
                output['meshes'],
                pred_seg,
                faces_pyvista,
            )

            sample_count += 1

    # compute and display summary statistics across all samples
    if all_dsc_results:
        print("\n" + "=" * 60)
        print("GENERATING SUMMARY STATISTICS")
        print("=" * 60)
        compute_and_save_summary_statistics(all_dsc_results, save_path="dsc_summary.txt")


if __name__ == "__main__":
    main()