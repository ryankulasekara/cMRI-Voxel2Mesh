import os
import numpy as np
import mcubes
import torch
import pyvista as pv
import SimpleITK as sitk
from scipy.ndimage import zoom
import skimage.measure

import config
from config import *
from model.mesh_utils import normalize_points
from model.template_mesh import TemplateMesh


# helper to match nrrd dimensions order... sitk has weird ordering
TARGET_SHAPE_ZYX = (NRRD_DIMENSIONS[2], NRRD_DIMENSIONS[1], NRRD_DIMENSIONS[0], NRRD_DIMENSIONS[3])

def reorient_to_identity(sitk_img, orientation="LPS"):
    """
    Helper function to reorient simple itk segmentations to their respective input images

    :param sitk_img: the segmentation image
    :param orientation: default orientation
    :return: reoriented segmentation
    """
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orientation)
    return orient_filter.Execute(sitk_img)


def load_images(file_path):
    """
    Loads the MRI image volumes from a directory

    :param file_path: directory containing mri image volumes
    :return: array of loaded/preprocessed image volumes as well as headers for aligning the segmentation labels
    """

    images = []
    headers = []
    img_files = sorted([f for f in os.listdir(file_path) if f.endswith('.nrrd') and not f.endswith('.seg.nrrd')])

    for filename in img_files:
        full_path = os.path.join(file_path, filename)

        # load with simple itk - ordering is (z,y,x) by default
        image_sitk = sitk.ReadImage(full_path)
        image_sitk = reorient_to_identity(image_sitk)
        img_np = sitk.GetArrayFromImage(image_sitk)  # remember that np array is in (z,y,x) order from the sitk image

        # pad/crop to target shape
        img_np = np.transpose(pad_to_size(img_np, TARGET_SHAPE_ZYX[0:3], pad_value=0))

        # z-score normalize voxel intensities
        mean, std = np.mean(img_np), np.std(img_np)
        img_np = (img_np - mean) / (std + 1e-8)

        images.append(img_np)
        headers.append(image_sitk)

    return np.array(images), headers


def load_labels(file_path, headers):
    """
    Loads the MRI labels from a directory

    :param file_path: directory containing MRI labels
    :param headers: headers for the corresponding MRI images
    :return: array of loaded/preprocessed labels
    """

    labels = []
    seg_files = sorted([f for f in os.listdir(file_path) if f.endswith('.seg.nrrd')])

    for i, filename in enumerate(seg_files):
        full_path = os.path.join(file_path, filename)

        # load with simple itk - again, remember these are in (z,y,x) order
        seg_sitk = sitk.ReadImage(full_path)
        seg_sitk = reorient_to_identity(seg_sitk)
        image_sitk = headers[i]  # matching MRI image header (contains alignment info)

        # resample the segmentation to align with the corresponding input image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(sitk.Transform())
        seg_resampled = resampler.Execute(seg_sitk)

        # convert to np array (still in (z,y,x) order)
        seg_np = sitk.GetArrayFromImage(seg_resampled)

        # get the correct labels depending on the desired cardiac structure (SEG_LABEL)
        if seg_np.ndim == 4:
            seg_fat = seg_np[..., 1]
            seg_np = seg_np[..., 3:5]
            seg_np = np.concatenate((seg_np, seg_fat[..., np.newaxis]),axis=-1)

        # flip axes to be in (x,y,z) such that the labels align properly
        seg_np = np.transpose(pad_to_size(seg_np, TARGET_SHAPE_ZYX, pad_value=0, label=True))
        labels.append(np.flipud(np.rot90(seg_np)))

    return np.array(labels)


def pad_to_size(volume, target_shape_zyx, pad_value=0, label=False):
    """
    Zero-pad or crop MRI volume to the desired target shape

    :param volume: input image or label volume
    :param target_shape_zyx: target dimensions
    :param pad_value: what number to pad with (0 by default)
    :return:
    """

    # make sure inputs are np arrays
    current_shape = np.array(volume.shape)
    target_shape = np.array(target_shape_zyx)

    # get center of before & after padding
    # we want the center to be in the same spot (pad around all edges, not just extending outwards right & down)
    pad_before = np.maximum((target_shape - current_shape) // 2, 0)
    pad_after = np.maximum(target_shape - current_shape - pad_before, 0)
    if label:
        rng = range(4)
        volume = np.pad(volume, [(pad_before[d], pad_after[d]) for d in rng], mode="constant", constant_values=pad_value)
    else:
        rng = range(3)
        volume = np.pad(volume, [(pad_before[d], pad_after[d]) for d in rng], mode="constant",constant_values=pad_value)
    current_shape = np.array(volume.shape)

    # cropping while keeping foreground
    crop_slices = []
    for d in rng:
        if current_shape[d] > target_shape[d]:
            # find indices of nonzero along axis d
            nonzero_idx = np.where(volume != 0)
            if len(nonzero_idx[0]) > 0:
                # compute min/max along axis d
                min_idx = nonzero_idx[d].min()
                max_idx = nonzero_idx[d].max()

                # make sure crop window contains foreground and fits target
                start = max(0, min(min_idx, current_shape[d] - target_shape[d]))
                end = start + target_shape[d]
            else:
                # center crop if no foreground exists
                start = (current_shape[d] - target_shape[d]) // 2
                end = start + target_shape[d]
            crop_slices.append(slice(start, end))
        else:
            crop_slices.append(slice(0, current_shape[d]))

    volume = volume[tuple(crop_slices)]
    return volume


def extract_surface_points(voxel_data, threshold=0.5, num_points=NUM_POINTS, spacing=(1.92308, 1.92308, 9.99985)):
    """
    Get marching cubes mesh from segmentations for each chamber

    :param voxel_data: segmentation volume
    :param threshold: level to decide between 1 or 0 for each chamber
    :param num_points: number of points to sample from marching cubes mesh
    :param spacing: spacing between points in x,y,z directions
    """
    # convert to tensor if not already
    if isinstance(voxel_data, np.ndarray):
        voxel_data = torch.tensor(voxel_data, dtype=torch.float32).to(DEVICE)

    # dimensions
    C, D, H, W = voxel_data.shape

    # only important if batch size isn't 1
    all_points = []

    # list of pts for each chamber
    chamber_points = []
    for c in range(C):
        volume = voxel_data[c].cpu().numpy()

        if volume.sum() == 0:
            # empty list of pts
            empty_verts = np.zeros((num_points, 3))
            chamber_points.append(empty_verts)
            continue

        # resample to isotropic spacing
        # smallest spacing as target (1.9 mm)
        target_spacing = min(spacing)
        zoom_factors = [s / target_spacing for s in spacing]
        volume_iso = zoom(volume, zoom=zoom_factors, order=0)

        # marching cubes, then scale using correct spacing
        try:
            verts, faces, _, _ = skimage.measure.marching_cubes(volume, level=0.0)
            # verts *= target_spacing
        except:
            try:
                verts, faces, _, _ = skimage.measure.marching_cubes(volume, level=-0.5)
                print("No segmentation for chamber", c+3)
            except:
                exit(0)

        # sample pts to be num_points
        # if extracted pts is greater than num_points
        if verts.shape[0] > num_points:
            idx = np.random.choice(verts.shape[0], num_points, replace=False)
            verts = verts[idx]
        # if extracted pts is less than num_points
        else:
            repeats = num_points // verts.shape[0] + 1
            verts = np.tile(verts, (repeats, 1))[:num_points]

        # convert to tensor
        verts = torch.tensor(verts, dtype=torch.float32).unsqueeze(0)

        # normalize points to be between -1 and 1 (get in same coordinate space as template mesh values)
        verts = map_coordinates(verts[0])
        all_points.append(torch.tensor(verts))

        # debugging
        pv_mesh = pv.PolyData(verts)
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color="red", opacity=0.8)
        template_mesh = TemplateMesh()
        template_verts = template_mesh.get_vertices()
        pv_template = pv.PolyData(template_verts.cpu().numpy())
        plotter.add_mesh(pv_template, color="blue", opacity=0.8)
        # plotter.show()


    return torch.cat(all_points, dim=0)


def map_coordinates(vertices):

    # this is the mri 'grid' of voxels in each dimension
    grid_x, grid_y, grid_z = 96, 96, 32

    # empty array w/ same dimensions as vertices
    vertices_norm = np.zeros_like(vertices)

    # map each point on [96,96,32] grid to a point in [(-1,1),(-1,1),(-1,1)] grid
    vertices_norm[:, 0] = 2.0 * (vertices[:, 0] / (grid_x - 1)) - 1.0  # x: [0,95] -> [-1,1]
    vertices_norm[:, 1] = 2.0 * (vertices[:, 1] / (grid_y - 1)) - 1.0  # y: [0,95] -> [-1,1]
    vertices_norm[:, 2] = 2.0 * (vertices[:, 2] / (grid_z - 1)) - 1.0  # z: [0,31] -> [-1,1]

    return vertices_norm


