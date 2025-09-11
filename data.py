import os
import numpy as np
import torch
import pyvista as pv
import SimpleITK as sitk
from scipy.ndimage import zoom
import skimage.measure

from config import *
from model.losses import normalize_points


# helper to match nrrd dimensions order that sitk pulls
TARGET_SHAPE_ZYX = (NRRD_DIMENSIONS[2], NRRD_DIMENSIONS[1], NRRD_DIMENSIONS[0])

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
        img_np = np.transpose(pad_to_size(img_np, TARGET_SHAPE_ZYX, pad_value=0))

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

        # get the correct label depending on the desired segmentation
        if seg_np.ndim == 4:
            seg_np = seg_np[..., SEG_LABEL]

        # flip axes to be in (x,y,z) such that the labels align properly
        seg_np = np.transpose(pad_to_size(seg_np, TARGET_SHAPE_ZYX, pad_value=0))
        labels.append(np.flipud(np.rot90(seg_np)))

    return np.array(labels)


def pad_to_size(volume, target_shape_zyx, pad_value=0):
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
    volume = np.pad(volume, [(pad_before[d], pad_after[d]) for d in range(3)], mode="constant", constant_values=pad_value)
    current_shape = np.array(volume.shape)

    # cropping while keeping foreground
    crop_slices = []
    for d in range(3):
        if current_shape[d] > target_shape[d]:
            # find indices of nonzero along axis d
            nonzero_idx = np.where(volume != 0)
            if len(nonzero_idx[0]) > 0:
                # compute min/max along axis d
                min_idx = nonzero_idx[d].min()
                max_idx = nonzero_idx[d].max()
                # Ensure crop window contains foreground and fits target
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
    Extracts outer surface points from a labeled voxel volume using marching cubes,
    resampling to isotropic spacing first so that the surface isn't limited to slice planes.

    :param voxel_data: array of segmentation/labeled volumes
    :param threshold: should be 0.5 (might need to change if adding multiple labels)
    :param num_points: number of points to extract
    :param spacing: spacing between points in x,y,z directions
    :return: array of extracted surface points
    """

    # convert to tensor - we want to use the GPU for this since it can take while if not
    if isinstance(voxel_data, np.ndarray):
        voxel_data = torch.tensor(voxel_data, dtype=torch.float32)

    # get batch size (getting surface pts from each individual volume)
    B = voxel_data.shape[0]
    all_points = []

    for b in range(B):
        if voxel_data.ndim == 5:  # (B, C, D, H, W)
            volume = voxel_data[b, 0].cpu().numpy()
        elif voxel_data.ndim == 4:  # (B, D, H, W)
            volume = voxel_data[b].cpu().numpy()
        else:
            raise ValueError(f"Unexpected voxel_data shape: {voxel_data.shape}")

        # resample to isotropic spacing
        # smallest spacing as target (1.9 mm)
        target_spacing = min(spacing)
        zoom_factors = [s / target_spacing for s in spacing]
        volume_iso = zoom(volume, zoom=zoom_factors, order=0)

        # ensure binary
        volume_iso = (volume_iso > threshold).astype(np.uint8)

        # marching cubes, then scale using correct spacing
        verts, faces, _, _ = skimage.measure.marching_cubes(volume_iso, level=0.5)
        verts *= target_spacing

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

        # normalize points to be between -1 and 1 (seems to help loss to not be NaN)
        verts = normalize_points(verts)
        all_points.append(verts)

    return torch.cat(all_points, dim=0)
