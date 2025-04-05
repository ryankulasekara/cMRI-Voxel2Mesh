import os
import nrrd
import nibabel as nib
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import trimesh
from scipy.ndimage import zoom
from skimage.measure import marching_cubes
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

from config import *


def load_labels(file_path, images):
    """
    Loads all .seg.nrrd files from a folder, extracts desired class label, stores them in a numpy array

    :param file_path: path to folder of .seg.nrrd files
    :param images: array of images that correspond to each label
    :return: numpy array of seg labels
    """

    # get all the .seg.nrrd filenames
    seg_nrrd_filenames = [f for f in os.listdir(file_path)]
    labels = []

    for i, filename in enumerate(seg_nrrd_filenames):
        # Load label data
        full_path = os.path.join(file_path, filename)
        seg_data, _ = nrrd.read(full_path)

        # Convert to SimpleITK images for resampling
        img_sitk = sitk.GetImageFromArray(images[i])
        seg_sitk = sitk.GetImageFromArray(seg_data.astype(np.float32))

        # Resample label to match image
        seg_resampled = sitk.Resample(
            seg_sitk,
            img_sitk,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0.0,
            seg_sitk.GetPixelID()
        )

        # Convert back to numpy and ensure 3D
        seg_array = sitk.GetArrayFromImage(seg_resampled)
        if len(seg_array.shape) == 4:  # If 4D (multiple labels)
            seg_array = seg_array[SEG_LABEL, ...]  # Extract specific label

        # Pad to target size
        seg_array = pad_to_size(seg_array, NRRD_DIMENSIONS)
        labels.append(seg_array)

    return np.array(labels)


def pad_to_size(data, target_size):
    """
    Padding function for labels & images

    :param data: input image
    :param target_size: size to zero pad to
    :return: zero-padded image
    """

    current_size = data.shape
    pad_amount = []
    for i in range(3):
        diff = target_size[i] - current_size[i]
        if diff > 0:
            pad_amount.append((diff // 2, diff - (diff // 2)))
        else:
            pad_amount.append((0, 0))

    return np.pad(data, pad_amount, mode='constant', constant_values=0)


def load_images(file_path):
    """
    Loads all .nrrd files from a folder & stores them in a numpy array.

    :param file_path: path to folder of .nrrd files
    :return: numpy array of images
    """

    # get all the .nrrd filenames
    nrrd_filenames = [f for f in os.listdir(file_path)]
    images = []

    for filename in nrrd_filenames:
        full_path = os.path.join(file_path, filename)
        image_data, _ = nrrd.read(full_path)

        # Ensure 3D (remove 4th dim if exists)
        if len(image_data.shape) == 4:
            image_data = image_data[0, ...]

        # Pad to target size
        image_data = pad_to_size(image_data, NRRD_DIMENSIONS)

        # Normalize
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data - mean) / (std + 1e-8)

        images.append(image_data)

    return np.array(images)


def load_nrrd(filename):
    """
    Loads a .nrrd file, separates into image data and header

    :param filename: filename of .nrrd file
    :return: image data and header
    """

    # load nrrd file
    # image_data is a numpy array w/ voxel intensities
    # header contains metadata
    image_data, header = nrrd.read(filename)

    return image_data, header


def preprocess(image, seg=False, target_size=NRRD_DIMENSIONS):
    """
    Preprocesses a .nrrd image to standard size specified in config.py by zero-padding
    Also z-score normalizes voxel values after zero-padding

    :param image: input .nrrd image
    :param seg: False if a normal image, True if a segmentation label
    :param target_size: desired output size
    :return: resized .nrrd image
    """

    # get current size of image
    current_size = image.shape

    # if it isn't a segmentation label, zero-pad & z-score normalize
    if not seg:
        # compute how much the image needs to be padded in each dimension
        pad_amount = [(max(0, target_size[i] - current_size[i]), 0) for i in range(3)]

        # apply the zero-padding
        padded_image = np.pad(image, pad_amount, mode='constant', constant_values=0)

        # z-score normalize voxel intensities
        mean = np.mean(padded_image)
        std = np.std(padded_image)
        image_data_normalized = (padded_image - mean) / std

        return image_data_normalized

    # if it is a segmentation, do padding accordingly & extract the desired label & return the image
    else:
        # compute how much the image needs to be padded in each dimension
        pad_amount = [(0, 0),
                      (max(0, target_size[0] - current_size[1]), 0),
                      (max(0, target_size[1] - current_size[2]), 0),
                      (max(0, target_size[2] - current_size[3]), 0)]

        # apply the zero-padding
        padded_image = np.pad(image, pad_amount, mode='constant', constant_values=0)

        # only get the desired label
        seg_image = padded_image[SEG_LABEL, :, :, :]
        return seg_image

def extract_surface_points(voxel_data, threshold=0.5, num_points=NUM_POINTS):
    """
    Extracts mesh representation from labeled voxel data in seg nrrd

    :param voxel_data: labeled volume (batch_size, x, y, z)
    :param threshold: voxels are 0 or 1, so pick 0.5 for this
    :param num_points: number of points to extract
    :return: resampled surface pts (batch_size, num_points, 3)
    """
    surface_points = []

    for i in range(voxel_data.shape[0]):
        volume = voxel_data[i]
        surface_voxels = volume > threshold

        # extract surface using marching cubes
        vertices, faces, _, _ = marching_cubes(surface_voxels, level=threshold)

        # sample points using nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(vertices)
        bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)
        random_points = np.random.uniform(bbox_min, bbox_max, (num_points, 3))
        _, indices = nbrs.kneighbors(random_points)
        resampled_points = vertices[indices.flatten()]
        voxel_spacing = np.array(SPACE_DIRECTIONS).diagonal()
        resampled_points *= voxel_spacing

        # z-score norm, prevent divide by 0
        resampled_points_mean = resampled_points.mean(axis=0)
        resampled_points_std = resampled_points.std(axis=0)
        # resampled_points_std[resampled_points_std == 0] = 1.0
        # resampled_points = preprocessing.MinMaxScaler().fit_transform(resampled_points)
        resampled_points = (resampled_points - resampled_points_mean) / resampled_points_std

        mesh = pv.PolyData(resampled_points)
        # mesh.plot()

        surface_points.append(resampled_points)

    return np.array(surface_points)


# def load_labels(file_path):
#     """
#     Loads all .nii.gz files from a folder, extracts desired class label, stores them in a numpy array.
#     """
#     # get all .nii.gz filenames
#     nii_filenames = [f for f in os.listdir(file_path) if f.endswith(".nii.gz")]
#
#     labels = []
#     for filename in nii_filenames:
#         full_path = os.path.join(file_path, filename)
#         nii_data = nib.load(full_path).get_fdata()
#
#         # preprocess (resize)
#         nii_data = preprocess(nii_data, seg=True)
#
#         labels.append(nii_data)
#
#     return np.array(labels)
#
#
# def load_images(file_path):
#     nii_filenames = [f for f in os.listdir(file_path) if f.endswith(".nii.gz")]
#
#     images = []
#     for filename in nii_filenames:
#         full_path = os.path.join(file_path, filename)
#         image_data = load_nii(full_path)  # Replace with your .nii.gz loader
#
#         print(f"Original shape of {filename}: {image_data.shape}")
#
#         image_data = preprocess(image_data)
#
#         print(f"Processed shape of {filename}: {image_data.shape}")
#
#         images.append(image_data)
#
#     return np.stack(images, axis=0)  # Ensures uniform shape
#
#
# def load_nii(filename):
#     """
#     Loads a .nii.gz file and returns the image data as a numpy array.
#
#     :param filename: filename of .nii.gz file
#     :return: image data as a numpy array
#     """
#
#     # Load .nii.gz file using nibabel
#     nii_img = nib.load(filename)
#
#     # Convert to numpy array (ensuring proper data type)
#     image_data = nii_img.get_fdata(dtype=np.float32)
#
#     return image_data
#
#
# def preprocess(image, seg=False, target_size=(128, 128, 64)):
#     """
#     Preprocesses a 3D image by downsampling to the target size and applying z-score normalization.
#
#     :param image: Input 3D image
#     :param seg: Boolean, False for normal image, True for segmentation label
#     :param target_size: Desired output size (100, 100, 100)
#     :return: Downsampled and normalized image
#     """
#     current_size = image.shape
#
#     # Compute scaling factors for each dimension
#     zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
#
#     # Downsample using scipy.ndimage.zoom
#     downsampled_image = zoom(image, zoom_factors, order=1)  # Use order=1 for bilinear interpolation
#
#     # If it's a normal image (not segmentation), apply z-score normalization
#     if not seg:
#         mean = np.mean(downsampled_image)
#         std = np.std(downsampled_image)
#         std = std if std > 0 else 1  # Prevent divide by zero
#         downsampled_image = (downsampled_image - mean) / std
#
#     return downsampled_image
#
#
# def extract_surface_points(voxel_data, threshold=0.5, num_points=NUM_POINTS):
#     """
#     Extracts mesh representation from labeled voxel data.
#
#     :param voxel_data: labeled volume (batch_size, x, y, z)
#     :param threshold: voxels are 0 or 1, so pick 0.5 for this
#     :param num_points: number of points to extract (should match template mesh size)
#     :return: resampled surface points (batch_size, num_points, 3)
#     """
#     surface_points = []
#
#     for i in range(voxel_data.shape[0]):
#         volume = voxel_data[i]
#         surface_voxels = volume > threshold
#
#         # Extract surface using marching cubes
#         vertices, faces, _, _ = marching_cubes(surface_voxels, level=threshold)
#
#         # Sample points using nearest neighbors (ensuring same # of pts as template mesh)
#         nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(vertices)
#         bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)
#         random_points = np.random.uniform(bbox_min, bbox_max, (num_points, 3))
#         _, indices = nbrs.kneighbors(random_points)
#         resampled_points = vertices[indices.flatten()]
#
#         # Z-score normalize
#         resampled_points_mean = resampled_points.mean(axis=0)
#         resampled_points_std = resampled_points.std(axis=0)
#         if np.any(resampled_points_std == 0):
#             resampled_points_std[resampled_points_std == 0] = 1.0  # Prevent division by zero
#
#         resampled_points = (resampled_points - resampled_points_mean) / resampled_points_std
#         resampled_points *= np.array(SPACE_DIRECTIONS).diagonal()  # Adjust based on voxel spacing
#
#         surface_points.append(resampled_points)
#
#     return np.array(surface_points)