import os
import nrrd
import numpy as np
from config import *


def load_labels(file_path):
    """
    Loads all .seg.nrrd files from a folder, extracts desired class label, stores them in a numpy array

    :param file_path: path to folder of .seg.nrrd files
    :return: numpy array of seg labels
    """

    # get all the .seg.nrrd filenames
    seg_nrrd_filenames = [f for f in os.listdir(file_path) if f.endswith(".seg.nrrd")]

    # load the .nrrd files
    labels = []
    for filename in seg_nrrd_filenames:
        # load file
        full_path = os.path.join(file_path, filename)
        seg_data, header = load_nrrd(full_path)

        # preprocess (resize)
        seg_data = preprocess(seg_data, seg=True)

        # add to list of images
        labels.append(seg_data)

    return np.array(labels)


def load_images(file_path):
    """
    Loads all .nrrd files from a folder & stores them in a numpy array.

    :param file_path: path to folder of .nrrd files
    :return: numpy array of images
    """

    # get all the .nrrd filenames
    nrrd_filenames = [f for f in os.listdir(file_path) if f.endswith(".nrrd")]

    # load the .nrrd files
    images = []
    for filename in nrrd_filenames:
        # load file
        full_path = os.path.join(file_path, filename)
        image_data, header = load_nrrd(full_path)

        # preprocess (resize & z-score normalize)
        image_data = preprocess(image_data)

        # add to list of images
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