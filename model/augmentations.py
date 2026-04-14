import torch
import torch.nn.functional as F
import numpy as np
import random
import math

from config import *

class CardiacAugmentations:
    def __init__(self, apply_augmentation_prob=1.0):
        self.prob = apply_augmentation_prob
        self.max_rotation_angle = 25.0

    def __call__(self, images, labels):
        """
        Apply augmentations to both images and labels... have to make sure dimensions are lined up
        This is super annoying but there isn't really a good way to have them lined up the whole way thru the
        pipeline... just gotta pay attention & permute when necessary

        images: [B, 1, 96, 96, 32]
        labels: [B, 96, 96, 32, 7]
        """

        # basically using an RNG to decide whether to do augmentation or not
        if random.random() > self.prob:
            return images, labels

        batch_size = images.shape[0]
        augmented_images = []
        augmented_labels = []

        for i in range(batch_size):
            img = images[i]  #
            lbl = labels[i].permute(1, 2, 3, 0)

            img, lbl = self.apply_random_augmentation(img, lbl)
            augmented_images.append(img)
            augmented_labels.append(lbl)

        images_aug = torch.stack(augmented_images)
        labels_aug = torch.stack(augmented_labels)

        return images_aug, labels_aug.permute(0, 4, 1, 2, 3)

    def apply_random_augmentation(self, image, label):
        """
        Apply a random augmentation to an img & corresponding label volume

        :param image: img volume
        :param label: label volume
        """

        # choose a type randomly
        aug_type = random.choice([
            'rotate', 'brightness'
        ])

        if aug_type == 'rotate':
            return self.rotate_augmentation(image, label)
        elif aug_type == 'brightness':
            return self.brightness_augmentation(image, label)

        return image, label

    def rotate_augmentation(self, image, label):
        """
        Rotation between -(max angle) & +(max angle)

        :param image: img volume
        :param label: label volume
        """

        # random angle between -max_rotation_angle and +max_rotation_angle
        angle_degrees = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)

        # convert to radians
        angle_rad = angle_degrees * math.pi / 180.0
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # create rotation matrix for 2D
        # we want to rotate in xy plane, not z plane
        theta = torch.tensor([[
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ]], dtype=torch.float32)

        # rotate image & label... kinda just have to do this for each slice
        image_rot = self.rotate_volume_2d(image, theta)
        label_rot = self.rotate_label_2d(label, theta)

        return image_rot, label_rot

    def rotate_volume_2d(self, volume, theta):
        """
        2D rotation for each slice in the img volume

        :param volume: img volume
        :param theta: angle to rotate (in radians)
        """

        # make sure it's on the GPU... can be a little computationally intense
        volume = volume.to(DEVICE)
        C, H, W, D = volume.shape
        rotated_slices = []

        for d in range(D):
            # get a slice
            slice_vol = volume[..., d]
            slice_vol = slice_vol.unsqueeze(0)

            # create sampling grid... need to resample bc of rotation
            grid = F.affine_grid(theta, slice_vol.size(), align_corners=False).to(DEVICE)

            # apply rotation
            slice_rot = F.grid_sample(slice_vol.float(), grid, mode='bilinear',
                                      align_corners=False, padding_mode='border')

            rotated_slices.append(slice_rot.squeeze(0))

        # stack slices back together
        return torch.stack(rotated_slices, dim=-1)

    def rotate_label_2d(self, label, theta):
        """
        2D rotation for each slice in label volume

        :param label: label volume
        :param theta: angle to rotate (in radians)
        """

        H, W, D, C = label.shape
        rotated_slices = []

        for d in range(D):
            # get single slice for all chambers' labels
            slice_label = label[:, :, d, :]
            slice_label = slice_label.permute(2, 0, 1).unsqueeze(0)

            # create sampling grid... need to resample bc of rotation
            grid = F.affine_grid(theta, slice_label.size(), align_corners=False).to(DEVICE)

            # apply rotation... nearest neighbor interpolation for labels
            slice_rot = F.grid_sample(slice_label.float(), grid, mode='nearest',
                                      align_corners=False, padding_mode='border')

            slice_rot = slice_rot.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            rotated_slices.append(slice_rot)

        # stack slices back together
        return torch.stack(rotated_slices, dim=2)

    def brightness_augmentation(self, image, label):
        """
        Random brightness adjustment

        :param image: img volume
        :param label: label volume... don't need to do anything w this one
        """

        # scale factor btwn 0.8 and 1.2
        factor = random.uniform(0.8, 1.2)
        image_aug = image * factor
        image_aug = torch.clamp(image_aug, -3.0, 3.0)

        return image_aug, label