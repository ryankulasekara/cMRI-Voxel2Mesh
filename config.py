# This file contains file paths & global variables

# File paths
TRAIN_IMAGES = r"data\train\images"
TRAIN_LABELS = r"data\train\labels"
TEMPLATE_MESH = r"spheres\icosahedron_2562.obj"

# Global variables
NRRD_DIMENSIONS = (100, 100, 22)  # All .nrrd images will be resized to 64x64x22
SEG_LABEL = 4   # '4' corresponds to left ventricle in our .seg.nrrd files
