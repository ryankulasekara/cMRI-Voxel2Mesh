# This file contains file paths & global variables

# File paths
TRAIN_IMAGES = r"data\train\images"
TEST_IMAGES = r"data\test\images"
TRAIN_LABELS = r"data\train\labels"
TEMPLATE_MESH = "spheres\icosahedron_2562.obj"

# Global variables
NRRD_DIMENSIONS = (100, 100, 28)  # All .nrrd images will be resized to 100x100x28
SEG_LABEL = 4   # '4' corresponds to left ventricle in our .seg.nrrd files
NUM_POINTS = 2562
SPACE_DIRECTIONS = [(-1.9230799999999997,-0,0), (-0,-1.9230799999999997,-0), (0,-0,5.0000273333706708)]

class Config:
    ndims = 3
    batch_size = 8
    num_input_channels = 1
    first_layer_channels = 16
    steps = 4
    num_classes = 2
    graph_conv_layer_count = 3
    batch_norm = True

config = Config()
