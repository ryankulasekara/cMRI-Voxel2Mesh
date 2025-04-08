# This file contains file paths & global variables

# File paths
# TRAIN_IMAGES = r"C:\Users\rkulase\Documents\MMWHS MRI\mr_train\images"
TRAIN_IMAGES = r"data\train\images"
TEST_IMAGES = r"data\test\images"
# TRAIN_LABELS = r"C:\Users\rkulase\Documents\MMWHS MRI\mr_train\labels"
TRAIN_LABELS = r"data\train\labels"
TEMPLATE_MESH = "spheres\icosahedron_642.obj"

# Global variables
NRRD_DIMENSIONS = (96, 96, 32)  # size to resize images & labels to
SEG_LABEL = 3   # '3' corresponds to left ventricle in our .seg.nrrd files
NUM_POINTS = 3000
SPACE_DIRECTIONS = [(-1.9230799999999997,-0,0), (-0,-1.9230799999999997,-0), (0,-0,5.0000273333706708)]

class Config:
    ndims = 3
    batch_size = 1
    num_input_channels = 1
    first_layer_channels = 16
    steps = 4
    num_classes = 2
    graph_conv_layer_count = 3
    batch_norm = True
    voxel_feature_dim = 16

config = Config()
