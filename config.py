# This file contains file paths & global variables
import torch

# File paths
TRAIN_IMAGES = r"data\train\images"
TEST_IMAGES = r"data\test\images"
TRAIN_LABELS = r"data\train\labels"
TEST_LABELS = r"data\test\labels"
TEMPLATE_MESH = "spheres\icosahedron_10242.obj"

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NRRD_DIMENSIONS = (96, 96, 32)  # size to resize images & labels to
SEG_LABEL = 3  # '3' corresponds to left ventricle in our .seg.nrrd files
NUM_POINTS = 10242
# SPACE_DIRECTIONS = [(-1.9230799999999997,-0,0), (-0,-1.9230799999999997,-0), (0,-0,5.0000273333706708)]
SPACE_DIRECTIONS = [(1.9230799999999997,-0,0), (-0,1.9230799999999997,-0), (0,-0,9.99985)]

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
