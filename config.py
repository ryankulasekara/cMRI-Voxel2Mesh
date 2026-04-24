# This file contains file paths & global variables
import torch

# File paths
TRAIN_IMAGES = r"data\train\images"
TEST_IMAGES = r"data\test\images"
TRAIN_LABELS = r"data\train\labels"
TEST_LABELS = r"data\test\labels"
TEMPLATE_MESH = "spheres\icosahedron_2562.obj"

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NRRD_DIMENSIONS = (96, 96, 32, 7)  # size to resize images & labels to
NUM_POINTS = 5134
SPACE_DIRECTIONS = [(1.9230799999999997,-0,0), (-0,1.9230799999999997,-0), (0,-0,9.99985)]

class Config:
    num_classes = 7
    num_mesh_classes = 6
    fat_class_index = 6
    ndims = 3
    batch_size = 1
    num_input_channels = 1
    first_layer_channels = 16
    steps = 4
    graph_conv_layer_count = 3
    batch_norm = True
    voxel_feature_dim = 16

config = Config()
