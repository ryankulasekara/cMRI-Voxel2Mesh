import torch
import torch.nn as nn

from model.voxel_encoder import VoxelEncoder
from model.voxel_decoder import VoxelDecoder

def model(input_data):
    # ensure cuda status
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # send the input data to gpu
    input_data = input_data.to(device)

    # get the voxel encoder module
    voxel_encoder = VoxelEncoder().to(device)

    # get the voxel decoder module
    voxel_decoder = VoxelDecoder().to(device)

    # pass input data through the encoder module
    encoder_output, skip_connections = voxel_encoder(input_data)

    # pass encoder output and skip connections through the voxel decoder module
    decoder_output = voxel_decoder(encoder_output, skip_connections)
    print(f"Output shape: {decoder_output.shape}")

