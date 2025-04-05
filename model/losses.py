import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

def chamfer_distance(pred_points, target_points):
    """
    Computes Chamfer Distance between predicted and target point sets.
    """
    batch_size, num_points, _ = pred_points.shape

    pred_expanded = pred_points.unsqueeze(2)  # (B, N, 1, 3)
    target_expanded = target_points.unsqueeze(1)  # (B, 1, M, 3)

    pred_expanded = pred_expanded / (pred_expanded.norm(dim=-1, keepdim=True) + 1e-8)
    target_expanded = target_expanded / (target_expanded.norm(dim=-1, keepdim=True) + 1e-8)

    distances = torch.norm(pred_expanded - target_expanded, dim=-1)  # (B, N, M)

    dist = torch.cdist(pred_expanded, target_expanded, p=2) + 1e-8

    return (dist.min(2)[0].mean() + dist.min(1)[0].mean()) / 2

def mesh_edge_loss(vertices, faces):
    """
    Computes edge length loss to encourage smoothness.
    """
    v0 = vertices[:, faces[:, 0], :]
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]

    edge1 = torch.norm(v0 - v1, dim=-1)
    edge2 = torch.norm(v1 - v2, dim=-1)
    edge3 = torch.norm(v2 - v0, dim=-1)

    return (edge1.mean() + edge2.mean() + edge3.mean()) / 3

def laplacian_smoothing(vertices, faces):
    """
    Laplacian smoothing loss to encourage smooth surfaces.
    """
    laplacian_loss = 0
    num_vertices = vertices.shape[1]

    # iterate over all vertices
    for i in range(num_vertices):
        neighbors = set()  # Use a set to avoid duplicate neighbors

        # iterate over all faces and find neighbors of vertex `i`
        for j in range(faces.shape[1]):  # iterate over the three vertices in each face
            # check if vertex i is part of the current face
            if i in faces[:, j]:
                # add the other two vertices of the face as neighbors
                for k in range(faces.shape[1]):
                    if faces[:, j][k] != i:
                        neighbors.add(faces[:, j][k])

        # compute the Laplacian loss for vertex
        if len(neighbors) > 0:
            neighbor_vertices = vertices[:, list(neighbors), :]
            # calculate the average position of neighbors
            laplacian_loss += torch.norm(vertices[:, i, :] - neighbor_vertices.mean(dim=1))

    # normalize by the number of vertices
    return laplacian_loss / num_vertices


def cross_entropy_loss(pred_voxels, target_voxels):
    smooth = 0.1
    target = target_voxels * (1 - smooth) + 0.5 * smooth
    return F.binary_cross_entropy_with_logits(
        pred_voxels,
        target,
        reduction='mean'
    )

    # return loss

