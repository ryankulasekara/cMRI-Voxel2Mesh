import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.mesh_utils import normalize_points
from config import *


def chamfer_distance(pred_points, target_points):
    """
    Computes Chamfer Distance between predicted and target point sets.
    """

    # compute distances from each point in predicted pts to target pts
    distances = torch.cdist(pred_points.to(DEVICE), target_points.to(DEVICE), p=2)

    # find nearest neighbor distances
    min_distances, _ = torch.min(distances, dim=2)

    # compute average of nearest neighbor distances
    chamfer_loss = min_distances.mean(dim=1)

    return chamfer_loss


def mesh_edge_loss(vertices, faces):
    """
    Computes edge length loss to encourage smoothness.

    vertices: (B, N, 3)
    faces: (F, 3)
    """
    # Move faces to same device as vertices
    faces = faces.to(vertices.device)

    # Gather vertices for each face (B, F, 3, 3)
    face_vertices = vertices[:, faces, :]

    # Compute edge differences
    edge1 = face_vertices[:, :, 0, :] - face_vertices[:, :, 1, :]
    edge2 = face_vertices[:, :, 1, :] - face_vertices[:, :, 2, :]
    edge3 = face_vertices[:, :, 2, :] - face_vertices[:, :, 0, :]

    # Compute squared lengths (avoid sqrt for speed)
    edge_len_sq = (edge1 ** 2).sum(-1) + (edge2 ** 2).sum(-1) + (edge3 ** 2).sum(-1)

    # Take mean of edge lengths (add small epsilon for numerical stability)
    loss = torch.mean(torch.sqrt(edge_len_sq / 3.0 + 1e-12))
    return loss


def laplacian_smoothing(vertices, faces):
    """
    Laplacian smoothness loss:
    Encourages each vertex to be close to the average of its neighbors.

    vertices: (B, N, 3)
    faces: (F, 3)
    """
    B, N, _ = vertices.shape
    device = vertices.device

    vertices = vertices.to(DEVICE)
    faces = faces.to(DEVICE)

    # Build adjacency list once
    adjacency = [[] for _ in range(N)]
    for f in faces:
        adjacency[f[0]].extend([f[1].item(), f[2].item()])
        adjacency[f[1]].extend([f[0].item(), f[2].item()])
        adjacency[f[2]].extend([f[0].item(), f[1].item()])

    loss = 0.0
    for i, neigh in enumerate(adjacency):
        if len(neigh) > 0:
            neigh_idx = torch.tensor(neigh, device=device, dtype=torch.long)
            neighbor_mean = vertices[:, neigh_idx, :].mean(dim=1)  # (B, 3)
            loss += torch.norm(vertices[:, i, :] - neighbor_mean, dim=-1).mean()

    return loss / N


def cross_entropy_loss(pred_voxels, target_voxels):
    # ensure target is float
    target = target_voxels.float().to(DEVICE)
    pred_voxels = pred_voxels.to(DEVICE)

    # weight '1s' higher because of class imbalance (way more 0s than 1s)
    pos_weight = torch.tensor([3.0], device=DEVICE)

    return F.binary_cross_entropy_with_logits(pred_voxels, target, reduction="mean", pos_weight=pos_weight)

