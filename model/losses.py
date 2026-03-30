import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import gc

from model.mesh_utils import normalize_points
from config import *


def chamfer_distance(pred_points, target_points):
    """
    Computes Chamfer Distance between predicted and target point sets.
    """

    # compute distances from each point in predicted pts to target pts
    # distances = torch.cdist(pred_points.to(DEVICE), target_points.to(DEVICE), p=2)
    distances = torch.cdist(pred_points.to(DEVICE), target_points.to(DEVICE))

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
    Vectorized Laplacian smoothing using adjacency matrix
    vertices: (B, N, 3)
    faces: (F, 3)
    """
    B, N, _ = vertices.shape
    device = vertices.device

    # Build adjacency matrix ONCE (could precompute this)
    with torch.no_grad():
        # Create edge list from faces
        edges = torch.cat([
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
        ], dim=0)

        # Remove duplicates and self-loops
        edges = edges[edges[:, 0] != edges[:, 1]]  # Remove self-loops
        edges = torch.unique(edges, dim=0)  # Remove duplicates

        # Create sparse adjacency matrix
        row = torch.cat([edges[:, 0], edges[:, 1]])
        col = torch.cat([edges[:, 1], edges[:, 0]])

        # Create degree matrix (number of neighbors per vertex)
        degree = torch.zeros(N, device=device)
        degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))

        # Avoid division by zero for isolated vertices
        degree = torch.clamp(degree, min=1.0)

    # Compute neighbor means using sparse operations
    neighbor_sum = torch.zeros(B, N, 3, device=device)

    # Sum contributions from neighbors (both directions due to symmetric edges)
    neighbor_sum.index_add_(1, row, vertices[:, col])

    # Compute means
    neighbor_means = neighbor_sum / degree.unsqueeze(0).unsqueeze(-1)

    # Laplacian loss: vertex - mean_of_neighbors
    loss = torch.norm(vertices - neighbor_means, dim=-1)

    # Average over vertices and batch
    return loss.mean()


def cross_entropy_loss(pred_voxels, target_voxels):
    # ensure target is float
    target = target_voxels.float().to(DEVICE)
    pred_voxels = pred_voxels.to(DEVICE)

    # # class weights
    # class_weights = torch.tensor([20.0, 20.0], device=DEVICE)
    #
    # return F.cross_entropy(pred_voxels, target, reduction="mean", weight=class_weights)

    ce_loss = 0
    for c in range(config.num_classes):
        if c > 1:
            pos_weight = torch.tensor([14.0], device=DEVICE)
        else:
            pos_weight = torch.tensor([1.0], device=DEVICE)
        ce_loss += F.binary_cross_entropy_with_logits(pred_voxels[:, c], target[:, c], pos_weight=pos_weight)

    return ce_loss


def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss for better small structure segmentation
    """
    pred = torch.sigmoid(pred) > 0.5  # Convert logits to probabilities

    # Flatten spatial dimensions
    pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)  # [B, C, H*W*D]
    target_flat = target.view(target.shape[0], target.shape[1], -1)  # [B, C, H*W*D]

    intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)  # [B, C]

    dice = (2. * intersection + smooth) / (union + smooth)  # [B, C]
    return 1 - dice.mean(dim=1)  # [C] - per-class dice loss

def normal_consistency_loss(vertices, faces):
    """
    Ultra-fast version using advanced indexing.
    """
    vertices = vertices.to(DEVICE).squeeze(0)
    faces = faces.to(DEVICE)

    # Compute face normals
    v0, v1, v2 = vertices[faces].unbind(1)  # Each: (F, 3)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1, eps=1e-8)

    # Create edges and find adjacent faces
    edges = torch.stack([
        torch.stack([faces[:, 0], faces[:, 1]], dim=1),
        torch.stack([faces[:, 1], faces[:, 2]], dim=1),
        torch.stack([faces[:, 2], faces[:, 0]], dim=1)
    ], dim=0)  # (3, F, 2)

    edges = edges.reshape(-1, 2)  # (3F, 2)
    edges_sorted, _ = torch.sort(edges, dim=1)

    # Use unique_consecutive for better performance on sorted edges
    edges_sorted, sort_idx = torch.sort(edges_sorted, dim=0)
    unique_edges, inverse_idx, counts = torch.unique(
        edges_sorted,
        dim=0,
        return_inverse=True,
        return_counts=True
    )

    manifold_mask = counts == 2
    if not manifold_mask.any():
        return torch.tensor(0.0, device=DEVICE)

    # Vectorized face pair finding
    manifold_edge_indices = torch.where(manifold_mask)[0]

    # For each manifold edge, get the two face indices
    edge_occurrences = []
    for edge_idx in manifold_edge_indices:
        occ = torch.where(inverse_idx == edge_idx)[0]
        edge_occurrences.append(occ)

    if not edge_occurrences:
        return torch.tensor(0.0, device=DEVICE)

    edge_occurrences = torch.stack(edge_occurrences)  # (E_manifold, 2)
    face_pairs = edge_occurrences // 3  # Convert edge indices to face indices

    # Compute loss vectorized
    n1 = face_normals[face_pairs[:, 0]]
    n2 = face_normals[face_pairs[:, 1]]
    dots = (n1 * n2).sum(dim=1)
    losses = 1.0 - torch.clamp(dots, -1.0, 1.0)

    return losses.mean()

