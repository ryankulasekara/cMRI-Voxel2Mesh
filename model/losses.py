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


    pos_weight = torch.tensor([10.0], device=DEVICE)
    # # class weights
    # class_weights = torch.tensor([20.0, 20.0], device=DEVICE)
    #
    # return F.cross_entropy(pred_voxels, target, reduction="mean", weight=class_weights)

    ce_loss = 0
    for i in range(config.num_classes):
        ce_loss += F.binary_cross_entropy_with_logits(pred_voxels[:, i], target[:, i], pos_weight=pos_weight)

    return ce_loss


# def normal_consistency_loss(vertices, faces):
#     """
#     Compute normal consistency loss for a mesh.
#
#     Args:
#         vertices: (V, 3) tensor of vertex positions
#         faces: (F, 3) tensor of face indices
#
#     Returns:
#         loss: scalar tensor representing normal consistency
#     """
#     vertices = vertices.squeeze(0).to(DEVICE)
#     faces = faces.to(DEVICE)
#
#     # Compute face normals
#     v0 = vertices[faces[:, 0]]  # (F, 3)
#     v1 = vertices[faces[:, 1]]  # (F, 3)
#     v2 = vertices[faces[:, 2]]  # (F, 3)
#
#     face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
#     face_normals = torch.nn.functional.normalize(face_normals, dim=1)  # Normalize
#
#     # Build edge-to-face mapping
#     edges = torch.cat([
#         faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
#     ], dim=0)  # (3F, 2)
#
#     # Sort edges to make them canonical (lower index first)
#     edges_sorted, _ = torch.sort(edges, dim=1)
#
#     # Find unique edges and their face indices
#     unique_edges, inverse_indices, counts = torch.unique(
#         edges_sorted, dim=0, return_inverse=True, return_counts=True
#     )
#
#     # We only care about edges shared by exactly 2 faces (manifold edges)
#     manifold_mask = counts == 2
#
#     if not manifold_mask.any():
#         return torch.tensor(0.0, device=DEVICE)
#
#     manifold_edges = unique_edges[manifold_mask]
#
#     with torch.no_grad():
#         # For each manifold edge, find the two faces that share it
#         edge_to_faces = {}
#         for i, edge in enumerate(edges_sorted):
#             face_idx = i // 3  # Each face contributes 3 edges
#             edge_tuple = tuple(edge.tolist())
#             if edge_tuple not in edge_to_faces:
#                 edge_to_faces[edge_tuple] = []
#             edge_to_faces[edge_tuple].append(face_idx)
#
#     # Compute consistency for each manifold edge
#     losses = []
#     for edge in manifold_edges:
#         edge_tuple = tuple(edge.tolist())
#         face_indices = edge_to_faces[edge_tuple]
#
#         if len(face_indices) == 2:
#             n1 = face_normals[face_indices[0]]
#             n2 = face_normals[face_indices[1]]
#
#             # Normal consistency: dot product should be close to 1
#             # We use 1 - dot to make it a loss (0 when normals are aligned)
#             dot_product = torch.dot(n1, n2)
#             loss = 1 - dot_product
#             losses.append(loss)
#
#     if not losses:
#         return torch.tensor(0.0, device=DEVICE)
#
#     return torch.stack(losses).mean()

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

