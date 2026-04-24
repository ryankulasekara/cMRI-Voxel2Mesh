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
    Computes Chamfer Distance between predicted and target meshes

    :param pred_points: model's mesh output
    :param target_points: target mesh (marching cubes of segmentation output)
    """

    # compute distances from each point in predicted pts to target pts
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
    # move to gpu
    faces = faces.to(vertices.device)

    # get vertices corresponding to each face
    face_vertices = vertices[:, faces, :]

    # calculate edge differences from each vertex
    edge1 = face_vertices[:, :, 0, :] - face_vertices[:, :, 1, :]
    edge2 = face_vertices[:, :, 1, :] - face_vertices[:, :, 2, :]
    edge3 = face_vertices[:, :, 2, :] - face_vertices[:, :, 0, :]

    # euclidean distance for edge lengths
    edge_len_sq = (edge1 ** 2).sum(-1) + (edge2 ** 2).sum(-1) + (edge3 ** 2).sum(-1)
    loss = torch.mean(torch.sqrt(edge_len_sq / 3.0 + 1e-12))

    return loss


def laplacian_smoothing(vertices, faces):
    """
    Vectorized Laplacian smoothing using adjacency matrix for better speed

    :param vertices: (B, N, 3)
    :param faces: (F, 3)
    """
    B, N, _ = vertices.shape
    device = vertices.device

    # make adjacency & degree matrices
    with torch.no_grad():
        # make edge list from faces
        edges = torch.cat([
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
        ], dim=0)

        # get rid of duplicates
        edges = edges[edges[:, 0] != edges[:, 1]]
        edges = torch.unique(edges, dim=0)

        # make adjacency matrix
        row = torch.cat([edges[:, 0], edges[:, 1]])
        col = torch.cat([edges[:, 1], edges[:, 0]])

        # make degree matrix (# of neighbors for each vertex)
        degree = torch.zeros(N, device=device)
        degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))

        # make sure not to divide by zero... was having issues w/ instability
        degree = torch.clamp(degree, min=1.0)

    # average distance from centroid of neighbors
    neighbor_sum = torch.zeros(B, N, 3, device=device)
    neighbor_sum.index_add_(1, row, vertices[:, col])
    neighbor_means = neighbor_sum / degree.unsqueeze(0).unsqueeze(-1)
    loss = torch.norm(vertices - neighbor_means, dim=-1)

    # average over all vertices
    return loss.mean()


def cross_entropy_loss(pred_voxels, target_voxels):
    # ensure target is float
    target = target_voxels.float().to(DEVICE)
    pred_voxels = pred_voxels.to(DEVICE)

    ce_loss = 0
    for c in range(config.num_classes):
        if c == 0:
            pos_weight = torch.tensor([1.8], device=DEVICE)
        elif c == 2:
            pos_weight = torch.tensor([3.0], device=DEVICE)
        elif c == 3:
            pos_weight = torch.tensor([4.0], device=DEVICE)
        elif c == 4:
            pos_weight = torch.tensor([4.0], device=DEVICE)
        elif c == 5:
            pos_weight = torch.tensor([2.25], device=DEVICE)
        elif c == 6:
            pos_weight = torch.tensor([2.0], device=DEVICE)
        else:
            pos_weight = torch.tensor([2.0], device=DEVICE)
        # pos_weight = torch.tensor([1.0], device=DEVICE)
        ce_loss += F.binary_cross_entropy_with_logits(pred_voxels[:, c], target[:, c], pos_weight=pos_weight)

    return ce_loss


def dice_loss(pred, target, smooth=1e-6):
    """
    Use DSC for loss w/ cross-entropy... trying this to make model better for upper chambers
    """
    pred = pred > 0.0

    pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.view(target.shape[0], target.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean(dim=1)


def normal_consistency_loss(vertices, faces):
    """
    Normals from each face, compare angles
    """
    vertices = vertices.to(DEVICE).squeeze(0)
    faces = faces.to(DEVICE)

    # compute normals from each face
    v0, v1, v2 = vertices[faces].unbind(1)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1, eps=1e-8)

    # Create edges and find adjacent faces
    edges = torch.stack([
        torch.stack([faces[:, 0], faces[:, 1]], dim=1),
        torch.stack([faces[:, 1], faces[:, 2]], dim=1),
        torch.stack([faces[:, 2], faces[:, 0]], dim=1)
    ], dim=0)

    edges = edges.reshape(-1, 2)
    edges_sorted, _ = torch.sort(edges, dim=1)

    # sort edges, get rid of duplicates
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

    # find face pairs
    manifold_edge_indices = torch.where(manifold_mask)[0]

    # get indices for faces
    edge_occurrences = []
    for edge_idx in manifold_edge_indices:
        occ = torch.where(inverse_idx == edge_idx)[0]
        edge_occurrences.append(occ)

    if not edge_occurrences:
        return torch.tensor(0.0, device=DEVICE)

    # convert face indices to edge indices
    edge_occurrences = torch.stack(edge_occurrences)
    face_pairs = edge_occurrences // 3

    # compute loss
    n1 = face_normals[face_pairs[:, 0]]
    n2 = face_normals[face_pairs[:, 1]]
    dots = (n1 * n2).sum(dim=1)
    losses = 1.0 - torch.clamp(dots, -1.0, 1.0)

    return losses.mean()

