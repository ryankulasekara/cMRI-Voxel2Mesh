import torch
import trimesh
from sklearn import preprocessing

from config import TEMPLATE_MESH
from model.mesh_utils import normalize_points


class TemplateMesh:
    def __init__(self, obj_file=TEMPLATE_MESH):
        self.mesh = trimesh.load(obj_file, process=False)

        self.vertices = torch.tensor(self.mesh.vertices, dtype=torch.float32) / 2 # (num points, 3)
        vertices_mean = self.vertices.mean(dim=0)
        vertices_std = self.vertices.std(dim=0)
        # min_v = self.vertices.min(dim=0, keepdim=True)[0]
        # max_v = self.vertices.max(dim=0, keepdim=True)[0]
        # self.vertices = 2 * (self.vertices - min_v) / (max_v - min_v + 1e-6) - 1


        self.faces = torch.tensor(self.mesh.faces, dtype=torch.long)
        self.faces = self.faces.unique(dim=0)

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces
