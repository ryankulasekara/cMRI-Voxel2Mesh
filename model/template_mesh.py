import torch
import trimesh

from config import TEMPLATE_MESH


class TemplateMesh:
    def __init__(self, obj_file=TEMPLATE_MESH):
        self.mesh = trimesh.load(obj_file, process=False)

        self.vertices = torch.tensor(self.mesh.vertices, dtype=torch.float32) # (num points, 3)
        vertices_mean = self.vertices.mean(dim=0)
        vertices_std = self.vertices.std(dim=0)
        self.vertices = (self.vertices - vertices_mean) / vertices_std

        self.faces = torch.tensor(self.mesh.faces, dtype=torch.long)
        self.faces = self.faces.unique(dim=0)

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces
