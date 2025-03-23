import torch
import trimesh
import pyvista as pv
import numpy as np

from model.model import Voxel2Mesh
from data import load_images
from model.template_mesh import TemplateMesh
from config import *

# make sure to use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# initialize template mesh
template = TemplateMesh()
template_faces = template.get_faces().cpu().numpy()

# make sure faces are readable by pyvista
faces_pyvista = []
for face in template_faces:
    faces_pyvista.append([3, *face])
faces_pyvista = np.array(faces_pyvista).flatten()

# load trained model
model = Voxel2Mesh(config).to(device)
model.load_state_dict(torch.load("voxel2mesh_model.pth", map_location=device))
model.eval()

# load test images
print("Loading test images...")
test_images = load_images(TEST_IMAGES)
test_images_tensor = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2, 4).to(device)
voxel_spacing = np.array(SPACE_DIRECTIONS).diagonal()

# run model on test images & visualize mesh
print("Running model on test images...")
with torch.no_grad():
    for i in range(len(test_images)):
        input_data = {'x': test_images_tensor[i].unsqueeze(0)}
        predicted_vertices = model.forward(input_data)
        predicted_vertices = predicted_vertices.cpu().numpy().squeeze()
        predicted_vertices = predicted_vertices * voxel_spacing
        pv_mesh = pv.PolyData(predicted_vertices, faces_pyvista)
        pv_mesh.plot()

        # convert vertices to trimesh & save as obj file
        mesh = trimesh.Trimesh(vertices=predicted_vertices, faces=template_faces)
        mesh.export(f"test_mesh_{i}.obj")
        print(f"Saved: test_mesh_{i}.obj")

print("Testing completed.")

