import torch
import torch.optim as optim

from config import *
from config import config
from data import load_images, load_labels, extract_surface_points
from model.model import Voxel2Mesh


# enable cuda to use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load data
print("Loading training images...")
train_images = load_images(TRAIN_IMAGES)
print("Loading training labels...")
train_labels = load_labels(TRAIN_LABELS)

# convert to tensor and move to gpu
train_images_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1).to(device)
train_images_tensor = train_images_tensor.permute(0, 1, 3, 2, 4)

train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)

print("Extracting surface points from training labels...")
with torch.no_grad():
    surface_points = extract_surface_points(train_labels)
surface_points_tensor = torch.tensor(surface_points, dtype=torch.float32).to(device)

# initialize data to pass into model
data = {
    'x': train_images_tensor,
    'y_voxels': train_labels_tensor,
    'surface_points': surface_points_tensor
}

# initialize model
model = Voxel2Mesh(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
num_epochs = 200
print("Training...")
for epoch in range(num_epochs):
    optimizer.zero_grad()

    loss, log = model.loss(data)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {log['loss']:.4f}, Chamfer = {log['chamfer_loss']:.4f}")

x = 1