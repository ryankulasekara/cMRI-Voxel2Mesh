import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import *
from config import config
from data import load_images, load_labels, extract_surface_points
from model.model import Voxel2Mesh


# enable cuda to use gpu
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
print("Loading training images...")
train_images = load_images(TRAIN_IMAGES)
print("Loading training labels...")
train_labels = load_labels(TRAIN_LABELS)

# split into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# convert to tensor and move to gpu
train_images_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2, 4)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
val_images_tensor = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2, 4)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

print("Extracting surface points from training labels...")
with torch.no_grad():
    train_surface_points = extract_surface_points(train_labels)
    val_surface_points = extract_surface_points(val_labels)

train_surface_points_tensor = torch.tensor(train_surface_points, dtype=torch.float32)
val_surface_points_tensor = torch.tensor(val_surface_points, dtype=torch.float32)

# create TensorDatasets
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor, train_surface_points_tensor)
val_dataset = TensorDataset(val_images_tensor, val_labels_tensor, val_surface_points_tensor)

# create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# initialize data to pass into model
data = {
    'x': train_images_tensor,
    'y_voxels': train_labels_tensor,
    'surface_points': train_surface_points_tensor
}

# initialize model
model = Voxel2Mesh(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# training loop
num_epochs = 150
print("Training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    # train w/ training DataLoader
    for images, labels, surface_points in train_loader:
        images, labels, surface_points = images.to(device), labels.to(device), surface_points.to(device)

        optimizer.zero_grad()
        loss, log = model.loss({'x': images, 'y_voxels': labels, 'surface_points': surface_points})
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # validation w/ validation DataLoader
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, surface_points in val_loader:
            images, labels, surface_points = images.to(device), labels.to(device), surface_points.to(device)
            loss, _ = model.loss({'x': images, 'y_voxels': labels, 'surface_points': surface_points})
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


torch.save(model.state_dict(), "voxel2mesh_model.pth")
print("Model saved successfully.")
