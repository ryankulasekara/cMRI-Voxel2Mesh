import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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
train_labels = load_labels(TRAIN_LABELS, train_images)

# split into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.3, random_state=42)

# convert to tensor and move to gpu
train_images_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2, 4)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).squeeze(1)  # Remove channel dim if exists
val_images_tensor = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2, 4)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).squeeze(1)

print("Extracting surface points from training labels...")
with torch.no_grad():
    train_surface_points = extract_surface_points(train_labels)
    val_surface_points = extract_surface_points(val_labels)

train_surface_points_tensor = torch.tensor(train_surface_points, dtype=torch.float32)
val_surface_points_tensor = torch.tensor(val_surface_points, dtype=torch.float32)

# create TensorDatasets to pass into DataLoaders
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor, train_surface_points_tensor)
val_dataset = TensorDataset(val_images_tensor, val_labels_tensor, val_surface_points_tensor)

# create DataLoaders to use w/ model
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# initialize data structure to pass into model
data = {
    'x': train_images_tensor,
    'y_voxels': train_labels_tensor,
    'surface_points': train_surface_points_tensor
}

# initialize model
model = Voxel2Mesh(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,
    eta_min=1e-6
)

# training loop
train_losses = []
val_losses = []
num_epochs = 10000
print("Training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # train w/ training DataLoader
    for images, labels, surface_points in train_loader:
        images, labels, surface_points = images.to(device), labels.to(device), surface_points.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss, log = model.loss({'x': images, 'y_voxels': labels, 'surface_points': surface_points})

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # validation w/ validation DataLoader
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, surface_points in val_loader:
            images, labels, surface_points = images.to(device), labels.to(device), surface_points.to(device)
            with torch.cuda.amp.autocast():
                loss, _ = model.loss({'x': images, 'y_voxels': labels, 'surface_points': surface_points})
                val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}: Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "voxel2mesh_model.pth")
print("Model saved successfully.")

# plot loss per epoch
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()
