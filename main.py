import torch
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

from config import *
from data import load_images, load_labels, extract_surface_points
from model.model import Voxel2Mesh, visualize_meshes
from model.template_mesh import TemplateMesh
from model.augmentations import CardiacAugmentations

# enable cuda to use gpu
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
print("\nLoading training images...")
train_images, train_image_headers = load_images(TRAIN_IMAGES)
print("Loading training labels...")
train_labels = load_labels(TRAIN_LABELS, train_image_headers)

# split into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.3, random_state=42)

# convert to tensor and move to gpu, permute to get dimensions in the right spots
train_images_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2, 4)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).permute(0,2,3,1,4)
val_images_tensor = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2, 4)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).permute(0,2,3,1,4)

# create TensorDatasets to pass into DataLoaders
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)

# create DataLoaders to use w/ model
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# initialize model, optimizer, & scheduler
model = Voxel2Mesh(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=100)
# scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# training loop
train_losses = []
val_losses = []
num_epochs = 400
start_time = time.time()
augmenter = CardiacAugmentations(apply_augmentation_prob=0.0)
print("\nTraining...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # train w/ training DataLoader
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # images, labels = augmenter(images, labels)

        optimizer.zero_grad()
        # was running out of memory on gpu so doing this w/ autocast
        with torch.cuda.amp.autocast():
            loss, log = model.loss({'x': images, 'y_voxels': labels})

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # validation w/ validation DataLoader
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # was running out of memory on gpu so doing this w/ autocast
            with torch.cuda.amp.autocast():
                loss, _ = model.loss({'x': images, 'y_voxels': labels})
                val_loss += loss.item()


    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}: Training Loss: {train_loss:.4f}, Chamfer: {log['chamfer_loss']:.4f}, "
          f"Cross-Entropy: {log['ce_loss']:.4f}, Dice: {log['dice_loss']:.4f}, Edge: {log['edge_loss']:.4f}, "
          f"Laplacian: {log['laplacian_loss']:.4f}, Normal: {log['normal_loss']:.4f}, Validation Loss: {val_loss:.4f}")

# time calculation
end_time = time.time()
training_time = end_time - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)
seconds = int(training_time % 60)
print(f"\nTotal training time: {hours}h {minutes}m {seconds}s")
print(f"Average time per epoch: {training_time/num_epochs:.2f} seconds")

# save model
torch.save(model.state_dict(), "voxel2mesh_model.pth")
print("\nModel saved successfully.")

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

# visualize mesh outputs
template_mesh = TemplateMesh()
faces = template_mesh.get_faces().cpu().numpy()
faces_pyvista = []
for face in faces:
    faces_pyvista.append([3, *face])
faces_pyvista = np.array(faces_pyvista).flatten()
pred = model({'x': images, 'y_voxels': labels})
visualize_meshes(pred['meshes'], faces_pyvista)
