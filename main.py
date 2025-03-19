import torch

from config import *
from data import load_images, load_labels
from model.model import model

# load training images & labels
train_images = load_images(TRAIN_IMAGES)
train_labels = load_labels(TRAIN_LABELS)

# convert train images to tensor
train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
train_images_tensor = train_images_tensor.unsqueeze(1)

# we need to be in the format: [batch size, channels (1), depth (22), height (100), width (100)]
train_images_tensor = train_images_tensor.permute(0, 1, 3, 2, 4)

# train the model
model = model(train_images_tensor)

x = 1