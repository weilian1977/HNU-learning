import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

# Download training data from open datasets.
training_data = datasets.MNIST (
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers=0)

print(len(test_loader))
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    break
  