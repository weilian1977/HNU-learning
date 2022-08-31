import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

# Download training data from open datasets.
training_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=0)

