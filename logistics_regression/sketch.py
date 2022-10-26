from typing import Any, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# Default log_dir argument is "runs" - but it's good to be specific
# torch.utils.tensorboard.SummaryWriter is imported above
writer = SummaryWriter()

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers=0)

x = np.zeros((28*28, 128))
y= np.zeros((1, 128))
w = np.zeros((28*28, 1))
b = 0

step=0
for data in test_loader:
    imgs, targets = data
    if targets[0]==1 or targets[0]==2:
        y[0][step] = targets[0]
        x[:, step] = imgs.numpy()[0, 0].reshape(784)
        step+=1
        if step==128:
            break


# print(x[:, 127], y)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def forward_pop(X: np.ndarray, W: np.ndarray, B: np.ndarray) -> np.ndarray:
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    return A


def backward(X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    m = Y.shape[1]
    dz: np.ndarray = A - Y # 行向量
    dw = np.dot(X, dz.T)
    db = dz
    dw = dw/m
    db = np.sum(db)/m
    return (dw, db)


for i in range(100):
    A = forward_pop(x, w, b)
    dw, db = backward(x, A, y)
    print(db)
    w -= 0.1*dw
    b -= 0.1*db
