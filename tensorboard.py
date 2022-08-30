import torch
import torchvision
import torch.utils.tensorboard

writer = tensorboard.SummaryWriter()

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()