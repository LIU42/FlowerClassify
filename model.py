import torch

from torch import nn
from torchvision import models

model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
model.classifier[6] = nn.Linear(4096, 10)

torch.save(model, "./weights/AlexNet-Pretrain.pt")
