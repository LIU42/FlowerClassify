import torch

from torch import nn
from torchvision import models

def load_pretrain_alexnet(classes_count) -> None:

    model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, classes_count)

    torch.save(model, "./weights/AlexNet-Pretrain.pt")

if __name__ == "__main__":

    load_pretrain_alexnet(classes_count = 10)
