import torch

from torch import nn
from torchvision import models

class ClassifyNet(nn.Module):

    def __init__(self, num_classes: int = 10, pretrain: bool = False) -> None:
        super().__init__()
        self.model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT if pretrain else None)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
    
if __name__ == "__main__":
    torch.save(ClassifyNet().state_dict(), "./weights/ClassifyNet-Pretrain.pt")
