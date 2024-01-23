import torch

from torch import Tensor
from torch import nn
from torchvision import models

class ClassifyNet(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.model(input_tensor)
    
if __name__ == "__main__":
    torch.save(ClassifyNet(), "./weights/ClassifyNet-Pretrain.pt")
