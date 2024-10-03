import torch.nn as nn
import torchvision.models as models


class ClassifyNet(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()

        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.backbone = models.resnet18()

        self.backbone.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, inputs):
        return self.backbone(inputs)
