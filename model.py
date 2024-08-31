import torch
import torch.nn as nn
import torchvision.models as models


class ClassifyNet(nn.Module):
    def __init__(self, num_classes=10, pretrain=False):
        super().__init__()

        if pretrain:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.backbone = models.resnet18()

        self.backbone.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, inputs):
        return self.backbone(inputs)


if __name__ == '__main__':
    torch.save(ClassifyNet(pretrain=True).state_dict(), 'weights/develop/pretrain.pt')
