import torch
import toml
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import FlowerNet


configs = toml.load('configs/config.toml')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = ImageFolder('datasets/test', transform=transform)
dataset_size = len(dataset)

dataloader = DataLoader(dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=False)
dataloader_size = len(dataloader)

device = torch.device(configs['device'])

model = FlowerNet(num_classes=configs['num-classes'], pretrained=False)
model = model.to(device)

log_interval = configs['log-interval']

print(f'\n---------- evaluation start at: {device} ----------\n')

with torch.no_grad():
    top1_accuracy = 0.0
    top2_accuracy = 0.0
    top3_accuracy = 0.0

    model.load_state_dict(torch.load(configs['load-checkpoint-path'], map_location=device, weights_only=True))
    model.eval()

    for batch, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, top1_indices = torch.topk(outputs, 1, dim=1)
        _, top2_indices = torch.topk(outputs, 2, dim=1)
        _, top3_indices = torch.topk(outputs, 3, dim=1)

        labels = labels.view(-1, 1)

        top1_accuracy += (top1_indices == labels).sum().item()
        top2_accuracy += (top2_indices == labels).sum().item()
        top3_accuracy += (top3_indices == labels).sum().item()

        if batch % log_interval == 0:
            print(f'[valid] [{batch:04d}/{dataloader_size:04d}]')

    top1_accuracy /= dataset_size
    top2_accuracy /= dataset_size
    top3_accuracy /= dataset_size

print('\n--------------------------------------')
print(f'top1 accuracy: {top1_accuracy:.3f}')
print(f'top2 accuracy: {top2_accuracy:.3f}')
print(f'top3 accuracy: {top3_accuracy:.3f}')

print('\n---------- evaluation finished ----------\n')
