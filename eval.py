import torch
import yaml

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import FlowerNet


with open('configs/eval.yaml', 'r') as configs:
    configs = yaml.load(configs, Loader=yaml.FullLoader)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder('datasets/test', transform=transform)
dataset_size = len(dataset)

dataloader = data.DataLoader(dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'])
dataloader_size = len(dataloader)

device = torch.device(configs['device'])

model = FlowerNet(num_classes=configs['num-classes'])
model = model.to(device)

print(f'\n---------- Evaluation Start At: {str(device).upper()} ----------\n')

with torch.no_grad():
    top1_accuracy = 0.0
    top2_accuracy = 0.0
    top3_accuracy = 0.0

    model.load_state_dict(torch.load(configs['checkpoint-path'], map_location=device, weights_only=True))
    model.eval()

    for index, (inputs, labels) in enumerate(dataloader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        _, top1_indices = torch.topk(outputs, 1, dim=1)
        _, top2_indices = torch.topk(outputs, 2, dim=1)
        _, top3_indices = torch.topk(outputs, 3, dim=1)

        labels = labels.view(-1, 1)

        top1_accuracy += (top1_indices == labels).sum().item()
        top2_accuracy += (top2_indices == labels).sum().item()
        top3_accuracy += (top3_indices == labels).sum().item()

        print(f'\rProgress: [{index}/{dataloader_size}]', end='')

    top1_accuracy /= dataset_size
    top2_accuracy /= dataset_size
    top3_accuracy /= dataset_size

print('\n\n---------- Evaluation Finish ----------\n')

print(f'Top1 Accuracy: {top1_accuracy:.3f}')
print(f'Top2 Accuracy: {top2_accuracy:.3f}')
print(f'Top3 Accuracy: {top3_accuracy:.3f}')
