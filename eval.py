import torch
import yaml

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import ClassifyNet


with open('configs/eval.yaml', 'r') as configs:
    configs = yaml.safe_load(configs)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder('datasets/test', transform=transform)
dataset_size = len(dataset)

dataloader = data.DataLoader(dataset, configs['batch-size'], num_workers=configs['num-workers'])
dataloader_size = len(dataloader)

device = torch.device(configs['device'])
accuracy = 0.0

model = ClassifyNet(num_classes=configs['num-classes'], pretrained=False)
model = model.to(device)

print(f'\n---------- Evaluation Start At: {str(device).upper()} ----------\n')

with torch.no_grad():
    model.load_state_dict(torch.load(configs['model-path'], map_location=device, weights_only=True))
    model.eval()

    for index, (inputs, labels) in enumerate(dataloader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

        print(f'\rProgress: [{index}/{dataloader_size}]', end='')

accuracy /= dataset_size

print(f'\nAccuracy: {accuracy:.3f}')
print('\n---------- Evaluation Finish ----------\n')
