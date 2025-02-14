import torch
import yaml

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import FlowerNet


with open('configs/train.yaml', 'r') as configs:
    configs = yaml.load(configs, Loader=yaml.FullLoader)

augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomRotation(10),
    transforms.RandomErasing(scale=(0.05, 0.2), ratio=(0.5, 2.0)),
])

normal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_transform = normal_transform
valid_transform = normal_transform

if configs['use-augment']:
    train_transform = augment_transform

train_dataset = datasets.ImageFolder('datasets/train', transform=train_transform)
valid_dataset = datasets.ImageFolder('datasets/valid', transform=valid_transform)

train_dataset_size = len(train_dataset)
valid_dataset_size = len(valid_dataset)

train_dataloader = data.DataLoader(train_dataset, batch_size=configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])
valid_dataloader = data.DataLoader(valid_dataset, batch_size=configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])

train_dataloader_size = len(train_dataloader)
valid_dataloader_size = len(valid_dataloader)

device = torch.device(configs['device'])

model = FlowerNet(num_classes=configs['num-classes'], pretrained=configs['load-pretrained'])
model = model.to(device)

if configs['load-checkpoint']:
    model.load_state_dict(torch.load(configs['load-path'], map_location=device, weights_only=True))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=configs['learning-rate'], weight_decay=configs['weight-decay'])

best_accuracy = 0.0

print(f'\n---------- Training Start At: {str(device).upper()} ----------\n')

for epoch in range(configs['num-epochs']):
    model.train()
    train_loss = 0.0

    for index, (inputs, labels) in enumerate(train_dataloader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        print(f'\rBatch Loss: {loss:.3f} [{index}/{train_dataloader_size}]', end='')

    model.eval()
    train_loss /= train_dataloader_size

    with torch.no_grad():
        valid_accuracy = 0.0

        for inputs, labels in valid_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            valid_accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

        valid_accuracy /= valid_dataset_size

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), configs['best-path'])

        torch.save(model.state_dict(), configs['last-path'])

    print(f'\tEpoch: {epoch:<6} Loss: {train_loss:<10.5f} Accuracy: {valid_accuracy:.3f}')

print('\n---------- Training Finish ----------\n')
print(f'Best Accuracy: {best_accuracy:.3f}')
