import torch
import toml
import tqdm

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import FlowerNet


configs = toml.load('configs/config.toml')

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

num_epochs = configs['num-epochs']

train_dataset = ImageFolder('datasets/train', transform=train_transform)
valid_dataset = ImageFolder('datasets/valid', transform=valid_transform)

train_dataset_size = len(train_dataset)
valid_dataset_size = len(valid_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=True)

train_dataloader_size = len(train_dataloader)
valid_dataloader_size = len(valid_dataloader)

best_accuracy = 0.0
last_accuracy = 0.0

device = torch.device(configs['device'])

model = FlowerNet(num_classes=configs['num-classes'], pretrained=configs['load-pretrained'])
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=configs['learning-rate'], weight_decay=configs['weight-decay'])

load_checkpoint_path = configs['load-checkpoint-path']
best_checkpoint_path = configs['best-checkpoint-path']
last_checkpoint_path = configs['last-checkpoint-path']

if configs['load-checkpoint']:
    model.load_state_dict(torch.load(load_checkpoint_path, map_location=device, weights_only=True))

print(f'\n---------- training start at: {device} ----------\n')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm.tqdm(train_dataloader, ncols=80):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    train_loss /= train_dataloader_size

    with torch.no_grad():
        accuracy = 0.0

        for images, labels in tqdm.tqdm(valid_dataloader, ncols=80):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

        accuracy /= valid_dataset_size

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_checkpoint_path)

        last_accuracy = accuracy
        torch.save(model.state_dict(), last_checkpoint_path)

    print(f'\nepoch: {epoch + 1}/{num_epochs:<6} loss: {train_loss:<10.5f} accuracy: {accuracy:.3f}\n')

print(f'best accuracy: {best_accuracy:.3f}')
print(f'last accuracy: {last_accuracy:.3f}')

print('\n---------- training finished ----------\n')
