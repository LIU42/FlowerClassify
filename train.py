import torch
import yaml

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import ClassifyNet


with open('configs/train.yaml', 'r') as configs:
    configs = yaml.safe_load(configs)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('datasets/train', transform=transform)
valid_dataset = datasets.ImageFolder('datasets/valid', transform=transform)

train_loader = data.DataLoader(train_dataset, configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])
valid_loader = data.DataLoader(valid_dataset, configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])

load_path = configs['load-path']
best_path = configs['best-path']
last_path = configs['last-path']

device = torch.device(configs['device'])

model = ClassifyNet(num_classes=configs['num-classes'], pretrained=configs['load-pretrained'])
model = model.to(device)

if configs['load-checkpoint']:
    model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=configs['learning-rate'])

max_accuracy = 0.0

print(f'\n---------- Training Start At: {str(device).upper()} ----------\n')

for epoch in range(configs['epochs']):
    model.train()
    training_loss = 0

    for index, (inputs, labels) in enumerate(train_loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        print(f'\rBatch Loss: {loss:.3f} [{index}/{len(train_loader)}]', end='')

    model.eval()
    training_loss /= len(train_loader)

    with torch.no_grad():
        valid_accuracy = 0

        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            valid_accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

        valid_accuracy /= len(valid_dataset)

    if valid_accuracy > max_accuracy:
        max_accuracy = valid_accuracy
        torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), last_path)

    print(f'\tEpoch: {epoch:<6} Loss: {training_loss:<10.5f} Accuracy: {valid_accuracy:.3f}')

print('\n---------- Training Finish ----------\n')
print(f'Max Accuracy: {max_accuracy:.3f}')
