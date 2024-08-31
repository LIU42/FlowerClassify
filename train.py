import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import ClassifyNet


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('datasets/train', transform=transform)
valid_dataset = datasets.ImageFolder('datasets/valid', transform=transform)

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = data.DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=0)

load_path = 'weights/develop/pretrain.pt'
best_path = 'weights/develop/best.pt'
last_path = 'weights/develop/last.pt'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = ClassifyNet().to(device)
model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0002)

epochs = 50
best_accuracy = 0

print(f'\n---------- Training Start At: {str(device).upper()} ----------\n')

for epoch in range(epochs):
    train_loss = 0
    model.train()

    for index, (inputs, labels) in enumerate(train_loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        print(f'\rBatch Loss: {loss:.3f} [{index}/{len(train_loader)}]', end='')

    train_loss /= len(train_loader)

    model.eval()
    valid_accuracy = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            valid_accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

    valid_accuracy /= len(valid_dataset)

    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), last_path)

    print(f'\tEpoch: {epoch:<6} Loss: {train_loss:<8.3f} Accuracy: {valid_accuracy:.3f}')

print('\n---------- Training Finish ----------\n')
print(f'Best Accuracy: {best_accuracy:.3f}')
