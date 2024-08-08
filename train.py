import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import ClassifyNet


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root='datasets/train', transform=transform)
valid_dataset = ImageFolder(root='datasets/valid', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

load_path = 'weights/dev/pretrain.pt'
best_path = 'weights/dev/best.pt'
last_path = 'weights/dev/last.pt'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

epochs = 50
best_accuracy = 0

model = ClassifyNet()
model.to(device)
model.load_state_dict(torch.load(load_path, map_location=device))

criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0002)

print(f'\n---------- Training Start At: {str(device).upper()} ----------\n')

for epoch in range(epochs):
    accumulate_loss = 0
    model.train()

    for step, (inputs, labels) in enumerate(train_loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accumulate_loss += loss.item()

        print(f'\rTrain Loss: {loss:.3f} [{step}/{len(train_loader)}]', end='')

    model.eval()
    correct_count = 0

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(valid_loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            correct_count += (torch.argmax(outputs, dim=1) == labels).sum().item()

        valid_accuracy = correct_count / len(valid_dataset)

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), best_path)

        torch.save(model.state_dict(), last_path)

    print(f'\tEpoch: {epoch:<6} Loss: {accumulate_loss / len(train_loader):<8.3f} Accuracy: {valid_accuracy:.3f}')

print('\n---------- Training Finish ----------\n')
print(f'Best Accuracy: {best_accuracy:.3f}')
