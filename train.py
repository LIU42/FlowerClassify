import torch

from torch import nn
from torch import optim
from torch.utils import data
from torchvision import transforms
from torchvision import datasets

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.ImageFolder(root = "./dataset/train", transform = train_transform)
train_len = len(train_set)
valid_set = datasets.ImageFolder(root = "./dataset/valid", transform = valid_transform)
valid_len = len(valid_set)
train_loader = data.DataLoader(train_set, batch_size = 32, shuffle = True, num_workers = 0)
valid_loader = data.DataLoader(valid_set, batch_size = 32, shuffle = True, num_workers = 0)

load_path = "./weights/AlexNet-Best.pt"
best_path = "./weights/AlexNet-Best.pt"
last_path = "./weights/AlexNet-Last.pt"
epoch_time = 10
best_accuracy = 0.96

model = torch.load(load_path)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = 0.00005)

print("\n---------- Training Start ----------\n")

for epoch in range(epoch_time):
    running_loss = 0
    model.train()

    for step, (inputs, labels) in enumerate(train_loader, start = 0):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        print(f"\rTrain Loss: {loss:.3f} [{step}/{len(train_loader)}]", end = "")

    model.eval()
    accuracy = 0
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(valid_loader, start = 0):
            outputs = model(inputs)
            predict = torch.max(outputs, dim = 1)[1]
            accuracy += (predict == labels).sum().item()

        valid_accuracy = accuracy / valid_len
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model, best_path)
        torch.save(model, last_path)

        print(f"\tEpoch: {epoch}\tLoss: {running_loss / len(train_loader):.3f}\tAccuracy: {valid_accuracy:.3f}")

print("\n---------- Training Finished ----------\n")
