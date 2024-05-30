import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import ClassifyNet

trainset_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

validset_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageFolder(root="./dataset/train", transform=trainset_transform)
train_length = len(train_dataset)
valid_dataset = ImageFolder(root="./dataset/valid", transform=validset_transform)
valid_length = len(valid_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=0)

load_path = "./weights/ClassifyNet-Best.pt"
best_path = "./weights/ClassifyNet-Best.pt"
last_path = "./weights/ClassifyNet-Last.pt"
epoch_times = 10
best_accuracy = 0.955

model = ClassifyNet()
model.load_state_dict(torch.load(load_path))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00005)

print("\n---------- Training Start ----------\n")

for epoch in range(epoch_times):
    train_loss = 0
    model.train()

    for step, (inputs, labels) in enumerate(train_loader, start=0):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(f"\rTrain Loss: {loss:.3f} [{step}/{len(train_loader)}]", end="")

    model.eval()
    correct_count = 0
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(valid_loader, start=0):
            outputs = model(inputs)
            predict = torch.max(outputs, dim=1)[1]
            correct_count += (predict == labels).sum().item()

        valid_accuracy = correct_count / valid_length
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), best_path)

        torch.save(model.state_dict(), last_path)
        print(f"\tEpoch: {epoch}\tLoss: {train_loss / len(train_loader):.3f}\tAccuracy: {valid_accuracy:.3f}")

print("\n---------- Training Finished ----------\n")
print(f"Best Accuracy: {best_accuracy:.3f}\n")
