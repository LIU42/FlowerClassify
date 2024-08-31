import torch
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import ClassifyNet


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder('datasets/test', transform=transform)
dataset_size = len(test_dataset)

test_loader = data.DataLoader(test_dataset, batch_size=16, num_workers=0)
loader_size = len(test_loader)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = ClassifyNet().to(device)
model.load_state_dict(torch.load('weights/develop/best.pt', map_location=device, weights_only=True))

accuracy = 0

print(f'\n---------- Evaluation Start At: {str(device).upper()} ----------\n')

with torch.no_grad():
    model.eval()

    for step, (inputs, labels) in enumerate(test_loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

        print(f'\rProgress: [{step}/{loader_size}]', end='')

accuracy /= dataset_size

print(f'\nAccuracy: {accuracy:.3f}')
print('\n---------- Evaluation Finish ----------\n')
