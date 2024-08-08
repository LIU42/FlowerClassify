import torch

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import ClassifyNet


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = ImageFolder(root='datasets/test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=16, pin_memory=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = ClassifyNet()
model.to(device)
model.load_state_dict(torch.load('weights/dev/best.pt', map_location=device))

print(f'\n---------- Test At: {str(device).upper()} ----------\n')

with torch.no_grad():
    model.eval()
    correct_count = 0

    for step, (inputs, labels) in enumerate(test_loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        correct_count += (torch.argmax(outputs, dim=1) == labels).sum().item()

        print(f'\rProgress: [{step}/{len(test_loader)}]', end='')

print(f'\tAccuracy: {correct_count / len(test_dataset):.3f}')
