import torch

from model import ClassifyNet


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = ClassifyNet()
model.to(device)
model.load_state_dict(torch.load('weights/develop/best.pt', map_location=device, weights_only=True))

torch.onnx.export(
    f='weights/product/classify-fp32.onnx',
    model=model,
    args=torch.ones((1, 3, 224, 224), dtype=torch.float32).to(device),
    input_names=['input'],
    output_names=['output'],
)
