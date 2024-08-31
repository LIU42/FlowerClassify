import torch

from model import ClassifyNet


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = ClassifyNet().to(device)
model.eval()
model.load_state_dict(torch.load('weights/develop/best.pt', map_location=device, weights_only=True))

dummy_input = torch.ones((1, 3, 224, 224), dtype=torch.float32, device=device)

torch.onnx.export(
    model=model,
    f='weights/product/classify-fp32.onnx',
    args=dummy_input,
    input_names=['input'],
    output_names=['output'],
)
