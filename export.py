import torch
import yaml

from model import ClassifyNet


with open('configs/export.yaml', 'r') as configs:
    configs = yaml.safe_load(configs)

device = torch.device(configs['device'])

model = ClassifyNet(num_classes=configs['num-classes'], pretrained=False)
model = model.to(device)

input = torch.ones(1, 3, 224, 224)
input = input.to(device)

model.load_state_dict(torch.load(configs['source-path'], map_location=device, weights_only=True))
model.eval()

if configs['precision'] == 'fp16':
    model = model.half()
    input = input.half()
else:
    model = model.float()
    input = input.float()

torch.onnx.export(model, input, configs['output-path'], input_names=['input'], output_names=['output'])
