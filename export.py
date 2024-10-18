import torch
import yaml

from models import ResNet


with open('configs/export.yaml', 'r') as configs:
    configs = yaml.load(configs, Loader=yaml.FullLoader)

    source_path = configs['source-path']
    num_classes = configs['num-classes']

    export_path_fp32 = configs['export-path-fp32']
    export_path_fp16 = configs['export-path-fp16']

model_fp32 = ResNet(num_classes=num_classes)
model_fp16 = ResNet(num_classes=num_classes)

model_fp32.load_state_dict(torch.load(source_path, map_location='cpu', weights_only=True))
model_fp32.eval()

model_fp16.load_state_dict(torch.load(source_path, map_location='cpu', weights_only=True))
model_fp16.eval()

model_fp32 = model_fp32.float()
model_fp16 = model_fp16.half()

inputs_fp32 = torch.randn(1, 3, 224, 224)
inputs_fp16 = torch.randn(1, 3, 224, 224)

inputs_fp32 = inputs_fp32.float()
inputs_fp16 = inputs_fp16.half()

torch.onnx.export(model_fp32, inputs_fp32, export_path_fp32, input_names=['input'], output_names=['output'])
torch.onnx.export(model_fp16, inputs_fp16, export_path_fp16, input_names=['input'], output_names=['output'])
