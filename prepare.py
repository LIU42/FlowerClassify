import torch
import yaml

from model import ClassifyNet


def load_configs():
    with open('configs/model.yaml', 'r') as configs:
        return yaml.safe_load(configs)


configs = load_configs()

model = ClassifyNet(num_classes=configs['num-classes'], pretrain=configs['pretrain'])
model.eval()

torch.save(model.state_dict(), configs['save-path'])
