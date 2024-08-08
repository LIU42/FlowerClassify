import yaml
import onnxruntime as ort

from utils import ImageUtils
from utils import ResultUtils


class FlowerClassifier:

    def __init__(self, device='CPU', precision='fp32'):
        if device == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        with open('datasets/classes.yaml', 'r') as classes_yaml:
            self.classes = yaml.load(classes_yaml, yaml.SafeLoader)['classes']

        self.session = ort.InferenceSession(f'weights/product/classify-{precision}.onnx', providers=providers)
        self.precision = precision

    def __call__(self, image):
        inputs = ImageUtils.preprocess(image, size=224, padding_color=127, precision=self.precision)

        outputs = self.session.run(None, {
            'input': inputs,
        })
        class_index, confidences = ResultUtils.parse_outputs(outputs[0].squeeze())

        return self.classes[class_index], f'{confidences[class_index]:.3f}'
