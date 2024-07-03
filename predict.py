import json
import onnxruntime as ort

from utils import ImageUtils
from utils import ResultUtils


class FlowerClassifier:

    def __init__(self, device='CPU', precision='fp32'):
        if device == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        with open('dataset/classes.json', mode='r', encoding='utf-8') as classes_file:
            self.classes = json.load(classes_file)

        self.session = ort.InferenceSession(f'weights/flower-classify-{precision}.onnx', providers=providers)
        self.precision = precision

    def __call__(self, image):
        return self.classify(image)

    def classify(self, image):
        inputs = ImageUtils.preprocess(image, size=224, padding_color=127, precision=self.precision)

        outputs = self.session.run(None, {
            'input': inputs,
        })
        class_index, confidence = ResultUtils.parse_outputs(outputs[0].squeeze())

        return self.classes[class_index], f'{confidence:.3f}'
