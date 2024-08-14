import onnxruntime as ort

from utils import ImageUtils
from utils import ResultUtils


class FlowerClassifier:

    def __init__(self, configs):
        if configs['device'] == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.configs = configs
        self.session = ort.InferenceSession(f'weights/product/classify-{self.precision}.onnx', providers=providers)

    def __call__(self, image):
        inputs = ImageUtils.preprocess(image, size=224, padding_color=127, precision=self.precision)

        outputs = self.session.run(None, {
            'input': inputs,
        })
        class_index, confidences = ResultUtils.parse_outputs(outputs[0].squeeze())

        return self.classes[class_index], f'{confidences[class_index]:.3f}'
    
    @property
    def precision(self):
        return self.configs['precision']
    
    @property
    def classes(self):
        return self.configs['classes']
