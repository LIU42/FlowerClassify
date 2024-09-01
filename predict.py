import onnxruntime as ort

import utils.preprocess as preprocess
import utils.postprocess as postprocess


class FlowerClassifier:
    def __init__(self, configs):
        if configs['device'] == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.configs = configs
        self.session = ort.InferenceSession(f'weights/product/classify-{self.precision}.onnx', providers=providers)

    def __call__(self, image):
        inputs = preprocess.preprocess(image, size=224, padding_color=127, precision=self.precision)

        outputs = self.session.run([], inputs)
        outputs = self.postprocessing(outputs)

        class_index, confidences = postprocess.parse_outputs(outputs)

        return self.classes[class_index], f'{confidences[class_index]:.3f}'
    
    @property
    def precision(self):
        return self.configs['precision']
    
    @property
    def classes(self):
        return self.configs['classes']

    @staticmethod
    def postprocessing(outputs):
        return outputs[0].squeeze()
