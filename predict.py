import onnxruntime as ort
import utils.process as process


class FlowerClassifier:
    def __init__(self, configs):
        if configs['device'] == 'CUDA':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.configs = configs
        self.session = ort.InferenceSession(f'weights/classify-{self.precision}.onnx', providers=providers)

    def __call__(self, image):
        inputs = process.preprocess(image, size=224, padding_color=127, precision=self.precision)

        outputs = self.predict(inputs)
        outputs = self.reshape(outputs)

        class_index, confidences = process.parse_outputs(outputs)

        return self.classes[class_index], f'{confidences[class_index]:.3f}'
    
    @property
    def precision(self):
        return self.configs['precision']
    
    @property
    def classes(self):
        return self.configs['classes']
    
    def predict(self, inputs):
        return self.session.run(None, {'input': inputs})

    @staticmethod
    def reshape(outputs):
        return outputs[0].squeeze()
