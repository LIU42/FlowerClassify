import cv2
import numpy as np


class ImageUtils:
    @staticmethod
    def letterbox(image, size, padding_color):
        current_size = max(image.shape[0], image.shape[1])

        x1 = (current_size - image.shape[1]) >> 1
        y1 = (current_size - image.shape[0]) >> 1

        x2 = x1 + image.shape[1]
        y2 = y1 + image.shape[0]

        background = np.full((current_size, current_size, 3), padding_color, dtype=np.uint8)
        background[y1:y2, x1:x2] = image

        return cv2.resize(background, (size, size))

    @staticmethod
    def convert_inputs(image, precision):
        inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        inputs = inputs / 255.0
        inputs = np.expand_dims(inputs, axis=0)

        if precision == 'fp16':
            return {'input': inputs.astype(np.float16)}
        else:
            return {'input': inputs.astype(np.float32)}

    @staticmethod
    def preprocess(image, size, padding_color, precision):
        return ImageUtils.convert_inputs(ImageUtils.letterbox(image, size, padding_color), precision)


class ResultUtils:
    @staticmethod
    def parse_outputs(outputs):
        probability_outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=0)

        return np.argmax(probability_outputs), probability_outputs
