import flask
import onnxruntime as ort
import cv2
import toml
import numpy as np


configs = toml.load('servers/configs/config.toml')

app = flask.Flask(__name__)
session = ort.InferenceSession(configs['model-path'], providers=configs['providers'])


def softmax(inputs):
    return np.exp(inputs) / np.sum(np.exp(inputs), axis=0)


def normalize(inputs):
    return (inputs - 127.5) / 127.5


def decode_stream(stream):
    return cv2.imdecode(np.frombuffer(stream.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def center_crop(image, size=224):
    current_size = min(image.shape[1], image.shape[0])

    x1 = (image.shape[1] - current_size) >> 1
    y1 = (image.shape[0] - current_size) >> 1

    x2 = x1 + current_size
    y2 = y1 + current_size

    return cv2.resize(image[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)


def preprocess(image):
    inputs = cv2.cvtColor(center_crop(image), cv2.COLOR_BGR2RGB).transpose((2, 0, 1))

    if configs['precision'] == 'fp16':
        inputs = normalize(inputs).astype(np.float16)
    else:
        inputs = normalize(inputs).astype(np.float32)

    return np.expand_dims(inputs, axis=0)


def inference(image):
    predictions = session.run(['outputs'], {'inputs': preprocess(image)})
    predictions = predictions[0]

    return output_postprocess(predictions.squeeze())


def output_postprocess(outputs):
    index = np.argmax(outputs)
    scores = softmax(outputs)

    return index.item(), round(scores[index].item(), 3)


def decode_and_inference(stream):
    index, score = inference(decode_stream(stream))

    return {
        'index': index,
        'score': score,
    }


@app.post('/flowerclassify')
def flower_classify():
    return flask.jsonify(decode_and_inference(flask.request.files['image'].stream))
