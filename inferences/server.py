import flask
import onnxruntime as ort
import cv2
import yaml
import numpy as np


app = flask.Flask(__name__)

with open('configs/server.yaml', 'r') as configs:
    configs = yaml.load(configs, Loader=yaml.FullLoader)

    precision = configs['precision']
    providers = configs['providers']

    flower_names = configs['flower-names']

session = ort.InferenceSession(f'inferences/models/flower-{precision}.onnx', providers=providers)


def decode_stream(stream):
    return cv2.imdecode(np.frombuffer(stream.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def letterbox(image, size=224, padding=127):
    current_size = max(image.shape[0], image.shape[1])

    x1 = (current_size - image.shape[1]) >> 1
    y1 = (current_size - image.shape[0]) >> 1

    x2 = x1 + image.shape[1]
    y2 = y1 + image.shape[0]

    background = np.full((current_size, current_size, 3), padding, dtype=np.uint8)
    background[y1:y2, x1:x2] = image

    return cv2.resize(background, (size, size))


def preprocess(image):
    image = letterbox(image)

    inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
    inputs = inputs / 255.0
    inputs = np.expand_dims(inputs, axis=0)

    if precision == 'fp16':
        return inputs.astype(np.float16)
    else:
        return inputs.astype(np.float32)


def softmax(array):
    return np.exp(array) / np.sum(np.exp(array), axis=0)


def inference(image):
    outputs = session.run(['output'], {'input': preprocess(image)})
    outputs = outputs[0]
    outputs = outputs.squeeze()

    confidences = softmax(outputs)
    result = np.argmax(confidences)

    return flower_names[result], f'{confidences[result]:.3f}'


@app.errorhandler(400)
def bad_request(exception):
    return f'{exception}', 400


@app.post('/flowerclassify')
def flower_classify():
    try:
        name, confidence = inference(decode_stream(flask.request.files['image'].stream))
    except KeyError:
        flask.abort(400)

    return flask.jsonify({'name': name, 'confidence': confidence})
