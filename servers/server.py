import flask
import onnxruntime as ort
import cv2
import toml
import numpy as np


configs = toml.load('servers/configs/config.toml')

precision = configs['precision']
providers = configs['providers']

app = flask.Flask(__name__)

model_path = configs['model-path']
image_size = configs['image-size']

session = ort.InferenceSession(model_path, providers=providers)


def decode_stream(stream):
    return cv2.imdecode(np.frombuffer(stream.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def center_crop(image):
    current_size = min(image.shape[0], image.shape[1])

    x1 = (image.shape[1] - current_size) >> 1
    y1 = (image.shape[0] - current_size) >> 1

    x2 = x1 + current_size
    y2 = y1 + current_size

    return cv2.resize(image[y1:y2, x1:x2], (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def normalize(inputs):
    return (inputs - 127.5) / 127.5


def convert_inputs(image):
    converted_inputs = cv2.cvtColor(center_crop(image), cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
    converted_inputs = normalize(converted_inputs)

    if precision == 'fp16':
        return np.expand_dims(converted_inputs, axis=0).astype(np.float16)
    else:
        return np.expand_dims(converted_inputs, axis=0).astype(np.float32)


def softmax(inputs):
    return np.exp(inputs) / np.sum(np.exp(inputs), axis=0)


def inference(image):
    classes_outputs = session.run(['outputs'], {'inputs': convert_inputs(image)})
    classes_outputs = classes_outputs[0]

    return outputs_postprocess(classes_outputs.squeeze())


def outputs_postprocess(outputs):
    index = np.argmax(outputs)
    confidences = softmax(outputs)

    return index.item(), confidences[index].item()


def classify(stream):
    return inference(decode_stream(stream))


@app.post('/flowerclassify')
def flower_classify():
    try:
        index, confidence = classify(flask.request.files['image'].stream)
    except KeyError:
        flask.abort(400)

    return flask.jsonify({'index': index, 'confidence': confidence})
