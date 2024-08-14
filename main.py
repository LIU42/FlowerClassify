import flask
import cv2
import numpy as np
import yaml

from predict import FlowerClassifier


def load_classifier():
    with open('config.yaml', 'r') as configs:
        return FlowerClassifier(yaml.safe_load(configs))


def read_stream(stream):
    return cv2.imdecode(np.frombuffer(stream.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


app = flask.Flask(__name__)
classifier = load_classifier()


@app.errorhandler(400)
def bad_request(exception):
    return f'{exception}', 400


@app.post('/flowerclassify')
def flower_classify():
    try:
        name, confidence = classifier(read_stream(flask.request.files['image'].stream))
    except KeyError:
        flask.abort(400)

    return flask.jsonify({'name': name, 'confidence': confidence})
