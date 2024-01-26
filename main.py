import flask
import numpy
import cv2

from predict import FlowerClassifier
from flask import Flask
from flask import Response

application = Flask(__name__)
classifier = FlowerClassifier()

@application.post("/flowerclassify")
def flower_classify() -> Response:
    try:
        image_file = flask.request.files["image"]
        image_stream = numpy.frombuffer(image_file.stream.read(), numpy.uint8)
        image_decode = cv2.imdecode(image_stream, cv2.IMREAD_UNCHANGED)
        image_origin = cv2.cvtColor(image_decode, cv2.COLOR_BGR2RGB)
    except KeyError:
        return flask.jsonify(result_code = 10001, result_data = "image not found")
    return flask.jsonify(result_code = 10000, result_data = classifier(image_origin))
    
if __name__ == "__main__":
    application.run("0.0.0.0", 9500)
