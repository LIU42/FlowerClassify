import flask
from flask import Flask
from flask import Response

from predict import FlowerClassifier
from utils import ImageUtils

app = Flask(__name__)
classifier = FlowerClassifier()

@app.post("/flowerclassify")
def flower_classify() -> Response:
    try:
        image = ImageUtils.from_stream(flask.request.files["image"].stream.read())
        return flask.jsonify(result_code=10000, result_data=classifier(image))
    except KeyError:
        return flask.jsonify(result_code=10001, result_data="image not found")


if __name__ == "__main__":
    app.run("0.0.0.0", 9500)
