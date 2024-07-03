import flask
import json

from flask import Flask
from flask import Response

from predict import FlowerClassifier
from utils import ImageUtils

app = Flask(__name__)
classifier = FlowerClassifier(device='CPU', precision='fp32')


@app.post('/flowerclassify')
def flower_classify():
    try:
        name, confidence = classifier(ImageUtils.read_stream(flask.request.files['image'].stream))
    except KeyError:
        return Response(status=400)
    
    response_body = json.dumps({
        'name': name,
        'confidence': confidence,
    })
    return Response(status=200, mimetype='application/json', response=response_body)
