import numpy
import cv2
import predict

from flask import Flask
from flask import Response
from flask import request
from flask import jsonify

application = Flask(__name__)

@application.post("/flowerclassify")
def flower_classify() -> Response:
    try:
        image_file = request.files["image"]
        image_stream = numpy.frombuffer(image_file.stream.read(), numpy.uint8)
        image_decode = cv2.imdecode(image_stream, cv2.IMREAD_UNCHANGED)
        image_origin = cv2.cvtColor(image_decode, cv2.COLOR_BGR2RGB)
    except KeyError:
        return jsonify(result_code = 10001, result_data = "image not found")
    
    name, confidence = predict.predict_image(image_origin)
    return jsonify(result_code = 10000, result_data = dict(name = name, confidence = f"{confidence:.3f}"))
    
if __name__ == "__main__":
    application.run("0.0.0.0", 9500)
