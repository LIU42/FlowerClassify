import numpy
import cv2
import predict

from flask import Flask
from flask import request

application = Flask(__name__)

@application.post("/flowerclassify")
def flower_classify() -> dict:

    image_file = request.files.get("image")

    if image_file == None:
        return { "result_code": 10001, "result_data": "image not found" }

    image_stream = numpy.frombuffer(image_file.stream.read(), numpy.uint8)
    image_decode = cv2.imdecode(image_stream, cv2.IMREAD_UNCHANGED)
    image_origin = cv2.cvtColor(image_decode, cv2.COLOR_BGR2RGB)

    return { "result_code": 10000, "result_data": predict.predict_image(image_origin) }
    
if __name__ == "__main__":

    application.run("0.0.0.0", 9500)
