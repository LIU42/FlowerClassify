import cv2
import numpy

class ImageUtils:

    @staticmethod
    def from_stream(image_bytes: bytes, convert: bool = True) -> cv2.Mat:
        stream_image = numpy.frombuffer(image_bytes, dtype=numpy.int8)
        decode_image = cv2.imdecode(stream_image, cv2.IMREAD_UNCHANGED)
        if convert:
            return cv2.cvtColor(decode_image, cv2.COLOR_BGR2RGB)
        return decode_image
    
    @staticmethod
    def letterbox(image: cv2.Mat, new_size: int = 224, background_color: int = 127) -> cv2.Mat:
        aspect_ratio = image.shape[1] / image.shape[0]
        if image.shape[1] > image.shape[0]:
            image_resize = cv2.resize(image, (new_size, int(new_size / aspect_ratio)))
        else:
            image_resize = cv2.resize(image, (int(new_size * aspect_ratio), new_size))

        background = numpy.ones((new_size, new_size, 3), dtype=numpy.uint8) * background_color
        x = (new_size - image_resize.shape[1]) // 2
        y = (new_size - image_resize.shape[0]) // 2
        background[y:y + image_resize.shape[0], x:x + image_resize.shape[1]] = image_resize

        return background


class PlottingUtils:

    @staticmethod
    def label(image: cv2.Mat, label: str, color: tuple[int, int, int] = (255, 0, 0)) -> cv2.Mat:
        return cv2.putText(image, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
