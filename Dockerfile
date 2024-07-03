FROM hdgigante/python-opencv:4.9.0-debian

RUN mkdir /home/app
RUN mkdir /home/app/dataset
RUN mkdir /home/app/weights

WORKDIR /home/app

COPY ./main.py /home/app/main.py
COPY ./predict.py /home/app/predict.py
COPY ./utils.py /home/app/utils.py

COPY ./weights/flower-classify-fp32.onnx /home/app/weights/flower-classify-fp32.onnx
COPY ./dataset/classes.json /home/app/dataset/classes.json

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages Flask onnxruntime

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD ["flask", "--app", "main", "run", "--host=0.0.0.0", "--port=9500"]
