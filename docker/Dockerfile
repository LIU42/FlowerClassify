FROM debian:12-slim

RUN mkdir /home/app
RUN mkdir /home/app/datasets
RUN mkdir /home/app/weights
RUN mkdir /home/app/weights/product

WORKDIR /home/app

COPY ./main.py /home/app/main.py
COPY ./predict.py /home/app/predict.py
COPY ./utils.py /home/app/utils.py

COPY ./weights/product/classify-fp32.onnx /home/app/weights/product/classify-fp32.onnx
COPY ./datasets/classes.yaml /home/app/datasets/classes.yaml

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update
RUN apt-get install python3 python3-pip python3-opencv -y

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages --no-cache-dir Flask onnxruntime PyYAML

RUN apt-get remove python3-pip -y
RUN apt-get clean -y
RUN apt-get autoremove -y
RUN apt-get autoclean -y

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD ["flask", "--app", "main", "run", "--host=0.0.0.0", "--port=9500"]
