FROM debian:12-slim

RUN mkdir /home/app
RUN mkdir /home/app/configs
RUN mkdir /home/app/inference
RUN mkdir /home/app/inference/models

WORKDIR /home/app

COPY ./configs/inference.yaml /home/app/configs/inference.yaml
COPY ./inference/models/flower-fp32.onnx /home/app/inference/models/flower-fp32.onnx
COPY ./inference/__init__.py /home/app/inference/__init__.py
COPY ./inference/server.py /home/app/inference/server.py

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

CMD ["flask", "--app", "inference.server", "run", "--host=0.0.0.0", "--port=9500"]
