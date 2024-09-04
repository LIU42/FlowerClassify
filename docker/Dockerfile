FROM debian:12-slim

RUN mkdir /home/app
RUN mkdir /home/app/configs
RUN mkdir /home/app/utils
RUN mkdir /home/app/weights
RUN mkdir /home/app/weights/deploy

WORKDIR /home/app

COPY ./server.py /home/app/server.py
COPY ./predict.py /home/app/predict.py
COPY ./utils/__init__.py /home/app/utils/__init__.py
COPY ./utils/process.py /home/app/utils/process.py
COPY ./configs/server.yaml /home/app/configs/server.yaml
COPY ./weights/deploy/classify-fp32.onnx /home/app/weights/deploy/classify-fp32.onnx

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

CMD ["flask", "--app", "server", "run", "--host=0.0.0.0", "--port=9500"]
