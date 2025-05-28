FROM debian:12-slim

RUN mkdir /home/app
RUN mkdir /home/app/servers
RUN mkdir /home/app/servers/models
RUN mkdir /home/app/servers/configs

WORKDIR /home/app

COPY ./servers/configs/config.toml /home/app/servers/configs/config.toml
COPY ./servers/models/flower-fp32.onnx /home/app/servers/models/flower-fp32.onnx
COPY ./servers/server.py /home/app/servers/server.py

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update
RUN apt-get install python3 python3-pip python3-opencv -y

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages --no-cache-dir Flask onnxruntime toml

RUN apt-get remove python3-pip -y
RUN apt-get clean -y
RUN apt-get autoremove -y
RUN apt-get autoclean -y

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD ["flask", "--app", "servers.server", "run", "--host=0.0.0.0", "--port=9500"]
