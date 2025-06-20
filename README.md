# Flower Classify

*<u>v1.4.0 新变化：调整数据增强重新训练模型。</u>*

## 项目简介

本项目为一个基于 ResNet 的花卉分类识别系统，能有效区分 10 中不同类别的花卉，采用 ResNet18 作为主干网络，包含模型的训练、测试以及线上部署（提供容器化部署）。

- 基于 [PyTorch](https://pytorch.org/) 框架进行模型的训练及测试。

- 模型采用 [ONNX](https://onnx.org.cn/onnx/index.html) 格式部署，采用 [ONNX Runtime](https://onnxruntime.ai/) 进行推理。

- 基于 [Flask](https://flask.palletsprojects.com/en/stable/) 框架实现 Web 接口。

- 使用 [Docker](https://www.docker.com/) 进行容器化部署。

训练数据集来自 [Kaggle](https://www.kaggle.com/)，融合了多个数据集并进行了数据清洗，基于预训练模型进行训练。

## 使用说明

### 安装环境依赖

首先使用 pip 安装本项目相关的依赖包：

```shell-session
pip install -r requirements.txt
```

### 启动 Web 服务

本项目 Web 服务的默认配置文件为 servers/configs/config.toml，其中各个字段描述如下。

| 字段名        | 字段描述                                      |
|:----------:|:-----------------------------------------:|
| precision  | 模型推理精度，取值为 "fp32" (单精度) 和 "fp16" (半精度) 。  |
| providers  | 模型推理 ONNX Runtime Execution Providers 列表。 |
| model-path | 模型加载路径。                                   |

从本项目 Release 中下载 [ONNX](https://onnx.org.cn/onnx/index.html) 格式的模型权重文件放入 servers/models 目录后，执行以下命令启动 Web 服务：

```shell-session
flask --app servers.server run --host="0.0.0.0" --port=9500
```

### 模型训练

若要使用自己的数据集训练模型，准备好数据集、调整好模型输出格式后，运行 train.py 即可开始训练，训练和验证默认的配置文件为 configs/config.toml，其中各个字段的描述如下。

| 字段名                  | 字段描述                                                                                    |
|:--------------------:|:---------------------------------------------------------------------------------------:|
| device               | 设备名称，与 PyTorch 的设备名称保持一致。                                                               |
| num-epochs           | 训练迭代次数。                                                                                 |
| num-workers          | 训练及评估数据加载进程数。                                                                           |
| batch-size           | 训练数据批大小。                                                                                |
| learning-rate        | 模型训练学习率。                                                                                |
| weight-decay         | 模型训练权重衰减。                                                                               |
| num-classes          | 模型输出类别数。                                                                                |
| log-interval         | 日志输出频率。                                                     |
| load-pretrained      | 是否使用预训练参数初始化模型权重。                                                                       |
| load-checkpoint      | 是否加载 checkpoint 继续训练，若为 true 则从 load-path 加载模型权重，覆盖 load-pretrained 值，反之则使用初始化模型权重开始训练。 |
| load-checkpoint-path | 训练初始模型的加载路径，同时也为待评估模型加载路径。                                                              |
| best-checkpoint-path | 训练中当前验证集最优模型保存路径。                                                                       |
| last-checkpoint-path | 训练中最后一次训练模型保存路径。                                                                        |

### 模型评估

模型训练完成后，运行 eval.py 以评估当前最优模型在测试集上的准确率，默认的配置文件及字段含义同上。

### 构建镜像

模型部署前需要转换为 [ONNX](https://onnx.org.cn/onnx/index.html) 格式放入 servers/models 目录中。构建镜像使用的 Dockerfile 位于 docker 目录中，请参考 [Docker 官方文档](https://docs.docker.com/) 进行镜像的构建和容器的运行。
