# Flower Classify

**基于 ResNet 的花卉分类识别系统，能有效区分 10 中不同类别的花卉。**

## 项目简介

本项目为一个基础的花卉分类识别系统，采用 ResNet18 作为主干网络，包含模型的训练、测试以及线上部署（提供容器化部署）。

- 基于 PyTorch 框架进行模型的训练及测试。

- 模型采用 ONNX 格式部署，采用 ONNX Runtime 进行推理。

- 基于 Flask 框架实现 Web 接口。

- 使用 Docker 进行容器化部署。

*<u>训练数据集来自 [Kaggle](https://www.kaggle.com/)，融合了多个数据集并进行了数据清洗，基于预训练模型进行训练，在当前数据集下准确率超过 98%。</u>*

## 使用说明

### 安装环境依赖

首先使用 pip 安装如下的依赖：

```python
# 推理部署环境依赖
opencv-python~=4.10.0.84
numpy~=1.23.4
Flask~=3.0.3
PyYAML~=6.0
onnxruntime~=1.14.1

# 训练环境依赖
torch~=2.4.0
torchvision~=0.19.0
onnx~=1.16.2
```

<u>*注：使用 pip 安装 opencv-python 可能会出现依赖不全的问题，推荐使用系统包管理器安装。*</u>

### 启动 Web 服务

本项目 Web 服务的默认配置文件为 <u>inferences/configs/server.yaml</u>，其中各个属性对应的含义如下：

```yaml
precision: "fp32"            # 推理运算精度，"fp32"（单精度）或 "fp16"（半精度）

session-providers:           # ONNX Runtime "InferenceSession" "providers" 参数
  - "CPUExecutionProvider"

flower-names:                # 花卉分类名称列表，包含所有花卉类别对应的标签（按顺序）
  - "Bellflower"
  - "Carnation"
  ...

model-path: "inferences/models/flower-fp32.onnx"    # 加载的模型路径
```

将模型权重文件放入 <u>inferences/models/</u> 目录后，执行以下命令启动 Web 服务：

```bash
flask --app inferences.server run --host="0.0.0.0" --port=9500
```

Web 服务接口描述如下：

```json5
/*
 * URL: http://<your-server-address>:9500/flowerclassify
 * METHOD: POST
 * BODY: form-data
 *   image: 待识别图像文件
 */

// 返回结果示例 (JSON)：
{
    "name": "Rose",
    "confidence": "1.000"
}
```

### 模型训练与评估

若要使用自己的数据集训练模型，准备好数据集、调整好模型输出格式后：

1. 根据需要调整 <u>configs/train.yaml</u> 中的各项参数，运行 train.py 即可开始训练，配置属性对应的含义如下：
   
   ```yaml
   device: "cpu"            # 设备名称，与 PyTroch 的设备名称保持一致
   num-epochs: 50           # 训练迭代次数
   learning-rate: 0.0002    # 学习率
   batch-size: 32           # 批大小
   weight-decay: 0.01       # 权重衰减
   
   use-augment: true        # 是否启用数据增强
   num-workers: 0           # DataLoader 加载子进程数
   num-classes: 10          # 模型分类类别数
   
   load-checkpoint: false         # 是否加载 checkpoint 继续训练，若为 true 则从 load-path 加载模型权重，反之则使用初始化模型权重开始训练
   load-pretrained: true          # 是否使用预训练参数初始化模型权重
   
   load-path: "checkpoints/last.pt"        # 待训练模型路径
   best-path: "checkpoints/best.pt"        # 当前验证集上最优模型保存路径
   last-path: "checkpoints/last.pt"        # 最后一次迭代模型保存路径
   ```

2. 运行 eval.py 以评估当前最优模型在测试集上的准确率（可选），默认的配置文件为 <u>configs/eval.yaml</u>，其中各个属性对应的含义如下：
   
   ```yaml
   device: "cpu"                              # 设备名称，与 PyTroch 的设备名称保持一致
   checkpoint-path: "checkpoints/best.pt"     # 待评估模型路径
   batch-size: 16                             # 批大小
   num-workers: 0                             # DataLoader 加载子进程数
   num-classes: 10                            # 模型分类类别数
   ```

### 模型推理部署

部署需要将训练好的模型转换为 ONNX 格式，PyTorch 模型转换为 ONNX 格式相对简单，可以根据自己的需要进行转换。若要使用 Docker 进行容器化部署：

```bash
# 构建镜像
cd FlowerClassify
docker build -t flowerclassify:1.3.0 -f docker/Dockerfile .

# 创建容器并运行
docker run --rm -p 9500:9500 --name flowerclassify flowerclassify:1.3.0
```

*<u>注：以上仅为一个示例，详情请参考 [Docker](https://docs.docker.com/) 文档。</u>*





# Flower Classify

**A ResNet-based flower classification and recognition system that can effectively distinguish 10 different categories of flowers.**

## Overviews

This project is a basic flower classification and recognition system, using ResNet18 as the backbone network, including model training, testing, and online deployment (containerized deployment is provided).

- Model training and testing based on PyTorch framework.

- The model is deployed in ONNX format and uses ONNX Runtime for inference.

- Web interface implementation based on Flask framework.

- Containerized deployment with Docker.

*<u>The training dataset is from [Kaggle](https://www.kaggle.com/), fused multiple datasets with data cleaning, and trained based on a pre-trained model with an accuracy of more than 98% with the current dataset.</u>*

## Usage

### Installation

First install the following dependencies using pip:

```python
# Inference and deployment environment dependencies
opencv-python~=4.10.0.84
numpy~=1.23.4
Flask~=3.0.3
PyYAML~=6.0
onnxruntime~=1.14.1

# Training environment dependencies
torch~=2.4.0
torchvision~=0.19.0
onnx~=1.16.2
```

<u>*Note: Using pip to install opencv-python may result in incomplete dependencies, it is recommended to use the system package manager to install it.*</u>

### Starting Web Services

The default configuration file for this project's Web service is <u>inferences/configs/server.yaml</u>, where each attribute has the following meaning:

```yaml
precision: "fp32"            # Precision of inference operations, “fp32” (single precision) or “fp16” (half precision)

session-providers:           # ONNX Runtime 'InferenceSession' 'providers' param
  - "CPUExecutionProvider"

flower-names:                # List of flower category names with labels corresponding to all flower categories (in order)
  - "Bellflower"
  - "Carnation"
  ...

model-path: "inferences/models/flower-fp32.onnx"    # Path to the model to be loaded
```

After placing the model files in the <u>inferences/models/</u> directory, execute the following command to start the web service:

```bash
flask --app inferences.server run --host="0.0.0.0" --port=9500
```

The Web service interface is described below:

```json5
/*
 * URL: http://<your-server-address>:9500/flowerclassify
 * METHOD: POST
 * BODY: form-data
 *   image: Image files to be recognized
 */

// Example of return result (JSON)：
{
    "name": "Rose",
    "confidence": "1.000"
}
```

### Model Training and Evaluation

To train the model using your own dataset, prepare the dataset and adjust the model output format:

1. Adjust the parameters in <u>configs/train.yaml</u> as needed, and run train.py to start training, with the following configuration attributes:
   
   ```yaml
   device: "cpu"            # Device name, consistent with PyTroch's device name
   num-epochs: 50           # Number of training iterations
   learning-rate: 0.0002    # Learning rate
   batch-size: 32           # Batch size
   weight-decay: 0.01       # Weight decay
   
   use-augment: true        # Whether to enable data augmentation
   num-workers: 0           # Number of data loading subprocesses
   num-classes: 10          # Number of model classification categories
   
   load-checkpoint: false         # Whether to load checkpoint to continue training, if true then load model weights from load-path, otherwise start training with initialized model weights
   load-pretrained: true          # Whether or not to initialize model weights with pre-training parameters
   
   load-path: "checkpoints/last.pt"        # Model path to be trained
   best-path: "checkpoints/best.pt"        # Optimal model saving path on the current validation set
   last-path: "checkpoints/last.pt"        # Last iteration model save path
   ```

2. Run eval.py to evaluate the accuracy of the current optimal model on the test set (optional), the default configuration file is <u>configs/eval.yaml</u>, where each attribute has the following meaning:
   
   ```yaml
   device: "cpu"                              # Device name, consistent with PyTroch's device name
   checkpoint-path: "checkpoints/best.pt"     # Model paths to be evaluated
   batch-size: 16                             # Batch size
   num-workers: 0                             # Number of data loading subprocesses
   num-classes: 10                            # Number of model classification categories
   ```

### Model Inference and Deployment

To deploy the model, you need to convert the trained model to ONNX format. Converting PyTorch models to ONNX format is relatively simple, you can do the model conversion by yourself. To use Docker for containerized deployment:

```bash
# Building an image
cd FlowerClassify
docker build -t flowerclassify:1.3.0 -f docker/Dockerfile .

# Create a container and run it
docker run --rm -p 9500:9500 --name flowerclassify flowerclassify:1.3.0
```

*<u>Note: The above is just an example, please refer to the [Docker](https://docs.docker.com/) documentation for more details.</u>*
