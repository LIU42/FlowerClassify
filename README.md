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

本项目 Web 服务的默认配置文件为 <u>configs/server.yaml</u>，其中各个属性对应的含义如下：

```yaml
precision: "fp32"            # 推理运算精度，"fp32"（单精度）或 "fp16"（半精度）
providers:                   # ONNX Runtime Providers 参数
  - "CPUExecutionProvider"

flower-names:                # 花卉分类名称列表，包含所有花卉类别对应的标签（按顺序）
  - "Bellflower"
  - "Carnation"
  ...
```

将模型权重文件放入 <u>weights/</u> 目录后，执行以下命令启动 Web 服务：

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
   epochs: 50               # 训练迭代次数
   learning-rate: 0.0002    # 学习率
   batch-size: 32           # 批大小
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
   device: "cpu"                         # 设备名称，与 PyTroch 的设备名称保持一致
   model-path: "checkpoints/best.pt"     # 待评估模型路径
   batch-size: 32                        # 批大小
   num-classes: 10                       # 模型分类类别数
   num-workers: 0                        # DataLoader 加载子进程数
   ```

### 模型推理部署

部署需要将训练好的模型转换为 ONNX 格式，运行 export.py 即可将模型导出为 ONNX 格式，默认的配置文件为 <u>configs/export.yaml</u>，其中各个属性对应的含义如下：

```yaml
source-path: "checkpoints/best.pt"              # 待导出的 PyTorch 格式模型路径
num-classes: 10                                 # 模型分类类别数

export-path-fp32: "inferences/models/flower-fp32.onnx"    # fp32 精度模型导出路径
export-path-fp16: "inferences/models/flower-fp16.onnx"    # fp16 精度模型导出路径
```

若要使用 Docker 进行容器化部署：

```bash
# 构建镜像
cd FlowerClassify
docker build -t flowerclassify:1.3.0 -f docker/Dockerfile .

# 创建容器并运行
docker run --rm -p 9500:9500 --name flowerclassify flowerclassify:1.3.0
```

*<u>注：以上仅为一个示例，详情请参考 [Docker](https://docs.docker.com/) 文档。</u>*


