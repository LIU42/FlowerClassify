# Flower Classify

**基于 ResNet 的花卉分类识别系统，能有效区分 10 中不同类别的花卉。**



#### 项目简介

本项目为一个基础的花卉分类识别系统，采用 ResNet18 作为主干网络，包含模型的训练、测试以及线上部署（提供容器化部署）。

- 基于 PyTorch 框架进行模型的训练及测试。

- 模型采用 ONNX 格式部署，采用 ONNX Runtime 进行推理。

- 基于 Flask 框架实现 Web 接口。

- 使用 Docker 进行容器化部署。

*<u>训练数据集来自 [Kaggle](https://www.kaggle.com/)，融合了多个数据集并进行了数据清洗，基于预训练模型进行训练，在当前数据集下准确率超过 98%。</u>*

 

#### 使用说明

首先安装相关的依赖：

```shell
pip install -r requirements.txt
```

将模型权重文件放入 weights/ 目录后，执行以下命令启动 Web 服务：

```shell
flask --app main run --host="0.0.0.0" --port=9500
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



若要使用自己的数据集训练模型，准备好数据集、调整好类别配置文件 classes.json 和模型输出格式后：

1. 运行 model.py 自动下载预训练模型权重并保存。

2. 根据需要调整 train.py 中的各项参数并运行即可开始训练。

3. 运行 test.py 以测试当前最优模型在测试集上的准确率（可选）。

4. 运行 export.py 将模型导出为 ONNX 格式用于部署。
   
   

若要使用 Docker 进行容器化部署：

```shell
# 构建镜像
cd FlowerClassify
docker build -t flowerclassify:1.1.0 -f docker/Dockerfile .

# 创建容器并运行
docker run -it --rm -p 9500:9500 --name flowerclassify flowerclassify:1.1.0
```

*<u>以上仅为一个示例，详情请参考 [Docker](https://docs.docker.com/) 文档。</u>*
