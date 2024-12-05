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

The default configuration file for this project's Web service is <u>configs/deploy.yaml</u>, where each attribute has the following meaning:

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
flask --app inferences.application run --host="0.0.0.0" --port=9500
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
   batch-size: 32                             # Batch size
   num-classes: 10                            # Number of model classification categories
   num-workers: 0                             # Number of data loading subprocesses
   ```

### Model Inference and Deployment

To deploy the model, you need to convert the trained model to ONNX format, Converting PyTorch models to ONNX format is relatively simple. To use Docker for containerized deployment:

```bash
# Building an image
cd FlowerClassify
docker build -t flowerclassify:1.3.0 -f docker/Dockerfile .

# Create a container and run it
docker run --rm -p 9500:9500 --name flowerclassify flowerclassify:1.3.0
```

*<u>Note: The above is just an example, please refer to the [Docker](https://docs.docker.com/) documentation for more details.</u>*


