import json
import os
import statistics
import time

import cv2
import torch
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import ClassifyNet
from utils import ImageUtils
from utils import PlottingUtils

class FlowerClassifier:

    def __init__(self, model_path: str = "./weights/ClassifyNet-Best.pt", classes_path: str = "./dataset/classes.json") -> None:
        self.model = ClassifyNet()
        self.predict_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.testset_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        with open(classes_path, mode="r", encoding="utf-8") as classes_file:
            self.classes_dict = json.load(classes_file)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def __call__(self, image: cv2.Mat) -> dict[str, str]:
        return self.predict_dict(image)

    def predict(self, image: cv2.Mat) -> tuple[str, float]:
        inputs = self.predict_transfrom(ImageUtils.letterbox(image))
        inputs = torch.unsqueeze(inputs, dim=0)

        with torch.no_grad():
            outputs = torch.squeeze(self.model(inputs))
            predict = torch.softmax(outputs, dim=0)
            classes_index = torch.argmax(predict).item()
            
        return self.classes_dict[str(classes_index)], predict[classes_index].item()
    
    def predict_dict(self, image: cv2.Mat) -> dict[str, str]:
        name, confidence = self.predict(image)
        return dict(name=name, confidence=f"{confidence:.3f}")

    def test_dataset(self) -> None:
        correct_count = 0
        test_dataset = ImageFolder(root="./dataset/test", transform=self.testset_transform)
        test_loader = DataLoader(test_dataset)

        with torch.no_grad():
            for step, (inputs, labels) in enumerate(test_loader, start=0):
                outputs = self.model(inputs)
                predict = torch.max(outputs, dim=1)[1]
                correct_count += (predict == labels).sum().item()
                print(f"\rProgress: [{step}/{len(test_loader)}]", end="")

        print(f"\tAccuracy: {correct_count / len(test_dataset):.3f}")

    def test_images_directory(self, images_path: str = "./images", result_path: str = "./results") -> None:
        cost_times = list()
        for image_name in os.listdir(images_path):
            image = cv2.cvtColor(cv2.imread(f"{images_path}/{image_name}"), cv2.COLOR_BGR2RGB)

            entry_time = time.perf_counter()
            name, confidence = self.predict(image)
            leave_time = time.perf_counter()

            cost_time = leave_time - entry_time
            cost_times.append(cost_time)

            cv2.imwrite(f"{result_path}/result_{image_name}", cv2.cvtColor(PlottingUtils.label(image, name), cv2.COLOR_RGB2BGR))
            print(f"Image: {image_name:<12} Name: {name:<15} Confidence: {confidence:<8.3f} Times: {cost_time:.3f}s")

        print(f"Average Times: {statistics.mean(cost_times):.3f}s")

    def test(self) -> None:
        self.test_dataset()
        self.test_images_directory()


if __name__ == "__main__":
    FlowerClassifier().test()
