import cv2
import os
import torch
import numpy
import json
import timeit

from model import ClassifyNet
from torch.utils import data
from torchvision import transforms
from torchvision import datasets

class TensorTransformer:

    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __call__(self, image: cv2.Mat) -> torch.Tensor:
        return self.to_tensor(image)

    def letterbox(self, image: cv2.Mat, new_size: int = 224, background_color: int = 127) -> cv2.Mat:
        aspect_ratio = image.shape[1] / image.shape[0]
        if image.shape[1] > image.shape[0]:
            image_resize = cv2.resize(image, (new_size, int(new_size / aspect_ratio)))
        else:
            image_resize = cv2.resize(image, (int(new_size * aspect_ratio), new_size))

        background = numpy.ones((new_size, new_size, 3), dtype = numpy.uint8) * background_color
        x = (new_size - image_resize.shape[1]) // 2
        y = (new_size - image_resize.shape[0]) // 2
        background[y:y + image_resize.shape[0], x:x + image_resize.shape[1]] = image_resize
        return background
    
    def to_tensor(self, image: cv2.Mat) -> torch.Tensor:
        image_tensor = self.transform(self.letterbox(image))
        image_tensor = torch.unsqueeze(image_tensor, dim = 0)
        return image_tensor
    
class FlowerClassifier:

    def __init__(self, model_path: str = "./weights/ClassifyNet-Best.pt", classes_path: str = "./dataset/classes.json") -> None:
        self.model = ClassifyNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.classes_dict = json.load(open(classes_path, "r"))
        self.transformer = TensorTransformer()

    def __call__(self, image: cv2.Mat) -> dict[str, str]:
        name, confidence = self.predict(image)
        return dict(name = name, confidence = f"{confidence:.3f}")

    def predict(self, image: cv2.Mat) -> tuple[str, float]:
        input_tensor = self.transformer(image)
        with torch.no_grad():
            outputs = torch.squeeze(self.model(input_tensor))
            predict = torch.softmax(outputs, dim = 0)
            classes_index = torch.argmax(predict).numpy()
        return self.classes_dict[str(classes_index)], predict[classes_index].item()

    def test_testing_set(self) -> None:
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_set = datasets.ImageFolder(root = "./dataset/test", transform = test_transform)
        test_length = len(test_set)
        test_loader = data.DataLoader(test_set)
        test_accuracy = 0

        with torch.no_grad():
            for step, (inputs, labels) in enumerate(test_loader, start = 0):
                outputs = self.model(inputs)
                predict = torch.max(outputs, dim = 1)[1]
                test_accuracy += (predict == labels).sum().item()
                print(f"\rProgress: [{step}/{len(test_loader)}]", end = "")

        print(f"\tAccuracy: {test_accuracy / test_length:.3f}")

    def test_image_dir(self, images_path: str = "./images", result_path: str = "./results") -> None:
        image_name_list = os.listdir(images_path)
        total_times = 0
        for image_name in image_name_list:
            image = cv2.cvtColor(cv2.imread(f"{images_path}/{image_name}"), cv2.COLOR_BGR2RGB)
            start_times = timeit.default_timer()
            name, confidence = self.predict(image)
            delta_times = timeit.default_timer() - start_times
            total_times += delta_times
            cv2.putText(image, name, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imwrite(f"{result_path}/result_{image_name}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Image: {image_name:<12}Name: {name:<15}Confidence: {confidence:<8.3f}Times: {delta_times:.3f}s")

        print(f"\nAverage Times: {total_times / len(image_name_list):.3f}s")

    def test(self) -> None:
        self.test_testing_set()
        self.test_image_dir()

if __name__ == "__main__":
    FlowerClassifier().test()
