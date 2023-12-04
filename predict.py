import cv2
import os
import torch
import numpy
import json

from torch import Tensor
from torch.utils import data
from torchvision import transforms
from torchvision import datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes_file = open("./dataset/classes.json", "r")
classes_dict = json.load(classes_file)

model = torch.load("./weights/AlexNet-Best.pt")
model.eval()

def letterbox(image: cv2.Mat, size: int) -> cv2.Mat:

    image_width = image.shape[1]
    image_height = image.shape[0]
    aspect_ratio = image_width / image_height

    if image_width > image_height:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))
    canvas = numpy.ones((size, size, 3), dtype = numpy.uint8) * 127

    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width, :] = resized_image

    return canvas

def transform_tensor(image: cv2.Mat) -> Tensor:

    image_tensor = transform(image)
    image_tensor = torch.unsqueeze(image_tensor, dim = 0)

    return image_tensor

def predict_image(image: cv2.Mat) -> dict:

    image_letter = letterbox(image, 224)
    input_tensor = transform_tensor(image_letter)

    with torch.no_grad():
        outputs = torch.squeeze(model(input_tensor))
        predict = torch.softmax(outputs, dim = 0)
        class_index = torch.argmax(predict).numpy()

    return { "name": classes_dict[str(class_index)], "confidence": "{:.3f}".format(predict[class_index].item()) }

def predict_test_set() -> None:

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_set = datasets.ImageFolder(root = "./dataset/test", transform = test_transform)
    test_len = len(test_set)
    test_loader = data.DataLoader(test_set)
    test_accuracy = 0

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_loader, start = 0):
            outputs = model(inputs)
            predict = torch.max(outputs, dim = 1)[1]
            test_accuracy += (predict == labels).sum().item()

            print("\rProgress: [{}/{}]".format(step, len(test_loader)), end = "")

    print("\tAccuracy: {:.3f}".format(test_accuracy / test_len))

def predict_images_folder() -> None:

    print("\n---------- Predict Images Start ----------\n")

    images_path = "./images/"
    result_path = "./results/"
    images_list = os.listdir(images_path)

    for image_name in images_list:
        image_origin = cv2.imread(images_path + image_name)
        image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        result_dict = predict_image(image_origin)
        image_result = cv2.putText(image_origin, result_dict["name"], (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)

        cv2.imwrite(result_path + "result_" + image_name, image_result)
        print("Image: {:<12}Name: {:<15}Confidence: {}".format(image_name, result_dict["name"], result_dict["confidence"]))

    print("\n---------- Predict Images Finished ----------\n")

if __name__ == "__main__":

    predict_test_set()
    predict_images_folder()
