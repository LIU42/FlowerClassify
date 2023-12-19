import cv2
import os

folder_path = "./valid/Lotus/"
image_list = os.listdir(folder_path)

for image_path in image_list:
    image = cv2.imread(folder_path + image_path)
    width = image.shape[1]
    height = image.shape[0]

    if width > height:
        size = height
        top = 0
        left = (width - size) // 2
    else:
        size = width
        left = 0
        top = (height - size) // 2

    new_image = image[top:top + size, left:left + size :]
    new_image = cv2.resize(new_image, (224, 224))
    cv2.imwrite(folder_path + image_path, new_image)
    