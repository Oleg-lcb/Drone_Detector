import torch
import cv2 as cv

from ultralytics import YOLO
from torchvision import transforms


# Задействование cuda по доступности
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Загрузка предобученной модели
def load_model(model_path):
    model = YOLO(model_path).to(device)
    return model


def path_to_tensor(source):
    # Считывание изображения
    if isinstance(source, str):
        # Для режима изображения
        image = cv.imread(source)
    else:
        # Для кадра видео
        image = source

    if image is None:
        print("Could not read the image")
        return

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Приводим к нужному и пропорциональному для модели размеру
    height, width = image.shape[:2]
    new_width = 640
    ratio = new_width / width
    new_height = int(ratio * height)
    image = cv.resize(image, (new_width, new_height))

    # Трансформируем в тензор с добавлением размерности и переводом на доступное устройство обработки
    image_tensor = torch.unsqueeze(transform(image).to(device=device), dim=0)
    image_tensor_new = torch.zeros((1, 3, 640, 640))
    image_tensor_new[:image_tensor.shape[0], :image_tensor.shape[1], :image_tensor.shape[2], :image_tensor.shape[3]] = image_tensor
    return image_tensor_new
