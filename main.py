import argparse
import os.path

from src.detection import DroneDetection
from src.utils import load_model


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='DroneDetection')
    # parser.add_argument('--source', type=str, required=True)
    # args = parser.parse_args()

    # Путь к модели
    model_path = os.path.abspath('models/best.pt')
    detector = DroneDetection(model_path=model_path)

    # if args.source == '0':
    #     results = detector.detect_image(args.source)

    detector.detect_image(source="123.jpg")

