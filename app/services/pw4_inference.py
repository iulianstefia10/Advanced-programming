# PW4 - run our exported CIFAR-10 CNN and a pretrained MobileNetV2 on one image
# Both models are ONNX files read by OpenCV, so the site does not need PyTorch.

import os
import time

import numpy as np
from PIL import Image, ImageOps

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CNN_PATH = os.path.join(APP_DIR, "pw4_cifar10_cnn.onnx")
MOBILENET_PATH = os.path.join(APP_DIR, "pw4_mobilenet_v2.onnx")
IMAGENET_CLASSES_PATH = os.path.join(APP_DIR, "pw4_imagenet_classes.txt")

CIFAR10_CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

_nets = {}
_imagenet_classes = None


def get_net(path):
    """Load an ONNX model once with OpenCV."""
    if path not in _nets:
        import cv2
        _nets[path] = cv2.dnn.readNetFromONNX(path)
    return _nets[path]


def imagenet_classes():
    global _imagenet_classes
    if _imagenet_classes is None:
        with open(IMAGENET_CLASSES_PATH, encoding="utf-8") as f:
            _imagenet_classes = [line.strip() for line in f if line.strip()]
    return _imagenet_classes


def center_square(image):
    """Cut the biggest square in the middle of the photo."""
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def cifar10_input(image):
    # like in the notebook: 32x32, ToTensor (0-1), Normalize(0.5, 0.5) -> values between -1 and 1
    small = center_square(image).resize((32, 32), Image.BILINEAR)
    x = np.asarray(small, dtype="float32") / 255.0
    x = (x - 0.5) / 0.5
    return x.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 32, 32)


def mobilenet_input(image):
    # torchvision preprocessing: shorter side 256, centre crop 224, ImageNet mean and std
    scale = 256 / min(image.size)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    crop = resized.crop((left, top, left + 224, top + 224))
    x = np.asarray(crop, dtype="float32") / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array([0.229, 0.224, 0.225], dtype="float32")
    return x.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def run_model(path, model_input, classes, repeats=10):
    """Prediction (top 3) and mean inference time on the server CPU."""
    net = get_net(path)
    net.setInput(model_input)
    net.forward()  # first run is slower (warm-up), not counted

    start = time.perf_counter()
    for _ in range(repeats):
        net.setInput(model_input)
        logits = net.forward()
    time_ms = (time.perf_counter() - start) / repeats * 1000

    probabilities = softmax(logits.flatten())
    top = probabilities.argsort()[::-1][:3]
    return {
        "top": [{"label": classes[i], "probability": round(float(probabilities[i]) * 100, 1)} for i in top],
        "time_ms": round(time_ms, 2),
    }


def classify(image):
    """Run the two models on the same image."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    return {
        "cnn": dict(
            run_model(CNN_PATH, cifar10_input(image), CIFAR10_CLASSES),
            name="Our PW4 CNN (CIFAR-10, exported to ONNX)",
            info="Trained by us, 2 epochs, 10 classes, 32×32 input, about 62,000 parameters",
        ),
        "mobilenet": dict(
            run_model(MOBILENET_PATH, mobilenet_input(image), imagenet_classes()),
            name="MobileNetV2 (pretrained on ImageNet)",
            info="Pretrained model from torchvision, 1000 classes, 224×224 input, about 3.5 million parameters",
        ),
    }
