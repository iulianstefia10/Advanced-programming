# Object detection with a small ready YOLO model (YOLOv8n, COCO classes)
# The model is an ONNX file read by OpenCV, so PyTorch is not needed.

import base64
import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(APP_DIR, "yolov8n.onnx")
INPUT_SIZE = 640  # the model was exported for 640x640 images

# the 80 classes of the COCO dataset, in the order of the model
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

_net = None


def get_net():
    """Load the ONNX model once with OpenCV."""
    global _net
    if _net is None:
        import cv2
        _net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    return _net


def letterbox(image):
    """Resize to fit in 640x640 without changing the proportions, grey borders around."""
    scale = INPUT_SIZE / max(image.width, image.height)
    new_w, new_h = round(image.width * scale), round(image.height * scale)
    pad_x, pad_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
    square = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    square.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return np.array(square), scale, pad_x, pad_y


def decode_output(output, scale, pad_x, pad_y, width, height, conf_threshold, iou_threshold):
    """YOLOv8 output (1, 84, 8400): 4 box values + 80 class scores for 8400 positions."""
    import cv2

    predictions = output[0].T  # (8400, 84)
    scores = predictions[:, 4:]
    class_ids = scores.argmax(axis=1)
    confidences = scores.max(axis=1)

    keep = confidences >= conf_threshold
    boxes_xywh = predictions[keep, :4]
    class_ids, confidences = class_ids[keep], confidences[keep]

    # centre x, centre y, width, height in the 640 image -> corners in the original image
    boxes = []
    for cx, cy, w, h in boxes_xywh:
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        boxes.append([float(x1), float(y1), float(w / scale), float(h / scale)])

    # non-maximum suppression: keep one box per object
    indexes = cv2.dnn.NMSBoxes(boxes, confidences.tolist(), conf_threshold, iou_threshold)

    detections = []
    for i in np.array(indexes).flatten():
        x, y, w, h = boxes[i]
        detections.append({
            "label": COCO_CLASSES[int(class_ids[i])],
            "confidence": round(float(confidences[i]) * 100, 1),
            "box": [
                max(0, round(x)), max(0, round(y)),
                min(width, round(x + w)), min(height, round(y + h)),
            ],
        })
    return sorted(detections, key=lambda d: -d["confidence"])


def detect(image, conf_threshold=0.25, iou_threshold=0.45):
    """Run YOLO on a PIL image and return the list of detected objects."""
    import cv2

    if not 0 < conf_threshold < 1:
        raise ValueError("The confidence threshold must be between 0 and 1.")

    image = ImageOps.exif_transpose(image).convert("RGB")  # phone photos can be rotated
    pixels, scale, pad_x, pad_y = letterbox(image)
    blob = cv2.dnn.blobFromImage(pixels, scalefactor=1 / 255.0, size=(INPUT_SIZE, INPUT_SIZE), swapRB=False)

    net = get_net()
    net.setInput(blob)
    output = net.forward()

    detections = decode_output(output, scale, pad_x, pad_y, image.width, image.height,
                               conf_threshold, iou_threshold)
    return image, detections


def draw_boxes(image, detections):
    """Draw the boxes and the labels, return the image as base64 JPEG."""
    image = image.copy()
    pen = ImageDraw.Draw(image)
    line = max(2, round(max(image.size) / 300))
    try:
        font = ImageFont.load_default(size=max(14, round(max(image.size) / 45)))
    except TypeError:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        text = f"{d['label']} {d['confidence']:.0f}%"
        pen.rectangle([x1, y1, x2, y2], outline=(255, 48, 79), width=line)
        left, top, right, bottom = pen.textbbox((x1, y1), text, font=font)
        label_top = max(0, y1 - (bottom - top) - 6)
        pen.rectangle([x1, label_top, x1 + (right - left) + 8, label_top + (bottom - top) + 6], fill=(255, 48, 79))
        pen.text((x1 + 4, label_top + 1), text, fill="white", font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
