# Tests for the YOLO object detection (service and /yolo/ page)
# Run: python -m unittest discover -s tests -v
# The tests that run the model need OpenCV, they are skipped without it.

import io
import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from flask import Flask  # noqa: E402

from routes.yolo import yolo_bp  # noqa: E402
from services.yolo import (  # noqa: E402
    COCO_CLASSES,
    INPUT_SIZE,
    MODEL_PATH,
    decode_output,
    letterbox,
)

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def fake_output(boxes):
    """A YOLOv8 output (1, 84, 8400) with the given (cx, cy, w, h, class, score) boxes."""
    output = np.zeros((1, 84, 8400), dtype="float32")
    for i, (cx, cy, w, h, class_id, score) in enumerate(boxes):
        output[0, :4, i] = [cx, cy, w, h]
        output[0, 4 + class_id, i] = score
    return output


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class ServiceTest(unittest.TestCase):

    def test_model_file_and_classes(self):
        self.assertTrue(os.path.exists(MODEL_PATH))
        self.assertEqual(len(COCO_CLASSES), 80)
        self.assertEqual(COCO_CLASSES[0], "person")

    def test_letterbox_keeps_the_proportions(self):
        pixels, scale, pad_x, pad_y = letterbox(Image.new("RGB", (1280, 720)))
        self.assertEqual(pixels.shape, (INPUT_SIZE, INPUT_SIZE, 3))
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, (640 - 360) // 2)

    @unittest.skipUnless(HAS_CV2, "OpenCV is not installed")
    def test_decode_boxes_back_to_the_photo(self):
        # photo 1280x720 -> scale 0.5, 140 px of grey band on top
        output = fake_output([(320, 320, 100, 50, 0, 0.9)])
        detections = decode_output(output, 0.5, 0, 140, 1280, 720, 0.25, 0.45)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "person")
        self.assertEqual(detections[0]["confidence"], 90.0)
        self.assertEqual(detections[0]["box"], [540, 310, 740, 410])

    @unittest.skipUnless(HAS_CV2, "OpenCV is not installed")
    def test_threshold_and_duplicate_boxes(self):
        output = fake_output([
            (320, 320, 100, 100, 5, 0.8),   # bus
            (322, 321, 100, 100, 5, 0.7),   # almost the same bus -> removed by NMS
            (100, 100, 50, 50, 16, 0.1),    # dog below the threshold -> removed
        ])
        detections = decode_output(output, 1.0, 0, 0, 640, 640, 0.25, 0.45)
        self.assertEqual([d["label"] for d in detections], ["bus"])


@unittest.skipUnless(HAS_CV2, "OpenCV is not installed")
class ModelTest(unittest.TestCase):

    def test_empty_picture_has_no_object(self):
        from services.yolo import detect
        _, detections = detect(Image.new("RGB", (640, 480), (200, 200, 200)))
        self.assertEqual(detections, [])

    def test_wrong_threshold_raises(self):
        from services.yolo import detect
        for threshold in (0, 1, 1.5):
            with self.assertRaises(ValueError):
                detect(Image.new("RGB", (64, 64)), conf_threshold=threshold)


class YoloPageTest(unittest.TestCase):

    def setUp(self):
        app = Flask(
            __name__,
            template_folder=os.path.join(APP_DIR, "templates"),
            static_folder=os.path.join(APP_DIR, "static"),
        )
        app.config["TESTING"] = True
        app.register_blueprint(yolo_bp)

        @app.route("/")
        def home():
            return "home"

        self.client = app.test_client()

    def post(self, data):
        return self.client.post("/yolo/", data=data, content_type="multipart/form-data").get_data(as_text=True)

    def test_page_has_the_form(self):
        html = self.client.get("/yolo/").get_data(as_text=True)
        self.assertIn('name="file"', html)
        self.assertIn('name="confidence"', html)

    def test_no_file_shows_error(self):
        self.assertIn("Choose a photo first.", self.post({"confidence": "25"}))

    def test_text_file_shows_error(self):
        html = self.post({"confidence": "25", "file": (io.BytesIO(b"hello"), "notes.txt")})
        self.assertIn("The file is not a valid image.", html)

    def test_bad_threshold_shows_error(self):
        html = self.post({"confidence": "abc", "file": (png_bytes(Image.new("RGB", (64, 64))), "a.png")})
        self.assertIn("must be a number", html)

    @unittest.skipUnless(HAS_CV2, "OpenCV is not installed")
    def test_photo_gives_a_result_image(self):
        image = Image.new("RGB", (400, 300), (30, 120, 200))
        ImageDraw.Draw(image).ellipse([150, 100, 250, 200], fill=(240, 200, 40))
        html = self.post({"confidence": "25", "file": (png_bytes(image), "photo.png")})
        self.assertIn('id="yoloResult"', html)
        self.assertNotIn(">Error<", html)


if __name__ == "__main__":
    unittest.main()
