# Tests for the PW4 report page (/pw4/) and the inference of the two exported models
# Run: python -m unittest discover -s tests -v
# The tests that run the models need OpenCV, they are skipped without it.

import io
import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from flask import Flask  # noqa: E402

from routes.pw4 import HARDWARE_RESULTS, pw4_bp  # noqa: E402
from services.pw4_inference import (  # noqa: E402
    CIFAR10_CLASSES,
    CNN_PATH,
    IMAGENET_CLASSES_PATH,
    MOBILENET_PATH,
    cifar10_input,
    imagenet_classes,
    mobilenet_input,
    softmax,
)

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def photo(width=400, height=300):
    """A simple coloured picture (sky, sea and a boat shape)."""
    image = Image.new("RGB", (width, height), (120, 170, 230))
    pen = ImageDraw.Draw(image)
    pen.rectangle([0, height // 2, width, height], fill=(30, 80, 140))
    pen.polygon([(100, 150), (300, 150), (260, 200), (140, 200)], fill=(240, 240, 240))
    return image


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class PreprocessingTest(unittest.TestCase):

    def test_model_files_exist(self):
        for path in (CNN_PATH, MOBILENET_PATH, IMAGENET_CLASSES_PATH):
            self.assertTrue(os.path.exists(path), path)

    def test_classes(self):
        self.assertEqual(len(CIFAR10_CLASSES), 10)
        self.assertEqual(len(imagenet_classes()), 1000)

    def test_cifar10_input_like_the_notebook(self):
        x = cifar10_input(photo())
        self.assertEqual(x.shape, (1, 3, 32, 32))
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) gives values between -1 and 1
        self.assertGreaterEqual(x.min(), -1.0)
        self.assertLessEqual(x.max(), 1.0)

    def test_mobilenet_input(self):
        x = mobilenet_input(photo(640, 480))
        self.assertEqual(x.shape, (1, 3, 224, 224))
        # white pixel with the ImageNet mean and std: (1 - 0.485) / 0.229
        white = mobilenet_input(Image.new("RGB", (300, 300), (255, 255, 255)))
        self.assertAlmostEqual(float(white[0, 0, 0, 0]), (1 - 0.485) / 0.229, places=4)

    def test_softmax(self):
        p = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
        self.assertEqual(int(p.argmax()), 2)


@unittest.skipUnless(HAS_CV2, "OpenCV is not installed")
class ModelsTest(unittest.TestCase):

    def test_both_models_give_top3_and_time(self):
        from services.pw4_inference import classify
        results = classify(photo())
        for key in ("cnn", "mobilenet"):
            top = results[key]["top"]
            self.assertEqual(len(top), 3)
            self.assertGreaterEqual(top[0]["probability"], top[1]["probability"])
            self.assertGreater(results[key]["time_ms"], 0)
        self.assertIn(results["cnn"]["top"][0]["label"], CIFAR10_CLASSES)
        self.assertIn(results["mobilenet"]["top"][0]["label"], imagenet_classes())


class Pw4PageTest(unittest.TestCase):

    def setUp(self):
        app = Flask(
            __name__,
            template_folder=os.path.join(APP_DIR, "templates"),
            static_folder=os.path.join(APP_DIR, "static"),
        )
        app.config["TESTING"] = True
        app.register_blueprint(pw4_bp)

        @app.route("/")
        def home():
            return "home"

        self.client = app.test_client()

    def post(self, data):
        return self.client.post("/pw4/", data=data, content_type="multipart/form-data").get_data(as_text=True)

    def test_report_is_on_the_page(self):
        html = self.client.get("/pw4/").get_data(as_text=True)
        self.assertEqual(html.count('class="hardware-row"'), len(HARDWARE_RESULTS))
        for text in ("What we had to fix", "Conclusion", "Colab TPU (v5e-1)", "torch.accelerator"):
            self.assertIn(text, html)
        self.assertIn('name="file"', html)
        self.assertIn(">\n                        Inference", html)

    def test_no_file_shows_error(self):
        self.assertIn("Choose an image first.", self.post({}))

    def test_text_file_shows_error(self):
        html = self.post({"file": (io.BytesIO(b"hello"), "notes.txt")})
        self.assertIn("The file is not a valid image.", html)

    @unittest.skipUnless(HAS_CV2, "OpenCV is not installed")
    def test_image_gives_two_predictions(self):
        html = self.post({"file": (png_bytes(photo()), "boat.png")})
        self.assertEqual(html.count('class="col-md-6 model-result"'), 2)
        self.assertEqual(html.count('class="time-ms"'), 2)


if __name__ == "__main__":
    unittest.main()
