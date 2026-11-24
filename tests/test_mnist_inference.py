# Tests for the MNIST preprocessing and the /mnist/ page (CNN + MLP)
# Run: python -m unittest discover -s tests -v
# The tests that run the models need TensorFlow, they are skipped without it.

import base64
import io
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from flask import Flask  # noqa: E402

from extensions import db  # noqa: E402
from models.app_data import StoredModel  # noqa: E402
from routes.mnist import mnist_bp  # noqa: E402
from services.mnist_inference import (  # noqa: E402
    cnn_input,
    decode_data_url,
    mlp_input,
    predict_both,
    prepare_image,
)
from services.model_store import seed_default_models  # noqa: E402

try:
    import tensorflow.keras  # noqa: F401
    HAS_TF = True
except ImportError:
    HAS_TF = False


def draw_seven(mode="L", size=280):
    """A black 7 on a white background, like a drawing on the canvas."""
    if mode == "RGBA":
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        color = (0, 0, 0, 255)
    else:
        image = Image.new("L", (size, size), 255)
        color = 0
    pen = ImageDraw.Draw(image)
    pen.line([(70, 60), (210, 60), (120, 230)], fill=color, width=18)
    return image


def to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class PreprocessingTest(unittest.TestCase):

    def test_output_is_28x28_between_0_and_1(self):
        pixels = prepare_image(draw_seven())
        self.assertEqual(pixels.shape, (28, 28))
        self.assertGreaterEqual(pixels.min(), 0.0)
        self.assertLessEqual(pixels.max(), 1.0)

    def test_white_digit_on_black_background(self):
        pixels = prepare_image(draw_seven())
        # the corners are background, so they must be black like in MNIST
        self.assertEqual(pixels[0, 0], 0.0)
        self.assertEqual(pixels[27, 27], 0.0)
        self.assertGreater(pixels.max(), 0.5)

    def test_transparent_background_works(self):
        # the canvas and many PNG files have a transparent background
        a = prepare_image(draw_seven("L"))
        b = prepare_image(draw_seven("RGBA"))
        self.assertTrue(np.allclose(a, b, atol=0.05))

    def test_digit_is_centred(self):
        pixels = prepare_image(draw_seven())
        ys, xs = np.indices(pixels.shape)
        self.assertAlmostEqual((ys * pixels).sum() / pixels.sum(), 13.5, delta=1)
        self.assertAlmostEqual((xs * pixels).sum() / pixels.sum(), 13.5, delta=1)

    def test_empty_image_raises(self):
        with self.assertRaises(ValueError):
            prepare_image(Image.new("L", (100, 100), 255))

    def test_model_input_shapes(self):
        pixels = prepare_image(draw_seven())
        self.assertEqual(cnn_input(pixels).shape, (1, 28, 28, 1))
        self.assertEqual(mlp_input(pixels).shape, (1, 784))
        self.assertLessEqual(mlp_input(pixels).max(), 1.0)

    def test_data_url(self):
        image = decode_data_url(to_data_url(draw_seven()))
        self.assertEqual(image.size, (280, 280))
        for bad in ("", "nothing", "data:image/png;base64,AAAA"):
            with self.assertRaises(ValueError):
                decode_data_url(bad)


@unittest.skipUnless(HAS_TF, "TensorFlow is not installed")
class ModelsTest(unittest.TestCase):

    def test_both_models_recognise_mnist_test_images(self):
        from tensorflow.keras.datasets import mnist
        (_, _), (X_test, y_test) = mnist.load_data()
        correct = {"cnn": 0, "mlp": 0}
        for i in range(50):
            # MNIST image turned into a big black-on-white drawing
            image = Image.fromarray(255 - X_test[i]).resize((280, 280))
            result = predict_both(prepare_image(image))
            for name in correct:
                correct[name] += result[name]["digit"] == y_test[i]
        self.assertGreaterEqual(correct["cnn"], 47)
        self.assertGreaterEqual(correct["mlp"], 45)

    def test_probabilities(self):
        result = predict_both(prepare_image(draw_seven()))
        for name in ("cnn", "mlp"):
            self.assertEqual(len(result[name]["probabilities"]), 10)
            self.assertAlmostEqual(sum(result[name]["probabilities"]), 100, delta=0.5)
            self.assertEqual(result[name]["digit"], 7)


class MnistPageTest(unittest.TestCase):

    def setUp(self):
        app = Flask(
            __name__,
            template_folder=os.path.join(APP_DIR, "templates"),
            static_folder=os.path.join(APP_DIR, "static"),
        )
        app.config["TESTING"] = True
        # the page reads the models from the database: use an in-memory SQLite
        # for both the default database and the "appdata" one
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
        app.config["SQLALCHEMY_BINDS"] = {"appdata": "sqlite://"}
        db.init_app(app)
        # uploads go to a temporary folder, not to the real static/uploads
        self.tmp = tempfile.TemporaryDirectory()
        app.root_path = self.tmp.name
        app.register_blueprint(mnist_bp)

        @app.route("/")
        def home():
            return "home"

        with app.app_context():
            db.create_all(bind_key="appdata")
            seed_default_models()

        self.app = app
        self.client = app.test_client()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
        self.tmp.cleanup()

    def test_default_models_are_seeded(self):
        with self.app.app_context():
            count = db.session.query(StoredModel).count()
        self.assertEqual(count, 2)

    def test_page_lists_models_from_the_database(self):
        with self.app.app_context():
            names = [m.name for m in db.session.scalars(db.select(StoredModel)).all()]
        html = self.client.get("/mnist/").get_data(as_text=True)
        for name in names:
            self.assertIn(name, html)

    def test_page_has_canvas_and_upload(self):
        html = self.client.get("/mnist/").get_data(as_text=True)
        self.assertIn('id="drawCanvas"', html)
        self.assertIn('name="drawing"', html)
        self.assertIn('name="file"', html)

    def test_nothing_sent_shows_error(self):
        html = self.client.post("/mnist/", data={}).get_data(as_text=True)
        self.assertIn("Draw a digit or choose an image file first.", html)

    def test_broken_drawing_shows_error(self):
        html = self.client.post("/mnist/", data={"drawing": "data:image/png;base64,AAAA"}).get_data(as_text=True)
        self.assertIn("The drawing could not be read.", html)

    def test_text_file_is_refused_and_not_saved(self):
        data = {"file": (io.BytesIO(b"not an image"), "notes.txt")}
        html = self.client.post("/mnist/", data=data, content_type="multipart/form-data").get_data(as_text=True)
        self.assertIn("The file is not a valid image.", html)
        self.assertEqual(os.listdir(os.path.join(self.tmp.name, "static", "uploads")), [])

    @unittest.skipUnless(HAS_TF, "TensorFlow is not installed")
    def test_drawing_gives_two_predictions(self):
        html = self.client.post("/mnist/", data={"drawing": to_data_url(draw_seven("RGBA"))}).get_data(as_text=True)
        self.assertEqual(html.count('class="mb-4 prediction-block"'), 2)
        self.assertEqual(html.count('line-height: 1;">7<'), 2)

    @unittest.skipUnless(HAS_TF, "TensorFlow is not installed")
    def test_upload_gives_two_predictions_and_is_saved(self):
        buffer = io.BytesIO()
        draw_seven().save(buffer, format="PNG")
        buffer.seek(0)
        data = {"file": (buffer, "seven.png")}
        html = self.client.post("/mnist/", data=data, content_type="multipart/form-data").get_data(as_text=True)
        self.assertEqual(html.count('class="mb-4 prediction-block"'), 2)
        self.assertIn("seven.png", os.listdir(os.path.join(self.tmp.name, "static", "uploads")))


if __name__ == "__main__":
    unittest.main()