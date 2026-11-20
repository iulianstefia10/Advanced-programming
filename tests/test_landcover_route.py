import io
import os
import sys
import unittest
from unittest import mock

from flask import Flask
from PIL import Image

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from routes.landcover import landcover_bp
from services import landcover as landcover_service


def create_test_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(APP_DIR, "templates"),
        static_folder=os.path.join(APP_DIR, "static"),
    )
    app.config["TESTING"] = True
    app.register_blueprint(landcover_bp)
    return app


def upload(data=b"x", name="map.png"):
    return {"image": (io.BytesIO(data), name)}


FAKE_RESULT = {
    "width": 200, "height": 130, "resized": False, "rows": 2, "columns": 3,
    "patch_size": 64, "patches": 6, "seconds": 0.1,
    "stats": [{"name": "Forest", "color": "#1b7837", "count": 6, "percent": 100.0}],
    "map_file": "landcover/abc_map.png", "npy_file": "landcover/abc_landcover.npy",
}


class LandcoverRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = create_test_app().test_client()

    def test_page_has_upload_form(self):
        response = self.client.get("/landcover/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'type="file"', response.data)
        self.assertIn(b'name="image"', response.data)
        self.assertIn(b"multipart/form-data", response.data)

    def test_page_has_patch_size_choice(self):
        response = self.client.get("/landcover/")
        self.assertIn(b'name="patch_size"', response.data)
        self.assertIn(b'value="128" selected', response.data)

    def test_patch_size_is_sent_to_the_service(self):
        with mock.patch.object(landcover_service, "analyse", return_value=FAKE_RESULT) as analyse:
            self.client.post("/landcover/predict",
                             data={**upload(), "patch_size": "64"},
                             content_type="multipart/form-data")
        self.assertEqual(analyse.call_args[0][3], "64")

    def test_page_warns_when_model_is_missing(self):
        with mock.patch.object(landcover_service, "model_available", return_value=False):
            response = self.client.get("/landcover/")
        self.assertIn(b"Model not found", response.data)

    def test_predict_without_file(self):
        response = self.client.post("/landcover/predict", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Choose a satellite screenshot", response.data)

    def test_predict_get_not_allowed(self):
        self.assertEqual(self.client.get("/landcover/predict").status_code, 405)

    def test_predict_shows_result(self):
        with mock.patch.object(landcover_service, "analyse", return_value=FAKE_RESULT):
            response = self.client.post("/landcover/predict", data=upload(),
                                        content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Forest", response.data)
        self.assertIn(b"landcover/abc_map.png", response.data)
        self.assertIn(b"abc_landcover.npy", response.data)

    def test_invalid_image_gives_400(self):
        with mock.patch.object(landcover_service, "analyse",
                               side_effect=landcover_service.InvalidImage("not an image")):
            response = self.client.post("/landcover/predict", data=upload(),
                                        content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not an image", response.data)

    def test_missing_model_gives_503(self):
        with mock.patch.object(landcover_service, "analyse",
                               side_effect=landcover_service.ModelNotFound("no model")):
            response = self.client.post("/landcover/predict", data=upload(),
                                        content_type="multipart/form-data")
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"no model", response.data)


if __name__ == "__main__":
    unittest.main()