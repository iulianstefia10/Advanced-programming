# Tests for the /cnn page
# Only the cnn blueprint is loaded, so no database is needed.
# The training is replaced by a mock: nothing is downloaded or trained here.

import os
import sys
import unittest
from unittest import mock

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from flask import Flask  # noqa: E402

from routes.cnn import cnn_bp  # noqa: E402
from services import cnn_service  # noqa: E402


def create_test_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(APP_DIR, "templates"),
        static_folder=os.path.join(APP_DIR, "static"),
    )
    app.config["TESTING"] = True
    app.register_blueprint(cnn_bp)
    return app


class CnnRouteTests(unittest.TestCase):

    def setUp(self):
        self.client = create_test_app().test_client()

    def test_page_is_displayed_with_the_form(self):
        response = self.client.get("/cnn/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("CIFAR-100", html)
        for field in ("epochs", "batch_size", "optimizer", "learning_rate",
                      "validation_split", "label_mode", "filters",
                      "kernel_size", "dense_units"):
            with self.subTest(field=field):
                self.assertIn(f'name="{field}"', html)

    def test_page_has_a_stop_button(self):
        html = self.client.get("/cnn/").get_data(as_text=True)
        self.assertIn('id="stop-button"', html)

    def test_valid_form_starts_the_training(self):
        with mock.patch.object(cnn_service, "start_training", return_value=True) as start:
            response = self.client.post("/cnn/train", data={
                "epochs": "2", "batch_size": "64", "optimizer": "adam",
                "learning_rate": "0.001", "validation_split": "0.2",
                "label_mode": "fine", "filters": "32", "kernel_size": "3",
                "dense_units": "256",
            })
        self.assertEqual(response.status_code, 302)          # redirected to the page
        self.assertTrue(response.headers["Location"].endswith("/cnn/"))
        start.assert_called_once()
        config = start.call_args[0][0]
        self.assertEqual(config["epochs"], 2)
        self.assertEqual(config["batch_size"], 64)

    def test_wrong_value_is_refused_and_nothing_starts(self):
        with mock.patch.object(cnn_service, "start_training") as start:
            response = self.client.post("/cnn/train", data={"epochs": "0"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("epochs", response.get_data(as_text=True))
        start.assert_not_called()

    def test_kernel_size_5_is_refused(self):
        with mock.patch.object(cnn_service, "start_training") as start:
            response = self.client.post("/cnn/train", data={"kernel_size": "5"})
        self.assertEqual(response.status_code, 400)
        start.assert_not_called()

    def test_training_already_running_gives_409(self):
        with mock.patch.object(cnn_service, "start_training", return_value=False):
            response = self.client.post("/cnn/train", data={"epochs": "2"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("already running", response.get_data(as_text=True))

    def test_train_does_not_answer_to_get(self):
        self.assertEqual(self.client.get("/cnn/train").status_code, 405)

    def test_status_returns_json(self):
        with mock.patch.object(cnn_service, "get_status", return_value={"status": "idle"}):
            response = self.client.get("/cnn/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "idle"})

    def test_status_gives_the_url_of_the_plot_when_finished(self):
        finished = {"status": "done", "plot_file": "cnn/history.png", "finished_at": 123.4}
        with mock.patch.object(cnn_service, "get_status", return_value=finished):
            data = self.client.get("/cnn/status").get_json()
        self.assertIn("/static/cnn/history.png", data["plot_url"])

    def test_stop_answers_with_the_result_of_the_request(self):
        for accepted in (True, False):
            with self.subTest(accepted=accepted):
                with mock.patch.object(cnn_service, "request_stop", return_value=accepted):
                    response = self.client.post("/cnn/stop")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json(), {"ok": accepted})


if __name__ == "__main__":
    unittest.main()