# Tests for the /mlp page (PW2)
# Only the mlp blueprint is loaded and the training is replaced by a fake one,
# so the tests are fast and do not need TensorFlow or the database.

import os
import sys
import unittest
from unittest import mock

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from flask import Flask  # noqa: E402

from routes.mlp import mlp_bp  # noqa: E402
from services import mlp as mlp_service  # noqa: E402


def create_test_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(APP_DIR, "templates"),
        static_folder=os.path.join(APP_DIR, "static"),
    )
    app.config["TESTING"] = True
    app.register_blueprint(mlp_bp)
    return app


def fake_train(params, data=None, train_limit=None):
    return {
        "params": params,
        "test_loss": 0.1,
        "test_accuracy": 0.97,
        "num_params": 111146,
        "train_time": 1.0,
        "history": {"accuracy": [0.9, 0.95], "val_accuracy": [0.92, 0.96],
                    "loss": [0.3, 0.1], "val_loss": [0.25, 0.12]},
    }


def fake_study(parameter, base=None, train_limit=None):
    return [
        {"parameter": parameter, "value": v, "test_accuracy": 0.9 + i / 100,
         "test_loss": 0.2, "num_params": 1000, "train_time": 1.0}
        for i, v in enumerate(mlp_service.STUDY_VALUES[parameter])
    ]


@mock.patch.object(mlp_service, "run_sensitivity", side_effect=fake_study)
@mock.patch.object(mlp_service, "train_mlp", side_effect=fake_train)
class MlpRouteTests(unittest.TestCase):

    def setUp(self):
        self.client = create_test_app().test_client()

    def post(self, **fields):
        data = {
            "hidden_layers": "128,64,32",
            "activation": "relu",
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "batch_size": "128",
            "epochs": "5",
            "validation_split": "0.1",
            "normalize": "on",
            "study_param": "activation",
            "action": "train",
        }
        data.update(fields)
        return self.client.post("/mlp/", data=data)

    def test_page_shows_the_form(self, *_):
        response = self.client.get("/mlp/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for name in ("hidden_layers", "activation", "optimizer", "loss", "batch_size",
                     "epochs", "validation_split", "normalize", "study_param"):
            self.assertIn(f'name="{name}"', html)

    def test_train(self, train, _):
        html = self.post().get_data(as_text=True)
        self.assertIn("Result · MNIST", html)
        self.assertIn("0.9700", html)
        self.assertIn("111,146", html)
        train.assert_called_once()

    def test_form_keeps_the_chosen_values(self, *_):
        html = self.post(activation="tanh", epochs="7").get_data(as_text=True)
        self.assertIn('<option value="tanh" selected', html)
        self.assertIn('value="7"', html)

    def test_sensitivity_study(self, _, study):
        html = self.post(action="study", study_param="batch_size").get_data(as_text=True)
        self.assertIn("Sensitivity study · batch_size", html)
        for value in ("32", "128", "512"):
            self.assertIn(f"<th>{value}</th>", html)
        study.assert_called_once()

    def test_invalid_values_show_an_error(self, train, _):
        cases = {
            "no epochs": {"epochs": "0"},
            "negative batch": {"batch_size": "-5"},
            "letters in layers": {"hidden_layers": "a,b"},
            "split too big": {"validation_split": "1.5"},
            "sparse loss": {"loss": "sparse_categorical_crossentropy"},
            "unknown study": {"action": "study", "study_param": "dropout"},
        }
        for name, fields in cases.items():
            with self.subTest(name):
                response = self.post(**fields)
                self.assertEqual(response.status_code, 200)
                html = response.get_data(as_text=True)
                self.assertIn("Invalid parameters", html)
                self.assertNotIn("Result ·", html)
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
