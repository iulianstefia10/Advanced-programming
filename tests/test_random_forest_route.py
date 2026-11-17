# Tests for the /rf page
# Only the rf blueprint is loaded, so no database is needed

import os
import sys
import unittest

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from flask import Flask  # noqa: E402

from routes.random_forest import rf_bp  # noqa: E402


def create_test_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(APP_DIR, "templates"),
        static_folder=os.path.join(APP_DIR, "static"),
    )
    app.config["TESTING"] = True
    app.register_blueprint(rf_bp)
    return app


class RandomForestRouteTests(unittest.TestCase):

    def setUp(self):
        self.client = create_test_app().test_client()

    def post(self, **fields):
        data = {
            "dataset": "iris",
            "test_size": "0.2",
            "n_estimators": "10",
            "max_depth": "",
            "random_state": "42",
            "action": "train",
        }
        data.update(fields)
        return self.client.post("/rf/", data=data)

    def test_page_shows_the_form(self):
        response = self.client.get("/rf/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for name in ("dataset", "test_size", "n_estimators", "max_depth", "random_state"):
            self.assertIn(f'name="{name}"', html)
        for dataset in ("iris", "digits", "diabetes"):
            self.assertIn(f'value="{dataset}"', html)

    def test_train_classification(self):
        response = self.post(dataset="digits")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Result · digits", html)
        self.assertIn("Accuracy on the test set", html)
        self.assertIn("precision", html)

    def test_train_diabetes_shows_r2(self):
        html = self.post(dataset="diabetes").get_data(as_text=True)
        self.assertIn("R² on the test set", html)
        self.assertIn("MSE", html)
        self.assertNotIn("Accuracy on the test set", html)

    def test_form_keeps_the_chosen_values(self):
        html = self.post(dataset="digits", n_estimators="25").get_data(as_text=True)
        self.assertIn('<option value="digits" selected', html)
        self.assertIn('value="25"', html)

    def test_max_depth_is_shown(self):
        html = self.post(max_depth="3").get_data(as_text=True)
        self.assertIn("max_depth 3", html)

    def test_sensitivity_study(self):
        html = self.post(dataset="iris", action="study").get_data(as_text=True)
        self.assertIn("Sensitivity study · iris", html)
        self.assertIn("test_size \\ n_estimators", html)
        self.assertEqual(html.count("<tr>"), 6)

    def test_invalid_values_show_an_error(self):
        cases = {
            "test_size too big": {"test_size": "2"},
            "test_size not a number": {"test_size": "abc"},
            "no trees": {"n_estimators": "0"},
            "too many trees": {"n_estimators": "1000"},
            "max_depth zero": {"max_depth": "0"},
            "unknown dataset": {"dataset": "mnist"},
        }
        for name, fields in cases.items():
            with self.subTest(name):
                response = self.post(**fields)
                self.assertEqual(response.status_code, 200)
                html = response.get_data(as_text=True)
                self.assertIn("Invalid parameters", html)
                self.assertNotIn("Result ·", html)


if __name__ == "__main__":
    unittest.main()
