# Tests for the PW2 service (MNIST MLP)
# The validation tests do not need TensorFlow.
# The model tests are skipped if TensorFlow is not installed.

import importlib.util
import os
import sys
import unittest

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from services import mlp  # noqa: E402

HAS_TF = importlib.util.find_spec("tensorflow") is not None

GOOD_FORM = {
    "hidden_layers": "128,64,32",
    "activation": "relu",
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "batch_size": "128",
    "epochs": "5",
    "validation_split": "0.1",
    "normalize": "on",
}


class ValidationTests(unittest.TestCase):

    def test_parse_hidden_layers(self):
        self.assertEqual(mlp.parse_hidden_layers("128,64,32"), [128, 64, 32])
        self.assertEqual(mlp.parse_hidden_layers(" 32 , 64 "), [32, 64])
        self.assertEqual(mlp.parse_hidden_layers(""), [])
        self.assertEqual(mlp.parse_hidden_layers("0"), [])

    def test_parse_hidden_layers_rejects_bad_input(self):
        for bad in ("abc", "128,,64", "-5", "128,0", "1.5", "99999"):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    mlp.parse_hidden_layers(bad)

    def test_good_form(self):
        params, errors = mlp.validate_params(GOOD_FORM)
        self.assertEqual(errors, [])
        self.assertEqual(params["batch_size"], 128)
        self.assertEqual(params["epochs"], 5)
        self.assertEqual(params["validation_split"], 0.1)
        self.assertTrue(params["normalize"])

    def test_unticked_checkbox_means_no_normalization(self):
        form = dict(GOOD_FORM)
        form.pop("normalize")
        params, errors = mlp.validate_params(form)
        self.assertEqual(errors, [])
        self.assertFalse(params["normalize"])

    def test_bad_values_are_rejected(self):
        cases = [
            ("epochs", "0"), ("epochs", "-3"), ("epochs", "abc"), ("epochs", "1000"),
            ("batch_size", "-5"), ("batch_size", "0"),
            ("validation_split", "1.5"), ("validation_split", "-0.1"), ("validation_split", "x"),
            ("activation", "softmaxx"), ("optimizer", "foo"),
            ("loss", "sparse_categorical_crossentropy"), ("hidden_layers", "a,b"),
        ]
        for field, value in cases:
            with self.subTest(f"{field}={value}"):
                params, errors = mlp.validate_params(dict(GOOD_FORM, **{field: value}))
                self.assertTrue(errors)
                # the params stay usable to show the form again
                self.assertEqual(set(params), set(mlp.DEFAULTS))


class StudyTests(unittest.TestCase):

    def test_only_one_parameter_changes(self):
        configs = mlp.study_configs("batch_size", mlp.DEFAULTS)
        self.assertEqual([v for v, _ in configs], [32, 128, 512])
        for value, cfg in configs:
            self.assertEqual(cfg["batch_size"], value)
            self.assertEqual(cfg["activation"], mlp.DEFAULTS["activation"])

    def test_neurons_order(self):
        layers = [cfg["hidden_layers"] for _, cfg in mlp.study_configs("neurons_order")]
        self.assertEqual(layers, ["128,64,32", "32,64,128"])

    def test_unknown_parameter(self):
        with self.assertRaises(ValueError):
            mlp.study_configs("dropout")

    def test_run_sensitivity_builds_the_rows(self):
        def fake_train(params, data=None, train_limit=None):
            return {"test_accuracy": 0.9, "test_loss": 0.3,
                    "num_params": 10, "train_time": 1.0}

        rows = mlp.run_sensitivity("hidden_layers", trainer=fake_train)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["value"], "(no hidden layer)")


@unittest.skipUnless(HAS_TF, "TensorFlow is not installed")
class ModelTests(unittest.TestCase):

    def test_parameter_counts(self):
        self.assertEqual(mlp.build_model([]).count_params(), 7850)              # subject
        self.assertEqual(mlp.build_model([128, 64, 32]).count_params(), 111146)  # PDF example

    def test_train_on_tiny_random_data(self):
        import numpy as np

        rng = np.random.default_rng(0)
        X = rng.random((200, 784), dtype=np.float32)
        y = np.eye(10, dtype=np.float32)[rng.integers(0, 10, 200)]
        params, errors = mlp.validate_params(dict(GOOD_FORM, epochs="2", hidden_layers="16"))
        self.assertEqual(errors, [])

        res = mlp.train_mlp(params, data=(X, y, X[:50], y[:50]))
        self.assertTrue(0.0 <= res["test_accuracy"] <= 1.0)
        self.assertEqual(len(res["history"]["accuracy"]), 2)
        self.assertEqual(res["num_params"], 784 * 16 + 16 + 16 * 10 + 10)


if __name__ == "__main__":
    unittest.main()
