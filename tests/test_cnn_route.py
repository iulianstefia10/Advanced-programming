# Tests for the CIFAR-100 CNN service (PW3): form checks, model, state of the training
# No dataset is downloaded and no training is run here.

import os
import sys
import unittest

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from services import cnn_service  # noqa: E402


class ParseConfigTest(unittest.TestCase):
    def test_empty_form_gives_the_defaults(self):
        self.assertEqual(cnn_service.parse_config({}), cnn_service.DEFAULTS)

    def test_text_values_are_converted(self):
        config = cnn_service.parse_config({
            "epochs": "3",
            "batch_size": "128",
            "optimizer": "sgd",
            "learning_rate": "0.01",
            "validation_split": "0.1",
            "label_mode": "coarse",
            "filters": "16",
            "kernel_size": "4",
            "dense_units": "512",
        })
        self.assertEqual(config["epochs"], 3)
        self.assertEqual(config["batch_size"], 128)
        self.assertEqual(config["optimizer"], "sgd")
        self.assertAlmostEqual(config["learning_rate"], 0.01)
        self.assertAlmostEqual(config["validation_split"], 0.1)
        self.assertEqual(config["label_mode"], "coarse")
        self.assertEqual(config["filters"], 16)
        self.assertEqual(config["kernel_size"], 4)
        self.assertEqual(config["dense_units"], 512)

    def test_limits_are_accepted(self):
        cnn_service.parse_config({"epochs": "1", "kernel_size": "2"})
        cnn_service.parse_config({"epochs": "100", "kernel_size": "4"})

    def test_wrong_values_are_refused(self):
        wrong_forms = [
            {"epochs": "0"},
            {"epochs": "101"},
            {"epochs": "abc"},
            {"epochs": "2.5"},
            {"batch_size": "7"},
            {"batch_size": "2000"},
            {"learning_rate": "0"},
            {"learning_rate": "2"},
            {"validation_split": "0.9"},
            {"filters": "3"},
            {"filters": "500"},
            {"kernel_size": "5"},   # would shrink 32x32 images to nothing
            {"dense_units": "4"},
            {"dense_units": "5000"},
            {"optimizer": "newton"},
            {"label_mode": "medium"},
        ]
        for form in wrong_forms:
            with self.subTest(form=form):
                with self.assertRaises(ValueError):
                    cnn_service.parse_config(form)

    def test_error_message_names_the_field(self):
        with self.assertRaises(ValueError) as context:
            cnn_service.parse_config({"epochs": "0"})
        self.assertIn("epochs", str(context.exception))


class BuildModelTest(unittest.TestCase):
    """Needs TensorFlow (the real one): the model is built but not trained."""

    def test_default_model_is_the_one_of_the_assignment(self):
        model = cnn_service.build_model(cnn_service.DEFAULTS)
        self.assertEqual(model.count_params(), 270372)

    def test_coarse_labels_change_only_the_output_layer(self):
        config = {**cnn_service.DEFAULTS, "label_mode": "coarse"}
        model = cnn_service.build_model(config)
        # 128 * 20 + 20 instead of 128 * 100 + 100 for the last layer
        self.assertEqual(model.count_params(), 270372 - 12900 + 2580)

    def test_more_filters_means_more_parameters(self):
        small = cnn_service.build_model({**cnn_service.DEFAULTS, "filters": 16})
        large = cnn_service.build_model({**cnn_service.DEFAULTS, "filters": 64})
        self.assertLess(small.count_params(), large.count_params())

    def test_every_allowed_kernel_size_builds(self):
        for kernel_size in (2, 3, 4):
            with self.subTest(kernel_size=kernel_size):
                config = {**cnn_service.DEFAULTS, "kernel_size": kernel_size}
                cnn_service.build_model(config)


class JobStateTest(unittest.TestCase):
    def setUp(self):
        self._reset()

    def tearDown(self):
        self._reset()

    @staticmethod
    def _reset():
        cnn_service._stop_event.clear()
        with cnn_service._lock:
            cnn_service._job.clear()
            cnn_service._job["status"] = "idle"

    @staticmethod
    def _set_status(status):
        with cnn_service._lock:
            cnn_service._job["status"] = status

    def test_nothing_running_at_start(self):
        self.assertFalse(cnn_service.is_running())
        self.assertEqual(cnn_service.get_status()["status"], "idle")

    def test_stop_is_refused_when_idle(self):
        self.assertFalse(cnn_service.request_stop())
        self.assertFalse(cnn_service._stop_event.is_set())

    def test_stop_during_download_and_training(self):
        for status in ("loading", "training"):
            with self.subTest(status=status):
                self._reset()
                self._set_status(status)
                self.assertTrue(cnn_service.request_stop())
                self.assertEqual(cnn_service.get_status()["status"], "stopping")
                self.assertTrue(cnn_service._stop_event.is_set())

    def test_second_training_is_refused_while_one_runs(self):
        self._set_status("training")
        self.assertTrue(cnn_service.is_running())
        self.assertFalse(cnn_service.start_training(cnn_service.DEFAULTS, "unused"))

    def test_status_is_a_copy(self):
        status = cnn_service.get_status()
        status["status"] = "hacked"
        self.assertEqual(cnn_service.get_status()["status"], "idle")


if __name__ == "__main__":
    unittest.main()