# Tests for the forecasting service (Sessions 2-3, Ex 2, 7.4 + several variables and cities)
# The neural network tests are skipped when TensorFlow is not installed.

import importlib.util
import os
import sys
import unittest

import numpy as np

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from forecast_fakes import FakeQueryApi  # noqa: E402
from services import forecast  # noqa: E402
from utils.dataset import format_dataset, make_windows  # noqa: E402

HAS_TF = importlib.util.find_spec("tensorflow") is not None
FAST = ["naive", "seasonal", "linear"]


def options():
    return forecast.discover(FakeQueryApi(), "iut", "weather")


def form(**changes):
    data = {"target_series": "annecy", "target_field": "temperature_2m",
            "extra_fields": ["relative_humidity_2m"], "extra_series": ["annemasse", "aix_les_bains"],
            "window": "24", "horizon": "12", "days": "40", "epochs": "5"}
    data.update(changes)
    return data


class WindowTests(unittest.TestCase):

    def test_same_windows_as_format_dataset(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        X_old, y_old = format_dataset(data, 3)
        X, y = make_windows(data, 0, 3, 3)
        self.assertEqual(X[:, :, 0].tolist(), [list(map(float, w)) for w in X_old])
        self.assertEqual(y.tolist(), [list(map(float, w)) for w in y_old])

    def test_several_variables(self):
        values = np.arange(20).reshape(10, 2)  # 10 hours, 2 variables
        X, y = make_windows(values, 1, 4, 2)
        self.assertEqual(X.shape, (5, 4, 2))
        self.assertEqual(y.shape, (5, 2))
        self.assertEqual(y[0].tolist(), [9.0, 11.0])  # target = column 1, hours 4 and 5

    def test_too_short(self):
        X, y = make_windows([1, 2, 3], 0, 3, 3)
        self.assertEqual(len(X), 0)

    def test_split_has_no_overlap(self):
        window, horizon = 24, 12
        n_train, test_start = forecast.split_points(500, window, horizon)
        last_train_hour = (n_train - 1) + window + horizon - 1
        self.assertLess(last_train_hour, test_start)  # first hour of the first test window

    def test_split_refuses_too_little_data(self):
        with self.assertRaises(ValueError):
            forecast.split_points(60, 24, 24)


class DiscoverAndLoadTests(unittest.TestCase):

    def test_discover(self):
        opts = options()
        self.assertEqual(opts["measurement"], "weather_current")
        self.assertEqual(opts["tag"], "location")  # columns starting with _ are skipped
        self.assertIn("annecy", opts["tag_values"])

    def test_default_params_use_annecy_and_other_cities(self):
        params = forecast.default_params(options())
        self.assertEqual(params["target_series"], "annecy")
        self.assertEqual(params["target_field"], "temperature_2m")
        self.assertIn("relative_humidity_2m", params["extra_fields"])
        self.assertEqual(sorted(params["extra_series"]), ["aix_les_bains", "annemasse"])

    def test_query_is_built_from_the_choices(self):
        params, errors = forecast.validate(form(), options())
        self.assertEqual(errors, [])
        query = forecast.build_query(options(), params)
        self.assertIn('r["location"] == "annemasse"', query)
        self.assertIn('r._field == "relative_humidity_2m"', query)
        self.assertIn("aggregateWindow(every: 1h", query)
        self.assertIn("range(start: -40d)", query)

    def test_load_frame(self):
        params, _ = forecast.validate(form(extra_fields=["relative_humidity_2m", "weather_code"]), options())
        frame = forecast.load_frame(FakeQueryApi(), "iut", options(), params)
        self.assertEqual(frame.columns[0], "temperature_2m @ annecy")  # target first
        self.assertEqual(len(frame.columns), 6)  # 2 numeric fields x 3 stations, text field dropped
        self.assertFalse(frame.isna().any().any())
        self.assertEqual(len(frame), 24 * 40)

    def test_no_data(self):
        params, _ = forecast.validate(form(), options())
        with self.assertRaises(ValueError):
            forecast.load_frame(FakeQueryApi(records=[]), "iut", options(), params)


class ValidationTests(unittest.TestCase):

    def test_good_form(self):
        params, errors = forecast.validate(form(), options())
        self.assertEqual(errors, [])
        self.assertEqual(params["window"], 24)
        self.assertEqual(params["extra_series"], ["aix_les_bains", "annemasse"])

    def test_target_is_removed_from_the_extra_inputs(self):
        params, errors = forecast.validate(form(extra_series=["annecy", "annemasse"],
                                                extra_fields=["temperature_2m"]), options())
        self.assertEqual(errors, [])
        self.assertEqual(params["extra_series"], ["annemasse"])
        self.assertEqual(params["extra_fields"], [])

    def test_bad_values(self):
        cases = [{"window": "0"}, {"window": "abc"}, {"horizon": "500"}, {"days": "-3"},
                 {"epochs": "0"}, {"target_series": "paris"}, {"target_field": "snow"},
                 {"extra_series": ['annecy") |> drop(']}, {"days": "3", "window": "48", "horizon": "48"}]
        for case in cases:
            with self.subTest(case):
                _, errors = forecast.validate(form(**case), options())
                self.assertTrue(errors)

    def test_flux_strings_are_escaped(self):
        self.assertEqual(forecast.flux_string('a"b'), '"a\\"b"')


class ForecastTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        opts = options()
        params, _ = forecast.validate(form(), opts)
        cls.frame = forecast.load_frame(FakeQueryApi(), "iut", opts, params)
        cls.params = params
        cls.result = forecast.run_forecast(cls.frame, params, models=FAST)

    def test_table_and_best(self):
        keys = [row["key"] for row in self.result["table"]]
        self.assertEqual(keys, FAST)
        self.assertIn(self.result["best"], FAST)
        for row in self.result["table"]:
            self.assertGreaterEqual(row["rmse"], row["mae"])
            self.assertEqual(len(row["mae_by_step"]), 12)

    def test_linear_beats_the_naive_baseline(self):
        mae = {row["key"]: row["mae"] for row in self.result["table"]}
        self.assertLess(mae["linear"], mae["naive"])

    def test_other_city_helps(self):
        # annemasse sees the weather 6 h before annecy in the fake data
        linear = next(row for row in self.result["table"] if row["key"] == "linear")
        self.assertLess(linear["mae"], linear["mae_target_only"])
        self.assertGreater(linear["gain"], 0)

    def test_chart_data(self):
        chart = self.result["chart"]
        self.assertEqual(len(chart["history"]), 24)
        self.assertEqual(len(chart["real"]), 12)
        self.assertEqual(len(chart["future_times"]), 12)
        self.assertEqual(sorted(chart["predictions"]), sorted(FAST))

    def test_seasonal_needs_a_day(self):
        params = dict(self.params, window=12, horizon=6)
        result = forecast.run_forecast(self.frame, params, models=FAST)
        self.assertNotIn("seasonal", [row["key"] for row in result["table"]])


class RobustnessTests(unittest.TestCase):
    """The first real run (20.11) showed that many inputs made the models worse
    than the naive forecast. These tests keep the fixes in place."""

    @staticmethod
    def noisy_frame(seed):
        import pandas as pd

        rng = np.random.default_rng(seed)
        n = 669
        t = pd.date_range("2026-10-21", periods=n, freq="h", tz="UTC")
        base = 6 + 4 * np.sin(2 * np.pi * np.arange(n + 6) / 24) + np.cumsum(rng.normal(0, 0.35, n + 6))
        cols = {"temperature_2m @ annecy": base[:n],
                "temperature_2m @ geneve": base[3:n + 3] + rng.normal(0, 0.3, n)}
        for i in range(30):  # useless series that drift
            cols[f"noise_{i}"] = np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame(cols, index=t)

    def test_many_useless_inputs_do_not_break_the_linear_model(self):
        params = dict(forecast.DEFAULT_PARAMS, window=24, horizon=24)
        for seed in (1, 2, 3):
            with self.subTest(seed=seed):
                result = forecast.run_forecast(self.noisy_frame(seed), params, models=FAST)
                mae = {row["key"]: row for row in result["table"]}
                self.assertLess(mae["linear"]["mae"], mae["naive"]["mae"])
                # 30 drifting random series is a worst case: a small loss is acceptable,
                # before the fixes the error was 2 to 3 times bigger
                self.assertLess(mae["linear"]["mae"], 1.25 * mae["linear"]["mae_target_only"])

    def test_alpha_is_chosen_by_time(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 50))
        y = X[:, :1] * 0.5 + rng.normal(0, 1, (200, 3))
        self.assertIn(forecast.choose_alpha(X, y, gap=10), list(forecast.ALPHAS))

    def test_model_inputs(self):
        X = np.random.default_rng(0).normal(size=(5, 24, 4))
        flat = forecast.model_inputs("linear", X, [0, 1, 2, 3])
        self.assertEqual(flat.shape, (5, 23 + 1 + 3 * 5))  # target window + level + 6 h of 3 others
        seq = forecast.model_inputs("lstm", X, [0, 1, 2, 3])
        self.assertEqual(seq.shape, (5, 24, 5))
        self.assertTrue(np.allclose(seq[:, -1, :4], 0))  # relative to the last value

    def test_defaults_keep_nearby_stations_only(self):
        opts = dict(options(), tag_values=["annecy", "bordeaux", "geneve", "paris", "rumilly"])
        params = forecast.default_params(opts)
        self.assertEqual(params["extra_series"], ["geneve", "rumilly"])


@unittest.skipUnless(HAS_TF, "TensorFlow is not installed")
class NeuralNetworkTests(unittest.TestCase):

    def test_mlp_and_lstm(self):
        opts = options()
        params, _ = forecast.validate(form(epochs="3"), opts)
        frame = forecast.load_frame(FakeQueryApi(), "iut", opts, params)
        result = forecast.run_forecast(frame, params, models=["naive", "mlp", "lstm"])
        rows = {row["key"]: row for row in result["table"]}
        for key in ("mlp", "lstm"):
            self.assertTrue(np.isfinite(rows[key]["mae"]))
            self.assertIn("mae_target_only", rows[key])
            self.assertLessEqual(rows[key]["epochs_run"], 3)
        self.assertEqual(len(result["chart"]["predictions"]["lstm"]), 12)


if __name__ == "__main__":
    unittest.main()
