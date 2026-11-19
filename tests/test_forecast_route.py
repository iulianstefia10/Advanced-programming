# Tests for the /forecast page. InfluxDB is replaced by a fake one and only the
# fast models are trained, so the tests need neither the server nor TensorFlow.

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask  # noqa: E402

from forecast_fakes import FakeQueryApi  # noqa: E402
from routes.forecast import forecast_bp  # noqa: E402
from services import forecast as forecast_service  # noqa: E402

REAL_RUN = forecast_service.run_forecast


def fast_run(frame, params):
    return REAL_RUN(frame, params, models=["naive", "seasonal", "linear"])


def create_test_app(api=None):
    app = Flask(__name__, template_folder=os.path.join(APP_DIR, "templates"),
                static_folder=os.path.join(APP_DIR, "static"))
    app.config["TESTING"] = True
    app.config["INFLUXDB_V2_ORG"] = "iut"
    api = api or FakeQueryApi()
    app.extensions["influxdb"] = SimpleNamespace(connection=SimpleNamespace(query_api=lambda: api))
    app.register_blueprint(forecast_bp)
    return app


FORM = {"bucket": "weather", "measurement": "weather_current", "tag": "location",
        "target_series": "annecy", "target_field": "temperature_2m",
        "extra_fields": ["relative_humidity_2m"], "extra_series": ["annemasse"],
        "window": "24", "horizon": "12", "days": "40", "epochs": "5", "action": "run"}


@mock.patch.object(forecast_service, "run_forecast", side_effect=fast_run)
class ForecastRouteTests(unittest.TestCase):

    def test_page_lists_stations_and_variables(self, _):
        html = create_test_app().test_client().get("/forecast/").get_data(as_text=True)
        for text in ('name="bucket"', 'name="target_series"', "annemasse", "relative_humidity_2m"):
            self.assertIn(text, html)
        self.assertIn('value="annecy" selected', html)

    def test_run(self, run):
        html = create_test_app().test_client().post("/forecast/", data=FORM).get_data(as_text=True)
        self.assertIn("Forecast · temperature_2m @ annecy", html)
        self.assertIn("Linear regression (ridge)", html)
        self.assertIn("Gain of the extra inputs", html)
        self.assertIn('id="fc-data"', html)
        run.assert_called_once()

    def test_invalid_values_show_an_error(self, run):
        for change in ({"window": "abc"}, {"target_series": "paris"}, {"horizon": "0"}):
            with self.subTest(change):
                html = create_test_app().test_client().post(
                    "/forecast/", data=dict(FORM, **change)).get_data(as_text=True)
                self.assertIn("Invalid parameters", html)
                self.assertNotIn("Forecast · temperature_2m", html)
        run.assert_not_called()

    def test_unknown_bucket_falls_back_to_weather(self, _):
        html = create_test_app().test_client().get("/forecast/?bucket=secret").get_data(as_text=True)
        self.assertIn('value="weather" selected', html)

    def test_influx_down(self, _):
        html = create_test_app(FakeQueryApi(fail=True)).test_client().get("/forecast/").get_data(as_text=True)
        self.assertIn("Cannot read InfluxDB", html)
        self.assertNotIn('name="target_series"', html)


if __name__ == "__main__":
    unittest.main()
