# Tests for the /apparent page (Session 1, 5.4)
# Only the apparent blueprint is loaded; InfluxDB is replaced by a fake one.

import os
import sys
import unittest
from types import SimpleNamespace

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from flask import Flask  # noqa: E402

from routes.apparent import apparent_bp  # noqa: E402


def fake_influx(records=None, fail=False):
    def query(org, query):
        if fail:
            raise ConnectionError("timeout")
        return [SimpleNamespace(records=records or [])]

    api = SimpleNamespace(query=query)
    return SimpleNamespace(connection=SimpleNamespace(query_api=lambda: api))


def rec(field, value):
    return SimpleNamespace(get_field=lambda: field, get_value=lambda: value,
                           get_time=lambda: "2026-11-18T08:00:00+00:00")


def create_test_app(influx=None):
    app = Flask(
        __name__,
        template_folder=os.path.join(APP_DIR, "templates"),
        static_folder=os.path.join(APP_DIR, "static"),
    )
    app.config["TESTING"] = True
    app.config["INFLUXDB_V2_ORG"] = "iut"
    if influx is not None:
        app.extensions["influxdb"] = influx
    app.register_blueprint(apparent_bp)
    return app


class ApparentRouteTests(unittest.TestCase):

    def post(self, app=None, **fields):
        data = {"temperature": "32", "humidity": "60", "wind": "10",
                "location": "annecy", "action": "compute"}
        data.update(fields)
        return (app or create_test_app()).test_client().post("/apparent/", data=data)

    def test_page_shows_the_form(self):
        html = create_test_app().test_client().get("/apparent/").get_data(as_text=True)
        for name in ("temperature", "humidity", "wind", "location"):
            self.assertIn(f'name="{name}"', html)

    def test_compute_heat_index(self):
        html = self.post().get_data(as_text=True)
        self.assertIn("Feels like", html)
        self.assertIn("Heat index (NOAA)", html)
        self.assertIn("37.1 °C", html)

    def test_compute_wind_chill(self):
        html = self.post(temperature="-10", humidity="50", wind="20").get_data(as_text=True)
        self.assertIn("-17.9 °C", html)

    def test_invalid_values_show_an_error(self):
        for fields in ({"temperature": "abc"}, {"humidity": "150"}, {"wind": "-3"},
                       {"location": "x\") |> drop()"}):
            with self.subTest(fields):
                html = self.post(**fields).get_data(as_text=True)
                self.assertIn("Invalid parameters", html)
                self.assertNotIn("Feels like", html)

    def test_current_weather_from_influx(self):
        influx = fake_influx([rec("temperature_2m", -2.0), rec("relative_humidity_2m", 70),
                              rec("wind_speed_10m", 25.0), rec("apparent_temperature", -8.4)])
        html = self.post(create_test_app(influx), action="influx").get_data(as_text=True)
        self.assertIn("Feels like", html)
        self.assertIn("Wind chill", html)
        self.assertIn("The weather service gives -8.4 °C", html)

    def test_influx_down_shows_an_error(self):
        html = self.post(create_test_app(fake_influx(fail=True)), action="influx").get_data(as_text=True)
        self.assertIn("Cannot read InfluxDB", html)

    def test_influx_missing_fields(self):
        influx = fake_influx([rec("temperature_2m", 10.0)])
        html = self.post(create_test_app(influx), action="influx").get_data(as_text=True)
        self.assertIn("No recent value in InfluxDB", html)


if __name__ == "__main__":
    unittest.main()
