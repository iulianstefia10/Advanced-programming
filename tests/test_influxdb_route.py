# Tests for the /influxdb pages when the InfluxDB server can't be reached
# Run: python -m unittest discover -s tests -v

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from flask import Flask
from influxdb_flask import InfluxDB

from routes.influxdb import influx_bp

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")


def create_test_app(url):
    app = Flask(
        __name__,
        template_folder=os.path.join(APP_DIR, "templates"),
        static_folder=os.path.join(APP_DIR, "static"),
    )
    app.config["TESTING"] = True
    app.config["INFLUXDB_V2_URL"] = url
    app.config["INFLUXDB_V2_ORG"] = "iut"
    app.config["INFLUXDB_V2_TOKEN"] = "test-token"
    app.config["INFLUXDB_V2_TIMEOUT"] = 2000
    app.extensions["influxdb"] = InfluxDB(app)
    app.register_blueprint(influx_bp)

    @app.route("/")
    def home():
        return "home"

    return app


class InfluxDBUnavailableTest(unittest.TestCase):

    def setUp(self):
        # nothing listens on this port, so the connection fails at once
        self.client = create_test_app("http://127.0.0.1:9").test_client()

    def test_weather_shows_message_instead_of_error(self):
        response = self.client.get("/influxdb/weather")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 503)
        self.assertIn("InfluxDB is unavailable", html)
        self.assertIn("http://127.0.0.1:9", html)

    def test_other_pages_show_the_same_message(self):
        for page in ("/influxdb/buckets", "/influxdb/measurements", "/influxdb/fields"):
            response = self.client.get(page)
            self.assertEqual(response.status_code, 503, page)
            self.assertIn("InfluxDB is unavailable", response.get_data(as_text=True))

    def test_ping_shows_offline_and_the_address(self):
        response = self.client.get("/influxdb/ping")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("OFFLINE", html)
        self.assertIn("http://127.0.0.1:9", html)

    def test_format_page_does_not_need_influxdb(self):
        response = self.client.get("/influxdb/format?x=2")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
