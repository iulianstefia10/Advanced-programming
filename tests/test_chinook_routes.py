# Tests for the Chinook pages (Sessions 2-3, exercise 1)
# They use the real chinook.db file of the project.
# Run: python -m unittest discover -s tests -v

import json
import os
import sys
import unittest

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from flask import Flask  # noqa: E402

from extensions import db  # noqa: E402
from routes.chinook import chinook_bp  # noqa: E402


def create_test_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(APP_DIR, "templates"),
        static_folder=os.path.join(APP_DIR, "static"),
    )
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.abspath(
        os.path.join(APP_DIR, "database", "chinook.db")
    )
    db.init_app(app)
    app.register_blueprint(chinook_bp)

    @app.route("/")
    def home():
        return "home"

    return app


class ChinookPagesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = create_test_app().test_client()

    def get_html(self, url):
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200, url)
        return response.get_data(as_text=True)

    def test_all_pages_open(self):
        for url in ("/chinook/", "/chinook/artists", "/chinook/albums",
                    "/chinook/artists_with_albums", "/chinook/albums_per_artist",
                    "/chinook/popular_artists", "/chinook/top_customers",
                    "/chinook/genres_stats", "/chinook/metadata"):
            self.get_html(url)

    def test_artists_and_albums(self):
        self.assertIn("AC/DC", self.get_html("/chinook/artists"))
        self.assertIn("For Those About To Rock We Salute You", self.get_html("/chinook/albums"))

    def test_artists_with_albums_join(self):
        html = self.get_html("/chinook/artists_with_albums")
        self.assertIn("AC/DC", html)
        self.assertIn("Let There Be Rock", html)

    def test_popular_artists_top_10(self):
        html = self.get_html("/chinook/popular_artists")
        # Iron Maiden is the artist with the most sales in Chinook
        self.assertIn("Iron Maiden", html)

    def test_genres_histogram(self):
        html = self.get_html("/chinook/genres_stats")
        self.assertIn("Rock", html)
        self.assertIn("data:image/png;base64,", html)

    def test_metadata_json(self):
        response = self.client.get("/chinook/api/metadata")
        self.assertEqual(response.status_code, 200)
        tables = json.loads(response.get_data(as_text=True))
        for table in ("Artist", "Album", "Track", "Genre", "Customer", "Invoice", "InvoiceLine"):
            self.assertIn(table, tables)
        columns = [column["name"] for column in tables["Artist"]]
        self.assertEqual(columns, ["ArtistId", "Name"])


if __name__ == "__main__":
    unittest.main()
