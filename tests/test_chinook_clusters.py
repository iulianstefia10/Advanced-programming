# Tests for the customer clustering (service and /chinook/clusters page)
# Run: python -m unittest discover -s tests -v

import os
import sys
import unittest

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from flask import Flask  # noqa: E402

from extensions import db  # noqa: E402
from routes.chinook import chinook_bp  # noqa: E402
from services.chinook_clusters import (  # noqa: E402
    FEATURES,
    cluster_customers,
    elbow_curve,
    get_customer_features,
)


def create_test_app():
    # the real chinook.db file from the project, only the chinook blueprint
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


class ClusterServiceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_test_app()
        with cls.app.app_context():
            cls.customers = get_customer_features()

    def fresh_customers(self):
        return [dict(c) for c in self.customers]

    def test_one_line_per_customer(self):
        self.assertEqual(len(self.customers), 59)

    def test_features_are_filled(self):
        for customer in self.customers:
            for feature in FEATURES:
                self.assertIn(feature, customer)
            self.assertGreater(customer["total_spent"], 0)
            self.assertGreater(customer["n_genres"], 0)
            self.assertTrue(0 < customer["favorite_share"] <= 100)
            self.assertNotEqual(customer["favorite_genre"], "-")

    def test_total_spent_matches_the_invoices(self):
        # the sum of all customers must be the total of the Invoice table
        with self.app.app_context():
            from sqlalchemy import func
            from models.chinook import Invoice
            total = float(db.session.query(func.sum(Invoice.Total)).scalar())
        self.assertAlmostEqual(sum(c["total_spent"] for c in self.customers), total, places=1)

    def test_every_customer_gets_a_cluster(self):
        customers, summary = cluster_customers(self.fresh_customers(), k=4)
        self.assertEqual(len(summary), 4)
        self.assertEqual(sum(line["size"] for line in summary), 59)
        self.assertTrue(all(0 <= c["cluster"] < 4 for c in customers))

    def test_same_result_with_same_random_state(self):
        first, _ = cluster_customers(self.fresh_customers(), k=3)
        second, _ = cluster_customers(self.fresh_customers(), k=3)
        self.assertEqual([c["cluster"] for c in first], [c["cluster"] for c in second])

    def test_wrong_k_raises(self):
        for k in (0, 1, 11):
            with self.assertRaises(ValueError):
                cluster_customers(self.fresh_customers(), k=k)

    def test_elbow_inertia_goes_down(self):
        ks, inertias = elbow_curve(self.fresh_customers())
        self.assertEqual(ks[0], 1)
        self.assertEqual(len(ks), len(inertias))
        self.assertGreater(inertias[0], inertias[-1])


class ClusterPageTest(unittest.TestCase):

    def setUp(self):
        self.client = create_test_app().test_client()

    def test_page_shows_the_clusters(self):
        response = self.client.get("/chinook/clusters?k=3")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Clusters (k = 3)", html)
        self.assertEqual(html.count('class="customer-row"'), 59)
        self.assertEqual(html.count("data:image/png;base64,"), 2)

    def test_default_k_is_3(self):
        html = self.client.get("/chinook/clusters").get_data(as_text=True)
        self.assertIn("Clusters (k = 3)", html)

    def test_invalid_k_shows_an_error(self):
        for value in ("1", "11", "abc"):
            html = self.client.get(f"/chinook/clusters?k={value}").get_data(as_text=True)
            self.assertIn("Invalid parameter", html, value)

    def test_dashboard_links_to_the_page(self):
        html = self.client.get("/chinook/").get_data(as_text=True)
        self.assertIn("/chinook/clusters", html)


if __name__ == "__main__":
    unittest.main()
