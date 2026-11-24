from flask import Flask, render_template
from models.chinook import Artist, Album, Genre, Track, Customer, Invoice, InvoiceLine
from models import app_data  # noqa: F401  (registers the tables of the "appdata" database)
from influxdb_flask import InfluxDB
import os
from routes.landcover import landcover_bp
from services.model_store import seed_default_models

from dotenv import load_dotenv
from extensions import db

load_dotenv()


def create_app():

    app = Flask(__name__)

    # imported models can be big (the given CNN is a few MB)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    # -------------------------
    # Database 1: Chinook (SQLite)
    # -------------------------

    basedir = os.path.abspath(
        os.path.dirname(__file__)
    )

    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "sqlite:///"
        + os.path.join(
            basedir,
            "database/chinook.db"
        )
    )

    # -------------------------
    # Database 2: our own tables ("appdata")
    # Postgres in Docker (DATABASE_URL), a local SQLite file otherwise
    # (tests, continuous integration, running without Docker)
    # -------------------------

    app.config["SQLALCHEMY_BINDS"] = {
        "appdata": os.environ.get(
            "DATABASE_URL",
            "sqlite:///" + os.path.join(basedir, "database/appdata.db"),
        )
    }

    db.init_app(app)

    with app.app_context():
        # only Chinook is reflected, never the "appdata" database
        db.reflect(bind_key=None)

        for table_name, table in db.metadata.tables.items():
            print(f"\nTable: {table_name}")

            for column in table.columns:
                print(f"  Column: {column.name}, Type: {column.type}")

        # only our own tables are created, never the Chinook ones
        try:
            db.create_all(bind_key="appdata")
            # first start: the two given digit models are copied into the database
            seed_default_models()
        except Exception as error:  # for example Postgres is not reachable
            db.session.rollback()
            app.logger.warning("App database unavailable: %s", error)

    # -------------------------
    # InfluxDB
    # -------------------------

    # the address can be changed in .env (the school network can change)
    app.config["INFLUXDB_V2_URL"] = os.environ.get(
        "INFLUXDB_V2_URL", "http://192.168.168.88:8086"
    )

    app.config["INFLUXDB_V2_ORG"] = "iut"

    app.config["INFLUXDB_V2_TOKEN"] = os.environ.get(
        "INFLUXDB_V2_TOKEN"
    )

    influxdb = InfluxDB(app)

    app.extensions["influxdb"] = influxdb

    # -------------------------
    # Main page
    # -------------------------

    @app.route("/")
    def home():
        return render_template("home.html")

    # -------------------------
    # Pages (one blueprint each)
    # -------------------------

    from routes.chinook import chinook_bp
    from routes.influxdb import influx_bp
    from routes.mnist import mnist_bp
    from routes.random_forest import rf_bp
    from routes.cnn import cnn_bp
    from routes.mlp import mlp_bp
    from routes.apparent import apparent_bp
    from routes.yolo import yolo_bp
    from routes.forecast import forecast_bp
    from routes.pw4 import pw4_bp

    app.register_blueprint(cnn_bp)
    app.register_blueprint(chinook_bp)
    app.register_blueprint(influx_bp)
    app.register_blueprint(mnist_bp)
    app.register_blueprint(rf_bp)
    app.register_blueprint(mlp_bp)
    app.register_blueprint(apparent_bp)
    app.register_blueprint(yolo_bp)
    app.register_blueprint(forecast_bp)
    app.register_blueprint(landcover_bp)
    app.register_blueprint(pw4_bp)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5004,
        debug=True
    )