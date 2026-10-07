from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from influxdb_flask import InfluxDB
import os

from extensions import db
from dotenv import load_dotenv

load_dotenv()

def create_app():

    app = Flask(__name__)

    # Database
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

    db.init_app(app)

    # InfluxDB
    app.config["INFLUXDB_V2_URL"] = (
        "http://192.168.168.88:8086"
    )

    app.config["INFLUXDB_V2_ORG"] = "iut"

    app.config["INFLUXDB_V2_TOKEN"] = os.environ.get("INFLUXDB_V2_TOKEN")

    influxdb = InfluxDB(app)
    app.extensions["influxdb"] = influxdb

    # Routes
    from routes.chinook import chinook_bp
    from routes.influxdb import influx_bp

    app.register_blueprint(chinook_bp)
    app.register_blueprint(influx_bp)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5047,
        debug=True
    )