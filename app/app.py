from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from influxdb_flask import InfluxDB
import os

from extensions import db


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

    app.config["INFLUXDB_V2_TOKEN"] = "AryKbDs91bu2VoKtFsVHF8_eSoQK4c_s-WEPkA5wXXX80lzcJtuu9k8AKDOMvrty-E3n9JsdqMWhC89y8e-K_A=="

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