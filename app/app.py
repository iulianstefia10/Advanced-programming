from flask import Flask, render_template
from models.chinook import Artist, Album, Genre, Track, Customer, Invoice, InvoiceLine
from influxdb_flask import InfluxDB
import os

from dotenv import load_dotenv
from extensions import db

load_dotenv()


def create_app():

    app = Flask(__name__)

    # -------------------------
    # Database
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

    db.init_app(app)

    # -------------------------
    # InfluxDB
    # -------------------------

    with app.app_context():
        db.reflect()

        for table_name, table in db.metadata.tables.items():
            print(f"\nTable: {table_name}")

            for column in table.columns:
                print(f"  Column: {column.name}, Type: {column.type}")

    app.config["INFLUXDB_V2_URL"] = (
        "http://192.168.168.88:8086"
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

   

    from routes.chinook import chinook_bp
    from routes.influxdb import influx_bp
    from routes.mnist import mnist_bp

    app.register_blueprint(chinook_bp)
    app.register_blueprint(influx_bp)
    app.register_blueprint(mnist_bp)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5004,
        debug=True
    )