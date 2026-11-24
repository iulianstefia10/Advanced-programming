import re

from flask import Blueprint, current_app, render_template, request

from services import apparent as apparent_service

apparent_bp = Blueprint(
    "apparent",
    __name__,
    url_prefix="/apparent",
)

LOCATIONS = ["annecy", "aix_les_bains", "annemasse"]


# Кнопка Use the current weather: читає InfluxDB, будь-яка помилка -> текст на сторінці
def _current_weather(location):
    """Current weather from InfluxDB, or (None, error message)."""
    influxdb = current_app.extensions.get("influxdb")
    if influxdb is None:
        return None, "InfluxDB is not configured."
    try:
        weather = apparent_service.current_weather(
            influxdb.connection.query_api(),
            current_app.config["INFLUXDB_V2_ORG"],
            location,
        )
    except Exception as e:  # network or token problem: never crash the page
        return None, f"Cannot read InfluxDB: {e}"
    missing = [k for k in ("temperature", "humidity", "wind") if k not in weather]
    if missing:
        return None, f"No recent value in InfluxDB for: {', '.join(missing)}."
    return weather, None


@apparent_bp.route("/", methods=["GET", "POST"])
def dashboard():
    values = {"temperature": 30.0, "humidity": 60.0, "wind": 10.0}
    location = "annecy"
    result = None
    weather = None
    errors = []

    if request.method == "POST":
        location = request.form.get("location", "annecy")
        # Regex: тільки малі літери і _, до 40 символів -> захист Flux-запиту від ін'єкції
        if not re.fullmatch(r"[a-z_]{1,40}", location):
            errors.append("Unknown location.")
            location = "annecy"

        # action=influx: дані з InfluxDB; інакше дані з форми
        if request.form.get("action") == "influx" and not errors:
            weather, error = _current_weather(location)
            if error:
                errors.append(error)
            else:
                values = {k: weather[k] for k in ("temperature", "humidity", "wind")}
        else:
            values, form_errors = apparent_service.validate(request.form)
            errors += form_errors

        if not errors:
            result = apparent_service.apparent_temperature(
                values["temperature"], values["humidity"], values["wind"]
            )
            result["curve"] = apparent_service.wind_curve(
                values["temperature"], values["humidity"]
            )

    return render_template(
        "apparent_dashboard.html",
        values=values,
        location=location,
        locations=LOCATIONS,
        result=result,
        weather=weather,
        errors=errors,
    )
