from flask import Blueprint, current_app, render_template, request

from services import forecast as forecast_service

forecast_bp = Blueprint(
    "forecast",
    __name__,
    url_prefix="/forecast",
)


def _query_api():
    influxdb = current_app.extensions.get("influxdb")
    if influxdb is None:
        raise RuntimeError("InfluxDB is not configured.")
    return influxdb.connection.query_api()


@forecast_bp.route("/", methods=["GET", "POST"])
def dashboard():
    # Bucket тільки з білого списку: weather, crypto, forex
    bucket = request.values.get("bucket", "weather")
    if bucket not in forecast_service.BUCKETS:
        bucket = "weather"

    errors = []
    options = None
    params = dict(forecast_service.DEFAULT_PARAMS)
    result = None
    org = current_app.config.get("INFLUXDB_V2_ORG", "iut")

    try:
        query_api = _query_api()
        # Спершу читаємо схему бази, щоб заповнити форму
        options = forecast_service.discover(
            query_api, org, bucket,
            request.values.get("measurement"), request.values.get("tag"),
        )
        params = forecast_service.default_params(options)
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:  # server down, wrong token...: never crash the page
        errors.append(f"Cannot read InfluxDB: {e}")

    if request.method == "POST" and options is not None:
        params, form_errors = forecast_service.validate(request.form, options)
        errors += form_errors
        if not form_errors:
            try:
                # Дані з InfluxDB -> навчання і порівняння 5 моделей
                frame = forecast_service.load_frame(query_api, org, options, params)
                result = forecast_service.run_forecast(frame, params)
            except ValueError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"Forecast failed: {e}")

    return render_template(
        "forecast_dashboard.html",
        buckets=forecast_service.BUCKETS,
        bucket=bucket,
        options=options,
        params=params,
        limits=forecast_service.LIMITS,
        result=result,
        errors=errors,
    )
