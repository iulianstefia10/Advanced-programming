from flask import Blueprint, current_app, render_template, request

influx_bp = Blueprint(
    "influxdb",
    __name__,
    url_prefix="/influxdb"
)


def get_influxdb():
    return current_app.extensions["influxdb"]


@influx_bp.route("/")
def dashboard():
    return render_template("influx_dashboard.html")


@influx_bp.route("/ping")
def ping():
    influxdb = get_influxdb()
    connected = influxdb.connection.ping()

    return render_template(
        "influx_status.html",
        connected=connected
    )


@influx_bp.route("/buckets")
def buckets():
    influxdb = get_influxdb()
    buckets_api = influxdb.connection.buckets_api()
    bucket_list = buckets_api.find_buckets().buckets

    return render_template(
        "data_list.html",
        title="InfluxDB Buckets",
        items=[
            {"name": bucket.name, "value": bucket.name}
            for bucket in bucket_list
        ]
    )



@influx_bp.route("/weather")
def weather():
    influxdb = get_influxdb()
    query_api = influxdb.connection.query_api()
    org = current_app.config["INFLUXDB_V2_ORG"]

    flux_query = '''
    from(bucket: "weather")
      |> range(start: -1w)
      |> filter(fn: (r) => r._measurement == "weather_current")
      |> filter(fn: (r) => r._field == "temperature_2m")
      |> filter(fn: (r) => r.location == "annecy")
      |> sort(columns: ["_time"])
    '''

    tables = query_api.query(
        org=org,
        query=flux_query
    )

    records = [
        record
        for table in tables
        for record in table.records
    ]

    data = [record.get_value() for record in records]
    time = [str(record.get_time()) for record in records]

    return render_template(
        "influx_data.html",
        data=data,
        time=time
    )



@influx_bp.route("/measurements")
def measurements():
    influxdb = get_influxdb()
    query_api = influxdb.connection.query_api()
    org = current_app.config["INFLUXDB_V2_ORG"]

    flux_query = '''
    import "influxdata/influxdb/schema"

    schema.measurements(bucket: "weather")
    '''

    tables = query_api.query(org=org, query=flux_query)

    measurement_list = [
        record.get_value()
        for table in tables
        for record in table.records
    ]

    return render_template(
        "data_list.html",
        title="InfluxDB Measurements",
        items=[
            {"name": name, "value": name}
            for name in measurement_list
        ]
    )


@influx_bp.route("/fields")
def fields():
    influxdb = get_influxdb()
    query_api = influxdb.connection.query_api()
    org = current_app.config["INFLUXDB_V2_ORG"]

    flux_query = '''
    import "influxdata/influxdb/schema"

    schema.fieldKeys(bucket: "weather")
    '''

    tables = query_api.query(org=org, query=flux_query)

    field_list = [
        record.get_value()
        for table in tables
        for record in table.records
    ]

    return render_template(
        "data_list.html",
        title="InfluxDB Fields",
        items=[
            {"name": field, "value": field}
            for field in field_list
        ]
    )


def format_dataset(data, x):
    X = []
    y = []

    for i in range(len(data) - 2 * x + 1):
        X.append(data[i:i + x])
        y.append(data[i + x:i + 2 * x])

    return X, y

@influx_bp.route("/format")
def format_test():
    data = [1, 2, 3, 4, 5, 6, 7, 8]

    x = request.args.get("x", default=3, type=int)
    x = max(1, min(x, len(data) // 2))

    X, y = format_dataset(data, x)

    return render_template(
        "influx_format.html",
        data=data,
        X=X,
        y=y,
        sequence_length=x
    )