from flask import Blueprint, current_app

influx_bp = Blueprint("influxdb", __name__)


def get_influxdb():
    return current_app.extensions["influxdb"]


@influx_bp.route("/ping")
def ping():
    influxdb = get_influxdb()

    response = influxdb.connection.ping()

    return {"connected": response}


@influx_bp.route("/buckets")
def buckets():
    influxdb = get_influxdb()

    buckets_api = influxdb.connection.buckets_api()
    buckets = buckets_api.find_buckets().buckets

    return {
        "buckets": [bucket.name for bucket in buckets]
    }


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
    '''

    tables = query_api.query(
        org=org,
        query=flux_query
    )

    data = [
        record.values["_value"]
        for table in tables
        for record in table.records
    ]

    time = [
        record.values["_time"]
        for table in tables
        for record in table.records
    ]

    return {
        "data": data,
        "time": [str(t) for t in time]
    }


@influx_bp.route("/measurements")
def measurements():
    influxdb = get_influxdb()

    query_api = influxdb.connection.query_api()

    org = current_app.config["INFLUXDB_V2_ORG"]

    flux_query = '''
    import "influxdata/influxdb/schema"

    schema.measurements(bucket: "weather")
    '''

    tables = query_api.query(
        org=org,
        query=flux_query
    )

    meas_list = [
        record.values["_value"]
        for table in tables
        for record in table.records
    ]

    return {
        "measurements": meas_list
    }


@influx_bp.route("/fields")
def fields():
    influxdb = get_influxdb()

    query_api = influxdb.connection.query_api()

    org = current_app.config["INFLUXDB_V2_ORG"]

    flux_query = '''
    import "influxdata/influxdb/schema"

    schema.fieldKeys(bucket: "weather")
    '''

    tables = query_api.query(
        org=org,
        query=flux_query
    )

    field_list = [
        record.values["_value"]
        for table in tables
        for record in table.records
    ]

    return {
        "fields": field_list
    }