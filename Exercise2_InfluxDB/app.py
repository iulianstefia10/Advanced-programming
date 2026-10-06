from flask import Flask
from influxdb_flask import InfluxDB


# CREATE FLASK APP

app = Flask(__name__)


# 7.1 CONNECT TO INFLUXDB
app.config["INFLUXDB_V2_URL"] = "http://192.168.168.88:8086"
app.config["INFLUXDB_V2_ORG"] = "iut"
app.config["INFLUXDB_V2_TOKEN"] = "AryKbDs91bu2VoKtFsVHF8_eSoQK4c_s-WEPkA5wXXX80lzcJtuu9k8AKDOMvrty-E3n9JsdqMWhC89y8e-K_A=="

influxdb = InfluxDB(app)

# 7.1 PING DATABASE
@app.route("/ping")
def ping():

    response = influxdb.connection.ping()

    return {
        "connected": response
    }


# 7.2 GET AVAILABLE BUCKETS
@app.route("/buckets")
def buckets():

    buckets_api = influxdb.connection.buckets_api()

    buckets = buckets_api.find_buckets().buckets

    return {
        "buckets": [bucket.name for bucket in buckets]
    }


# 7.2 GET WEATHER TIME SERIES
@app.route("/weather")
def weather():

    query_api = influxdb.connection.query_api()

    org = app.config["INFLUXDB_V2_ORG"]

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


# 7.3 GET MEASUREMENTS

@app.route("/measurements")
def measurements():

    query_api = influxdb.connection.query_api()

    org = app.config["INFLUXDB_V2_ORG"]

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

# 7.3 GET FIELDS

@app.route("/fields")
def fields():

    query_api = influxdb.connection.query_api()

    org = app.config["INFLUXDB_V2_ORG"]

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


# 7.4 FORMAT DATASET

def format_dataset(data, x):

    X = []
    y = []

    for i in range(len(data) - 2 * x + 1):

        X.append(data[i:i + x])

        y.append(data[i + x:i + 2 * x])

    return X, y

# TEST 7.4
@app.route("/format")
def format_test():

    data = [1, 2, 3, 4, 5, 6, 7, 8]

    x = 3

    X, y = format_dataset(data, x)

    return {
        "X": X,
        "y": y
    }


# START SERVER

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5004,
        debug=True
    )
