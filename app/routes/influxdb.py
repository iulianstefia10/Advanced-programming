# Сторінки InfluxDB (Sessions 2–3, вправа 2): 7.1 ping, 7.2 buckets,
# 7.3 дані за тиждень + measurements/fields, 7.4 нарізка на вікна.
# Клієнт InfluxDB створюється в app.py (адреса, org і токен з .env).

from flask import Blueprint, current_app, render_template, request
from influxdb_client.rest import ApiException   # помилка, яку повернув сервер (напр. неправильний токен)
from urllib3.exceptions import HTTPError        # помилка мережі (сервер недоступний, таймаут)

# група сторінок: усі адреси цього файлу починаються з /influxdb
influx_bp = Blueprint(
    "influxdb",
    __name__,
    url_prefix="/influxdb"
)


# беремо клієнт InfluxDB, який app.py зберіг у app.extensions
def get_influxdb():
    return current_app.extensions["influxdb"]


# when the InfluxDB server can't be reached, show a short message instead of the error page
# якщо сервер недоступний або токен неправильний: замість сторінки з помилкою
# показуємо OFFLINE з адресою сервера і код 503 (Service Unavailable)
@influx_bp.errorhandler(HTTPError)
@influx_bp.errorhandler(ApiException)
def influxdb_unavailable(error):
    return render_template(
        "influx_status.html",
        connected=False,
        error=str(error),
        url=current_app.config["INFLUXDB_V2_URL"]
    ), 503


# /influxdb/ : меню з шістьма картками
@influx_bp.route("/")
def dashboard():
    return render_template("influx_dashboard.html")


# 7.1 /influxdb/ping : чи відповідає сервер (True/False → ONLINE/OFFLINE)
@influx_bp.route("/ping")
def ping():
    influxdb = get_influxdb()
    connected = influxdb.connection.ping()   # не падає, просто повертає False

    return render_template(
        "influx_status.html",
        connected=connected,
        url=current_app.config["INFLUXDB_V2_URL"]
    )


# 7.2 /influxdb/buckets : список buckets, які бачить наш токен
@influx_bp.route("/buckets")
def buckets():
    influxdb = get_influxdb()
    buckets_api = influxdb.connection.buckets_api()      # інструмент для роботи з buckets
    bucket_list = buckets_api.find_buckets().buckets     # відповідь сервера → беремо сам список

    # спільний шаблон data_list.html чекає список словників name/value
    return render_template(
        "data_list.html",
        title="InfluxDB Buckets",
        items=[
            {"name": bucket.name, "value": bucket.name}
            for bucket in bucket_list
        ]
    )



# 7.3 /influxdb/weather : температура в Аннесі за останній тиждень (графік + таблиця)
@influx_bp.route("/weather")
def weather():
    influxdb = get_influxdb()
    query_api = influxdb.connection.query_api()          # інструмент для Flux-запитів
    org = current_app.config["INFLUXDB_V2_ORG"]          # організація "iut"

    # Flux-запит, крок за кроком (|> = "потім"):
    #   bucket weather → останній тиждень → measurement weather_current
    #   → поле temperature_2m → станція (тег location) annecy → сортувати за часом
    # ЗМІНА: інший період: range(start: -24h) або -2w
    # ЗМІНА: інша величина: "relative_humidity_2m", "wind_speed_10m", "cloud_cover"…
    #        (тоді в influx_data.html поміняти °C на потрібну одиницю)
    # ЗМІНА: інше місто: "chambery", "geneva", "annemasse"… (і текст ANNECY у шаблоні)
    # ЗМІНА: середнє за годину: додати рядок |> aggregateWindow(every: 1h, fn: mean)
    flux_query = '''
    from(bucket: "weather")
      |> range(start: -1w)
      |> filter(fn: (r) => r._measurement == "weather_current")
      |> filter(fn: (r) => r._field == "temperature_2m")
      |> filter(fn: (r) => r.location == "annecy")
      |> sort(columns: ["_time"])
    '''

    # відповідь = список таблиць (одна на кожну серію), у кожній — записи (рядки)
    tables = query_api.query(
        org=org,
        query=flux_query
    )

    # усі рядки всіх таблиць → один простий список
    records = [
        record
        for table in tables
        for record in table.records
    ]

    data = [record.get_value() for record in records]       # значення (_value): температури
    time = [str(record.get_time()) for record in records]   # час (_time), як текст

    # шаблон показує останнє/max/min, графік Chart.js і таблицю
    # ЗМІНА: середнє можна додати в шаблоні: {{ data|sum / data|length }}
    return render_template(
        "influx_data.html",
        data=data,
        time=time
    )



# 7.3 /influxdb/measurements : які measurements є в bucket weather
@influx_bp.route("/measurements")
def measurements():
    influxdb = get_influxdb()
    query_api = influxdb.connection.query_api()
    org = current_app.config["INFLUXDB_V2_ORG"]

    # schema.measurements повертає один рядок на кожну назву measurement
    # ЗМІНА: інший bucket: "crypto", "forex", "system", "zwave"
    flux_query = '''
    import "influxdata/influxdb/schema"

    schema.measurements(bucket: "weather")
    '''

    tables = query_api.query(org=org, query=flux_query)

    measurement_list = [
        record.get_value()                  # у кожному рядку — одна назва
        for table in tables
        for record in table.records
    ]

    return render_template(
        "data_list.html",
        title="InfluxDB Measurements",
        items=[
            {"name": name, "value": name}   # у назви немає окремого значення → двічі назва
            for name in measurement_list
        ]
    )


# 7.3 /influxdb/fields : які поля (величини) є в bucket weather
@influx_bp.route("/fields")
def fields():
    influxdb = get_influxdb()
    query_api = influxdb.connection.query_api()
    org = current_app.config["INFLUXDB_V2_ORG"]

    # schema.fieldKeys: назви полів (temperature_2m, wind_speed_10m…)
    # ЗМІНА: інший bucket (як у measurements)
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


# 7.4 нарізати ряд на вікна: X = x значень (вхід), y = наступні x значень (що передбачити)
# приклад: [1..8], x=3 → X=[1,2,3],[2,3,4],[3,4,5]  y=[4,5,6],[5,6,7],[6,7,8]
# вікон: len(data) - 2*x + 1 (кожному потрібно 2*x значень)
def format_dataset(data, x):
    X = []
    y = []

    for i in range(len(data) - 2 * x + 1):
        X.append(data[i:i + x])            # x значень від позиції i
        y.append(data[i + x:i + 2 * x])    # наступні x значень

    return X, y

# /influxdb/format?x=3 : показати вікна на прикладі (InfluxDB тут не потрібен)
@influx_bp.route("/format")
def format_test():
    # ЗМІНА: інший приклад даних, напр. справжні температури [7.4, 7.1, 6.8, ...]
    data = [1, 2, 3, 4, 5, 6, 7, 8]

    x = request.args.get("x", default=3, type=int)   # x з адреси (?x=3); не число → 3
    x = max(1, min(x, len(data) // 2))               # тримати x від 1 до 4 (захист від 0 і завеликих)

    X, y = format_dataset(data, x)

    return render_template(
        "influx_format.html",
        data=data,
        X=X,
        y=y,
        sequence_length=x
    )
