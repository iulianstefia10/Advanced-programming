"""Time-series forecasting with several variables and several stations.

Sessions 2-3, Exercise 2 (7.2-7.4) prepared the data: read a series from
InfluxDB and cut it into windows of x points to predict the x next points.
Here we use those windows to forecast, as asked by the teacher: the target
(e.g. the temperature of Annecy) is predicted from several variables
(temperature, humidity, wind, pressure...) of several stations.

Steps:
1. discover()   - list measurements, tags, stations and fields of a bucket
2. load_frame() - one hourly table: one column per (variable, station)
3. run_forecast() - windows, chronological split, scaling, 5 models, metrics

TensorFlow is imported only when a neural network is trained.
"""

import time

import numpy as np
import pandas as pd

from utils.dataset import make_windows

SEED = 42
BUCKETS = ["weather", "crypto", "forex"]

# Preferred choices when they exist in the bucket (otherwise the first one found).
# A list means "the first of these that exists".
PREFERRED = {
    "weather": {
        "measurement": ["weather_current"],
        "tag": ["location"],
        "target_series": ["annecy"],
        "target_field": ["temperature_2m"],
        "extra_fields": ["relative_humidity_2m", "surface_pressure", "pressure_msl", "wind_speed_10m"],
        # stations less than about 60 km from Annecy: far cities (Paris, Bordeaux...)
        # say nothing about the next hours in Annecy and only add noise
        "extra_series": ["aix_les_bains", "annemasse", "chambery", "geneve", "rumilly"],
    },
    "crypto": {
        "target_series": ["BTCUSDT", "BTC", "bitcoin"],
        "target_field": ["price", "close", "last_price", "last", "price_usd"],
    },
    "forex": {
        "target_series": ["USD"],
        "target_field": ["rate", "close", "price"],
    },
}
MAX_EXTRA_FIELDS = 3

LIMITS = {
    "window": (3, 168),     # hours of history given to the model
    "horizon": (1, 72),     # hours to predict
    "days": (3, 365),       # days of data read from InfluxDB
    "epochs": (1, 300),     # max epochs of the neural networks (early stopping)
}
DEFAULT_PARAMS = {"window": 24, "horizon": 24, "days": 60, "epochs": 50}
# the other series give their last hours only (the target keeps the whole window)
EXTRA_HOURS = 6

MAX_EXTRA_SERIES = 6
MIN_TRAIN_WINDOWS = 30
TEST_SHARE = 0.2

MODELS = [
    ("naive", "Naive (last value)"),
    ("seasonal", "Seasonal naive (same hour, previous day)"),
    ("linear", "Linear regression (ridge)"),
    ("mlp", "MLP (dense network)"),
    ("lstm", "LSTM (recurrent network)"),
]
LEARNED = ["linear", "mlp", "lstm"]


# --------------------------------------------------------------------------
# 1. What is in the bucket
# --------------------------------------------------------------------------
def flux_string(value):
    """Quote a value for a Flux query."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _first(candidates, available, default=None):
    """First preferred value that exists (candidates may be a string or a list)."""
    if isinstance(candidates, str):
        candidates = [candidates]
    for c in candidates or []:
        if c in available:
            return c
    return default


def _values(query_api, org, flux):
    tables = query_api.query(org=org, query=flux)
    return sorted({str(r.get_value()) for t in tables for r in t.records})


def discover(query_api, org, bucket, measurement=None, tag=None):
    """Measurements, tag key, tag values (stations, coins...) and fields."""
    schema = 'import "influxdata/influxdb/schema"\n'
    b = flux_string(bucket)
    preferred = PREFERRED.get(bucket, {})

    measurements = _values(query_api, org, schema + f"schema.measurements(bucket: {b})")
    if not measurements:
        raise ValueError(f"The bucket '{bucket}' has no data in the last 30 days.")
    if measurement not in measurements:
        measurement = _first(preferred.get("measurement"), measurements, measurements[0])
    m = flux_string(measurement)

    tag_keys = [
        k for k in _values(query_api, org, schema +
                           f"schema.measurementTagKeys(bucket: {b}, measurement: {m})")
        if not k.startswith("_")
    ]
    if tag not in tag_keys:
        tag = _first(preferred.get("tag"), tag_keys, tag_keys[0] if tag_keys else None)

    tag_values = []
    if tag:
        tag_values = _values(
            query_api, org, schema +
            f"schema.measurementTagValues(bucket: {b}, measurement: {m}, tag: {flux_string(tag)})"
        )

    fields = _values(query_api, org, schema +
                     f"schema.measurementFieldKeys(bucket: {b}, measurement: {m})")

    return {
        "bucket": bucket,
        "measurements": measurements,
        "measurement": measurement,
        "tag_keys": tag_keys,
        "tag": tag,
        "tag_values": tag_values,
        "fields": fields,
    }


def default_params(options):
    """Values shown in the form the first time."""
    params = dict(DEFAULT_PARAMS)
    preferred = PREFERRED.get(options["bucket"], {})
    fields, values = options["fields"], options["tag_values"]

    params["target_field"] = _first(preferred.get("target_field"), fields, fields[0] if fields else "")
    params["target_series"] = _first(preferred.get("target_series"), values, values[0] if values else "")
    params["extra_fields"] = [f for f in preferred.get("extra_fields", [])
                              if f in fields and f != params["target_field"]][:MAX_EXTRA_FIELDS]
    params["extra_series"] = [v for v in preferred.get("extra_series", [])
                              if v in values and v != params["target_series"]][:MAX_EXTRA_SERIES]
    return params


def _getlist(form, name):
    if hasattr(form, "getlist"):
        return form.getlist(name)
    value = form.get(name, [])
    return value if isinstance(value, list) else [value]


def validate(form, options):
    """Check the form against what exists in InfluxDB. Returns (params, errors)."""
    params, errors = dict(DEFAULT_PARAMS), []

    for name, (lo, hi) in LIMITS.items():
        raw = str(form.get(name, DEFAULT_PARAMS[name])).strip()
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"{name} must be a whole number (got '{raw}').")
            continue
        if not lo <= value <= hi:
            errors.append(f"{name} must be between {lo} and {hi}.")
            continue
        params[name] = value

    fields, values = options["fields"], options["tag_values"]

    params["target_field"] = form.get("target_field", "")
    if params["target_field"] not in fields:
        errors.append("Choose a variable to predict that exists in the bucket.")

    params["target_series"] = form.get("target_series", "") if options["tag"] else ""
    if options["tag"] and params["target_series"] not in values:
        errors.append(f"Choose a {options['tag']} that exists in the bucket.")

    extra_fields = _getlist(form, "extra_fields")
    extra_series = _getlist(form, "extra_series")
    if any(f not in fields for f in extra_fields) or any(v not in values for v in extra_series):
        errors.append("Unknown variable or station in the inputs.")
    params["extra_fields"] = [f for f in fields if f in extra_fields and f != params["target_field"]]
    params["extra_series"] = [v for v in values if v in extra_series and v != params["target_series"]]
    if len(params["extra_series"]) > MAX_EXTRA_SERIES:
        errors.append(f"Choose at most {MAX_EXTRA_SERIES} other {options['tag'] or 'series'}.")

    hours = params["days"] * 24
    needed = 2 * (params["window"] + params["horizon"]) + MIN_TRAIN_WINDOWS
    if not errors and hours < needed:
        errors.append(f"{params['days']} days is too short for a window of {params['window']} h "
                      f"and a horizon of {params['horizon']} h: read at least {needed // 24 + 1} days.")
    return params, errors


# --------------------------------------------------------------------------
# 2. One hourly table
# --------------------------------------------------------------------------
def column_name(field, series):
    return f"{field} @ {series}" if series else field


def build_query(options, params):
    fields = [params["target_field"]] + params["extra_fields"]
    field_filter = " or ".join(f"r._field == {flux_string(f)}" for f in fields)
    lines = [
        f"from(bucket: {flux_string(options['bucket'])})",
        f"  |> range(start: -{int(params['days'])}d)",
        f"  |> filter(fn: (r) => r._measurement == {flux_string(options['measurement'])})",
        f"  |> filter(fn: (r) => {field_filter})",
    ]
    if options["tag"]:
        series = [params["target_series"]] + params["extra_series"]
        tag = flux_string(options["tag"])
        tag_filter = " or ".join(f"r[{tag}] == {flux_string(v)}" for v in series)
        lines.append(f"  |> filter(fn: (r) => {tag_filter})")
    # one value per hour; "last" also works if a field is not a number
    lines.append("  |> aggregateWindow(every: 1h, fn: last, createEmpty: false)")
    return "\n".join(lines)


def load_frame(query_api, org, options, params):
    """Hourly table: index = time (UTC), one column per variable @ station."""
    tables = query_api.query(org=org, query=build_query(options, params))
    tag = options["tag"]
    rows = []
    for table in tables:
        for record in table.records:
            try:
                value = float(record.get_value())
            except (TypeError, ValueError):
                continue
            series = record.values.get(tag, "") if tag else ""
            rows.append((record.get_time(), column_name(record.get_field(), series), value))

    if not rows:
        raise ValueError("InfluxDB returned no data for this choice.")

    data = pd.DataFrame(rows, columns=["time", "column", "value"])
    data["time"] = pd.to_datetime(data["time"], utc=True)
    frame = data.pivot_table(index="time", columns="column", values="value", aggfunc="mean")
    frame = frame.sort_index().resample("1h").mean()
    frame = frame.interpolate(limit=3, limit_area="inside")  # fill holes of up to 3 hours

    target = column_name(params["target_field"], params["target_series"])
    if target not in frame.columns:
        raise ValueError(f"No data for the target {target}.")

    # drop inputs that are almost empty, then the hours where something is missing
    keep = [c for c in frame.columns if c == target or frame[c].notna().mean() >= 0.8]
    dropped = sorted(set(frame.columns) - set(keep))
    frame = frame[[target] + [c for c in keep if c != target]].dropna()
    frame.attrs["dropped"] = dropped
    return frame


# --------------------------------------------------------------------------
# 3. Windows, split, models
# --------------------------------------------------------------------------
def split_points(n_windows, window, horizon, test_share=TEST_SHARE):
    """Chronological split with a gap, so no hour is both in train and in test.

    Window i uses the hours i .. i + window + horizon - 1. The first test window
    starts after the last hour used by the training windows.
    """
    n_test = max(1, int(round(n_windows * test_share)))
    test_start = n_windows - n_test
    n_train = test_start - (window + horizon - 1)
    if n_train < MIN_TRAIN_WINDOWS:
        raise ValueError(
            f"Not enough hours of data: {n_windows + window + horizon - 1} hours give only "
            f"{max(n_train, 0)} training windows (need {MIN_TRAIN_WINDOWS}). "
            "Read more days or use a smaller window / horizon."
        )
    return n_train, test_start


def metrics(y_true, y_pred):
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae_by_step": [float(v) for v in np.mean(np.abs(err), axis=0)],
    }


def predict_naive(X, horizon, target_col=0):
    return np.repeat(X[:, -1, target_col][:, None], horizon, axis=1)


def predict_seasonal(X, horizon, target_col=0, period=24):
    """Value of the same hour one day before (needs window >= 24 and horizon <= 24)."""
    window = X.shape[1]
    idx = [window - period + k for k in range(horizon)]
    return X[:, idx, target_col]


def _keras_model(kind, input_shape, horizon):
    from tensorflow.keras import Input
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2

    inputs = Input(shape=input_shape)
    if kind == "mlp":  # input already flat: (features,)
        x = Dense(64, activation="relu", kernel_regularizer=l2(1e-3))(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(1e-3))(x)
    else:              # LSTM reads (hours, series)
        x = LSTM(32)(inputs)
        x = Dropout(0.2)(x)
    outputs = Dense(horizon)(x)
    return Model(inputs=inputs, outputs=outputs)


ALPHAS = np.logspace(-1, 5, 13)


def choose_alpha(X_train, y_train, gap):
    """Strength of the ridge regularisation, chosen BY TIME.

    The usual RidgeCV uses leave-one-out on overlapping windows, which almost
    sees the answer and picks too small an alpha. Here: train on the first 80 %
    of the training windows, validate on the last 20 % (after a gap), keep the
    alpha with the smallest error.
    """
    from sklearn.linear_model import Ridge

    n = len(X_train)
    n_val = max(1, int(n * 0.2))
    n_fit = n - n_val - gap
    if n_fit < 10:
        return 10.0
    best, best_err = 10.0, np.inf
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(X_train[:n_fit], y_train[:n_fit])
        err = np.mean(np.abs(model.predict(X_train[n - n_val:]) - y_train[n - n_val:]))
        if err < best_err:
            best, best_err = float(alpha), err
    return best


def train_predict(kind, X_train, y_train, X_test, epochs, gap=0):
    """Train one learned model on scaled data and predict the scaled test targets."""
    if kind == "linear":
        from sklearn.linear_model import Ridge

        alpha = choose_alpha(X_train, y_train, gap)
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model.predict(X_test), {"alpha": alpha}

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.utils.set_random_seed(SEED)
    model = _keras_model(kind, X_train.shape[1:], y_train.shape[1])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=32, verbose=0,
        validation_split=0.1,  # the last 10 % of the training windows, i.e. the most recent ones
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    )
    return model.predict(X_test, verbose=0), {"epochs_run": len(history.history["loss"])}


def model_inputs(kind, X_scaled, cols):
    """What each model reads.

    Every series is given RELATIVE to its last value in the window (its recent
    changes), so a model is not confused when the level drifts (colder in
    November than in October). The level of the target is kept as one input.
    Linear and MLP: the whole window of the target + the last EXTRA_HOURS hours
    of the other series, flattened (fewer inputs = less overfitting).
    LSTM: the whole window of every chosen series, hour by hour.
    """
    X = X_scaled[:, :, cols]
    level = X[:, -1, 0]
    X = X - X[:, -1:, :]
    window = X.shape[1]
    if kind == "lstm":
        level_step = np.repeat(level[:, None, None], window, axis=1)
        return np.concatenate([X, level_step], axis=2)
    parts = [X[:, :-1, 0], level[:, None]]
    if X.shape[2] > 1:
        k = min(EXTRA_HOURS, window)
        parts.append(X[:, window - k:-1, 1:].reshape(len(X), -1))
    return np.concatenate(parts, axis=1)


def run_forecast(frame, params, models=None):
    """Train and compare the models. frame: first column = target."""
    models = models or [key for key, _ in MODELS]
    window, horizon = params["window"], params["horizon"]
    values = frame.to_numpy(dtype=float)
    columns = list(frame.columns)

    X_all, y = make_windows(values, 0, window, horizon)
    n_train, test_start = split_points(len(X_all), window, horizon)

    # scaling fitted on the hours used by the training windows only
    train_rows = values[: n_train + window + horizon - 1]
    mean = train_rows.mean(axis=0)
    std = train_rows.std(axis=0)
    std[std == 0] = 1.0
    X_scaled = (X_all - mean) / std

    # the learned models predict the CHANGE from the last known value
    # (if they learn nothing, they give the naive forecast instead of nonsense)
    last = X_all[:, -1, 0]
    change = y - last[:, None]
    change_std = float(change[:n_train].std()) or 1.0
    change_scaled = change / change_std

    y_test = y[test_start:]
    results = {}

    def add(key, variant, pred, seconds, extra=None):
        results.setdefault(key, {})[variant] = dict(metrics(y_test, pred), seconds=round(seconds, 2), **(extra or {}))
        results[key].setdefault("prediction", pred)

    # baselines (no training)
    start = time.time()
    if "naive" in models:
        add("naive", "all", predict_naive(X_all[test_start:], horizon), time.time() - start)
    if "seasonal" in models and window >= 24 and horizon <= 24:
        add("seasonal", "all", predict_seasonal(X_all[test_start:], horizon), time.time() - start)

    # learned models, with all inputs and with the target only
    variants = [("all", list(range(len(columns))))]
    if len(columns) > 1:
        variants.append(("target_only", [0]))
    for key in [k for k in LEARNED if k in models]:
        for variant, cols in variants:
            start = time.time()
            inputs = model_inputs(key, X_scaled, cols)
            pred_scaled, extra = train_predict(
                key, inputs[:n_train], change_scaled[:n_train], inputs[test_start:], params["epochs"],
                gap=window + horizon - 1,
            )
            pred = last[test_start:, None] + np.asarray(pred_scaled) * change_std
            if variant == "all":
                results.setdefault(key, {})["prediction"] = pred
            add(key, variant, pred, time.time() - start, extra)

    names = dict(MODELS)
    table = []
    for key, _ in MODELS:
        if key not in results:
            continue
        row = {"key": key, "name": names[key], **results[key]["all"]}
        if "target_only" in results[key]:
            only = results[key]["target_only"]["mae"]
            row["mae_target_only"] = only
            row["gain"] = (only - row["mae"]) / only * 100 if only else 0.0
        table.append(row)
    best = min(table, key=lambda r: r["mae"])["key"]

    # chart: the last test window (history, real future, every prediction)
    w = len(X_all) - 1  # last window
    times = frame.index
    history_times = times[w:w + window]
    future_times = times[w + window:w + window + horizon]
    chart = {
        "history": [round(float(v), 2) for v in values[w:w + window, 0]],
        "real": [round(float(v), 2) for v in y[w]],
        "history_times": [t.strftime("%d/%m %Hh") for t in history_times],
        "future_times": [t.strftime("%d/%m %Hh") for t in future_times],
        "predictions": {k: [round(float(v), 2) for v in results[k]["prediction"][-1]] for k in results},
        "mae_by_step": {r["key"]: [round(v, 3) for v in r["mae_by_step"]] for r in table},
    }

    return {
        "table": table,
        "best": best,
        "names": names,
        "chart": chart,
        "columns": columns,
        "dropped": frame.attrs.get("dropped", []),
        "rows": len(frame),
        "start": times[0].strftime("%Y-%m-%d %H:%M"),
        "end": times[-1].strftime("%Y-%m-%d %H:%M"),
        "n_train": int(n_train),
        "n_test": int(len(y_test)),
        "window": window,
        "horizon": horizon,
        "target": columns[0],
    }
