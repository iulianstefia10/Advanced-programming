"""Session 1, 5.4 - Apparent temperature (how hot or cold it feels).

Three classic models, each valid in its own range:
- Heat index (NOAA / Rothfusz): hot and humid air, T >= 27 C and RH >= 40 %.
  The NOAA formula is written in Fahrenheit, so we convert C -> F -> C.
- Wind chill (Environment Canada / NWS 2001): cold and windy air,
  T <= 10 C and wind > 4.8 km/h.
- Steadman apparent temperature (Australian Bureau of Meteorology): every
  other case; uses temperature, humidity and wind. Open-Meteo uses the same idea.
"""

import math

LIMITS = {
    "temperature": (-50.0, 60.0),   # C
    "humidity": (0.0, 100.0),       # %
    "wind": (0.0, 200.0),           # km/h
}

# (upper bound in C, label, css level) - first match wins
COMFORT_LEVELS = [
    (-27.0, "Danger: frostbite in minutes", "danger"),
    (-10.0, "Very cold", "cold"),
    (0.0, "Cold", "cold"),
    (10.0, "Cool", "cool"),
    (20.0, "Fresh", "ok"),
    (27.0, "Comfortable", "ok"),
    (32.0, "Caution: fatigue possible", "warm"),
    (41.0, "Extreme caution: heat cramps possible", "hot"),
    (54.0, "Danger: heat exhaustion likely", "danger"),
    (math.inf, "Extreme danger: heat stroke", "danger"),
]


def c_to_f(t):
    return t * 9 / 5 + 32


def f_to_c(t):
    return (t - 32) * 5 / 9


def heat_index(t_c, rh):
    """NOAA heat index in C (formula in F, as published by the NWS)."""
    t = c_to_f(t_c)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    if (simple + t) / 2 < 80:
        return f_to_c(simple)

    hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh
          - 0.22475541 * t * rh - 6.83783e-3 * t * t
          - 5.481717e-2 * rh * rh + 1.22874e-3 * t * t * rh
          + 8.5282e-4 * t * rh * rh - 1.99e-6 * t * t * rh * rh)
    if rh < 13 and 80 <= t <= 112:
        hi -= ((13 - rh) / 4) * math.sqrt((17 - abs(t - 95)) / 17)
    elif rh > 85 and 80 <= t <= 87:
        hi += ((rh - 85) / 10) * ((87 - t) / 5)
    return f_to_c(hi)


def wind_chill(t_c, wind_kmh):
    """Environment Canada wind chill index in C (wind in km/h at 10 m)."""
    v = wind_kmh ** 0.16
    return 13.12 + 0.6215 * t_c - 11.37 * v + 0.3965 * t_c * v


def steadman(t_c, rh, wind_kmh):
    """Steadman apparent temperature (shade), as used by the Australian BoM."""
    e = rh / 100 * 6.105 * math.exp(17.27 * t_c / (237.7 + t_c))  # vapour pressure, hPa
    ws = wind_kmh / 3.6                                             # m/s
    return t_c + 0.33 * e - 0.70 * ws - 4.00


def comfort(value):
    for upper, label, level in COMFORT_LEVELS:
        if value < upper:
            return label, level
    return COMFORT_LEVELS[-1][1:]


def apparent_temperature(t_c, rh, wind_kmh):
    """Pick the model valid for these conditions and return all the details."""
    hi_valid = t_c >= 27 and rh >= 40
    wc_valid = t_c <= 10 and wind_kmh > 4.8

    models = [
        {"name": "Heat index (NOAA)", "key": "heat_index",
         "value": heat_index(t_c, rh), "valid": hi_valid,
         "range": "T ≥ 27 °C and humidity ≥ 40 %"},
        {"name": "Wind chill (Environment Canada)", "key": "wind_chill",
         "value": wind_chill(t_c, wind_kmh) if wind_kmh > 0 else t_c, "valid": wc_valid,
         "range": "T ≤ 10 °C and wind > 4.8 km/h"},
        {"name": "Steadman apparent temperature", "key": "steadman",
         "value": steadman(t_c, rh, wind_kmh), "valid": True,
         "range": "any conditions (used when the others do not apply)"},
    ]

    if wc_valid:
        chosen = models[1]
    elif hi_valid:
        chosen = models[0]
    else:
        chosen = models[2]

    label, level = comfort(chosen["value"])
    return {
        "temperature": t_c,
        "humidity": rh,
        "wind": wind_kmh,
        "value": round(chosen["value"], 1),
        "difference": round(chosen["value"] - t_c, 1),
        "formula": chosen["name"],
        "formula_key": chosen["key"],
        "comfort": label,
        "level": level,
        "models": [dict(m, value=round(m["value"], 1)) for m in models],
    }


def wind_curve(t_c, rh, max_wind=60, step=5):
    """Felt temperature of the same air at wind speeds 0..max_wind km/h.

    The heat index does not use the wind at all, so a curve made with it would
    be flat. We draw the model that really depends on the wind: the wind chill
    in the cold (T <= 10 C), Steadman otherwise.
    """
    if t_c <= 10:
        name = "Wind chill (Environment Canada)"
        points = [{"wind": w, "value": round(wind_chill(t_c, w) if w > 4.8 else t_c, 1)}
                  for w in range(0, max_wind + 1, step)]
    else:
        name = "Steadman apparent temperature"
        points = [{"wind": w, "value": round(steadman(t_c, rh, w), 1)}
                  for w in range(0, max_wind + 1, step)]
    return {"model": name, "points": points}


def validate(form):
    """Read temperature, humidity, wind from a form. Returns (values, errors)."""
    values, errors = {}, []
    labels = {"temperature": "Temperature", "humidity": "Humidity", "wind": "Wind speed"}
    defaults = {"temperature": 30.0, "humidity": 60.0, "wind": 10.0}
    for name, (lo, hi) in LIMITS.items():
        raw = str(form.get(name, defaults[name])).strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{labels[name]} must be a number (got '{raw}').")
            values[name] = defaults[name]
            continue
        if math.isnan(value) or not lo <= value <= hi:
            errors.append(f"{labels[name]} must be between {lo:g} and {hi:g}.")
            value = defaults[name]
        values[name] = value
    return values, errors


# --------------------------------------------------------------------------
# Current weather from InfluxDB (Sessions 2-3, Exercise 2)
# --------------------------------------------------------------------------
# Field names of the "weather_current" measurement. Check them on /influxdb/fields
# and change them here if they differ.
FIELDS = {
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "wind": "wind_speed_10m",
    "reference": "apparent_temperature",   # value computed by the weather service, if present
}


def current_weather(query_api, org, location="annecy"):
    """Last value of each field for a location. Missing fields are left out."""
    names = " or ".join(f'r._field == "{f}"' for f in FIELDS.values())
    flux = f'''
    from(bucket: "weather")
      |> range(start: -1d)
      |> filter(fn: (r) => r._measurement == "weather_current")
      |> filter(fn: (r) => r.location == "{location}")
      |> filter(fn: (r) => {names})
      |> last()
    '''
    tables = query_api.query(org=org, query=flux)
    by_field = {
        record.get_field(): (record.get_value(), record.get_time())
        for table in tables for record in table.records
    }
    result = {}
    for key, field in FIELDS.items():
        if field in by_field:
            value, time = by_field[field]
            result[key] = float(value)
            result["time"] = str(time)
    return result
