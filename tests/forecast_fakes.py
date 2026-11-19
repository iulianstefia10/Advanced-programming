# Fake InfluxDB used by the forecast tests: answers the schema queries
# and returns hourly weather for two stations (no network needed).

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

STATIONS = ["aix_les_bains", "annecy", "annemasse"]
FIELDS = ["relative_humidity_2m", "temperature_2m", "weather_code", "wind_speed_10m"]


class Record:
    def __init__(self, time, field, value, location):
        self._time, self._field, self._value = time, field, value
        self.values = {"location": location, "_field": field}

    def get_time(self):
        return self._time

    def get_field(self):
        return self._field

    def get_value(self):
        return self._value


def weather_records(hours=24 * 40):
    rng = np.random.default_rng(0)
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    base = 10 + 6 * np.sin(2 * np.pi * np.arange(hours + 6) / 24) + np.cumsum(rng.normal(0, 0.3, hours + 6))
    series = {
        "annecy": base[:hours],
        "annemasse": base[6:hours + 6] + rng.normal(0, 0.2, hours),  # sees the weather 6 h earlier
        "aix_les_bains": base[:hours] + 1 + rng.normal(0, 0.5, hours),
    }
    records = []
    for loc, temps in series.items():
        for h in range(hours):
            t = start + timedelta(hours=h)
            records.append(Record(t, "temperature_2m", float(temps[h]), loc))
            records.append(Record(t, "relative_humidity_2m", float(60 + 10 * np.sin(h / 7)), loc))
            records.append(Record(t, "weather_code", "cloudy", loc))  # text field: must be ignored
    return records


class FakeQueryApi:
    def __init__(self, records=None, fail=False):
        self.records = weather_records() if records is None else records
        self.fail = fail
        self.queries = []

    def query(self, org, query):
        self.queries.append(query)
        if self.fail:
            raise ConnectionError("connection refused")

        def table(values):
            return [SimpleNamespace(records=[SimpleNamespace(get_value=lambda v=v: v) for v in values])]

        if "schema.measurements" in query:
            return table(["weather_current"])
        if "measurementTagKeys" in query:
            return table(["_field", "_measurement", "_start", "_stop", "location"])
        if "measurementTagValues" in query:
            return table(STATIONS)
        if "measurementFieldKeys" in query:
            return table(FIELDS)
        wanted = [r for r in self.records
                  if f'"{r.get_field()}"' in query and f'"{r.values["location"]}"' in query]
        return [SimpleNamespace(records=wanted)]
