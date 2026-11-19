"""Forecasting results for the README, printed as Markdown tables.

Run inside the container (about 1-3 minutes):
    python forecast_report.py
It reads the real InfluxDB data (same address and token as the site), trains the
five models on the Annecy temperature, then on one crypto and one forex series.
"""

import os
import sys

from influxdb_client import InfluxDBClient

from services import forecast

URL = os.environ.get("INFLUXDB_V2_URL", "http://192.168.168.88:8086")
ORG = os.environ.get("INFLUXDB_V2_ORG", "iut")


def report(query_api, bucket, title, days=30, window=24, horizon=24):
    print(f"\n## {title}\n")
    try:
        options = forecast.discover(query_api, ORG, bucket)
    except Exception as e:
        print(f"Skipped: {e}")
        return
    params = forecast.default_params(options)
    params.update(days=days, window=window, horizon=horizon)
    try:
        frame = forecast.load_frame(query_api, ORG, options, params)
        result = forecast.run_forecast(frame, params)
    except Exception as e:
        print(f"Skipped: {e}")
        return

    print(f"Target `{result['target']}`, {result['rows']} hours ({result['start']} to {result['end']} UTC), "
          f"{window} h -> {horizon} h, {len(result['columns'])} input series: "
          + ", ".join(f"`{c}`" for c in result["columns"]) + ".\n")
    print("| Model | MAE | RMSE | MAE, target only | Gain of the extra inputs | Time (s) |")
    print("|---|---|---|---|---|---|")
    for row in result["table"]:
        only = f"{row['mae_target_only']:.3f}" if "mae_target_only" in row else "–"
        gain = f"{row['gain']:+.1f} %" if "gain" in row else "–"
        best = "**" if row["key"] == result["best"] else ""
        print(f"| {best}{row['name']}{best} | {row['mae']:.3f} | {row['rmse']:.3f} | {only} | {gain} | {row['seconds']} |")
    sys.stdout.flush()


def main():
    token = os.environ.get("INFLUXDB_V2_TOKEN")
    with InfluxDBClient(url=URL, token=token, org=ORG, timeout=120_000) as client:
        api = client.query_api()
        report(api, "weather", "Annecy temperature, 24 h -> next 24 h")
        report(api, "weather", "Annecy temperature, 24 h -> next 6 h", horizon=6)
        report(api, "crypto", "Crypto (first series of the bucket)")
        report(api, "forex", "Forex (first series of the bucket)")


if __name__ == "__main__":
    main()
