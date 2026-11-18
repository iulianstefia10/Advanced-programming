# Tests for the apparent temperature model (Session 1, 5.4)
# Reference values come from the NOAA heat index table
# and the Environment Canada wind chill table.

import os
import sys
import unittest
from types import SimpleNamespace

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from services import apparent  # noqa: E402


class FormulaTests(unittest.TestCase):

    def test_heat_index_matches_noaa_table(self):
        # (air in F, humidity %, heat index in F from the NOAA table)
        for t_f, rh, expected_f in [(90, 60, 100), (100, 40, 109), (86, 90, 105)]:
            with self.subTest(t=t_f, rh=rh):
                hi_c = apparent.heat_index(apparent.f_to_c(t_f), rh)
                self.assertAlmostEqual(apparent.c_to_f(hi_c), expected_f, delta=1)

    def test_wind_chill_matches_environment_canada_table(self):
        for t, wind, expected in [(-10, 20, -18), (-20, 30, -33), (0, 10, -3)]:
            with self.subTest(t=t, wind=wind):
                self.assertAlmostEqual(apparent.wind_chill(t, wind), expected, delta=0.5)

    def test_steadman(self):
        self.assertAlmostEqual(apparent.steadman(20, 50, 0), 19.85, places=1)
        # more wind = colder
        self.assertLess(apparent.steadman(20, 50, 30), apparent.steadman(20, 50, 0))


class ChoiceTests(unittest.TestCase):

    def test_formula_depends_on_conditions(self):
        self.assertEqual(apparent.apparent_temperature(32, 60, 10)["formula_key"], "heat_index")
        self.assertEqual(apparent.apparent_temperature(-5, 50, 20)["formula_key"], "wind_chill")
        self.assertEqual(apparent.apparent_temperature(20, 50, 10)["formula_key"], "steadman")

    def test_heat_index_not_used_in_dry_air(self):
        self.assertEqual(apparent.apparent_temperature(32, 20, 10)["formula_key"], "steadman")

    def test_wind_chill_not_used_without_wind(self):
        self.assertEqual(apparent.apparent_temperature(-5, 50, 2)["formula_key"], "steadman")

    def test_comfort_levels(self):
        self.assertEqual(apparent.comfort(-30)[1], "danger")
        self.assertEqual(apparent.comfort(22)[0], "Comfortable")
        self.assertEqual(apparent.comfort(60)[0], "Extreme danger: heat stroke")

    def test_wind_curve_goes_down(self):
        for t, model in [(0, "Wind chill"), (30, "Steadman")]:
            with self.subTest(t=t):
                curve = apparent.wind_curve(t, 60)
                points = curve["points"]
                self.assertIn(model, curve["model"])
                self.assertEqual(points[0]["wind"], 0)
                self.assertEqual(points[-1]["wind"], 60)
                # more wind always feels colder, also in the heat
                self.assertLess(points[-1]["value"], points[0]["value"])


class ValidationTests(unittest.TestCase):

    def test_good_values(self):
        values, errors = apparent.validate({"temperature": "25,5", "humidity": "40", "wind": "0"})
        self.assertEqual(errors, [])
        self.assertEqual(values["temperature"], 25.5)

    def test_bad_values(self):
        for field, value in [("temperature", "100"), ("temperature", "abc"),
                             ("humidity", "120"), ("humidity", "-1"),
                             ("wind", "-5"), ("wind", "nan")]:
            with self.subTest(f"{field}={value}"):
                form = {"temperature": "20", "humidity": "50", "wind": "10", field: value}
                _, errors = apparent.validate(form)
                self.assertTrue(errors)


class InfluxTests(unittest.TestCase):

    def test_current_weather_reads_last_values(self):
        def rec(field, value):
            return SimpleNamespace(get_field=lambda: field, get_value=lambda: value,
                                   get_time=lambda: "2026-11-18T08:00:00+00:00")

        table = SimpleNamespace(records=[rec("temperature_2m", 4.2),
                                         rec("relative_humidity_2m", 81),
                                         rec("wind_speed_10m", 12.5)])
        api = SimpleNamespace(query=lambda org, query: [table])
        weather = apparent.current_weather(api, "iut", "annecy")
        self.assertEqual(weather["temperature"], 4.2)
        self.assertEqual(weather["humidity"], 81.0)
        self.assertEqual(weather["wind"], 12.5)
        self.assertNotIn("reference", weather)


if __name__ == "__main__":
    unittest.main()
