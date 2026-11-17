# Tests for app/services/random_forest.py
# Run: python -m unittest discover -s tests -v

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.random_forest import (  # noqa: E402
    DATASETS,
    N_ESTIMATORS,
    TEST_SIZES,
    sensitivity_study,
    train_random_forest,
)


class TrainRandomForestTests(unittest.TestCase):

    def test_classification_datasets_use_accuracy(self):
        for dataset in ("iris", "digits"):
            with self.subTest(dataset=dataset):
                result = train_random_forest(dataset, test_size=0.2, n_estimators=10)
                self.assertEqual(result["task"], "classification")
                self.assertEqual(result["metric_name"], "Accuracy")
                self.assertGreaterEqual(result["score"], 0.0)
                self.assertLessEqual(result["score"], 1.0)
                self.assertIn("precision", result["report"])

    def test_diabetes_is_treated_as_regression(self):
        result = train_random_forest("diabetes", test_size=0.2, n_estimators=10)
        self.assertEqual(result["task"], "regression")
        self.assertEqual(result["metric_name"], "R²")
        self.assertIsNone(result["report"])
        self.assertIn("mse", result)
        self.assertGreater(result["mse"], 0)
        self.assertGreater(result["score"], 0)

    def test_sizes_of_train_and_test_sets(self):
        result = train_random_forest("iris", test_size=0.1, n_estimators=10)
        self.assertEqual(result["n_samples"], 150)
        self.assertEqual(result["n_features"], 4)
        self.assertEqual(result["n_test"], 15)
        self.assertEqual(result["n_train"], 135)

    def test_same_random_state_gives_same_score(self):
        first = train_random_forest("digits", test_size=0.3, n_estimators=10, random_state=7)
        second = train_random_forest("digits", test_size=0.3, n_estimators=10, random_state=7)
        self.assertEqual(first["score"], second["score"])

    def test_split_is_stratified(self):
        # iris has 50 flowers per class, so 10 of each in the test set
        result = train_random_forest("iris", test_size=0.2, n_estimators=10)
        supports = [
            int(line.split()[-1])
            for line in result["report"].splitlines()
            if line.strip()[:1].isdigit()
        ]
        self.assertEqual(supports, [10, 10, 10])

    def test_report_matches_accuracy(self):
        result = train_random_forest("digits", test_size=0.2, n_estimators=10)
        accuracy_line = next(
            line for line in result["report"].splitlines()
            if line.strip().startswith("accuracy")
        )
        self.assertAlmostEqual(float(accuracy_line.split()[1]), result["score"], places=2)

    def test_max_depth_is_used(self):
        result = train_random_forest("digits", test_size=0.2, n_estimators=10, max_depth=2)
        self.assertEqual(result["max_depth"], 2)
        full = train_random_forest("digits", test_size=0.2, n_estimators=10)
        self.assertLess(result["score"], full["score"])

    def test_more_trees_is_better_than_one_tree(self):
        one_tree = train_random_forest("digits", test_size=0.2, n_estimators=1)
        forest = train_random_forest("digits", test_size=0.2, n_estimators=100)
        self.assertGreater(forest["score"], one_tree["score"])

    def test_unknown_dataset_raises_error(self):
        with self.assertRaises(ValueError):
            train_random_forest("mnist")


class SensitivityStudyTests(unittest.TestCase):

    def test_table_has_one_score_per_combination(self):
        study = sensitivity_study("iris")
        self.assertEqual(len(study["rows"]), len(TEST_SIZES) * len(N_ESTIMATORS))
        self.assertEqual([line["test_size"] for line in study["table"]], TEST_SIZES)
        for line in study["table"]:
            self.assertEqual(len(line["scores"]), len(N_ESTIMATORS))

    def test_metric_name_depends_on_dataset(self):
        self.assertEqual(sensitivity_study("iris", [0.2], [10])["metric_name"], "Accuracy")
        self.assertEqual(sensitivity_study("diabetes", [0.2], [10])["metric_name"], "R²")

    def test_scores_are_the_same_as_a_single_training(self):
        study = sensitivity_study("digits", test_sizes=[0.3], n_estimators_list=[10])
        single = train_random_forest("digits", test_size=0.3, n_estimators=10)
        self.assertEqual(study["table"][0]["scores"][0], single["score"])

    def test_all_datasets_are_available(self):
        self.assertEqual(set(DATASETS), {"iris", "digits", "diabetes"})


if __name__ == "__main__":
    unittest.main()
