# PW1 - Random Forest on iris, digits and diabetes

from sklearn.datasets import load_diabetes, load_digits, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

DATASETS = {
    "iris": {"loader": load_iris, "task": "classification"},
    "digits": {"loader": load_digits, "task": "classification"},
    "diabetes": {"loader": load_diabetes, "task": "regression"},
}

TEST_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5]
N_ESTIMATORS = [1, 10, 50, 100]


def train_random_forest(dataset, test_size=0.1, n_estimators=10,
                        max_depth=None, random_state=42):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    config = DATASETS[dataset]
    data = config["loader"]()
    is_classification = config["task"] == "classification"

    # stratify keeps the same class proportions in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target if is_classification else None,
    )

    # diabetes is a regression dataset, so we use a regressor for it
    model_class = RandomForestClassifier if is_classification else RandomForestRegressor
    model = model_class(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {
        "dataset": dataset,
        "task": config["task"],
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_size": test_size,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }

    if is_classification:
        result["metric_name"] = "Accuracy"
        result["score"] = accuracy_score(y_test, y_pred)
        # y_test first, then y_pred
        result["report"] = classification_report(y_test, y_pred, zero_division=0)
    else:
        result["metric_name"] = "R²"
        result["score"] = r2_score(y_test, y_pred)
        result["mse"] = mean_squared_error(y_test, y_pred)
        result["report"] = None

    return result


def sensitivity_study(dataset, test_sizes=TEST_SIZES, n_estimators_list=N_ESTIMATORS,
                      random_state=42):
    # train the model for every test_size and n_estimators
    rows = []
    for test_size in test_sizes:
        for n_estimators in n_estimators_list:
            result = train_random_forest(
                dataset,
                test_size=test_size,
                n_estimators=n_estimators,
                random_state=random_state,
            )
            rows.append({
                "test_size": test_size,
                "n_estimators": n_estimators,
                "score": result["score"],
            })
    # one line per test_size for the table on the page
    table = [
        {
            "test_size": test_size,
            "scores": [r["score"] for r in rows if r["test_size"] == test_size],
        }
        for test_size in test_sizes
    ]
    return {
        "dataset": dataset,
        "metric_name": "Accuracy" if DATASETS[dataset]["task"] == "classification" else "R²",
        "test_sizes": list(test_sizes),
        "n_estimators_list": list(n_estimators_list),
        "rows": rows,
        "table": table,
    }
