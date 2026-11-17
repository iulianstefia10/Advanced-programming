from flask import Blueprint, render_template, request

from services.random_forest import DATASETS, sensitivity_study, train_random_forest

rf_bp = Blueprint(
    "rf",
    __name__,
    url_prefix="/rf",
)


def _read_params(form):
    # read the form and check the values
    dataset = form.get("dataset", "digits")
    if dataset not in DATASETS:
        raise ValueError("Please choose iris, digits or diabetes.")

    test_size = float(form.get("test_size", 0.1))
    if not 0.05 <= test_size <= 0.9:
        raise ValueError("test_size must be between 0.05 and 0.9.")

    n_estimators = int(form.get("n_estimators", 10))
    if not 1 <= n_estimators <= 500:
        raise ValueError("n_estimators must be between 1 and 500.")

    max_depth = form.get("max_depth", "").strip()
    max_depth = int(max_depth) if max_depth else None
    if max_depth is not None and max_depth < 1:
        raise ValueError("max_depth must be at least 1 (or empty for no limit).")

    random_state = int(form.get("random_state", 42))

    return dataset, test_size, n_estimators, max_depth, random_state


@rf_bp.route("/", methods=["GET", "POST"])
def dashboard():
    params = {
        "dataset": "digits",
        "test_size": 0.1,
        "n_estimators": 10,
        "max_depth": "",
        "random_state": 42,
    }
    result = None
    study = None
    error = None

    if request.method == "POST":
        try:
            dataset, test_size, n_estimators, max_depth, random_state = _read_params(request.form)
            params = {
                "dataset": dataset,
                "test_size": test_size,
                "n_estimators": n_estimators,
                "max_depth": max_depth or "",
                "random_state": random_state,
            }

            if request.form.get("action") == "study":
                study = sensitivity_study(dataset, random_state=random_state)
            else:
                result = train_random_forest(
                    dataset,
                    test_size=test_size,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                )
        except ValueError as e:
            error = str(e)

    return render_template(
        "rf_dashboard.html",
        datasets=DATASETS,
        params=params,
        result=result,
        study=study,
        error=error,
    )
