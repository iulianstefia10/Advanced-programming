# Сторінка /rf/ (PW1): читає форму, перевіряє значення, викликає сервіс, показує rf_dashboard.html.

from flask import Blueprint, render_template, request

from services.random_forest import DATASETS, sensitivity_study, train_random_forest

# група сторінок Random Forest, адреса /rf/
rf_bp = Blueprint(
    "rf",
    __name__,
    url_prefix="/rf",
)


# прочитати форму й перевірити значення на сервері
# (обмеження min/max у HTML можна обійти, тому перевіряємо ще раз тут)
# неправильне значення → ValueError → червоне повідомлення на сторінці
def _read_params(form):
    # read the form and check the values
    dataset = form.get("dataset", "digits")
    if dataset not in DATASETS:
        # ЗМІНА: якщо додали датасет у DATASETS — оновити цей текст
        raise ValueError("Please choose iris, digits or diabetes.")

    # ЗМІНА: межі test_size (зараз 0.05–0.9), також min/max у rf_dashboard.html
    test_size = float(form.get("test_size", 0.1))        # текст → число; "abc" → ValueError
    if not 0.05 <= test_size <= 0.9:
        raise ValueError("test_size must be between 0.05 and 0.9.")

    # ЗМІНА: межі кількості дерев (зараз 1–500)
    n_estimators = int(form.get("n_estimators", 10))
    if not 1 <= n_estimators <= 500:
        raise ValueError("n_estimators must be between 1 and 500.")

    # порожньо = без обмеження глибини (None)
    max_depth = form.get("max_depth", "").strip()
    max_depth = int(max_depth) if max_depth else None
    if max_depth is not None and max_depth < 1:
        raise ValueError("max_depth must be at least 1 (or empty for no limit).")

    random_state = int(form.get("random_state", 42))

    # ЗМІНА: новий параметр (напр. min_samples_split) — прочитати тут, перевірити й повернути
    return dataset, test_size, n_estimators, max_depth, random_state


# /rf/ : GET — відкрити сторінку; POST — натиснули Train model або Sensitivity study
@rf_bp.route("/", methods=["GET", "POST"])
def dashboard():
    # значення у формі при першому відкритті
    # ЗМІНА: значення за замовчуванням (напр. "n_estimators": 100); також у функції сервісу
    params = {
        "dataset": "digits",
        "test_size": 0.1,
        "n_estimators": 10,
        "max_depth": "",
        "random_state": 42,
    }
    result = None    # результат кнопки Train
    study = None     # таблиця кнопки Study
    error = None     # текст помилки

    if request.method == "POST":
        try:
            dataset, test_size, n_estimators, max_depth, random_state = _read_params(request.form)
            # зберегти вибір користувача, щоб форма його «пам'ятала»
            params = {
                "dataset": dataset,
                "test_size": test_size,
                "n_estimators": n_estimators,
                "max_depth": max_depth or "",
                "random_state": random_state,
            }

            # обидві кнопки в одній формі: name="action", value="train" або "study"
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
            error = str(e)    # показати в червоній картці замість падіння сайту

    # Jinja заповнює rf_dashboard.html цими значеннями
    return render_template(
        "rf_dashboard.html",
        datasets=DATASETS,     # для списку датасетів у формі
        params=params,
        result=result,
        study=study,
        error=error,
    )
