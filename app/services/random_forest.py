# PW1 - Random Forest on iris, digits and diabetes
# PW1: навчити випадковий ліс (багато дерев рішень, які голосують) на одному з трьох
# датасетів scikit-learn і порахувати результат. Без Flask-коду: викликається з routes/random_forest.py.

from sklearn.datasets import load_diabetes, load_digits, load_iris   # ЗМІНА: новий датасет, напр. load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,          # частка правильних відповідей (класифікація)
    classification_report,   # precision / recall / f1 для кожного класу
    mean_squared_error,      # MSE: середня квадратична помилка (регресія)
    r2_score,                # R²: наскільки модель краща за «завжди середнє» (регресія)
)
from sklearn.model_selection import train_test_split

# датасети: функція, що їх завантажує, і тип задачі
# diabetes — регресія (відповідь — число), тому для нього регресор і R²
# ЗМІНА: додати датасет: "wine": {"loader": load_wine, "task": "classification"}
#        (форма й перевірка беруть список звідси самі; оновити лише текст помилки в маршруті)
DATASETS = {
    "iris": {"loader": load_iris, "task": "classification"},
    "digits": {"loader": load_digits, "task": "classification"},
    "diabetes": {"loader": load_diabetes, "task": "regression"},
}

# значення для sensitivity study: 5 test_size × 4 кількості дерев = 20 навчань
# ЗМІНА: інші значення (таблиця на сторінці зміниться сама)
TEST_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5]
N_ESTIMATORS = [1, 10, 50, 100]


# навчити один ліс і повернути словник з результатом для сторінки
# ЗМІНА: новий параметр, напр. min_samples_split=2 → додати сюди, у model_class(...) і в маршрут
def train_random_forest(dataset, test_size=0.1, n_estimators=10,
                        max_depth=None, random_state=42):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    config = DATASETS[dataset]
    data = config["loader"]()                    # data.data = ознаки (X), data.target = відповіді (y)
    is_classification = config["task"] == "classification"

    # stratify keeps the same class proportions in train and test
    # поділ на навчальну і тестову частини; random_state = той самий поділ щоразу
    # stratify: однакові пропорції класів у train і test (для регресії класів немає → None)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target if is_classification else None,
    )

    # diabetes is a regression dataset, so we use a regressor for it
    # класифікатор (голосування дерев) або регресор (середнє дерев)
    model_class = RandomForestClassifier if is_classification else RandomForestRegressor
    model = model_class(
        n_estimators=n_estimators,     # кількість дерев
        max_depth=max_depth,           # макс. глибина дерева (None = без обмеження)
        random_state=random_state,     # той самий ліс щоразу
    )
    model.fit(X_train, y_train)        # навчання: побудувати дерева
    y_pred = model.predict(X_test)     # прогноз на даних, яких модель не бачила

    # інформація для рядка над результатом («150 samples, 4 features · train 120 / test 30»)
    result = {
        "dataset": dataset,
        "task": config["task"],
        "n_samples": data.data.shape[0],    # рядки = приклади
        "n_features": data.data.shape[1],   # колонки = ознаки
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_size": test_size,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    # ЗМІНА: точність на навчальних даних (видно перенавчання):
    #        result["train_score"] = model.score(X_train, y_train)
    # ЗМІНА: найважливіші ознаки: model.feature_importances_

    if is_classification:
        result["metric_name"] = "Accuracy"
        result["score"] = accuracy_score(y_test, y_pred)
        # y_test first, then y_pred
        # спочатку правильні відповіді, потім прогнози (у завданні було навпаки)
        result["report"] = classification_report(y_test, y_pred, zero_division=0)
    else:
        result["metric_name"] = "R²"
        result["score"] = r2_score(y_test, y_pred)
        result["mse"] = mean_squared_error(y_test, y_pred)
        result["report"] = None

    return result


# sensitivity study: навчити ліс для кожної пари (test_size, кількість дерев)
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
                "score": result["score"],      # accuracy або R²
            })
    # one line per test_size for the table on the page
    # згрупувати 20 результатів у таблицю: рядок = test_size, 4 числа = 1/10/50/100 дерев
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
        "n_estimators_list": list(n_estimators_list),   # заголовки колонок таблиці
        "rows": rows,
        "table": table,
    }
