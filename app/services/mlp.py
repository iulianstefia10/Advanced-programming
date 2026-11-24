"""PW2 - Training an MLP classifier on MNIST with TensorFlow/Keras.

Service layer: parameter validation, model building, training and the
sensitivity study. TensorFlow is imported lazily inside the functions so the
Flask app starts fast and the validation logic can be tested without TF.
"""

import time

SEED = 42

ACTIVATIONS = ["relu", "sigmoid", "tanh"]
OPTIMIZERS = ["adam", "sgd", "rmsprop"]
# Labels are one-hot encoded (to_categorical), so only losses that accept
# one-hot targets are allowed. sparse_categorical_crossentropy would crash.
# Дозволені значення форми; sparse_* немає, бо мітки one-hot
LOSSES = ["categorical_crossentropy", "mean_squared_error"]

DEFAULTS = {
    "hidden_layers": "128,64,32",
    "activation": "relu",
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
    "normalize": True,
}

# Teacher's baseline (PDF p.17): no hidden layer, no normalisation, 20 epochs.
TEACHER = {
    "hidden_layers": "",
    "activation": "relu",
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "batch_size": 128,
    "epochs": 20,
    "validation_split": 0.1,
    "normalize": False,
}

LIMITS = {
    "epochs": (1, 50),
    "batch_size": (1, 10000),
    "neurons": (1, 2048),
    "layers": (0, 10),
}

# Sensitivity study: one hyper-parameter varies, the others keep their base value.
STUDY_VALUES = {
    "activation": ["relu", "sigmoid", "tanh"],
    "hidden_layers": ["", "128", "128,64", "128,64,32"],
    "neurons_order": ["128,64,32", "32,64,128"],
    "loss": ["categorical_crossentropy", "mean_squared_error"],
    "optimizer": ["adam", "sgd", "rmsprop"],
    "batch_size": [32, 128, 512],
    "epochs": [5, 10, 20],
    "validation_split": [0.1, 0.2, 0.3],
    "normalize": [False, True],
}

_DATA_CACHE = {}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
# Текст '128,64,32' -> список шарів; порожньо = модель викладача без прихованих шарів
def parse_hidden_layers(text):
    """'128,64,32' -> [128, 64, 32]; '' -> [] (no hidden layer)."""
    text = (text or "").strip()
    if text == "" or text == "0":
        return []
    layers = []
    for part in text.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(
                f"Hidden layers: '{part}' is not a positive integer "
                "(expected e.g. 128,64,32)."
            )
        n = int(part)
        lo, hi = LIMITS["neurons"]
        if not lo <= n <= hi:
            raise ValueError(f"Hidden layers: each layer needs {lo}-{hi} neurons.")
        layers.append(n)
    if len(layers) > LIMITS["layers"][1]:
        raise ValueError(f"Hidden layers: at most {LIMITS['layers'][1]} layers.")
    return layers


def _int_field(form, name, errors):
    raw = str(form.get(name, DEFAULTS[name])).strip()
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{name} must be an integer (got '{raw}').")
        return DEFAULTS[name]
    lo, hi = LIMITS[name]
    if not lo <= value <= hi:
        errors.append(f"{name} must be between {lo} and {hi} (got {value}).")
    return value


# Перевірка форми на сервері: помилки показуються на сторінці, сайт не падає
def validate_params(form):
    """Read a dict-like form, return (params, errors).

    params always contains every key (defaults are used when a value is bad),
    so the template can re-display the form.
    """
    errors = []
    params = {}

    raw_layers = str(form.get("hidden_layers", DEFAULTS["hidden_layers"]))
    try:
        parse_hidden_layers(raw_layers)
    except ValueError as exc:
        errors.append(str(exc))
    params["hidden_layers"] = raw_layers.strip()

    for name, allowed in (
        ("activation", ACTIVATIONS),
        ("optimizer", OPTIMIZERS),
        ("loss", LOSSES),
    ):
        value = str(form.get(name, DEFAULTS[name])).strip()
        if value not in allowed:
            errors.append(f"{name} must be one of {', '.join(allowed)} (got '{value}').")
            value = DEFAULTS[name]
        params[name] = value

    params["batch_size"] = _int_field(form, "batch_size", errors)
    params["epochs"] = _int_field(form, "epochs", errors)

    raw_vs = str(form.get("validation_split", DEFAULTS["validation_split"])).strip()
    try:
        vs = float(raw_vs)
        if not 0.0 <= vs < 0.5:
            errors.append(f"validation_split must be in [0, 0.5) (got {vs}).")
            vs = DEFAULTS["validation_split"]
    except ValueError:
        errors.append(f"validation_split must be a number (got '{raw_vs}').")
        vs = DEFAULTS["validation_split"]
    params["validation_split"] = vs

    # Checkbox: present -> normalise. Accept bools too (used by the study code).
    norm = form.get("normalize", False)
    params["normalize"] = norm is True or str(norm).lower() in ("on", "true", "1", "yes")

    return params, errors


# --------------------------------------------------------------------------
# Data and model
# --------------------------------------------------------------------------
# 60 000 train + 10 000 test; картинка 28x28 -> вектор 784; мітки one-hot
def load_mnist(normalize=True):
    """MNIST flattened to 784 values, labels one-hot. Cached in memory."""
    if "raw" not in _DATA_CACHE:
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], 28 * 28)).astype("float32")
        X_test = X_test.reshape((X_test.shape[0], 28 * 28)).astype("float32")
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
        _DATA_CACHE["raw"] = (X_train, y_train, X_test, y_test)

    X_train, y_train, X_test, y_test = _DATA_CACHE["raw"]
    if normalize:
        # Main fix vs. the teacher's code: pixels 0-255 -> 0-1.
        # Нормалізація 0-255 -> 0-1: стабільні градієнти (0.864 -> 0.928)
        X_train, X_test = X_train / 255.0, X_test / 255.0
    return X_train, y_train, X_test, y_test


# Мережа: вхід 784 -> Dense шари -> Dense(10, softmax) = 10 ймовірностей
def build_model(hidden_layers, activation="relu", input_dim=784, num_classes=10):
    """Dense-only network. hidden_layers=[] reproduces the teacher's model."""
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model

    xi = Input(shape=(input_dim,))
    x = xi
    for units in hidden_layers:
        x = Dense(units, activation=activation)(x)
    y = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=[xi], outputs=[y])  # created once (teacher did it twice)


# Навчання: seed -> модель -> compile -> fit -> evaluate на тесті -> результат для сторінки
def train_mlp(params, data=None, train_limit=None):
    """Train with validated params and return a JSON-friendly result dict.

    data: optional (X_train, y_train, X_test, y_test) - used by the tests.
    train_limit: use only the first N training images (faster studies).
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)  # reproducible results

    if data is None:
        data = load_mnist(normalize=params["normalize"])
    X_train, y_train, X_test, y_test = data
    if train_limit:
        X_train, y_train = X_train[:train_limit], y_train[:train_limit]

    model = build_model(
        parse_hidden_layers(params["hidden_layers"]),
        params["activation"],
        input_dim=X_train.shape[1],
        num_classes=y_train.shape[1],
    )
    # loss = що мінімізуємо, optimizer = як оновлюємо ваги
    model.compile(
        loss=params["loss"],
        optimizer=params["optimizer"],
        metrics=["accuracy"],
    )

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        verbose=0,
        validation_split=params["validation_split"],
    )
    duration = time.time() - start
    # Тест (10 000 картинок) модель бачить тільки тут: чесна оцінка
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    hist = history.history
    return {
        "params": dict(params),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "num_params": int(model.count_params()),
        "train_time": round(duration, 1),
        "history": {
            "accuracy": [float(v) for v in hist.get("accuracy", [])],
            "val_accuracy": [float(v) for v in hist.get("val_accuracy", [])],
            "loss": [float(v) for v in hist.get("loss", [])],
            "val_loss": [float(v) for v in hist.get("val_loss", [])],
        },
    }


# --------------------------------------------------------------------------
# Sensitivity study
# --------------------------------------------------------------------------
def study_configs(parameter, base=None):
    """Return the list of param dicts to train for one studied parameter."""
    if parameter not in STUDY_VALUES:
        raise ValueError(
            f"Unknown parameter '{parameter}'. Choose one of: {', '.join(STUDY_VALUES)}."
        )
    base = dict(base or DEFAULTS)
    configs = []
    for value in STUDY_VALUES[parameter]:
        cfg = dict(base)
        if parameter == "neurons_order":
            cfg["hidden_layers"] = value
        else:
            cfg[parameter] = value
        configs.append((value, cfg))
    return configs


# Sensitivity study: міняємо один параметр, решта фіксовані, одна модель на значення
def run_sensitivity(parameter, base=None, train_limit=None, data=None, trainer=None):
    """Train one model per value of `parameter`; return table rows."""
    trainer = trainer or train_mlp
    rows = []
    for value, cfg in study_configs(parameter, base):
        res = trainer(cfg, data=data, train_limit=train_limit)
        rows.append(
            {
                "parameter": parameter,
                "value": value if value != "" else "(no hidden layer)",
                "test_accuracy": res["test_accuracy"],
                "test_loss": res["test_loss"],
                "num_params": res["num_params"],
                "train_time": res["train_time"],
            }
        )
    return rows
