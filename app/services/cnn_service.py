"""CIFAR-100 CNN training service (Practical Work 3).

Training runs in a background thread so the Flask request returns immediately.
The state of the current job lives in memory (module level), so run the app
with a single process (Flask dev server, or one gunicorn worker).

The user can stop a running training: request_stop() sets a flag, a Keras
callback notices it after the current batch and stops fit().
"""

import copy
import os
import threading
import time

import numpy as np
from matplotlib.figure import Figure

from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# Defaults reproduce the model from the assignment (test accuracy about 0.29).
DEFAULTS = {
    "epochs": 10,
    "batch_size": 50,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "label_mode": "fine",
    "filters": 32,
    "kernel_size": 3,
    "dense_units": 256,
}

OPTIMIZERS = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
LABEL_MODES = {"fine": 100, "coarse": 20}  # label_mode -> number of classes

IMG_SHAPE = (32, 32, 3)

# ---------------------------------------------------------------------------
# Shared job state
# ---------------------------------------------------------------------------

# Job status: idle | loading | training | stopping | evaluating | done | stopped | error
_ACTIVE = ("loading", "training", "stopping", "evaluating")

_lock = threading.Lock()
_stop_event = threading.Event()
_job = {"status": "idle"}
_data_cache = {"label_mode": None, "data": None}


def _update(**fields):
    with _lock:
        _job.update(fields)


def _set_phase(phase, **fields):
    """Change the job status without cancelling a pending stop request."""
    with _lock:
        if _job.get("status") == "stopping" and phase in ("loading", "training"):
            phase = "stopping"
        _job.update(status=phase, **fields)


def get_status():
    """Return a copy of the current job state (safe to serialize)."""
    with _lock:
        return copy.deepcopy(_job)


def is_running():
    return get_status()["status"] in _ACTIVE


def request_stop():
    """Ask the running job to stop. Returns True if a stop was requested."""
    with _lock:
        if _job.get("status") not in ("loading", "training"):
            return False
        _stop_event.set()
        _job["status"] = "stopping"
        return True


# ---------------------------------------------------------------------------
# Form parsing and validation
# ---------------------------------------------------------------------------

def _number(form, key, cast, low, high):
    raw = form.get(key, DEFAULTS[key])
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number.")
    if not low <= value <= high:
        raise ValueError(f"'{key}' must be between {low} and {high}.")
    return value


def _choice(form, key, allowed):
    value = form.get(key, DEFAULTS[key])
    if value not in allowed:
        raise ValueError(f"'{key}' must be one of: {', '.join(allowed)}.")
    return value


def parse_config(form):
    """Validate a form (dict-like) and return a clean config dict.

    Raises ValueError with a readable message when a value is invalid.
    kernel_size is limited to 2-4: with 3 conv + pooling blocks on 32x32
    images and 'valid' padding, a 5x5 kernel shrinks the image to nothing.
    """
    return {
        "epochs": _number(form, "epochs", int, 1, 100),
        "batch_size": _number(form, "batch_size", int, 8, 1024),
        "optimizer": _choice(form, "optimizer", OPTIMIZERS),
        "learning_rate": _number(form, "learning_rate", float, 1e-6, 1.0),
        "validation_split": _number(form, "validation_split", float, 0.05, 0.5),
        "label_mode": _choice(form, "label_mode", LABEL_MODES),
        "filters": _number(form, "filters", int, 4, 128),
        "kernel_size": _number(form, "kernel_size", int, 2, 4),
        "dense_units": _number(form, "dense_units", int, 8, 1024),
    }


# ---------------------------------------------------------------------------
# Data and model
# ---------------------------------------------------------------------------

def _load_data(label_mode):
    """Load CIFAR-100 (uint8), keeping only the last label_mode in memory."""
    if _data_cache["label_mode"] != label_mode:
        _data_cache["data"] = cifar100.load_data(label_mode=label_mode)
        _data_cache["label_mode"] = label_mode
    (x_train, y_train), (x_test, y_test) = _data_cache["data"]
    # Normalize to [0, 1]: inputs should never be fed raw 0-255 to the network.
    return (
        (x_train.astype("float32") / 255, y_train),
        (x_test.astype("float32") / 255, y_test),
    )


def build_model(config):
    """Same architecture as the assignment, with configurable sizes.

    With filters=32, kernel_size=3, dense_units=256 it is identical to the
    original model (32-64-128 filters, Dense 256 -> 128 -> classes).
    """
    no_classes = LABEL_MODES[config["label_mode"]]
    kernel = (config["kernel_size"], config["kernel_size"])
    filters = config["filters"]
    dense_units = config["dense_units"]

    model = Sequential()
    model.add(Input(shape=IMG_SHAPE))
    for block in range(3):
        model.add(Conv2D(filters * 2 ** block, kernel_size=kernel, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(max(dense_units // 2, 1), activation="relu"))
    model.add(Dense(no_classes, activation="softmax"))
    return model


class _ProgressCallback(Callback):
    """Reports progress to the job state and stops fit() on request."""

    def on_train_batch_end(self, batch, logs=None):
        # Checked after every batch, so a stop takes effect within seconds.
        if _stop_event.is_set():
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if "val_loss" not in logs:
            return  # an epoch interrupted before validation has no usable row
        row = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs["val_loss"]),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "partial": _stop_event.is_set(),  # True if the epoch was cut short
        }
        with _lock:
            _job["history"].append(row)
            _job["epoch"] = row["epoch"]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_history_plot(history, out_path):
    """Save train and validation curves. Uses Figure directly (no pyplot, no GUI)."""
    epochs = [row["epoch"] for row in history]
    fig = Figure(figsize=(11, 4))
    ax_loss, ax_acc = fig.subplots(1, 2)

    ax_loss.plot(epochs, [r["loss"] for r in history], marker="o", label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], marker="o", label="validation")
    ax_loss.set_title("Loss history")
    ax_loss.set_xlabel("No. epoch")
    ax_loss.set_ylabel("Loss value")
    ax_loss.legend()

    ax_acc.plot(epochs, [r["accuracy"] * 100 for r in history], marker="o", label="train")
    ax_acc.plot(epochs, [r["val_accuracy"] * 100 for r in history], marker="o", label="validation")
    ax_acc.set_title("Accuracy history")
    ax_acc.set_xlabel("No. epoch")
    ax_acc.set_ylabel("Accuracy value (%)")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)


# ---------------------------------------------------------------------------
# Training job
# ---------------------------------------------------------------------------

def _run(config, root_path):
    try:
        _set_phase("loading")
        (x_train, y_train), (x_test, y_test) = _load_data(config["label_mode"])

        # A stop requested during the download cannot interrupt it, but we
        # can skip the training entirely once the data is there.
        if _stop_event.is_set():
            _update(status="stopped", finished_at=time.time())
            return

        model = build_model(config)
        optimizer = OPTIMIZERS[config["optimizer"]](learning_rate=config["learning_rate"])
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        _set_phase("training", params=int(model.count_params()))

        model.fit(
            x_train,
            y_train,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            verbose=0,
            validation_split=config["validation_split"],
            callbacks=[_ProgressCallback()],
        )

        was_stopped = _stop_event.is_set()

        _set_phase("evaluating")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        plot_file = None
        history = get_status()["history"]
        if history:
            plot_dir = os.path.join(root_path, "static", "cnn")
            os.makedirs(plot_dir, exist_ok=True)
            _save_history_plot(history, os.path.join(plot_dir, "history.png"))
            plot_file = "cnn/history.png"

        # Saving the model is a bonus: a failure here must not fail the job.
        # A model stopped on purpose is not saved, so it never replaces a good one.
        if not was_stopped:
            try:
                model_dir = os.path.join(root_path, "saved_models")
                os.makedirs(model_dir, exist_ok=True)
                model.save(os.path.join(model_dir, "cifar100_cnn.keras"))
            except Exception:
                pass

        _update(
            status="stopped" if was_stopped else "done",
            test_loss=float(test_loss),
            test_accuracy=float(test_accuracy),
            plot_file=plot_file,
            finished_at=time.time(),
        )
    except Exception as exc:  # shown to the user in the dashboard
        _update(status="error", error=f"{type(exc).__name__}: {exc}")


def start_training(config, root_path):
    """Start a background training job. Returns False if one is already running."""
    with _lock:
        if _job["status"] in _ACTIVE:
            return False
        _stop_event.clear()
        _job.clear()
        _job.update(
            status="loading",
            config=config,
            epoch=0,
            total_epochs=config["epochs"],
            history=[],
            started_at=time.time(),
        )

    thread = threading.Thread(target=_run, args=(config, root_path), daemon=True)
    thread.start()
    return True