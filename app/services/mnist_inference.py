# Prepare an image (upload or drawing) and run the CNN and the MLP on it

import base64
import binascii
import io
import os
import tempfile

import numpy as np
from PIL import Image, ImageOps

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILES = {
    "cnn": os.path.join(APP_DIR, "MNIST_keras_CNN.h5"),
    "mlp": os.path.join(APP_DIR, "mnist_mlp.keras"),
}
MODEL_NAMES = {
    "cnn": "CNN (MNIST_keras_CNN.h5, given model)",
    "mlp": "MLP 128-64-32 (our PW2 model)",
}

_models = {}
_stored_models = {}


def decode_data_url(data_url):
    """The canvas sends 'data:image/png;base64,....' -> PIL image."""
    if not data_url or "," not in data_url:
        raise ValueError("The drawing is empty.")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        return Image.open(io.BytesIO(raw))
    except (binascii.Error, OSError):
        raise ValueError("The drawing could not be read.")


def prepare_image(image):
    """Image -> 28x28 array between 0 and 1, white digit on a black background (like MNIST)."""
    # a transparent background (canvas, PNG) becomes white
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    gray = np.array(image.convert("L"), dtype="float32")

    # MNIST digits are white on black: invert a dark digit on a light background
    if gray.mean() > 127:
        gray = 255 - gray
    gray[gray < 50] = 0

    if gray.max() == 0:
        raise ValueError("No digit found in the image.")

    # keep only the digit, then make it fit in a 20x20 box like in MNIST
    rows = np.where(gray.max(axis=1) > 0)[0]
    cols = np.where(gray.max(axis=0) > 0)[0]
    digit = Image.fromarray(gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].astype("uint8"))
    digit = ImageOps.contain(digit, (20, 20), Image.LANCZOS)

    # paste it in the middle of a 28x28 black image
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    result = np.array(canvas, dtype="float32")

    # move the centre of mass to the centre, as in the MNIST dataset
    total = result.sum()
    ys, xs = np.indices(result.shape)
    shift_y = int(round(13.5 - (ys * result).sum() / total))
    shift_x = int(round(13.5 - (xs * result).sum() / total))
    result = np.roll(np.roll(result, shift_y, axis=0), shift_x, axis=1)

    return result / 255.0


def cnn_input(pixels):
    # the CNN wants an image: (1, 28, 28, 1)
    return pixels.reshape(1, 28, 28, 1).astype("float32")


def mlp_input(pixels):
    # the MLP wants a vector of 784 values (already divided by 255)
    return pixels.reshape(1, 784).astype("float32")


def get_model(name):
    """Load a model once and keep it in memory."""
    if name not in _models:
        from tensorflow.keras.models import load_model
        _models[name] = load_model(MODEL_FILES[name], compile=False)
    return _models[name]


def predict_digit(model, model_input):
    probabilities = np.asarray(model.predict(model_input, verbose=0)[0], dtype=float)
    return {
        "digit": int(np.argmax(probabilities)),
        "confidence": round(float(probabilities.max()) * 100, 2),
        "probabilities": [round(float(p) * 100, 2) for p in probabilities],
    }


def predict_both(pixels):
    """Prediction of the CNN and of the MLP for the same prepared image."""
    inputs = {"cnn": cnn_input(pixels), "mlp": mlp_input(pixels)}
    return {
        name: dict(predict_digit(get_model(name), inputs[name]), model=MODEL_NAMES[name])
        for name in ("cnn", "mlp")
    }


# ---------------------------------------------------------------------------
# Models stored in the database (see services/model_store.py)
# ---------------------------------------------------------------------------

def load_from_bytes(data, filename):
    """Keras loads from a file, so the bytes go through a temporary folder."""
    extension = os.path.splitext(filename)[1].lower()
    if extension not in (".h5", ".keras"):
        raise ValueError("Only .h5 and .keras models are accepted.")
    from tensorflow.keras.models import load_model
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model" + extension)
        with open(path, "wb") as f:
            f.write(data)
        try:
            return load_model(path, compile=False)
        except Exception as error:
            raise ValueError("The file is not a valid Keras model.") from error


def check_digit_model(model):
    """The page sends 28x28 images: the model must take 784 values and give 10 classes."""
    if int(np.prod(model.input_shape[1:])) != 784 or model.output_shape[-1] != 10:
        raise ValueError("The model must take 28x28 images and give 10 classes.")


def get_stored_model(stored):
    """Load a model from the database once and keep it in memory."""
    if stored.id not in _stored_models:
        # stored.data is deferred: it is only read from the database here
        _stored_models[stored.id] = load_from_bytes(stored.data, stored.filename)
    return _stored_models[stored.id]


def forget_model(model_id):
    """Remove a deleted model from the memory."""
    _stored_models.pop(model_id, None)


def model_input(pixels, model):
    """Shape the pixels like the input of this model: (1, 28, 28, 1) or (1, 784)."""
    return pixels.reshape((1,) + tuple(model.input_shape[1:])).astype("float32")


def predict_with(pixels, stored_models):
    """Run the prepared image through the chosen models: {name: result}."""
    results = {}
    for stored in stored_models:
        model = get_stored_model(stored)
        results[stored.name] = dict(
            predict_digit(model, model_input(pixels, model)), model=stored.name
        )
    return results


def preview_png(pixels):
    """The prepared 28x28 image as base64 PNG, to show what the models see."""
    image = Image.fromarray((pixels * 255).astype("uint8")).resize((140, 140), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")