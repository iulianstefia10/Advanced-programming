import os
import io
import base64
import numpy as np
from PIL import Image, ImageOps
from flask import Blueprint, render_template, request, current_app, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

mnist_bp = Blueprint(
    "mnist",
    __name__,
    url_prefix="/mnist",
)

_model = None


def get_mnist_model():
    """Load the Keras H5 model once and cache it in memory."""
    global _model
    if _model is None:
        model_path = os.path.join(current_app.root_path, "MNIST_keras_CNN.h5")
        _model = load_model(model_path, compile=False)
    return _model


@mnist_bp.route("/", methods=["GET", "POST"])
def dashboard():
    prediction = None
    confidence = None
    image_data = None
    filename = None
    error = None

    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            error = "Please select a valid image file before uploading."
        else:
            file = request.files["file"]
            filename = secure_filename(file.filename)
            save_path = os.path.join(upload_folder, filename)

            try:
                raw_bytes = file.read()
                with open(save_path, "wb") as f:
                    f.write(raw_bytes)

                image_data = base64.b64encode(raw_bytes).decode("utf-8")

                # Preprocess image for MNIST model (28x28 grayscale)
                img = Image.open(io.BytesIO(raw_bytes)).convert("L")
                img = ImageOps.autocontrast(img)
                img = img.resize((28, 28))
                im2arr = np.array(img)

                # Invert colors if background is light (MNIST is white digit on black background)
                if np.mean(im2arr) > 127:
                    im2arr = 255 - im2arr

                # Clean background noise with a soft threshold
                im2arr = np.where(im2arr < 50, 0, im2arr)

                im2arr = im2arr.reshape(1, 28, 28, 1).astype("float32") / 255.0

                model = get_mnist_model()
                predict_x = model.predict(im2arr)
                prediction = int(np.argmax(predict_x, axis=1)[0])
                confidence = round(float(np.max(predict_x)) * 100, 2)

            except Exception as e:
                error = f"Error processing image: {str(e)}"

    uploaded_files = [
        f for f in os.listdir(upload_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".jfif", ".webp"))
    ]

    return render_template(
        "mnist_dashboard.html",
        prediction=prediction,
        confidence=confidence,
        image_data=image_data,
        filename=filename,
        uploaded_files=uploaded_files,
        error=error,
    )


@mnist_bp.route("/delete/<path:filename>", methods=["POST"])
def delete_image(filename):
    """Delete a single uploaded image."""
    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    safe_name = secure_filename(filename)
    file_path = os.path.join(upload_folder, safe_name)

    if os.path.exists(file_path):
        os.remove(file_path)

    return redirect(url_for("mnist.dashboard"))


@mnist_bp.route("/clear", methods=["POST"])
def clear_images():
    """Delete all uploaded images."""
    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    if os.path.exists(upload_folder):
        for f in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return redirect(url_for("mnist.dashboard"))