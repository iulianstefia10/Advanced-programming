import base64
import io
import os

from flask import Blueprint, current_app, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from services.mnist_inference import decode_data_url, predict_both, prepare_image, preview_png

mnist_bp = Blueprint(
    "mnist",
    __name__,
    url_prefix="/mnist",
)


@mnist_bp.route("/", methods=["GET", "POST"])
def dashboard():
    predictions = None
    image_data = None
    model_view = None
    filename = None
    error = None

    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    if request.method == "POST":
        try:
            if request.form.get("drawing"):
                # digit drawn on the canvas (sent as a PNG data URL)
                image = decode_data_url(request.form["drawing"])
                filename = "drawing"
                image_data = request.form["drawing"].split(",", 1)[1]
            elif "file" in request.files and request.files["file"].filename != "":
                # uploaded image file, also saved in static/uploads
                file = request.files["file"]
                filename = secure_filename(file.filename)
                raw_bytes = file.read()
                image = Image.open(io.BytesIO(raw_bytes))
                # save it only when it really is an image
                with open(os.path.join(upload_folder, filename), "wb") as f:
                    f.write(raw_bytes)
                image_data = base64.b64encode(raw_bytes).decode("utf-8")
            else:
                raise ValueError("Draw a digit or choose an image file first.")

            # the same 28x28 image goes to both models
            pixels = prepare_image(image)
            model_view = preview_png(pixels)
            predictions = predict_both(pixels)

        except (ValueError, UnidentifiedImageError, OSError) as e:
            error = str(e) if isinstance(e, ValueError) else "The file is not a valid image."

    uploaded_files = [
        f for f in os.listdir(upload_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".jfif", ".webp"))
    ]

    return render_template(
        "mnist_dashboard.html",
        predictions=predictions,
        image_data=image_data,
        model_view=model_view,
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