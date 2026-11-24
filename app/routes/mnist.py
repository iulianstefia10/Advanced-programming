import base64
import io
import os

from flask import Blueprint, current_app, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import defer
from werkzeug.utils import secure_filename

from extensions import db
from models.app_data import StoredModel
from services.mnist_inference import (
    decode_data_url,
    forget_model,
    predict_with,
    prepare_image,
    preview_png,
)
from services.model_store import add_model

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
    error = request.args.get("model_error")   # set by upload_model below

    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    try:
        # defer(data): the list must not load every model file from the database
        models = db.session.scalars(
            db.select(StoredModel).options(defer(StoredModel.data)).order_by(StoredModel.id)
        ).all()
    except SQLAlchemyError:
        db.session.rollback()
        models = []
        error = "The model database is unavailable."

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

            # the same 28x28 image goes to every chosen model
            pixels = prepare_image(image)
            model_view = preview_png(pixels)

            choice = request.form.get("model_id", "all")
            chosen = models if choice == "all" else [m for m in models if str(m.id) == choice]
            if not chosen:
                raise ValueError("No model available: import one first.")
            predictions = predict_with(pixels, chosen)

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
        models=models,
        error=error,
    )


@mnist_bp.route("/models/upload", methods=["POST"])
def upload_model():
    """Import a .h5 / .keras model and store it in the database."""
    file = request.files.get("model_file")
    try:
        if file is None or file.filename == "":
            raise ValueError("Choose a .h5 or .keras file.")
        filename = secure_filename(file.filename)
        name = request.form.get("name", "").strip() or os.path.splitext(filename)[0]
        add_model(name, request.form.get("description", "").strip(), filename, file.read())
    except ValueError as e:
        return redirect(url_for("mnist.dashboard", model_error=str(e)))
    except SQLAlchemyError:
        db.session.rollback()
        return redirect(url_for("mnist.dashboard", model_error="The database is unavailable."))
    return redirect(url_for("mnist.dashboard"))


@mnist_bp.route("/models/delete/<int:model_id>", methods=["POST"])
def delete_model(model_id):
    """Delete one model from the database."""
    stored = db.session.get(StoredModel, model_id)
    if stored is not None:
        db.session.delete(stored)
        db.session.commit()
        forget_model(model_id)
    return redirect(url_for("mnist.dashboard"))


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