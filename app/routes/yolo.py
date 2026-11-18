import io

from flask import Blueprint, render_template, request
from PIL import Image, UnidentifiedImageError

from services.yolo import detect, draw_boxes

yolo_bp = Blueprint(
    "yolo",
    __name__,
    url_prefix="/yolo",
)


@yolo_bp.route("/", methods=["GET", "POST"])
def dashboard():
    result_image = None
    detections = None
    error = None
    confidence = request.form.get("confidence", "25")

    if request.method == "POST":
        try:
            # read the threshold from the form (in %)
            try:
                threshold = float(confidence) / 100
            except ValueError:
                raise ValueError("The confidence threshold must be a number.")

            file = request.files.get("file")
            if file is None or file.filename == "":
                raise ValueError("Choose a photo first.")

            image = Image.open(io.BytesIO(file.read()))
            image, detections = detect(image, conf_threshold=threshold)
            result_image = draw_boxes(image, detections)

        except UnidentifiedImageError:
            error = "The file is not a valid image."
        except ValueError as e:
            error = str(e)

    return render_template(
        "yolo_dashboard.html",
        result_image=result_image,
        detections=detections,
        confidence=confidence,
        error=error,
    )
