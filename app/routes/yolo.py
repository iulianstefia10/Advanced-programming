# Сторінка /yolo/ (B3): завантажити фото → services/yolo.py знаходить об'єкти → фото з рамками + таблиця.

import io

from flask import Blueprint, render_template, request
from PIL import Image, UnidentifiedImageError

from services.yolo import detect, draw_boxes

# група сторінок YOLO, адреса /yolo/
yolo_bp = Blueprint(
    "yolo",
    __name__,
    url_prefix="/yolo",
)


# GET — порожня форма; POST — надіслали фото
@yolo_bp.route("/", methods=["GET", "POST"])
def dashboard():
    result_image = None   # фото з рамками (base64)
    detections = None     # список знайдених об'єктів для таблиці
    error = None
    # поріг у відсотках з форми; при першому відкритті 25
    # ЗМІНА: інший поріг за замовчуванням, напр. "50" (також value у yolo_dashboard.html)
    confidence = request.form.get("confidence", "25")

    if request.method == "POST":
        try:
            # read the threshold from the form (in %)
            # 25 % → 0.25, бо модель працює з числами 0–1
            try:
                threshold = float(confidence) / 100
            except ValueError:
                raise ValueError("The confidence threshold must be a number.")

            # файл з <input type="file" name="file">
            file = request.files.get("file")
            if file is None or file.filename == "":
                raise ValueError("Choose a photo first.")

            # фото читається з пам'яті, на диск не зберігається
            image = Image.open(io.BytesIO(file.read()))
            # ЗМІНА: передати IoU з форми: detect(image, conf_threshold=threshold, iou_threshold=...)
            image, detections = detect(image, conf_threshold=threshold)
            result_image = draw_boxes(image, detections)
            # ЗМІНА: кількість кожного класу: collections.Counter(d["label"] for d in detections)

        except UnidentifiedImageError:
            error = "The file is not a valid image."    # напр. завантажили .txt
        except ValueError as e:
            error = str(e)

    return render_template(
        "yolo_dashboard.html",
        result_image=result_image,
        detections=detections,
        confidence=confidence,
        error=error,
    )
