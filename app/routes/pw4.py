# Сторінка /pw4/: звіт PW4 (таблиця CPU/GPU/TPU з ноутбука) + кнопка розпізнати своє фото двома моделями.

import io

from flask import Blueprint, render_template, request
from PIL import Image, UnidentifiedImageError

from services.pw4_inference import classify

# група сторінок PW4, адреса /pw4/
pw4_bp = Blueprint(
    "pw4",
    __name__,
    url_prefix="/pw4",
)

# results we measured with the notebook (static report)
# числа виміряні в ноутбуці (не рахуються на сайті): час навчання 2 епох (с),
# час перевірки на 10 000 тестових фото (с), точність (%)
# ЗМІНА: новий рядок, напр. {"hardware": "Colab GPU (A100)", "training": ..., "inference": ..., "accuracy": ...}
HARDWARE_RESULTS = [
    {"hardware": "CPU (our laptop)", "training": 113, "inference": 9, "accuracy": 55},
    {"hardware": "Colab CPU (Intel Xeon 2.2 GHz)", "training": 168, "inference": 10, "accuracy": 55},
    {"hardware": "Colab GPU (NVIDIA T4)", "training": 119, "inference": 7, "accuracy": 54},
    {"hardware": "Colab TPU (v5e-1)", "training": 96, "inference": 8, "accuracy": 55},
]


# GET — звіт; POST — надіслали фото
@pw4_bp.route("/", methods=["GET", "POST"])
def dashboard():
    results = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            error = "Choose an image first."
        else:
            try:
                image = Image.open(io.BytesIO(file.read()))    # фото з пам'яті, не з диска
                results = classify(image)                      # {"cnn": ..., "mobilenet": ...}
            except (UnidentifiedImageError, OSError):
                error = "The file is not a valid image."

    return render_template(
        "pw4_report.html",
        hardware_results=HARDWARE_RESULTS,
        results=results,
        error=error,
    )
