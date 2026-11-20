import os

from flask import Blueprint, current_app, render_template, request

from services import landcover as landcover_service

landcover_bp = Blueprint(
    "landcover",
    __name__,
    url_prefix="/landcover",
)


def _model_dir():
    return current_app.config.get(
        "LANDCOVER_MODEL_DIR", os.path.join(current_app.root_path, "models_store")
    )


def _output_dir():
    return current_app.config.get(
        "LANDCOVER_OUTPUT_DIR", os.path.join(current_app.static_folder, "landcover")
    )


def _page(result=None, error=None, status=200):
    return render_template(
        "landcover_dashboard.html",
        result=result,
        error=error,
        model_ok=landcover_service.model_available(_model_dir()),
        model_name=landcover_service.MODEL_NAME,
        patch_sizes=landcover_service.PATCH_SIZES,
        default_patch_size=landcover_service.DEFAULT_PATCH_SIZE,
    ), status


@landcover_bp.route("/", methods=["GET"])
def dashboard():
    return _page()


@landcover_bp.route("/predict", methods=["POST"])
def predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return _page(error="Choose a satellite screenshot first.", status=400)


    data = upload.read(landcover_service.MAX_UPLOAD_BYTES + 1)

    try:
        result = landcover_service.analyse(
            data, _model_dir(), _output_dir(), request.form.get("patch_size")
        )
    except landcover_service.ModelNotFound as exc:
        return _page(error=str(exc), status=503)
    except landcover_service.InvalidImage as exc:
        return _page(error=str(exc), status=400)

    return _page(result=result)