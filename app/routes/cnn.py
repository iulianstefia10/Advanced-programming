from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for

from services import cnn_service

cnn_bp = Blueprint(
    "cnn",
    __name__,
    url_prefix="/cnn",
)


def _status_payload():
    """Current job state, with a ready-to-use URL for the history plot."""
    status = cnn_service.get_status()
    if status.get("plot_file"):
        stamp = int(status.get("finished_at", 0))
        status["plot_url"] = url_for("static", filename=status["plot_file"]) + f"?t={stamp}"
    return status


@cnn_bp.route("/", methods=["GET"])
def dashboard():
    status = _status_payload()
    # Pre-fill the form with the last config used, or the defaults.
    values = {**cnn_service.DEFAULTS, **status.get("config", {})}
    return render_template(
        "cnn_dashboard.html",
        values=values,
        status=status,
        optimizers=list(cnn_service.OPTIMIZERS),
        label_modes=cnn_service.LABEL_MODES,
        error=None,
    )


@cnn_bp.route("/train", methods=["POST"])
def train():
    try:
        config = cnn_service.parse_config(request.form)
    except ValueError as exc:
        status = _status_payload()
        values = {**cnn_service.DEFAULTS, **request.form.to_dict()}
        return render_template(
            "cnn_dashboard.html",
            values=values,
            status=status,
            optimizers=list(cnn_service.OPTIMIZERS),
            label_modes=cnn_service.LABEL_MODES,
            error=str(exc),
        ), 400

    started = cnn_service.start_training(config, current_app.root_path)
    if not started:
        status = _status_payload()
        return render_template(
            "cnn_dashboard.html",
            values={**cnn_service.DEFAULTS, **config},
            status=status,
            optimizers=list(cnn_service.OPTIMIZERS),
            label_modes=cnn_service.LABEL_MODES,
            error="A training is already running. Wait for it to finish.",
        ), 409

    return redirect(url_for("cnn.dashboard"))


@cnn_bp.route("/stop", methods=["POST"])
def stop():
    """Ask the running training to stop (used by the Stop button)."""
    return jsonify({"ok": cnn_service.request_stop()})


@cnn_bp.route("/status", methods=["GET"])
def status():
    """JSON endpoint polled by the dashboard while training runs."""
    return jsonify(_status_payload())