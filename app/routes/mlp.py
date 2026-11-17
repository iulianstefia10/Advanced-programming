from flask import Blueprint, render_template, request

from services import mlp as mlp_service

mlp_bp = Blueprint(
    "mlp",
    __name__,
    url_prefix="/mlp",
)

# training images used by the sensitivity study (out of 60,000),
# so one click stays under about a minute on the CPU
STUDY_TRAIN_LIMIT = 10000


@mlp_bp.route("/", methods=["GET", "POST"])
def dashboard():
    params = dict(mlp_service.DEFAULTS)
    study_param = "activation"
    full_data = False
    result = None
    study = None
    errors = []

    if request.method == "POST":
        params, errors = mlp_service.validate_params(request.form)
        study_param = request.form.get("study_param", "activation")
        full_data = request.form.get("full_data") is not None
        action = request.form.get("action", "train")

        if action == "study" and study_param not in mlp_service.STUDY_VALUES:
            errors.append(f"Unknown parameter to study: '{study_param}'.")

        if not errors:
            try:
                if action == "study":
                    study = mlp_service.run_sensitivity(
                        study_param,
                        base=params,
                        train_limit=None if full_data else STUDY_TRAIN_LIMIT,
                    )
                else:
                    result = mlp_service.train_mlp(params)
            except Exception as e:  # never crash the site
                errors.append(f"Training failed: {e}")

    return render_template(
        "mlp_dashboard.html",
        params=params,
        result=result,
        study=study,
        study_param=study_param,
        full_data=full_data,
        errors=errors,
        activations=mlp_service.ACTIVATIONS,
        optimizers=mlp_service.OPTIMIZERS,
        losses=mlp_service.LOSSES,
        study_values=mlp_service.STUDY_VALUES,
        study_limit=STUDY_TRAIN_LIMIT,
    )
