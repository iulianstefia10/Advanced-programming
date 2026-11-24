# Keep the digit models in the database (table stored_model)

import os

from extensions import db
from models.app_data import StoredModel
from services.mnist_inference import (
    MODEL_FILES,
    MODEL_NAMES,
    check_digit_model,
    load_from_bytes,
)


def seed_default_models():
    """First start only: copy the two given model files into the database."""
    if db.session.scalar(db.select(db.func.count(StoredModel.id))):
        return
    for key in ("cnn", "mlp"):
        path = MODEL_FILES[key]
        with open(path, "rb") as f:
            data = f.read()
        db.session.add(StoredModel(
            name=MODEL_NAMES[key],
            description="Built-in model",
            filename=os.path.basename(path),
            size_bytes=len(data),
            data=data,
        ))
    db.session.commit()


def add_model(name, description, filename, data):
    """Check that the file is a digit model, then store it."""
    name = name[:100]
    if db.session.scalar(db.select(StoredModel.id).where(StoredModel.name == name)):
        raise ValueError("A model with this name already exists.")
    check_digit_model(load_from_bytes(data, filename))
    db.session.add(StoredModel(
        name=name,
        description=description[:255] or None,
        filename=filename,
        size_bytes=len(data),
        data=data,
    ))
    db.session.commit()