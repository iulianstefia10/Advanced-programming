import os
import threading
import time
import uuid

import numpy as np
from PIL import Image

MODEL_SIZE = 64
PATCH_SIZES = (64, 96, 128, 192, 256)
DEFAULT_PATCH_SIZE = 128
BATCH_SIZE = 64
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 50_000_000
MAX_SIDE = 2400
RESULT_LIFETIME = 3600

MODEL_NAME = "eurosat_vgg16.keras"
CLASSES_NAME = "class_indices.npy"


DEFAULT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_COLORS = [
    "#e6c229",
    "#1b7837",
    "#a6d96a",
    "#7f7f7f",
    "#d73027",
    "#fdae61",
    "#8c510a",
    "#f46d9b",
    "#4575b4",
    "#08306b",
]


class ModelNotFound(Exception):
    pass


class InvalidImage(ValueError):
    pass


_model_lock = threading.Lock()
_predict_lock = threading.Lock()
_cache = {"path": None, "mtime": None, "model": None}


def model_path(model_dir):
    return os.path.join(model_dir, MODEL_NAME)


def model_available(model_dir):
    return os.path.isfile(model_path(model_dir))


def _load_model(path):

    from tensorflow.keras.models import load_model

    return load_model(path)


def get_model(model_dir):
    path = model_path(model_dir)
    if not os.path.isfile(path):
        raise ModelNotFound(
            f"Model file not found. Put {MODEL_NAME} in the app/models_store folder."
        )
    mtime = os.path.getmtime(path)
    with _model_lock:
        if _cache["path"] != path or _cache["mtime"] != mtime:
            _cache.update(path=path, mtime=mtime, model=_load_model(path))
        return _cache["model"]


def load_class_names(model_dir):
    path = os.path.join(model_dir, CLASSES_NAME)
    if os.path.isfile(path):
        indices = np.load(path, allow_pickle=True).item()
        names = [None] * len(indices)
        for name, index in indices.items():
            names[index] = name
        return names
    return list(DEFAULT_CLASSES)


def read_image(data):
    if not data:
        raise InvalidImage("No image was sent.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImage("The image is too big (10 MB maximum).")

    import io

    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width * height > MAX_PIXELS:
            raise InvalidImage("The image has too many pixels.")
        image = image.convert("RGB")
    except InvalidImage:
        raise
    except Exception:
        raise InvalidImage("This file is not a valid image (use a PNG or JPEG screenshot).")

    if min(image.size) < MODEL_SIZE:
        raise InvalidImage(
            f"The image is too small: it must be at least {MODEL_SIZE}x{MODEL_SIZE} pixels."
        )

    resized = False
    if max(image.size) > MAX_SIDE:
        scale = MAX_SIDE / max(image.size)
        new_size = (max(MODEL_SIZE, round(image.size[0] * scale)),
                    max(MODEL_SIZE, round(image.size[1] * scale)))
        image = image.resize(new_size, Image.LANCZOS)
        resized = True
    return np.asarray(image), resized


def parse_patch_size(value):
    if value in (None, ""):
        return DEFAULT_PATCH_SIZE
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise InvalidImage("The patch size must be a number.")
    if size not in PATCH_SIZES:
        allowed = ", ".join(str(n) for n in PATCH_SIZES)
        raise InvalidImage(f"The patch size must be one of: {allowed}.")
    return size


def cut_patches(image, patch_size=DEFAULT_PATCH_SIZE, stride=None):
    stride = stride or patch_size
    height, width = image.shape[:2]
    patches = []
    rows = 0
    columns = 0
    for y in range(0, height - patch_size + 1, stride):
        rows += 1
        columns = 0
        for x in range(0, width - patch_size + 1, stride):
            columns += 1
            patch = image[y:y + patch_size, x:x + patch_size]
            if patch_size != MODEL_SIZE:
                patch = np.asarray(Image.fromarray(patch).resize((MODEL_SIZE, MODEL_SIZE)))
            patches.append(patch)
    return np.array(patches), rows, columns


def classify_patches(model, patches):
    classes = []
    for start in range(0, len(patches), BATCH_SIZE):
        batch = patches[start:start + BATCH_SIZE].astype("float32") / 255.0
        with _predict_lock:
            probabilities = np.asarray(model.predict(batch, verbose=0))
        classes.extend(np.argmax(probabilities, axis=1).tolist())
    return np.array(classes, dtype=np.uint8)


def run_inference(image, model, patch_size=DEFAULT_PATCH_SIZE):
    patches, rows, columns = cut_patches(image, patch_size)
    classes = classify_patches(model, patches)
    return classes.reshape(rows, columns)


def class_statistics(landcover, class_names):
    counts = np.bincount(landcover.ravel(), minlength=len(class_names))
    total = int(counts.sum()) or 1
    rows = []
    for index, name in enumerate(class_names):
        if counts[index]:
            rows.append({
                "name": name,
                "color": CLASS_COLORS[index % len(CLASS_COLORS)],
                "count": int(counts[index]),
                "percent": round(100 * int(counts[index]) / total, 1),
            })
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def _delete_old_results(output_dir):
    now = time.time()
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if name.endswith((".png", ".npy")) and now - os.path.getmtime(path) > RESULT_LIFETIME:
            try:
                os.remove(path)
            except OSError:
                pass


def _save_map(image, landcover, class_names, path, patch_size):
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(class_names))]
    cmap = ListedColormap(colors)
    rows, columns = landcover.shape
    covered_h, covered_w = rows * patch_size, columns * patch_size

    figure = Figure(figsize=(13, 5.5))
    ax_image, ax_map = figure.subplots(1, 2)
    ax_image.imshow(image[:covered_h, :covered_w])
    ax_image.set_title("Screenshot")
    ax_image.axis("off")
    ax_map.imshow(landcover, cmap=cmap, vmin=0, vmax=len(class_names) - 1,
                  interpolation="nearest", extent=(0, covered_w, covered_h, 0))
    ax_map.set_title(f"Land cover ({rows} x {columns} patches)")
    ax_map.axis("off")
    ax_map.legend(handles=[Patch(color=colors[i], label=name) for i, name in enumerate(class_names)],
                  loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.savefig(path, dpi=110, bbox_inches="tight")


def analyse(data, model_dir, output_dir, patch_size=DEFAULT_PATCH_SIZE):
    patch_size = parse_patch_size(patch_size)
    model = get_model(model_dir)
    image, resized = read_image(data)
    if min(image.shape[:2]) < patch_size:
        raise InvalidImage(f"The image is too small for patches of {patch_size} pixels.")
    class_names = load_class_names(model_dir)

    started = time.time()
    landcover = run_inference(image, model, patch_size)
    duration = time.time() - started

    os.makedirs(output_dir, exist_ok=True)
    _delete_old_results(output_dir)
    result_id = uuid.uuid4().hex[:12]
    map_file = f"{result_id}_map.png"
    npy_file = f"{result_id}_landcover.npy"
    np.save(os.path.join(output_dir, npy_file), landcover)
    _save_map(image, landcover, class_names, os.path.join(output_dir, map_file), patch_size)

    folder = os.path.basename(os.path.normpath(output_dir))
    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "resized": resized,
        "rows": landcover.shape[0],
        "columns": landcover.shape[1],
        "patch_size": patch_size,
        "patches": int(landcover.size),
        "seconds": round(duration, 1),
        "stats": class_statistics(landcover, class_names),
        "map_file": f"{folder}/{map_file}",
        "npy_file": f"{folder}/{npy_file}",
    }