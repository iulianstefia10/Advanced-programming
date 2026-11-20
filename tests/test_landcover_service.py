import io
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)

from services import landcover


class FakeModel:

    def __init__(self):
        self.seen = []

    def predict(self, batch, verbose=0):
        self.seen.append(batch)
        out = np.zeros((len(batch), 10), dtype="float32")
        for i, patch in enumerate(batch):
            out[i, 1 if patch.mean() > 0.5 else 3] = 1.0
        return out


def png_bytes(width, height, mode="RGB", color=(255, 255, 255)):
    buffer = io.BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


class ReadImageTests(unittest.TestCase):
    def test_rgb_image(self):
        image, resized = landcover.read_image(png_bytes(200, 100))
        self.assertEqual(image.shape, (100, 200, 3))
        self.assertFalse(resized)

    def test_rgba_is_converted_to_three_channels(self):
        image, _ = landcover.read_image(png_bytes(128, 128, "RGBA", (255, 0, 0, 128)))
        self.assertEqual(image.shape, (128, 128, 3))

    def test_empty_file_refused(self):
        with self.assertRaises(landcover.InvalidImage):
            landcover.read_image(b"")

    def test_not_an_image_refused(self):
        with self.assertRaises(landcover.InvalidImage):
            landcover.read_image(b"this is not an image")

    def test_too_small_refused(self):
        with self.assertRaises(landcover.InvalidImage):
            landcover.read_image(png_bytes(63, 200))

    def test_too_big_file_refused(self):
        with self.assertRaises(landcover.InvalidImage):
            landcover.read_image(b"0" * (landcover.MAX_UPLOAD_BYTES + 1))

    def test_large_image_is_resized(self):
        image, resized = landcover.read_image(png_bytes(landcover.MAX_SIDE + 400, 640))
        self.assertTrue(resized)
        self.assertEqual(max(image.shape[:2]), landcover.MAX_SIDE)


class InferenceTests(unittest.TestCase):
    def test_cut_patches_loops_y_then_x(self):
        image = np.zeros((130, 200, 3), dtype=np.uint8)
        patches, rows, columns = landcover.cut_patches(image, 64)
        self.assertEqual((rows, columns), (2, 3))
        self.assertEqual(patches.shape, (6, 64, 64, 3))

    def test_matrix_is_indexed_y_then_x(self):
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        image[:, 64:] = 255
        result = landcover.run_inference(image, FakeModel(), 64)
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result.tolist(), [[3, 1]])

    def test_large_patches_are_resized_to_model_size(self):
        model = FakeModel()
        result = landcover.run_inference(np.zeros((256, 384, 3), dtype=np.uint8), model, 128)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(model.seen[0].shape[1:], (64, 64, 3))

    def test_patch_size_values(self):
        self.assertEqual(landcover.parse_patch_size(None), landcover.DEFAULT_PATCH_SIZE)
        self.assertEqual(landcover.parse_patch_size("96"), 96)
        for wrong in ("abc", "100", "0", "-64"):
            with self.assertRaises(landcover.InvalidImage):
                landcover.parse_patch_size(wrong)

    def test_pixels_are_only_divided_by_255(self):
        model = FakeModel()
        landcover.run_inference(np.full((64, 64, 3), 255, dtype=np.uint8), model, 64)
        self.assertEqual(float(model.seen[0].max()), 1.0)
        self.assertEqual(str(model.seen[0].dtype), "float32")

    def test_more_patches_than_one_batch(self):
        image = np.zeros((64, 64 * 70, 3), dtype=np.uint8)
        model = FakeModel()
        result = landcover.run_inference(image, model, 64)
        self.assertEqual(result.shape, (1, 70))
        self.assertEqual(len(model.seen), 2)

    def test_statistics(self):
        stats = landcover.class_statistics(np.array([[1, 1, 3], [1, 3, 9]]), landcover.DEFAULT_CLASSES)
        self.assertEqual([row["name"] for row in stats], ["Forest", "Highway", "SeaLake"])
        self.assertEqual(stats[0]["count"], 3)
        self.assertEqual(stats[0]["percent"], 50.0)


class AnalyseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model_dir = os.path.join(self.tmp.name, "models_store")
        self.output_dir = os.path.join(self.tmp.name, "static", "landcover")
        os.makedirs(self.model_dir)
        self._original = landcover._load_model
        landcover._load_model = lambda path: FakeModel()
        landcover._cache.update(path=None, mtime=None, model=None)

    def tearDown(self):
        landcover._load_model = self._original
        self.tmp.cleanup()

    def put_model(self):
        open(os.path.join(self.model_dir, landcover.MODEL_NAME), "wb").close()

    def test_missing_model(self):
        self.assertFalse(landcover.model_available(self.model_dir))
        with self.assertRaises(landcover.ModelNotFound):
            landcover.analyse(png_bytes(128, 128), self.model_dir, self.output_dir)

    def test_class_names_default_and_from_file(self):
        self.assertEqual(landcover.load_class_names(self.model_dir), landcover.DEFAULT_CLASSES)
        indices = {"B": 1, "A": 0}
        np.save(os.path.join(self.model_dir, landcover.CLASSES_NAME), indices)
        self.assertEqual(landcover.load_class_names(self.model_dir), ["A", "B"])

    def test_analyse_saves_map_and_matrix(self):
        self.put_model()
        result = landcover.analyse(png_bytes(200, 130), self.model_dir, self.output_dir, 64)
        self.assertEqual((result["rows"], result["columns"], result["patches"]), (2, 3, 6))
        self.assertEqual(result["stats"][0]["name"], "Forest")
        self.assertEqual(result["stats"][0]["percent"], 100.0)
        matrix = np.load(os.path.join(self.output_dir, os.path.basename(result["npy_file"])))
        self.assertEqual(matrix.shape, (2, 3))
        map_path = os.path.join(self.output_dir, os.path.basename(result["map_file"]))
        self.assertTrue(os.path.getsize(map_path) > 0)
        self.assertTrue(result["map_file"].startswith("landcover/"))

    def test_image_smaller_than_patch_refused(self):
        self.put_model()
        with self.assertRaises(landcover.InvalidImage):
            landcover.analyse(png_bytes(100, 100), self.model_dir, self.output_dir, 128)

    def test_model_is_loaded_only_once(self):
        self.put_model()
        calls = []
        landcover._load_model = lambda path: calls.append(path) or FakeModel()
        landcover.get_model(self.model_dir)
        landcover.get_model(self.model_dir)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()