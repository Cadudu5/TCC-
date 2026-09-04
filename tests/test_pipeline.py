import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from apps.inferencia_gui import compute_area_statistics
from features.extract import ARTICLE_FEATURE_NAMES, extract_features, segment_superpixels
from rotulador_lite import (
    new_labels_table,
    read_labels_table,
    write_labels_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FeatureExtractionTests(unittest.TestCase):
    def test_parallel_and_sequential_are_equal(self):
        rows, columns = np.indices((48, 64))
        image = np.stack(
            ((rows * 5) % 256, (columns * 3) % 256, ((rows + columns) * 7) % 256),
            axis=-1,
        ).astype(np.uint8)
        labels = segment_superpixels(image, n_segments=32)
        sequential = extract_features(image, labels, max_workers=1)
        parallel = extract_features(image, labels, max_workers=2)
        self.assertEqual(list(sequential.columns), list(parallel.columns))
        np.testing.assert_allclose(
            sequential.select_dtypes(include="number"),
            parallel.select_dtypes(include="number"),
            rtol=0,
            atol=0,
        )
        self.assertTrue(set(ARTICLE_FEATURE_NAMES).issubset(sequential.columns))

    def test_area_is_computed_by_pixels(self):
        labels = np.array([[1, 1, 2], [1, 3, 3]], dtype=np.int32)
        positive_pixels, tissue_pixels, percentage = compute_area_statistics(labels, {2}, {1})
        self.assertEqual(positive_pixels, 3)
        self.assertEqual(tissue_pixels, 5)
        self.assertAlmostEqual(percentage, 60.0)


class ArtifactContractTests(unittest.TestCase):
    def test_existing_neutrophil_rf_contract(self):
        metadata_path = PROJECT_ROOT / "models" / "artifacts" / "rf_best_cv_meta.json"
        model_path = PROJECT_ROOT / "models" / "artifacts" / "rf_best_cv.joblib"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
        self.assertEqual(len(metadata["feature_names"]), int(model.n_features_in_))

    def test_article_background_contract_when_present(self):
        metadata_path = PROJECT_ROOT / "models" / "artifacts" / "fundo_xgb_artigo_meta.json"
        model_path = PROJECT_ROOT / "models" / "artifacts" / "fundo_xgb_artigo.joblib"
        if not metadata_path.exists() or not model_path.exists():
            self.skipTest("Modelo reconstruído ainda não foi treinado")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
        self.assertEqual(metadata["feature_names"], ARTICLE_FEATURE_NAMES)
        self.assertEqual(int(model.n_features_in_), len(ARTICLE_FEATURE_NAMES))


class LabelerContractTests(unittest.TestCase):
    def setUp(self):
        self.superpixels = np.array([[1, 1, 2], [3, 3, 2]], dtype=np.int32)

    def test_label_table_round_trip_has_only_the_contract_columns(self):
        table = new_labels_table(self.superpixels)
        table.loc[table["superpixel_id"] == 2, "label"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotulos_teste.csv"
            write_labels_atomic(table, path)
            restored = read_labels_table(path, np.unique(self.superpixels))
        self.assertEqual(list(restored.columns), ["superpixel_id", "label"])
        np.testing.assert_array_equal(restored, table)

    def test_label_table_rejects_another_segmentation(self):
        table = new_labels_table(self.superpixels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotulos_teste.csv"
            write_labels_atomic(table, path)
            with self.assertRaisesRegex(ValueError, "não corresponde"):
                read_labels_table(path, np.array([1, 2, 4], dtype=np.int32))

    def test_label_table_rejects_fractional_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotulos_invalidos.csv"
            pd.DataFrame(
                {"superpixel_id": [1, 2, 3], "label": [0, 0.5, 1]}
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "números inteiros"):
                read_labels_table(path, np.array([1, 2, 3], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
