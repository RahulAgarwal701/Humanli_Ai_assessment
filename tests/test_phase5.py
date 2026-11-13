"""Evaluation contracts and best-checkpoint provenance on isolated fixtures."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from typing import Any, Sequence

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data.manifest import save_json, save_manifest
from data.records import ManifestRow
from evaluation.checkpoint import load_best_checkpoint, sha256_file
from evaluation.evaluator import Evaluator
from evaluation.metrics import AccuracyMetric, ClassificationReportMetric, ConfusionMatrixMetric, Metric
from evaluation.pipeline import run_evaluation
from evaluation.reporting import save_confusion_matrix
from models.feature_extractor import FeatureExtractorClassifier
from training.checkpoint import BestLossCheckpointer
from training.metrics import EpochRecord
from training.reporting import save_history


NAMES = ("dog", "cat", "wild")


class IndexedModel(nn.Module):
    """Known logits make metric values independently predictable."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor([[3., 0., 0.], [0., 3., 0.], [0., 3., 0.], [0., 0., 3.]]))
        self.observed_modes: list[tuple[bool, bool]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.observed_modes.append((self.training, torch.is_grad_enabled()))
        return self.logits[images[:, 0].long()]


class EvaluatorTests(unittest.TestCase):
    """Metrics are extensible and inference is strictly eval/no_grad."""

    def make_loader(self) -> DataLoader:
        dataset = TensorDataset(torch.arange(4).reshape(-1, 1).float(), torch.tensor([0, 1, 2, 2]))
        return DataLoader(dataset, batch_size=3, shuffle=False)

    def test_known_accuracy_report_and_confusion_counts(self) -> None:
        model = IndexedModel().train()
        before = model.logits.clone()
        evaluator = Evaluator(model, [AccuracyMetric(), ClassificationReportMetric(), ConfusionMatrixMetric()], torch.device("cpu"))
        result = evaluator.evaluate(self.make_loader(), NAMES)
        self.assertEqual(result.metrics["accuracy"], .75)
        self.assertEqual(result.metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 1, 1]])
        self.assertEqual(result.metrics["classification_report"]["cat"]["precision"], .5)
        self.assertEqual(result.metrics["classification_report"]["wild"]["recall"], .5)
        self.assertEqual(result.metrics["classification_report"]["wild"]["support"], 2)
        self.assertEqual(model.observed_modes, [(False, False), (False, False)])
        self.assertTrue(torch.equal(before, model.logits))

    def test_new_metric_injects_without_evaluator_changes(self) -> None:
        class ErrorCount(Metric):
            """An independently added metric uses the same tiny contract."""

            name = "error_count"

            def compute(self, targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]) -> int:
                return sum(target != predicted for target, predicted in zip(targets, predictions))

        result = Evaluator(IndexedModel(), [ErrorCount()], torch.device("cpu")).evaluate(self.make_loader(), NAMES)
        self.assertEqual(result.metrics, {"error_count": 1})

    def test_missing_prediction_classes_keep_class_names_and_zero_counts(self) -> None:
        report = ClassificationReportMetric().compute([0, 0], [0, 0], NAMES)
        self.assertEqual(report["wild"]["precision"], 0)
        self.assertEqual(report["wild"]["support"], 0)
        self.assertEqual(ConfusionMatrixMetric().compute([0, 0], [0, 0], NAMES), [[2, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_duplicate_metric_names_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            Evaluator(IndexedModel(), [AccuracyMetric(), AccuracyMetric()], torch.device("cpu"))

    def test_empty_loader_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty loader"):
            Evaluator(IndexedModel(), [AccuracyMetric()], torch.device("cpu")).evaluate([], NAMES)

    def test_invalid_output_shape_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "N-by-class"):
            Evaluator(nn.Identity(), [AccuracyMetric()], torch.device("cpu")).evaluate(self.make_loader(), NAMES)


class CheckpointAndPipelineTests(unittest.TestCase):
    """Validate minimum-loss selection before any final-test data is used."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.config = Config(phase=5, dataset_root=self.root / "dataset", manifest_path=self.root / "manifest.csv",
                             phase4_output_dir=self.root / "training", phase5_output_dir=self.root / "evaluation",
                             image_size=8, device="cpu", pretrained_weights=None)
        self.model = self.make_model(self.config)
        self.path = self.config.phase4_output_dir / self.config.checkpoint_filename
        BestLossCheckpointer(self.path).save_if_best(self.model, epoch=2, val_loss=.3, config=self.config)
        save_history([EpochRecord(1, 1., .6, .8, .6), EpochRecord(2, .5, .9, .3, .9),
                      EpochRecord(3, .4, .95, .5, .8)], self.config.phase4_output_dir / "history.csv")
        save_json({"status": "completed", "best_epoch": 2, "best_val_loss": .3,
                   "test_dataset_constructed": False}, self.config.phase4_output_dir / "training_status.json")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def make_model(config: Config) -> FeatureExtractorClassifier:
        """RGB-average classifier with an exact, deterministic class mapping."""
        model = FeatureExtractorClassifier(nn.AdaptiveAvgPool2d(1), nn.Linear(3, config.num_classes), freeze_features=True)
        with torch.no_grad():
            model.fc.weight.copy_(torch.eye(3) * 5)
            model.fc.bias.zero_()
        return model

    def test_loads_best_earlier_epoch_instead_of_last(self) -> None:
        with patch("evaluation.checkpoint.build_model", side_effect=self.make_model):
            loaded = load_best_checkpoint(self.config)
        self.assertEqual(loaded.provenance["epoch"], 2)
        self.assertEqual(loaded.provenance["dev_val_loss"], .3)
        self.assertTrue(loaded.provenance["selection_verified_against_history"])
        self.assertFalse(loaded.provenance["test_used_for_checkpoint_selection"])

    def test_rejects_last_epoch_when_it_is_not_best(self) -> None:
        payload = torch.load(self.path, weights_only=True)
        payload.update(epoch=3, val_loss=.5)
        torch.save(payload, self.path)
        with self.assertRaisesRegex(ValueError, "lowest dev-validation"):
            load_best_checkpoint(self.config)

    def test_rejects_class_order_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "class names/order"):
            load_best_checkpoint(replace(self.config, class_names=("cat", "dog", "wild")))

    def test_rejects_training_that_constructed_a_test_dataset(self) -> None:
        save_json({"status": "completed", "best_epoch": 2, "best_val_loss": .3,
                   "test_dataset_constructed": True}, self.config.phase4_output_dir / "training_status.json")
        with self.assertRaisesRegex(ValueError, "dev-only"):
            load_best_checkpoint(self.config)

    def test_restores_saved_preprocessing_and_does_not_download_weights(self) -> None:
        with patch("evaluation.checkpoint.build_model", side_effect=self.make_model) as builder:
            loaded = load_best_checkpoint(replace(self.config, image_size=224))
        self.assertEqual(loaded.config.image_size, 8)
        self.assertIsNone(builder.call_args.args[0].pretrained_weights)

    def test_pipeline_saves_metrics_figure_and_keeps_checkpoint_unchanged(self) -> None:
        rows: list[ManifestRow] = []
        for index, label in enumerate(NAMES):
            for image_index in range(2):
                path = self.config.test_dir / label / f"{image_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                pixel = tuple(255 if channel == index else 0 for channel in range(3))
                Image.new("RGB", (10, 8), pixel).save(path)
                rows.append(ManifestRow(path.relative_to(self.config.dataset_root).as_posix(), label, f"{label}-{image_index}", "test"))
        # No train folder exists: the final-test pipeline must never require it.
        save_manifest(rows, self.config.manifest_path)
        before = sha256_file(self.path)
        with patch("evaluation.checkpoint.build_model", side_effect=self.make_model):
            report = run_evaluation(self.config)
        self.assertEqual(report["sample_count"], 6)
        self.assertEqual(report["metrics"]["accuracy"], 1.0)
        self.assertEqual(report["metrics"]["confusion_matrix"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        self.assertEqual(report["checkpoint"]["epoch"], 2)
        self.assertEqual(sha256_file(self.path), before)
        self.assertTrue(report["checkpoint_unchanged"])
        saved = json.loads((self.config.phase5_output_dir / "test_metrics.json").read_text())
        self.assertEqual(saved["metrics"]["classification_report"]["dog"]["support"], 2)
        with Image.open(self.config.phase5_output_dir / "confusion_matrix.png") as image:
            image.verify()
