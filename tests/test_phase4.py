"""Training/checkpoint contracts on synthetic data, never the real test folder."""

from dataclasses import replace
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from typing import Any, Callable

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data.dataset import build_training_datasets
from data.loaders import build_dataloaders
from data.manifest import save_manifest
from data.records import ManifestRow
from models.builder import build_model
from models.feature_extractor import FeatureExtractorClassifier
from training.checkpoint import BestLossCheckpointer
from training.checkpoint import CheckpointInfo
from training.metrics import ClassificationMeter, EpochRecord, MetricHistory
from training.pipeline import run_training
from training.reporting import LoggingEpochReporter, assess_learning_curves, save_curves, save_history
from training.trainer import Trainer


class TinyClassifier(FeatureExtractorClassifier):
    """Small real torch model makes loop tests fast and records mode/gradient state."""

    def __init__(self) -> None:
        super().__init__(nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1)),
                         nn.Linear(4, 3), freeze_features=True)
        self.forward_modes: list[tuple[bool, bool]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.forward_modes.append((self.training, torch.is_grad_enabled()))
        return super().forward(images)


class TrainingTests(unittest.TestCase):
    """Dependency-injected optimization, checkpoint selection, reload and curves."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.config = Config(phase=4, pretrained_weights=None, phase4_output_dir=self.root,
                             torch_hub_dir=self.root / "hub", device="cpu")
        torch.manual_seed(self.config.seed)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_trainer(self) -> tuple[Trainer, TinyClassifier, MetricHistory]:
        """Five default epochs, Adam over only the head and partial final batches."""
        model = TinyClassifier()
        train = TensorDataset(torch.randn(40, 3, 8, 8), torch.arange(40) % 3)
        dev = TensorDataset(torch.randn(17, 3, 8, 8), torch.arange(17) % 3)
        loaders = build_dataloaders({"train": train, "dev": dev}, self.config)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=self.config.lr)
        history = MetricHistory()
        trainer = Trainer(model, optimizer, nn.CrossEntropyLoss(), loaders,
                          BestLossCheckpointer(self.root / "best.pt"), self.config,
                          torch.device("cpu"), history, LoggingEpochReporter())
        return trainer, model, history

    def test_five_epochs_checkpoint_and_modes(self) -> None:
        """The complete loop logs every epoch, keeps features frozen and selects minimum dev loss."""
        trainer, model, history = self.make_trainer()
        before = {name: value.clone() for name, value in model.features.state_dict().items()}
        head_before = model.fc.weight.detach().clone()
        with self.assertLogs("training.reporting", logging.INFO) as logs:
            best = trainer.fit()
        self.assertEqual(len(history.records), 5)
        self.assertEqual(len(logs.records), 5)
        self.assertEqual(best.val_loss, min(record.val_loss for record in history.records))
        self.assertEqual(best.epoch, min(history.records, key=lambda record: record.val_loss).epoch)
        self.assertTrue(best.path.is_file())
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.features.state_dict().items()))
        self.assertFalse(torch.equal(head_before, model.fc.weight))
        self.assertTrue(any(mode for mode, _ in model.forward_modes))
        self.assertTrue(any(not mode for mode, _ in model.forward_modes))
        self.assertTrue(all(grad_enabled == mode for mode, grad_enabled in model.forward_modes))
        self.assertEqual(sum(len(loader.dataset) for loader in trainer.loaders.values()), 57)

    def test_checkpoint_reload_into_fresh_model_has_identical_outputs(self) -> None:
        """A saved checkpoint is complete, safe-loadable and independent of later model changes."""
        model = TinyClassifier().eval()
        probe = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            expected = model(probe).clone()
        strategy = BestLossCheckpointer(self.root / "roundtrip.pt")
        saved = strategy.save_if_best(model, epoch=2, val_loss=0.7, config=self.config)
        self.assertIsNotNone(saved)
        self.assertTrue(strategy.path.is_file())
        with torch.no_grad():
            model.fc.weight.add_(10)
        payload = torch.load(strategy.path, map_location="cpu", weights_only=True)
        self.assertEqual(set(payload), {"model_state_dict", "epoch", "val_loss", "class_names", "config"})
        self.assertEqual(payload["epoch"], 2)
        self.assertEqual(payload["class_names"], ["dog", "cat", "wild"])
        self.assertEqual(payload["config"]["epochs"], 5)
        self.assertEqual(payload["config"]["lr"], 0.001)
        self.assertIsInstance(payload["config"]["phase4_output_dir"], str)
        fresh = TinyClassifier().eval()
        fresh.load_state_dict(payload["model_state_dict"], strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(expected, fresh(probe)))

    def test_real_resnet_checkpoint_roundtrip(self) -> None:
        """The same storage contract preserves every ResNet18 parameter and BatchNorm buffer."""
        model = build_model(self.config).eval()
        probe = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            expected = model(probe)
        strategy = BestLossCheckpointer(self.root / "resnet.pt")
        strategy.save_if_best(model, epoch=1, val_loss=1.0, config=self.config)
        fresh = build_model(self.config).eval()
        payload = torch.load(strategy.path, map_location="cpu", weights_only=True)
        fresh.load_state_dict(payload["model_state_dict"], strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(expected, fresh(probe)))

    def test_trained_checkpoint_restores_best_epoch_outputs(self) -> None:
        """Reload the actual best training snapshot, even if later epochs change the head."""
        trainer, model, history = self.make_trainer()
        probe = torch.randn(4, 3, 8, 8)

        class CapturingCheckpointer(BestLossCheckpointer):
            """Record best-epoch outputs independently from serialization."""

            expected: torch.Tensor | None = None

            def save_if_best(self, model: nn.Module, *, epoch: int, val_loss: float,
                             config: Config) -> CheckpointInfo | None:
                saved = super().save_if_best(model, epoch=epoch, val_loss=val_loss, config=config)
                if saved is not None:
                    with torch.no_grad():
                        self.expected = model(probe).clone()
                return saved

        strategy = CapturingCheckpointer(self.root / "trained.pt")
        trainer.checkpointer = strategy
        best = trainer.fit()
        payload = torch.load(best.path, map_location="cpu", weights_only=True)
        fresh = TinyClassifier().eval()
        fresh.load_state_dict(payload["model_state_dict"], strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(strategy.expected, fresh(probe)))
        self.assertEqual(payload["val_loss"], min(row.val_loss for row in history.records))

    def test_checkpointer_retains_strict_minimum_and_earliest_tie(self) -> None:
        """Worse losses and equal losses never overwrite the selected state."""
        model = TinyClassifier()
        strategy = BestLossCheckpointer(self.root / "selection.pt")
        for epoch, loss in enumerate((0.8, 0.4, 0.6, 0.4, 0.7), start=1):
            strategy.save_if_best(model, epoch=epoch, val_loss=loss, config=self.config)
        payload = torch.load(strategy.path, weights_only=True)
        self.assertEqual(payload["epoch"], 2)
        self.assertEqual(payload["val_loss"], 0.4)
        with self.assertRaisesRegex(ValueError, "finite loss"):
            strategy.save_if_best(model, epoch=6, val_loss=float("nan"), config=self.config)

    def test_metrics_weight_partial_batches(self) -> None:
        """Two differently sized batches must not receive equal averaging weights."""
        meter = ClassificationMeter()
        meter.update(torch.tensor([[2., 0., 0.], [0., 2., 0.]]), torch.tensor([0, 1]), 1.0)
        meter.update(torch.tensor([[2., 0., 0.]]), torch.tensor([2]), 4.0)
        result = meter.compute()
        self.assertEqual(result.loss, 2.0)
        self.assertAlmostEqual(result.accuracy, 2 / 3)
        self.assertEqual(result.sample_count, 3)
        with self.assertRaisesRegex(ValueError, "empty"):
            ClassificationMeter().compute()

    def test_trainer_rejects_test_loader(self) -> None:
        """Even an injected test loader is refused by the training loop."""
        trainer, _, _ = self.make_trainer()
        loaders = {**trainer.loaders, "test": trainer.loaders["dev"]}
        with self.assertRaisesRegex(ValueError, "test is forbidden"):
            Trainer(trainer.model, trainer.optimizer, trainer.criterion, loaders, trainer.checkpointer,
                    self.config, trainer.device, trainer.metrics, trainer.reporter)

    def test_optimizer_rejects_frozen_parameters(self) -> None:
        """Incorrect optimizer wiring is detected before the first update."""
        trainer, model, _ = self.make_trainer()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        with self.assertRaisesRegex(ValueError, "no frozen parameters"):
            Trainer(model, optimizer, trainer.criterion, trainer.loaders, trainer.checkpointer,
                    self.config, trainer.device, trainer.metrics, trainer.reporter)

    def test_curve_files_are_saved_from_actual_test_history(self) -> None:
        """Headless plotting and CSV persistence use the measured five-epoch history."""
        trainer, _, history = self.make_trainer()
        trainer.fit()
        save_history(history.records, self.root / "history.csv")
        paths = save_curves(history.records, self.root, dpi=self.config.plot_dpi)
        self.assertEqual(len((self.root / "history.csv").read_text().splitlines()), 6)
        for path in paths:
            with Image.open(path) as image:
                image.verify()

    def test_curve_assessments_are_qualified(self) -> None:
        """Trend labels are explained heuristics, not claims based on unobserved runs."""
        healthy = [EpochRecord(1, 1.0, .65, .9, .65), EpochRecord(5, .5, .85, .6, .8)]
        overfit = [EpochRecord(1, 1.0, .65, .7, .65), EpochRecord(3, .6, .85, .5, .8), EpochRecord(5, .4, .95, .8, .7)]
        underfit = [EpochRecord(1, 1.1, .3, 1.1, .3), EpochRecord(5, 1.0, .4, 1.1, .35)]
        self.assertIn("Healthy", assess_learning_curves(healthy, self.config))
        self.assertIn("overfitting", assess_learning_curves(overfit, self.config))
        self.assertIn("underfitting", assess_learning_curves(underfit, self.config))
        self.assertIn("Inconclusive", assess_learning_curves([], self.config))

    def test_training_dataset_builder_never_accesses_val(self) -> None:
        """A forbidden nonexistent val path in the manifest cannot be resolved or inspected."""
        root = self.root / "dataset"
        rows: list[ManifestRow] = []
        for split in ("train", "dev"):
            for label in self.config.class_names:
                path = root / "train" / label / f"{split}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), "red").save(path)
                rows.append(ManifestRow(path.relative_to(root).as_posix(), label, f"{label}-{split}", split))
        rows.append(ManifestRow("val/dog/DO_NOT_ACCESS.png", "dog", "test-content", "test"))
        config = replace(self.config, dataset_root=root, manifest_path=self.root / "manifest.csv")
        save_manifest(rows, config.manifest_path)
        def guard(method: str) -> Callable[..., Any]:
            original = getattr(Path, method)

            def deny_val(path: Path, *args: Any, **kwargs: Any) -> Any:
                if "val" in path.parts:
                    raise AssertionError(f"Final-test path accessed via {method}")
                return original(path, *args, **kwargs)
            return deny_val

        with ExitStack() as stack:
            stack.enter_context(patch("data.dataset.TestFolderDataset", side_effect=AssertionError("Test Dataset constructed")))
            for method in ("resolve", "stat", "iterdir", "rglob", "open"):
                stack.enter_context(patch.object(Path, method, guard(method)))
            datasets = build_training_datasets(config)
            loaders = build_dataloaders(datasets, config)
            for loader in loaders.values():
                next(iter(loader))
        self.assertEqual(set(datasets), {"train", "dev"})

    def test_real_pipeline_records_pretrained_blocker_before_data_access(self) -> None:
        """Unavailable ImageNet weights produce explicit failure, not fake training results."""
        config = replace(self.config, pretrained_weights="IMAGENET1K_V1")
        with patch("training.pipeline.build_model", side_effect=OSError("network unavailable")), \
             patch("training.pipeline.build_training_datasets", side_effect=AssertionError("data accessed")):
            with self.assertRaisesRegex(OSError, "network unavailable"):
                run_training(config)
        status = json.loads((self.root / "training_status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertIsNone(status["best_epoch"])
        self.assertIsNone(status["best_val_loss"])
        self.assertFalse((self.root / self.config.checkpoint_filename).exists())

    def test_pipeline_wiring_with_train_dev_files_and_mocked_weight_acquisition(self) -> None:
        """Execute production composition while replacing only pretrained acquisition.

        This is synthetic fixture data and is not evidence of ImageNet loading
        or AFHQ training. All outputs are scoped to this temporary directory.
        """
        dataset_root = self.root / "images"
        rows: list[ManifestRow] = []
        for split in ("train", "dev"):
            for index, label in enumerate(self.config.class_names):
                path = dataset_root / "train" / label / f"{split}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), (index * 70, 100, 40)).save(path)
                rows.append(ManifestRow(path.relative_to(dataset_root).as_posix(), label, f"{split}-{label}", split))
        rows.append(ManifestRow("val/dog/NEVER_OPEN.png", "dog", "test-only", "test"))
        config = replace(self.config, pretrained_weights="IMAGENET1K_V1", dataset_root=dataset_root,
                         manifest_path=self.root / "manifest.csv", image_size=8)
        save_manifest(rows, config.manifest_path)
        model = TinyClassifier()
        original_adam = torch.optim.Adam
        with patch("training.pipeline.build_model", return_value=model), \
             patch("training.pipeline.torch.optim.Adam", wraps=original_adam) as optimizer_factory, \
             patch("data.dataset.TestFolderDataset", side_effect=AssertionError("test touched")):
            report = run_training(config)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["history"]), 5)
        self.assertEqual(report["best_val_loss"], min(row["val_loss"] for row in report["history"]))
        self.assertEqual(report["dataset_counts"], {"train": 3, "dev": 3})
        self.assertFalse(report["test_dataset_constructed"])
        self.assertEqual(optimizer_factory.call_args.kwargs["lr"], .001)
        self.assertEqual(optimizer_factory.call_args.kwargs["betas"], config.adam_betas)
        self.assertTrue((self.root / config.checkpoint_filename).is_file())
        self.assertTrue((self.root / "loss_curve.png").is_file())
        self.assertTrue((self.root / "accuracy_curve.png").is_file())

    def test_real_pipeline_rejects_random_weight_fallback(self) -> None:
        """The architecture-only option cannot silently become a real training run."""
        with self.assertRaisesRegex(ValueError, "requires pretrained weights"):
            run_training(self.config)


if __name__ == "__main__":
    unittest.main()
