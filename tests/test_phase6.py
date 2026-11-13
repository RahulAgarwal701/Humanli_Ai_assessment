"""Prediction fault isolation, checkpoint-owned metadata, demo safety and CLI tests."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from cli import parse_options
from config import Config
from data.audit import file_sha1
from data.manifest import save_json, save_manifest
from data.records import ManifestRow
from data.transforms import ImageTransform, build_transforms
from evaluation.checkpoint import LoadedCheckpoint, sha256_file
from inference.pipeline import run_prediction_demo, select_demo_inputs
from inference.predictor import Predictor
from inference.reporting import save_prediction_grid
from models.feature_extractor import FeatureExtractorClassifier
from training.checkpoint import BestLossCheckpointer
from training.metrics import EpochRecord
from training.reporting import save_history


class FixedLogits(nn.Module):
    """Known logits and mode observations independent of image contents."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor([[0., 2., 1.]]))
        self.calls: list[tuple[bool, bool, tuple[int, ...]]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls.append((self.training, torch.is_grad_enabled(), tuple(images.shape)))
        return self.values.expand(len(images), -1)


class PredictorTests(unittest.TestCase):
    """One bad file never prevents prediction on the next valid image."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.config = Config(phase=6, device="cpu", phase6_output_dir=self.root / "output")
        self.model = FixedLogits()
        self.saved_config = replace(self.config, class_names=("wild", "dog", "cat"))
        self.loaded = LoadedCheckpoint(self.model, self.saved_config, {"epoch": 2})
        self.good = self.root / "good.png"
        Image.new("RGB", (31, 19), (255, 20, 50)).save(self.good)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def predictor(self) -> Predictor:
        """Inject the same saved contract a real checkpoint reader provides."""
        return Predictor(self.config, checkpoint_loader=lambda config: self.loaded)

    def test_predicted_label_comes_from_checkpoint_order(self) -> None:
        predictor = self.predictor()
        result = predictor.predict(self.good)
        self.assertTrue(result.ok)
        self.assertEqual(predictor.class_names, ("wild", "dog", "cat"))
        self.assertEqual(result.predicted_class, "dog")  # Winning logit index is 1.

    def test_softmax_top_three_are_sorted_and_sum_to_one(self) -> None:
        result = self.predictor().predict(self.good)
        expected = torch.softmax(torch.tensor([0., 2., 1.]), dim=0)
        self.assertEqual([item.class_name for item in result.top3], ["dog", "cat", "wild"])
        self.assertAlmostEqual(result.confidence, expected[1].item())
        self.assertAlmostEqual(sum(item.probability for item in result.top3), 1., places=6)
        self.assertTrue(all(0 <= item.probability <= 1 for item in result.top3))

    def test_missing_file_returns_error_record(self) -> None:
        result = self.predictor().predict(self.root / "missing.png")
        self.assertFalse(result.ok)
        self.assertIn("FileNotFoundError", result.error)
        self.assertIsNone(result.predicted_class)
        self.assertIsNone(result.confidence)
        self.assertEqual(result.top3, ())

    def test_nonimage_file_returns_error_record(self) -> None:
        path = self.root / "notes.txt"
        path.write_text("This is not an image.")
        result = self.predictor().predict(path)
        self.assertFalse(result.ok)
        self.assertIn("UnidentifiedImageError", result.error)

    def test_directory_returns_error_record(self) -> None:
        result = self.predictor().predict(self.root)
        self.assertFalse(result.ok)
        self.assertIn("IsADirectoryError", result.error)

    def test_rgb_grayscale_rgba_palette_and_la_are_converted_to_rgb(self) -> None:
        observed: list[str] = []

        def factory(mode: str, config: Config) -> ImageTransform:
            self.assertEqual(mode, "test")
            actual = build_transforms(mode, config)

            def transform(image: Image.Image) -> torch.Tensor:
                observed.append(image.mode)
                return actual(image)
            return transform

        predictor = Predictor(self.config, checkpoint_loader=lambda config: self.loaded, transform_factory=factory)
        for mode in ("RGB", "L", "RGBA", "P", "LA"):
            path = self.root / f"{mode}.png"
            Image.new("RGB", (29, 17), (255, 20, 50)).convert(mode).save(path)
            self.assertTrue(predictor.predict(path).ok)
        self.assertEqual(observed, ["RGB"] * 5)
        self.assertTrue(all(shape == (1, 3, 224, 224) for _, _, shape in self.model.calls))

    def test_mixed_batch_preserves_order_and_continues_after_failures(self) -> None:
        text = self.root / "bad.jpg"
        text.write_bytes(b"not a JPEG")
        paths = [self.good, self.root / "missing.png", text, self.good]
        results = self.predictor().predict_many(paths)
        self.assertEqual([result.ok for result in results], [True, False, False, True])
        self.assertEqual([result.path for result in results], [str(path) for path in paths])

    def test_eval_no_grad_and_repeated_predictions_are_identical(self) -> None:
        predictor = self.predictor()
        first = predictor.predict(self.good)
        self.model.train()  # A caller cannot accidentally leave prediction in train mode.
        torch.manual_seed(91)
        second = predictor.predict(self.good)
        self.assertEqual(first, second)
        self.assertTrue(all(not training and not grad for training, grad, _ in self.model.calls))

    def test_truncated_image_does_not_abort_batch(self) -> None:
        path = self.root / "truncated.jpg"
        Image.new("RGB", (100, 100), "red").save(path)
        path.write_bytes(path.read_bytes()[:-12])
        results = self.predictor().predict_many([path, self.good])
        self.assertFalse(results[0].ok)
        self.assertTrue(results[1].ok)

    def test_unexpected_per_image_plugin_or_model_failure_is_isolated(self) -> None:
        predictor = self.predictor()
        original = self.model.forward
        with patch.object(self.model, "forward", side_effect=[IndexError("one bad image"), original(torch.zeros(1, 3, 224, 224))]):
            results = predictor.predict_many([self.good, self.good])
        self.assertFalse(results[0].ok)
        self.assertTrue(results[1].ok)

    def test_invalid_checkpoint_fails_at_construction(self) -> None:
        with self.assertRaisesRegex(OSError, "bad checkpoint"):
            Predictor(self.config, checkpoint_loader=lambda config: (_ for _ in ()).throw(OSError("bad checkpoint")))

    def test_tied_probabilities_use_saved_class_order(self) -> None:
        self.model.values.zero_()
        result = self.predictor().predict(self.good)
        self.assertEqual([item.class_name for item in result.top3], list(self.saved_config.class_names))

    def test_error_placeholder_does_not_abort_visual_grid(self) -> None:
        predictor = self.predictor()
        results = predictor.predict_many([self.good, self.root / "missing.png"])
        path = save_prediction_grid(results, self.saved_config, source_description="Synthetic test fixture")
        with Image.open(path) as image:
            image.verify()


class DemoAndCheckpointTests(unittest.TestCase):
    """Demo selection avoids train/dev and default loading trusts saved class names."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.config = Config(phase=6, device="cpu", dataset_root=self.root / "dataset",
                             phase4_output_dir=self.root / "training", phase6_output_dir=self.root / "output",
                             manifest_path=self.root / "manifest.csv", new_images_dir=self.root / "new_images")
        self.rows = [ManifestRow(f"val/{label}/{i}.png", label, f"test-{label}-{i}", "test")
                     for label in self.config.class_names for i in range(2)]
        self.rows.append(ManifestRow("train/dog/train.png", "dog", "train-hash", "train"))
        save_manifest(self.rows, self.config.manifest_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fallback_uses_only_six_test_manifest_rows(self) -> None:
        paths, source = select_demo_inputs(self.config)
        self.assertEqual(len(paths), 6)
        self.assertTrue(all(path.is_relative_to(self.config.test_dir) for path in paths))
        self.assertIn("Reused final-test", source)

    def test_explicit_missing_input_directory_does_not_fall_back(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Input directory"):
            select_demo_inputs(self.config, input_dir=self.root / "absent")

    def test_new_images_directory_is_preferred(self) -> None:
        self.config.new_images_dir.mkdir()
        path = self.config.new_images_dir / "photo.png"
        Image.new("RGB", (8, 8), "blue").save(path)
        paths, source = select_demo_inputs(self.config)
        self.assertEqual(paths, [path])
        self.assertIn("new_images", source)

    def test_real_checkpoint_class_order_is_read_independently_of_defaults(self) -> None:
        saved = replace(self.config, class_names=("wild", "cat", "dog"), image_size=8)
        model = FeatureExtractorClassifier(nn.AdaptiveAvgPool2d(1), nn.Linear(3, 3), freeze_features=True)
        checkpoint = self.config.phase4_output_dir / self.config.checkpoint_filename
        BestLossCheckpointer(checkpoint).save_if_best(model, epoch=1, val_loss=.2, config=saved)
        save_history([EpochRecord(1, .4, .8, .2, .9)], self.config.phase4_output_dir / "history.csv")
        save_json({"status": "completed", "best_epoch": 1, "best_val_loss": .2,
                   "test_dataset_constructed": False}, self.config.phase4_output_dir / "training_status.json")
        with patch("evaluation.checkpoint.build_model", return_value=model):
            predictor = Predictor(self.config)
        self.assertEqual(predictor.class_names, saved.class_names)
        self.assertEqual(predictor.config.image_size, 8)

    def test_demo_rejects_renamed_training_copy_and_keeps_other_files(self) -> None:
        good = self.root / "new.png"
        Image.new("RGB", (8, 8), "red").save(good)
        copied = self.root / "renamed_training.png"
        Image.new("RGB", (8, 8), "blue").save(copied)
        self.rows[-1] = replace(self.rows[-1], group_id=file_sha1(copied))
        save_manifest(self.rows, self.config.manifest_path)
        checkpoint = self.config.phase4_output_dir / self.config.checkpoint_filename
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint fingerprint")
        loaded = LoadedCheckpoint(FixedLogits(), self.config, {"sha256": sha256_file(checkpoint), "epoch": 1})
        actual = Predictor(self.config, checkpoint_loader=lambda config: loaded)
        with patch("inference.pipeline.Predictor", return_value=actual):
            report = run_prediction_demo(self.config, paths=[copied, good, self.root / "missing.png"])
        self.assertEqual(report["success_count"], 1)
        self.assertEqual(report["error_count"], 2)
        self.assertIn("train/dev", report["predictions"][0]["error"])
        self.assertTrue(report["checkpoint_unchanged"])
        self.assertTrue((self.config.phase6_output_dir / "predictions.json").is_file())


class CLITests(unittest.TestCase):
    """Every stage remains available by readable command or original numeric flag."""

    def test_all_stage_commands_match_numeric_flags(self) -> None:
        commands = ("split", "transforms", "model", "train", "evaluate", "predict")
        for phase, command in enumerate(commands, start=1):
            self.assertEqual(parse_options([command]).config.phase, phase)
            self.assertEqual(parse_options(["--phase", str(phase)]).config.phase, phase)

    def test_prediction_files_and_checkpoint_arguments(self) -> None:
        options = parse_options(["predict", "--images", "one.jpg", "two.png", "--checkpoint", "run/best.pt"])
        self.assertEqual(options.images, (Path("one.jpg"), Path("two.png")))
        self.assertEqual(options.config.checkpoint_filename, "best.pt")
        self.assertEqual(options.config.phase4_output_dir, Path("run").resolve())

    def test_conflicting_stage_options_fail(self) -> None:
        with self.assertRaises(SystemExit):
            parse_options(["predict", "--phase", "4"])

    def test_prediction_inputs_are_rejected_for_training(self) -> None:
        with self.assertRaises(SystemExit):
            parse_options(["train", "--images", "one.jpg"])

    def test_files_and_directory_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_options(["predict", "--images", "one.jpg", "--input-dir", "images"])

    def test_main_dispatches_prediction_inputs_without_running_other_stages(self) -> None:
        from main import main
        with patch("main.run_phase_six", return_value={}) as predict:
            self.assertEqual(main(["predict", "--images", "example.png"]), 0)
        self.assertEqual(predict.call_args.args[0].phase, 6)
        self.assertEqual(predict.call_args.kwargs["paths"], (Path("example.png"),))
