"""Offline model contracts use real ResNet18 with explicit random initialization."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from config import Config
from models.builder import (ModelBuilder, ResNet18FeatureExtractorBuilder, build_model,
                            create_model_builder, register_model)
from models.feature_extractor import FeatureExtractorClassifier
from models.sanity import inspect_model, select_device


class ModelTests(unittest.TestCase):
    """Freeze, BatchNorm, output, configuration, factory and loading behavior."""

    def setUp(self) -> None:
        """Prevent downloads and keep every test artifact inside a temporary directory."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = Config(phase=3, pretrained_weights=None, device="cpu",
                             torch_hub_dir=self.root / "hub", phase3_output_dir=self.root / "reports")
        torch.manual_seed(self.config.seed)

    def tearDown(self) -> None:
        """Clean only artifacts created by this test."""
        self.temporary.cleanup()

    def test_only_fc_parameters_are_trainable(self) -> None:
        """All feature parameters are frozen, while both head parameters require gradients."""
        model = build_model(self.config)
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad},
                         {"fc.weight", "fc.bias"})
        self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))
        self.assertEqual(model.fc.in_features, 512)
        self.assertEqual(model.fc.out_features, 3)

    def test_total_and_trainable_parameter_counts(self) -> None:
        """Expected counts detect accidental duplicate modules or an unchanged 1000-class head."""
        model = build_model(self.config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11_178_051)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1_539)

    def test_output_shape_for_different_batch_sizes(self) -> None:
        """Real 224-pixel forwards preserve N and produce exactly three class logits."""
        model = build_model(self.config)
        for count in (1, 2, 4):
            with self.subTest(batch_size=count), torch.no_grad():
                output = model(torch.randn(count, 3, 224, 224))
                self.assertEqual(tuple(output.shape), (count, 3))
                self.assertTrue(torch.isfinite(output).all().item())

    def test_frozen_batchnorm_stays_eval_after_mode_switches(self) -> None:
        """Repeated trainer-style train()/eval() calls cannot reactivate frozen BatchNorm."""
        model = build_model(self.config)
        for mode in (True, False, True, True):
            self.assertIs(model.train(mode), model)
            self.assertEqual(model.training, mode)
            self.assertEqual(model.fc.training, mode)
            norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
            self.assertEqual(len(norms), 20)
            self.assertTrue(all(not m.training for m in norms))

    def test_frozen_batchnorm_buffers_do_not_change(self) -> None:
        """requires_grad=False alone is insufficient; compare actual running buffers."""
        model = build_model(self.config).train()
        before = {name: value.clone() for name, value in model.named_buffers()}
        with torch.no_grad():
            model(torch.randn(4, 3, 224, 224))
        self.assertTrue(before)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.named_buffers()))

    def test_backward_produces_only_head_gradients(self) -> None:
        """A single autograd check verifies the head can learn without feature gradients."""
        model = build_model(self.config).train()
        model(torch.randn(2, 3, 224, 224)).sum().backward()
        self.assertIsNotNone(model.fc.weight.grad)
        self.assertIsNotNone(model.fc.bias.grad)
        self.assertTrue(all(p.grad is None for p in model.features.parameters()))

    def test_freeze_behavior_comes_from_config(self) -> None:
        """An explicit unfrozen configuration enables feature gradients and BatchNorm updates."""
        model = build_model(replace(self.config, freeze_features=False)).train()
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertTrue(all(m.training for m in norms))
        before = norms[0].num_batches_tracked.clone()
        with torch.no_grad():
            model(torch.randn(2, 3, 224, 224))
        self.assertTrue(torch.equal(norms[0].num_batches_tracked, before + 1))

    def test_num_classes_comes_from_config(self) -> None:
        """Head width uses the configured class count, not a hardcoded three."""
        config = replace(self.config, num_classes=5, class_names=("a", "b", "c", "d", "e"))
        model = build_model(config)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(2, 3, 224, 224)).shape), (2, 5))

    def test_pretrained_api_receives_explicit_weights_enum(self) -> None:
        """Mock only weight acquisition; do not pretend this loads actual ImageNet weights."""
        def offline_source(*, weights: ResNet18_Weights | None) -> nn.Module:
            return resnet18(weights=None)

        with patch("models.builder.resnet18", side_effect=offline_source) as source:
            model = ResNet18FeatureExtractorBuilder(replace(self.config, pretrained_weights="IMAGENET1K_V1")).build()
        source.assert_called_once_with(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.assertEqual(model.fc.out_features, 3)

    def test_pretrained_failure_is_propagated_without_fallback(self) -> None:
        """A failed ImageNet download must not quietly create an untrained classifier."""
        config = replace(self.config, pretrained_weights="IMAGENET1K_V1")
        with patch("models.builder.resnet18", side_effect=OSError("offline")) as source:
            with self.assertRaisesRegex(OSError, "offline"):
                inspect_model(config, create_model_builder(config))
        self.assertEqual(source.call_count, 1)
        error = json.loads((config.phase3_output_dir / "model_build_error.json").read_text())
        self.assertFalse(error["passed"])
        self.assertFalse(error["pretrained_weights_loaded"])
        self.assertFalse((config.phase3_output_dir / "model_report.json").exists())

    def test_registry_extends_without_editing_factory(self) -> None:
        """Another architecture satisfies the same small abstract build interface."""
        @register_model("phase3_test_tiny")
        class TinyBuilder(ModelBuilder):
            """Lightweight fixture architecture following the common feature/head contract."""

            def __init__(self, config: Config) -> None:
                self.config = config

            def build(self) -> FeatureExtractorClassifier:
                return FeatureExtractorClassifier(nn.AdaptiveAvgPool2d(1), nn.Linear(3, self.config.num_classes),
                                                   freeze_features=self.config.freeze_features)

        builder = create_model_builder(replace(self.config, architecture="phase3_test_tiny"))
        self.assertIsInstance(builder, ModelBuilder)
        self.assertEqual(tuple(builder.build()(torch.randn(4, 3, 224, 224)).shape), (4, 3))
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_model("phase3_test_tiny")(TinyBuilder)

    def test_unknown_architecture_and_weights_fail(self) -> None:
        """Configuration typos fail early with an actionable message."""
        with self.assertRaisesRegex(ValueError, "Unknown architecture"):
            build_model(replace(self.config, architecture="missing"))
        with self.assertRaisesRegex(ValueError, "Unknown ResNet18 weights"):
            build_model(replace(self.config, pretrained_weights="missing"))

    def test_architecture_report_is_explicitly_not_pretrained(self) -> None:
        """Offline evidence records the real parameter/mode checks without claiming pretraining."""
        report = inspect_model(self.config, create_model_builder(self.config))
        self.assertTrue(report["passed"])
        self.assertFalse(report["pretrained_weights_loaded"])
        self.assertEqual(report["verification_mode"], "architecture_only")
        self.assertEqual(report["input_shape"], [4, 3, 224, 224])
        self.assertEqual(report["output_shape"], [4, 3])
        self.assertTrue(report["batchnorm_all_eval_after_train"])
        self.assertTrue(report["buffers_unchanged_after_train_forward"])
        self.assertTrue((self.config.phase3_output_dir / "architecture_check_report.json").exists())
        self.assertFalse((self.config.phase3_output_dir / "model_report.json").exists())

    def test_device_selection_and_unavailable_gpu(self) -> None:
        """CPU works everywhere, auto uses available GPUs, and explicit unavailable devices fail."""
        self.assertEqual(select_device("cpu"), torch.device("cpu"))
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "unavailable"):
                select_device("cuda")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(select_device("auto"), torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
