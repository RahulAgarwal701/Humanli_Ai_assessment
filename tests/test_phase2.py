"""Phase 2 contracts, runnable with standard-library unittest or pytest."""

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms as T

from config import Config
from data.dataset import ManifestImageDataset, build_datasets
from data.loaders import build_dataloaders
from data.manifest import read_manifest, save_manifest
from data.records import ManifestRow
from data.sanity import check_datasets
from data.transforms import ImageTransform, build_eval_transform, build_transforms, register_transform


def patterned_image() -> Image.Image:
    """Asymmetric, non-square image makes spatial/color transformations observable."""
    y, x = np.mgrid[:40, :60]
    pixels = np.stack((x * 4, y * 6, (x + y) * 2), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels)


class TransformTests(unittest.TestCase):
    """Verify numerical contracts and deterministic versus random mode behavior."""

    def test_output_shape_dtype_and_finiteness(self) -> None:
        """Every registered built-in mode yields normalized float32 CHW images."""
        for mode in ("train", "dev", "test"):
            with self.subTest(mode=mode):
                result = build_transforms(mode)(patterned_image())
                self.assertEqual(tuple(result.shape), (3, 224, 224))
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(torch.isfinite(result).all().item())

    def test_eval_is_exactly_deterministic_despite_rng_changes(self) -> None:
        """No randomness may appear in dev or test preprocessing."""
        for mode in ("dev", "test"):
            transform = build_transforms(mode)
            first = transform(patterned_image())
            torch.manual_seed(987)
            torch.rand(100)
            second = transform(patterned_image())
            self.assertTrue(torch.equal(first, second))

    def test_dev_and_test_preprocessing_match(self) -> None:
        """Evaluation modes share the same preprocessing definition."""
        self.assertTrue(torch.equal(build_transforms("dev")(patterned_image()),
                                    build_transforms("test")(patterned_image())))

    def test_exact_imagenet_normalization(self) -> None:
        """Known pixel values verify channel order, scaling and normalization."""
        config = Config()
        pixel = (255, 128, 0)
        result = build_transforms("dev", config)(Image.new("RGB", (12, 20), pixel))
        expected = (torch.tensor(pixel, dtype=torch.float32) / 255
                    - torch.tensor(config.normalization_mean)) / torch.tensor(config.normalization_std)
        self.assertTrue(torch.equal(result[:, 0, 0], expected))

    def test_augmentation_order_and_magnitudes(self) -> None:
        """Only train includes mild PIL augmentation before ToTensor/Normalize."""
        config = Config()
        steps = build_transforms("train", config).transforms
        self.assertEqual([type(step) for step in steps], [T.Resize, T.RandomHorizontalFlip,
                         T.RandomRotation, T.ColorJitter, T.ToTensor, T.Normalize])
        self.assertEqual(steps[0].size, (224, 224))
        self.assertEqual(steps[1].p, config.horizontal_flip_probability)
        self.assertEqual(steps[2].degrees, [-10.0, 10.0])
        self.assertEqual(steps[3].brightness, (0.8, 1.2))
        self.assertEqual(steps[3].contrast, (0.8, 1.2))
        self.assertEqual([type(step) for step in build_transforms("dev", config).transforms],
                         [T.Resize, T.ToTensor, T.Normalize])

    def test_seeded_training_is_reproducible_and_varies_between_views(self) -> None:
        """Train transforms consume RNG but can be reproduced with the same seed."""
        transform = build_transforms("train")
        torch.manual_seed(42)
        first = transform(patterned_image())
        second = transform(patterned_image())
        torch.manual_seed(42)
        repeated = transform(patterned_image())
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, second))

    def test_registry_adds_mode_without_editing_factory(self) -> None:
        """A plugin can register a builder through the public extension point."""
        register_transform("phase2_test_custom_eval")(build_eval_transform)
        result = build_transforms("phase2_test_custom_eval")(patterned_image())
        self.assertEqual(tuple(result.shape), (3, 224, 224))
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_transform("phase2_test_custom_eval")(build_eval_transform)

    def test_unknown_mode_is_rejected(self) -> None:
        """A mode typo must not accidentally enable training augmentation."""
        with self.assertRaisesRegex(ValueError, "Unknown transform mode"):
            build_transforms("typo")


class DatasetTests(unittest.TestCase):
    """Exercise manifest selection, RGB handling, folder testing and DataLoaders."""

    def setUp(self) -> None:
        """Create tiny independent RGB, grayscale and RGBA images in every split."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = Config(dataset_root=self.root / "dataset", manifest_path=self.root / "manifest.csv",
                             phase2_output_dir=self.root / "phase2", phase=2)
        rows: list[ManifestRow] = []
        for split in ("train", "dev", "test"):
            source = "val" if split == "test" else "train"
            for label in self.config.class_names:
                for mode in ("RGB", "L", "RGBA"):
                    path = self.config.dataset_root / source / label / f"{split}_{mode}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    patterned_image().convert(mode).save(path)
                    rows.append(ManifestRow(path.relative_to(self.config.dataset_root).as_posix(),
                                            label, hashlib.sha1(path.read_bytes()).hexdigest(), split))
        save_manifest(rows, self.config.manifest_path)

    def tearDown(self) -> None:
        """Discard fixture images and diagnostics."""
        self.temporary.cleanup()

    def test_datasets_select_manifest_assignments_and_val_folder(self) -> None:
        """No resplitting; final-test Dataset enumerates only the original val source."""
        datasets = build_datasets(self.config)
        rows = read_manifest(self.config.manifest_path)
        for mode, dataset in datasets.items():
            expected = {(self.config.dataset_root / row.path).resolve() for row in rows if row.split == mode}
            self.assertEqual({sample.path for sample in dataset.samples}, expected)
            self.assertEqual(len(dataset), 9)
            self.assertEqual(dataset.class_to_idx, {"dog": 0, "cat": 1, "wild": 2})
        self.assertTrue(all(sample.path.is_relative_to(self.config.test_dir.resolve())
                            for sample in datasets["test"].samples))

    def test_dataset_can_read_manifest_path_directly(self) -> None:
        """The manifest-backed Dataset also accepts a CSV path for direct use."""
        dataset = ManifestImageDataset(self.config.dataset_root, self.config.manifest_path, "dev",
                                       self.config.class_names, build_transforms("dev", self.config))
        self.assertEqual(len(dataset), 9)
        self.assertEqual(tuple(dataset[0][0].shape), (3, 224, 224))

    def test_dataset_factory_accepts_injected_transform_factory(self) -> None:
        """Dataset composition depends on the small callable contract, not a concrete builder."""
        received: list[str] = []

        def recording_factory(mode: str, config: Config) -> ImageTransform:
            received.append(mode)
            return build_transforms(mode, config)

        datasets = build_datasets(self.config, recording_factory)
        self.assertEqual(received, ["train", "dev", "test"])
        self.assertEqual(set(datasets), {"train", "dev", "test"})

    def test_rgb_grayscale_and_rgba_all_become_three_channels(self) -> None:
        """Conversion belongs to the common Dataset, so all sources behave equally."""
        for mode, dataset in build_datasets(self.config).items():
            for i, sample in enumerate(dataset.samples):
                with self.subTest(split=mode, file=sample.path.name):
                    image, target = dataset[i]
                    self.assertEqual(tuple(image.shape), (3, 224, 224))
                    self.assertEqual(image.dtype, torch.float32)
                    self.assertIn(target, range(3))

    def test_same_dev_and_test_image_load_is_identical(self) -> None:
        """Repeated complete Dataset reads give byte-exact tensor equality."""
        datasets = build_datasets(self.config)
        for mode in ("dev", "test"):
            first, first_target = datasets[mode][0]
            second, second_target = datasets[mode][0]
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(first_target, second_target)

    def test_loaders_shuffle_only_train_and_keep_partial_batches(self) -> None:
        """Sampler types enforce ordering; partial batches must not drop images."""
        loaders = build_dataloaders(build_datasets(self.config), self.config)
        self.assertIsInstance(loaders["train"].sampler, RandomSampler)
        for mode in ("dev", "test"):
            self.assertIsInstance(loaders[mode].sampler, SequentialSampler)
        for loader in loaders.values():
            self.assertEqual(loader.batch_size, 32)
            images, labels = next(iter(loader))
            self.assertEqual(tuple(images.shape), (9, 3, 224, 224))
            self.assertEqual(labels.dtype, torch.int64)

    def test_train_sampler_order_is_seeded(self) -> None:
        """New loaders built with the same config reproduce training sample order."""
        datasets = build_datasets(self.config)
        first = build_dataloaders(datasets, self.config)
        second = build_dataloaders(datasets, self.config)
        self.assertEqual(list(first["train"].sampler), list(second["train"].sampler))

    def test_changed_val_inventory_fails(self) -> None:
        """Unmanifested test additions require an explicit audit decision."""
        patterned_image().save(self.config.test_dir / "dog" / "new.png")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            build_datasets(self.config)

    def test_missing_manifest_image_fails(self) -> None:
        """Missing sources are rejected while composing the datasets."""
        (self.config.train_dir / "dog" / "dev_RGB.png").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or out-of-root"):
            build_datasets(self.config)

    def test_path_traversal_fails(self) -> None:
        """Manifest inputs cannot escape the audited dataset root."""
        rows = list(read_manifest(self.config.manifest_path))
        rows[0] = replace(rows[0], path="../outside.png")
        save_manifest(rows, self.config.manifest_path)
        with self.assertRaisesRegex(ValueError, "violates"):
            build_datasets(self.config)

    def test_sanity_checks_and_augmented_grid(self) -> None:
        """The end-to-end data check writes a viewable preview and evidence report."""
        datasets = build_datasets(self.config)
        report = check_datasets(datasets, build_dataloaders(datasets, self.config), self.config)
        self.assertTrue(report["passed"])
        self.assertTrue(report["splits"]["dev"]["repeated_image_exactly_equal"])
        self.assertTrue(report["splits"]["test"]["repeated_image_exactly_equal"])
        self.assertEqual(report["grid"]["row_order"], ["dog", "cat", "wild"])
        self.assertTrue((self.config.phase2_output_dir / "sanity_report.json").is_file())
        with Image.open(self.config.phase2_output_dir / "augmented_training_grid.png") as grid:
            grid.verify()


if __name__ == "__main__":
    unittest.main()
