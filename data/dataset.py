"""Manifest-driven training/dev and folder-driven final-test image datasets."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

from config import Config
from data.manifest import read_manifest
from data.records import ManifestRow
from data.transforms import ImageTransform, build_transforms


@dataclass(frozen=True)
class ImageSample:
    """Resolved source path and the shared integer target."""

    path: Path
    target: int


class RGBImageDataset(Dataset[tuple[Tensor, int]]):
    """Load pixels on demand; own RGB conversion while accepting injected transforms."""

    def __init__(self, samples: Sequence[ImageSample], class_names: Sequence[str],
                 transform: ImageTransform) -> None:
        self.samples = tuple(samples)
        self.classes = tuple(class_names)
        self.class_to_idx = {label: index for index, label in enumerate(self.classes)}
        self.transform = transform
        if not self.samples:
            raise ValueError("Dataset must contain at least one image")

    def __len__(self) -> int:
        """Return image count without opening files."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Apply EXIF orientation and RGB conversion to grayscale/RGBA inputs too."""
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                tensor = self.transform(rgb)
        except OSError as exc:
            raise OSError(f"Cannot load dataset image {sample.path}: {exc}") from exc
        return tensor, sample.target


def _resolve_manifest_path(root: Path, row: ManifestRow, folder: str) -> Path:
    """Enforce the audited source/class layout and prevent path traversal."""
    relative = PurePosixPath(row.path)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in row.path
            or len(relative.parts) < 3 or relative.parts[:2] != (folder, row.label)):
        raise ValueError(f"Manifest path violates the expected source/class layout: {row.path}")
    resolved_root = root.resolve()
    path = resolved_root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to((resolved_root / folder).resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-root manifest image: {row.path}")
    return path


class ManifestImageDataset(RGBImageDataset):
    """Select train or dev solely from Phase 1 assignments; do not resplit."""

    def __init__(self, root: Path, records: Sequence[ManifestRow] | Path, split: str,
                 class_names: Sequence[str], transform: ImageTransform,
                 source_folder: str = "train") -> None:
        if split not in {"train", "dev"}:
            raise ValueError("ManifestImageDataset supports train/dev; final test comes from its folder")
        records = read_manifest(records) if isinstance(records, Path) else records
        class_to_idx = {label: index for index, label in enumerate(class_names)}
        samples: list[ImageSample] = []
        for row in records:
            if row.split != split:
                continue
            if row.label not in class_to_idx:
                raise ValueError(f"Unknown manifest class: {row.label}")
            samples.append(ImageSample(_resolve_manifest_path(root, row, source_folder), class_to_idx[row.label]))
        super().__init__(samples, class_names, transform)


class TestFolderDataset(RGBImageDataset):
    """Enumerate val/<class> directly and reconcile it with the reserved test rows."""

    def __init__(self, root: Path, folder: str, class_names: Sequence[str],
                 transform: ImageTransform, manifest_records: Sequence[ManifestRow]) -> None:
        source = root / folder
        if not source.is_dir():
            raise ValueError(f"Final-test source does not exist: {source}")
        visible = [entry for entry in source.iterdir() if not entry.name.startswith(".")]
        if {entry.name for entry in visible if entry.is_dir()} != set(class_names) or any(
            not entry.is_dir() for entry in visible
        ):
            raise ValueError("Final-test folder must contain exactly the configured class directories")
        expected: dict[Path, str] = {}
        for row in manifest_records:
            if row.split == "test":
                if row.label not in class_names:
                    raise ValueError(f"Unknown final-test manifest class: {row.label}")
                path = _resolve_manifest_path(root, row, folder)
                if path in expected:
                    raise ValueError("Repeated final-test manifest image")
                expected[path] = row.label
        samples: list[ImageSample] = []
        observed: dict[Path, str] = {}
        for target, label in enumerate(class_names):
            class_dir = source / label
            paths = [path for path in sorted(class_dir.rglob("*")) if path.is_file()
                     and not any(part.startswith(".") for part in path.relative_to(class_dir).parts)]
            if not paths:
                raise ValueError(f"Final-test class is empty: {label}")
            for path in paths:
                resolved = path.resolve()
                if not resolved.is_relative_to(root.resolve()) or resolved in observed:
                    raise ValueError(f"Out-of-root or repeated final-test image: {path}")
                observed[resolved] = label
                samples.append(ImageSample(resolved, target))
        if observed != expected:
            raise ValueError("Final-test folder inventory differs from Phase 1 manifest; re-audit explicitly")
        super().__init__(samples, class_names, transform)


TransformFactory = Callable[[str, Config], ImageTransform]


def build_datasets(config: Config, transform_factory: TransformFactory = build_transforms) -> dict[str, RGBImageDataset]:
    """Compose datasets from one manifest and an interchangeable transform factory."""
    records = read_manifest(config.manifest_path)
    datasets: dict[str, RGBImageDataset] = {
        split: ManifestImageDataset(config.dataset_root, records, split, config.class_names,
                                    transform_factory(split, config), config.train_folder)
        for split in ("train", "dev")
    }
    datasets["test"] = TestFolderDataset(config.dataset_root, config.test_folder, config.class_names,
                                        transform_factory("test", config), records)
    return datasets


def build_training_datasets(
    config: Config, transform_factory: TransformFactory = build_transforms
) -> dict[str, RGBImageDataset]:
    """Load train/dev assignments only; never resolve or inspect a final-test path."""
    records = read_manifest(config.manifest_path)
    return {
        split: ManifestImageDataset(config.dataset_root, records, split, config.class_names,
                                    transform_factory(split, config), config.train_folder)
        for split in ("train", "dev")
    }
