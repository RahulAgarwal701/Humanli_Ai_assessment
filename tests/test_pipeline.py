"""End-to-end preparation and optional-generator reproducibility checks."""

import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import pytest

from config import Config
from data.records import ImageRecord, SplitIndices
from data.split import DatasetSplitter, StratifiedGroupSplitter
from main import run_phase_one
import reproducibility


class RecordingSplitter(DatasetSplitter):
    """Test double proves only original training records reach the injected strategy."""

    def __init__(self) -> None:
        self.received: tuple[ImageRecord, ...] = ()

    def split(self, records: Sequence[ImageRecord], *, dev_ratio: float, seed: int) -> SplitIndices:
        """Record the interface boundary, then use the production algorithm."""
        self.received = tuple(records)
        return StratifiedGroupSplitter().split(records, dev_ratio=dev_ratio, seed=seed)


def test_pipeline_saves_reproducible_manifest_and_injects_strategy(tmp_path: Path) -> None:
    """Real hashing, decoding, strategy injection, validation and CSV writing agree."""
    root = tmp_path / "dataset"
    config = Config(dataset_root=root, output_dir=tmp_path / "run1")
    for split in ("train", "val"):
        for class_index, label in enumerate(config.class_names):
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(20 if split == "train" else 2):
                Image.new("RGB", (8, 8), (class_index * 70, i * 10, 80 if split == "train" else 160)).save(
                    folder / f"{i:02d}.png"
                )
    strategy = RecordingSplitter()
    first_report = run_phase_one(config, strategy)
    second_config = replace(config, output_dir=tmp_path / "run2")
    second_report = run_phase_one(second_config, RecordingSplitter())
    assert all(record.path.startswith("train/") for record in strategy.received)
    assert len(strategy.received) == 60
    assert first_report == second_report
    assert (config.output_dir / "split_manifest.csv").read_bytes() == (
        second_config.output_dir / "split_manifest.csv"
    ).read_bytes()
    with (config.output_dir / "split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 66
    assert all((root / row["path"]).is_file() for row in rows)
    assert sum(row["split"] == "test" for row in rows) == 6
    metadata = json.loads((config.output_dir / "run_metadata.json").read_text())
    assert metadata["config"]["seed"] == config.seed
    assert metadata["strategy"] == "RecordingSplitter"
    (root / "train/dog/broken.png").write_bytes(b"invalid")
    failed_config = replace(config, output_dir=tmp_path / "failed")
    with pytest.raises(ValueError, match="Audit failed"):
        run_phase_one(failed_config, strategy)
    assert (failed_config.output_dir / "audit_report.json").exists()
    assert not (failed_config.output_dir / "split_manifest.csv").exists()


def test_seeds_python_numpy_and_optional_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional torch is seeded while data-only environments remain supported."""
    calls: list[int] = []

    class FakeCuda:
        """Minimal CUDA seed interface."""

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            calls.append(seed)

    class FakeTorch:
        """Minimal torch seed interface."""

        cuda = FakeCuda()

        @staticmethod
        def manual_seed(seed: int) -> None:
            calls.append(seed)

    monkeypatch.setattr(reproducibility.importlib, "import_module", lambda name: FakeTorch())
    reproducibility.seed_everything(42)
    first = (reproducibility.random.random(), np.random.random())
    reproducibility.seed_everything(42)
    assert first == (reproducibility.random.random(), np.random.random())
    assert calls == [42, 42, 42, 42]


def test_missing_torch_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """No PyTorch installation is required to perform a data-only phase."""
    def missing_torch(name: str) -> None:
        raise ModuleNotFoundError("No module named torch", name="torch")

    monkeypatch.setattr(reproducibility.importlib, "import_module", missing_torch)
    reproducibility.seed_everything(42)
