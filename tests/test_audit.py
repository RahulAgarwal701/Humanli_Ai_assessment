"""Real files exercise decode failures, byte hashing and audit invariants."""

import csv
from pathlib import Path
import shutil

import pytest
from PIL import Image

from config import Config
from data.audit import audit_dataset, file_sha1
from data.manifest import save_manifest
from data.records import ManifestRow


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """One distinct RGB image per class and source, plus one exact copy."""
    for source_idx, source in enumerate(("train", "val")):
        for label_idx, label in enumerate(Config().class_names):
            folder = tmp_path / source / label
            folder.mkdir(parents=True)
            Image.new("RGB", (12, 10), (30 + label_idx * 50, 60 + source_idx * 80, 100)).save(folder / "a.png")
    shutil.copyfile(tmp_path / "train/dog/a.png", tmp_path / "train/dog/copy.png")
    return tmp_path


def test_audit_reads_real_images_and_exact_groups(dataset: Path) -> None:
    """Audit reports decoded format/size, raw counts and the duplicate group."""
    result = audit_dataset(dataset, "train", "val", Config().class_names)
    assert result.report["passed"]
    assert result.report["sources"]["train"]["class_counts"] == {"dog": 2, "cat": 1, "wild": 1}
    assert result.report["sources"]["train"]["formats"] == {"PNG": 4}
    assert result.report["sources"]["train"]["sizes"] == {"12x10": 4}
    assert len(result.report["duplicate_groups"]) == 1
    assert result.report["extra_duplicate_copies"] == 1
    assert result.train[1].group_id == result.train[0].group_id


def test_corrupt_file_fails_audit(dataset: Path) -> None:
    """Invalid files are reported, never silently admitted to splitting."""
    (dataset / "train/cat/broken.jpg").write_bytes(b"not an image")
    result = audit_dataset(dataset, "train", "val", Config().class_names)
    assert not result.report["passed"]
    assert result.report["errors"][0]["path"] == "train/cat/broken.jpg"
    assert len(result.train) == 4


def test_truncated_jpeg_fails_full_decode(dataset: Path) -> None:
    """Recognizable image headers are insufficient when pixel data is truncated."""
    path = dataset / "train/cat/truncated.jpg"
    Image.new("RGB", (100, 100), "blue").save(path)
    path.write_bytes(path.read_bytes()[:-12])
    result = audit_dataset(dataset, "train", "val", Config().class_names)
    assert not result.report["passed"]
    assert any(error["path"].endswith("truncated.jpg") for error in result.report["errors"])


def test_test_overlap_is_detected(dataset: Path) -> None:
    """Hash comparisons detect copies even when filenames differ."""
    shutil.copyfile(dataset / "train/cat/a.png", dataset / "val/cat/renamed.png")
    result = audit_dataset(dataset, "train", "val", Config().class_names)
    assert not result.report["passed"]
    assert result.report["train_test_overlap_hashes"] == [file_sha1(dataset / "train/cat/a.png")]


def test_conflicting_labels_are_reported(dataset: Path) -> None:
    """An exact duplicate assigned to two classes requires an explicit data fix."""
    shutil.copyfile(dataset / "train/dog/a.png", dataset / "train/cat/conflict.png")
    result = audit_dataset(dataset, "train", "val", Config().class_names)
    assert not result.report["passed"]
    assert len(result.report["conflicting_label_groups"]) == 1


def test_hidden_metadata_is_listed_and_ignored(dataset: Path) -> None:
    """Common platform metadata is not misreported as a corrupt training image."""
    (dataset / "train/dog/.DS_Store").write_bytes(b"metadata")
    result = audit_dataset(dataset, "train", "val", Config().class_names)
    assert result.report["passed"]
    assert result.report["ignored_hidden_entries"] == ["train/dog/.DS_Store"]


def test_unexpected_class_is_reported(dataset: Path) -> None:
    """New classes cannot silently change class indices in later phases."""
    (dataset / "train/extra_class").mkdir()
    result = audit_dataset(dataset, "train", "val", Config().class_names)
    assert not result.report["passed"]
    assert result.report["errors"][0]["path"] == "train/extra_class"


def test_manifest_schema_and_byte_stability(tmp_path: Path) -> None:
    """Saved CSV is portable, deterministic and has precisely the requested schema."""
    rows = [ManifestRow("train/dog/a.jpg", "dog", "hash", "dev")]
    destination = tmp_path / "splits.csv"
    save_manifest(rows, destination)
    first = destination.read_bytes()
    save_manifest(rows, destination)
    assert destination.read_bytes() == first
    with destination.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["path", "label", "group_id", "split"]
        assert list(reader) == [{"path": "train/dog/a.jpg", "label": "dog", "group_id": "hash", "split": "dev"}]
