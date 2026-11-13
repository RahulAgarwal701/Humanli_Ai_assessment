"""Contract, leakage and class-balance tests on deterministic synthetic records."""

from collections import Counter
from dataclasses import replace

import pytest

from config import Config
from data.records import ImageRecord, SplitIndices
from data.split import DatasetSplitter, StratifiedGroupSplitter
from data.validation import validate_split


@pytest.fixture
def records() -> tuple[ImageRecord, ...]:
    """Unequal class counts with duplicate groups of different sizes."""
    return tuple(
        ImageRecord(f"train/{label}/{i}_{copy}.jpg", label, f"{label}-{i}")
        for label, count in (("dog", 120), ("cat", 110), ("wild", 100))
        for i in range(count)
        for copy in range(2 if i % 9 == 0 else 1)
    )


@pytest.fixture
def test_pool() -> tuple[ImageRecord, ...]:
    """Independent final-test groups."""
    return tuple(ImageRecord(f"val/{label}/0.jpg", label, f"test-{label}")
                 for label in Config().class_names)


def check_split(records: tuple[ImageRecord, ...], test_pool: tuple[ImageRecord, ...],
                indices: SplitIndices) -> dict:
    """Apply production validation with centralized tolerances."""
    config = Config()
    return validate_split(records, test_pool, indices, class_names=config.class_names,
                          dev_ratio=config.dev_ratio,
                          class_tolerance=config.class_proportion_tolerance,
                          ratio_tolerance=config.split_ratio_tolerance)


def test_no_group_leakage(records: tuple[ImageRecord, ...], test_pool: tuple[ImageRecord, ...]) -> None:
    """All duplicate copies stay on one side, with exactly-once coverage."""
    indices = StratifiedGroupSplitter().split(records, dev_ratio=0.1, seed=42)
    train_groups = {records[i].group_id for i in indices.train}
    dev_groups = {records[i].group_id for i in indices.dev}
    assert train_groups.isdisjoint(dev_groups)
    assert set(indices.train).isdisjoint(indices.dev)
    assert sorted((*indices.train, *indices.dev)) == list(range(len(records)))
    assert check_split(records, test_pool, indices)["overlap_hash_counts"] == {
        "train_dev": 0, "dev_test": 0, "train_test": 0,
    }


def test_class_proportions_within_tolerance(records: tuple[ImageRecord, ...]) -> None:
    """Check sample proportions independently, rather than mirroring validation."""
    config = Config()
    indices = StratifiedGroupSplitter().split(records, dev_ratio=config.dev_ratio, seed=config.seed)
    baseline = Counter(record.label for record in records)
    for selection in (indices.train, indices.dev):
        observed = Counter(records[i].label for i in selection)
        for label in config.class_names:
            assert abs(observed[label] / len(selection) - baseline[label] / len(records)) <= (
                config.class_proportion_tolerance
            )
    assert abs(len(indices.dev) / len(records) - config.dev_ratio) <= config.split_ratio_tolerance


def test_repeated_seed_produces_identical_split(records: tuple[ImageRecord, ...]) -> None:
    """Explicit seed and stable records fully determine assignments."""
    strategy: DatasetSplitter = StratifiedGroupSplitter()
    assert strategy.split(records, dev_ratio=0.1, seed=42) == strategy.split(records, dev_ratio=0.1, seed=42)


def test_validation_rejects_duplicate_group_leakage(test_pool: tuple[ImageRecord, ...]) -> None:
    """A broken strategy cannot bypass independent leakage verification."""
    samples = tuple(ImageRecord(f"train/{label}/{copy}.jpg", label, label)
                    for copy in range(2) for label in Config().class_names)
    with pytest.raises(ValueError, match="leakage"):
        check_split(samples, test_pool, SplitIndices((0, 1, 2), (3, 4, 5)))


@pytest.mark.parametrize("test_side", ["dev", "train"])
def test_validation_rejects_test_overlap(records: tuple[ImageRecord, ...],
                                        test_pool: tuple[ImageRecord, ...], test_side: str) -> None:
    """Neither train nor dev may share hashes with final testing."""
    indices = StratifiedGroupSplitter().split(records, dev_ratio=0.1, seed=42)
    selection = indices.dev if test_side == "dev" else indices.train
    leaked = (replace(test_pool[0], group_id=records[selection[0]].group_id), *test_pool[1:])
    with pytest.raises(ValueError, match="leakage"):
        check_split(records, leaked, indices)


def test_validation_rejects_bad_class_balance(records: tuple[ImageRecord, ...],
                                             test_pool: tuple[ImageRecord, ...]) -> None:
    """All classes can be present while their proportions are still unacceptable."""
    dev = tuple(i for i, record in enumerate(records)
                if (record.label == "dog" and int(record.group_id.split("-")[1]) < 28)
                or (record.label != "dog" and record.group_id.endswith("-0")))
    dev_set = set(dev)
    train = tuple(i for i in range(len(records)) if i not in dev_set)
    with pytest.raises(ValueError, match="proportions"):
        check_split(records, test_pool, SplitIndices(train, dev))


@pytest.mark.parametrize("bad", [SplitIndices((0, 1), (1, 2)), SplitIndices((0,), ()),
                                  SplitIndices((0, 0), (1, 2))])
def test_validation_rejects_invalid_coverage(records: tuple[ImageRecord, ...],
                                            test_pool: tuple[ImageRecord, ...], bad: SplitIndices) -> None:
    """Missing, repeated and overlapping indices fail before manifest persistence."""
    with pytest.raises(ValueError, match="exactly once"):
        check_split(records, test_pool, bad)


def test_insufficient_groups_is_actionable() -> None:
    """Too few unique images must not silently produce an invalid ten-fold split."""
    with pytest.raises(ValueError, match="at least 10 unique"):
        StratifiedGroupSplitter().split([ImageRecord("a.jpg", "dog", "same")], dev_ratio=0.1, seed=42)


def test_conflicting_group_labels_are_rejected(records: tuple[ImageRecord, ...]) -> None:
    """The interface rejects a content group associated with conflicting targets."""
    conflicting = (*records, ImageRecord("train/cat/conflict.jpg", "cat", records[0].group_id))
    with pytest.raises(ValueError, match="conflicting"):
        StratifiedGroupSplitter().split(conflicting, dev_ratio=0.1, seed=42)


def test_unsupported_ratio_is_actionable(records: tuple[ImageRecord, ...]) -> None:
    """A different ratio can use a replacement strategy behind the same interface."""
    with pytest.raises(ValueError, match="1 / an integer"):
        StratifiedGroupSplitter().split(records, dev_ratio=0.13, seed=42)
