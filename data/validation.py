"""Independent leakage, coverage and class-balance checks for split strategies."""

from collections import Counter
from typing import Any, Sequence

from data.records import ImageRecord, SplitIndices


def validate_split(
    train_pool: Sequence[ImageRecord],
    test_pool: Sequence[ImageRecord],
    indices: SplitIndices,
    *,
    class_names: Sequence[str],
    dev_ratio: float,
    class_tolerance: float,
    ratio_tolerance: float,
) -> dict[str, Any]:
    """Fail closed on leakage, incomplete coverage or unacceptable imbalance."""
    train_set, dev_set = set(indices.train), set(indices.dev)
    expected = set(range(len(train_pool)))
    if (len(train_set) != len(indices.train) or len(dev_set) != len(indices.dev)
            or train_set & dev_set or train_set | dev_set != expected):
        raise ValueError("Indices must cover the training pool exactly once")
    pools = {
        "train": [train_pool[i] for i in indices.train],
        "dev": [train_pool[i] for i in indices.dev],
        "test": list(test_pool),
    }
    if any(not records for records in pools.values()):
        raise ValueError("Every split must be nonempty")
    if any(record.label not in class_names for records in pools.values() for record in records):
        raise ValueError("Encountered an unconfigured class label")
    paths = [record.path for records in pools.values() for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("A path occurs more than once across the splits")
    group_sets = {name: {record.group_id for record in records} for name, records in pools.items()}
    overlap = {
        "train_dev": len(group_sets["train"] & group_sets["dev"]),
        "dev_test": len(group_sets["dev"] & group_sets["test"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"Content-hash group leakage detected: {overlap}")
    baseline = Counter(record.label for record in train_pool)
    proportions = {label: baseline[label] / len(train_pool) for label in class_names}
    summaries: dict[str, Any] = {}
    for name, records in pools.items():
        counts = Counter(record.label for record in records)
        if any(counts[label] == 0 for label in class_names):
            raise ValueError(f"{name} does not contain every configured class")
        observed = {label: counts[label] / len(records) for label in class_names}
        deviations = {label: abs(observed[label] - proportions[label]) for label in class_names}
        if name != "test" and any(delta > class_tolerance for delta in deviations.values()):
            raise ValueError(f"{name} class proportions exceed tolerance: {deviations}")
        summaries[name] = {
            "count": len(records),
            "unique_groups": len(group_sets[name]),
            "class_counts": {label: counts[label] for label in class_names},
            "class_proportions": observed,
            "absolute_deviation_from_training_pool": deviations,
        }
    observed_ratio = len(pools["dev"]) / len(train_pool)
    if abs(observed_ratio - dev_ratio) > ratio_tolerance:
        raise ValueError(f"Dev fraction {observed_ratio:.6f} exceeds target-ratio tolerance")
    return {
        "passed": True,
        "coverage_exactly_once": True,
        "overlap_hash_counts": overlap,
        "baseline_class_proportions": proportions,
        "requested_dev_ratio": dev_ratio,
        "actual_dev_ratio": observed_ratio,
        "class_proportion_tolerance": class_tolerance,
        "split_ratio_tolerance": ratio_tolerance,
        "splits": summaries,
    }
