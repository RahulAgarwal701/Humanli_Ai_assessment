"""Interchangeable, deterministic group-aware splitting strategies."""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from math import isclose
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from data.records import ImageRecord, SplitIndices


class DatasetSplitter(ABC):
    """Minimal interface: select train/dev indices without reading or writing files."""

    @abstractmethod
    def split(
        self, records: Sequence[ImageRecord], *, dev_ratio: float, seed: int
    ) -> SplitIndices:
        """Return a complete group-safe partition; strategies must support ratio 0.10."""
        raise NotImplementedError


class StratifiedGroupSplitter(DatasetSplitter):
    """Choose the best stratified fold, balancing class counts and target dev size.

    Supports reciprocal ratios (1/2, 1/3, ..., 1/10, ...). All candidate folds
    use only original training labels and groups, never final-test records.
    """

    def split(
        self, records: Sequence[ImageRecord], *, dev_ratio: float, seed: int
    ) -> SplitIndices:
        """Select one fold using normalized per-class dev-count error."""
        if not records or not 0 < dev_ratio < 1:
            raise ValueError("Nonempty records and dev_ratio in (0, 1) are required")
        n_splits = round(1 / dev_ratio)
        if n_splits < 2 or not isclose(1 / n_splits, dev_ratio):
            raise ValueError("This strategy requires dev_ratio = 1 / an integer >= 2")
        labels = np.asarray([record.label for record in records])
        groups = np.asarray([record.group_id for record in records])
        group_labels: dict[str, set[str]] = defaultdict(set)
        for record in records:
            group_labels[record.group_id].add(record.label)
        if any(len(values) > 1 for values in group_labels.values()):
            raise ValueError("A content group has conflicting class labels")
        unique_counts = Counter(next(iter(values)) for values in group_labels.values())
        if any(count < n_splits for count in unique_counts.values()):
            raise ValueError(f"Each class needs at least {n_splits} unique content groups")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        totals = Counter(labels)
        best: SplitIndices | None = None
        best_score = float("inf")
        for train_indices, dev_indices in splitter.split(np.zeros(len(records)), labels, groups):
            dev_counts = Counter(labels[dev_indices])
            if set(dev_counts) != set(totals) or set(labels[train_indices]) != set(totals):
                continue
            score = sum(
                abs(dev_counts[label] / count - dev_ratio) for label, count in totals.items()
            )
            if score < best_score:
                best_score = score
                best = SplitIndices(tuple(map(int, train_indices)), tuple(map(int, dev_indices)))
        if best is None:
            raise ValueError("No candidate fold contains every class in both train and dev")
        return best
