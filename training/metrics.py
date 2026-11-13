"""Sample-weighted classification metrics, separate from optimization and logging."""

from dataclasses import asdict, dataclass
import math
from typing import Protocol

from torch import Tensor


@dataclass(frozen=True)
class ClassificationMetrics:
    """Mean cross-entropy and accuracy fraction for all observed samples."""

    loss: float
    accuracy: float
    sample_count: int


@dataclass(frozen=True)
class EpochRecord:
    """One epoch's training and development-validation measurements."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class ClassificationMeter:
    """Accumulate detached scalar metrics, weighting partial batches correctly."""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.correct = 0
        self.count = 0

    def update(self, logits: Tensor, labels: Tensor, mean_loss: float) -> None:
        """Consume one batch without retaining its autograd graph."""
        if not math.isfinite(mean_loss):
            raise ValueError("Encountered nonfinite batch loss")
        count = labels.shape[0]
        self.loss_sum += mean_loss * count
        self.correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        self.count += count

    def compute(self) -> ClassificationMetrics:
        """Return full-split averages; reject empty loaders."""
        if self.count == 0:
            raise ValueError("Cannot compute metrics for an empty loader")
        return ClassificationMetrics(self.loss_sum / self.count, self.correct / self.count, self.count)


class MetricRecorder(Protocol):
    """Minimal tracker interface needed by a training loop."""

    def record(self, epoch: int, train: ClassificationMetrics, dev: ClassificationMetrics) -> EpochRecord:
        """Record one epoch and return its reporting row."""
        ...


class MetricHistory:
    """Own epoch history independently of the training loop and persistence."""

    def __init__(self) -> None:
        self.records: list[EpochRecord] = []

    def record(self, epoch: int, train: ClassificationMetrics, dev: ClassificationMetrics) -> EpochRecord:
        """Append one immutable row and expose it to reporting/checkpoint callers."""
        row = EpochRecord(epoch, train.loss, train.accuracy, dev.loss, dev.accuracy)
        self.records.append(row)
        return row

    def as_dicts(self) -> list[dict[str, int | float]]:
        """Provide JSON-compatible metric records."""
        return [asdict(record) for record in self.records]
