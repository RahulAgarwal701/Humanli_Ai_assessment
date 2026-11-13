"""Small metric contract and scikit-learn implementations independent of torch."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Metric(ABC):
    """Named, injectable metric requiring only integer predictions and class names."""

    name: str

    @abstractmethod
    def compute(self, targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]) -> Any:
        """Return a JSON-compatible result without mutating inputs or model state."""
        raise NotImplementedError


class AccuracyMetric(Metric):
    """Overall sample accuracy, stored as a fraction."""

    name = "accuracy"

    def compute(self, targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]) -> float:
        return float(accuracy_score(targets, predictions))


class ClassificationReportMetric(Metric):
    """Per-class precision, recall, F1 and support, plus macro/weighted averages."""

    name = "classification_report"

    def compute(self, targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]) -> dict[str, Any]:
        return classification_report(targets, predictions, labels=list(range(len(class_names))),
                                     target_names=list(class_names), output_dict=True, zero_division=0)


class ConfusionMatrixMetric(Metric):
    """Raw counts with rows=true class and columns=predicted class."""

    name = "confusion_matrix"

    def compute(self, targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]) -> list[list[int]]:
        return confusion_matrix(targets, predictions, labels=list(range(len(class_names)))).tolist()
