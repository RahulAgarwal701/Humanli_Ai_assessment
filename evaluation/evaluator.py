"""Read-only inference with metrics injected through their small interface."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from evaluation.metrics import Metric


@dataclass(frozen=True)
class EvaluationResult:
    """Collected labels and predictions support reporting without another model pass."""

    targets: tuple[int, ...]
    predictions: tuple[int, ...]
    metrics: dict[str, Any]


class Evaluator:
    """Evaluate once with no optimizer, tuning or hardcoded metric implementations."""

    def __init__(self, model: nn.Module, metrics: Sequence[Metric], device: torch.device) -> None:
        names = [metric.name for metric in metrics]
        if not names or len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("Metric names must be nonempty and unique")
        self.model = model
        self.metrics = tuple(metrics)
        self.device = device

    def evaluate(self, loader: Iterable[tuple[Tensor, Tensor]], class_names: Sequence[str]) -> EvaluationResult:
        """Collect predictions in deterministic loader order under eval/no_grad."""
        if not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("Class names must be nonempty and unique")
        self.model.eval()
        targets: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for images, labels in loader:
                logits = self.model(images.to(self.device))
                if tuple(logits.shape) != (len(labels), len(class_names)) or not torch.isfinite(logits).all():
                    raise ValueError("Expected finite N-by-class logits")
                targets.extend(labels.detach().cpu().tolist())
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        if not targets:
            raise ValueError("Cannot evaluate an empty loader")
        if any(not 0 <= target < len(class_names) for target in targets):
            raise ValueError("Target labels are outside the configured class indices")
        results = {metric.name: metric.compute(targets, predictions, class_names) for metric in self.metrics}
        return EvaluationResult(tuple(targets), tuple(predictions), results)
