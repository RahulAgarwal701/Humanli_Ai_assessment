"""Logging, history persistence and learning curves, separate from the loop."""

from abc import ABC, abstractmethod
from dataclasses import asdict
import csv
import logging
import os
from pathlib import Path
from typing import Sequence

from training.metrics import EpochRecord
from config import Config


LOGGER = logging.getLogger(__name__)


class EpochReporter(ABC):
    """Minimal reporting interface; consumers need not know any logging backend."""

    @abstractmethod
    def report(self, record: EpochRecord) -> None:
        """Publish one completed epoch's metrics."""
        raise NotImplementedError


class LoggingEpochReporter(EpochReporter):
    """Report train/dev loss and accuracy using standard logging."""

    def report(self, record: EpochRecord) -> None:
        LOGGER.info("Epoch %d | train loss=%.6f accuracy=%.2f%% | dev loss=%.6f accuracy=%.2f%%",
                    record.epoch, record.train_loss, record.train_accuracy * 100,
                    record.val_loss, record.val_accuracy * 100)


def save_history(records: Sequence[EpochRecord], destination: Path) -> None:
    """Save exact per-epoch measurements in a portable CSV."""
    if not records:
        raise ValueError("Cannot save an empty training history")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def save_curves(records: Sequence[EpochRecord], directory: Path, *, dpi: int) -> tuple[Path, Path]:
    """Save loss and accuracy curves from actual history using a headless backend."""
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(directory / "render_cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if not records:
        raise ValueError("Cannot plot an empty training history")
    directory.mkdir(parents=True, exist_ok=True)
    epochs = [record.epoch for record in records]
    paths: list[Path] = []
    for metric in ("loss", "accuracy"):
        figure, axis = plt.subplots()
        axis.plot(epochs, [getattr(record, f"train_{metric}") for record in records], marker="o", label="Train")
        axis.plot(epochs, [getattr(record, f"val_{metric}") for record in records], marker="o", label="Dev-validation")
        axis.set(xlabel="Epoch", ylabel="Cross-entropy loss" if metric == "loss" else "Accuracy (fraction)",
                 title=f"Training and dev-validation {metric}", xticks=epochs)
        if metric == "accuracy":
            axis.set_ylim(0, 1)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = directory / f"{metric}_curve.png"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        paths.append(path)
    return paths[0], paths[1]


def assess_learning_curves(records: Sequence[EpochRecord], config: Config) -> str:
    """Give a qualified, short-run interpretation using explicit configured thresholds."""
    if len(records) < 2:
        return "Inconclusive: at least two epochs are required to assess a trend."
    first, last = records[0], records[-1]
    if max(last.train_accuracy, last.val_accuracy) < config.underfit_accuracy_threshold:
        return ("Possible underfitting: both final accuracies remain below the configured "
                f"{config.underfit_accuracy_threshold:.0%} threshold. More evidence is needed.")
    train_improves = last.train_loss < first.train_loss - config.curve_loss_tolerance
    if train_improves and last.val_loss > min(row.val_loss for row in records) + config.curve_loss_tolerance:
        return ("Possible overfitting: training loss improved, while final dev loss rose above its "
                "best value. Use the saved minimum-dev-loss checkpoint.")
    if train_improves and last.val_loss < first.val_loss - config.curve_loss_tolerance:
        return "Healthy learning trend: both train and dev loss decreased; five epochs cannot establish full convergence."
    return "Inconclusive or plateauing curves: no clear joint loss improvement; inspect the saved curves before drawing conclusions."
