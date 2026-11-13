"""Printed classification metrics and labeled confusion-matrix figures."""

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report

from config import Config
from evaluation.evaluator import EvaluationResult


LOGGER = logging.getLogger(__name__)


def log_results(result: EvaluationResult, class_names: Sequence[str], *, digits: int) -> None:
    """Print the named scikit-learn report and a count matrix with explicit axes."""
    LOGGER.info("Overall test accuracy: %.4f%% (%d/%d)", result.metrics["accuracy"] * 100,
                sum(t == p for t, p in zip(result.targets, result.predictions)), len(result.targets))
    LOGGER.info("Classification report:\n%s", classification_report(result.targets, result.predictions,
                labels=list(range(len(class_names))), target_names=list(class_names), digits=digits, zero_division=0))
    LOGGER.info("Confusion matrix: rows=true, columns=predicted; class order=%s\n%s", list(class_names),
                np.asarray(result.metrics["confusion_matrix"]))


def save_confusion_matrix(matrix: Sequence[Sequence[int]], class_names: Sequence[str], config: Config) -> Path:
    """Save integer counts with stable class order and labeled true/predicted axes."""
    directory = config.phase5_output_dir
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(directory / "render_cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axis = plt.subplots()
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=list(class_names)).plot(
        ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set(title="Final test confusion matrix", xlabel="Predicted class", ylabel="True class")
    figure.tight_layout()
    path = directory / "confusion_matrix.png"
    figure.savefig(path, dpi=config.plot_dpi)
    plt.close(figure)
    return path
