"""Human-readable prediction logging and an image/label/confidence contact sheet."""

from collections.abc import Sequence
import logging
import math
import os
from pathlib import Path
import warnings

from PIL import Image, ImageOps

from config import Config
from inference.predictor import Prediction


LOGGER = logging.getLogger(__name__)


def log_predictions(predictions: Sequence[Prediction]) -> None:
    """Log successes with ranked probabilities and failures without raising."""
    for result in predictions:
        if result.ok:
            LOGGER.info("%s -> %s (%.4f%%); top probabilities=%s", result.path, result.predicted_class,
                        result.confidence * 100,
                        [(item.class_name, round(item.probability, 8)) for item in result.top3])
        else:
            LOGGER.warning("%s -> ERROR: %s", result.path, result.error)


def save_prediction_grid(predictions: Sequence[Prediction], config: Config, *, source_description: str) -> Path:
    """Save input images with predicted classes/confidences; bad files get placeholders."""
    if not predictions:
        raise ValueError("No input paths were provided for the prediction grid")
    directory = config.phase6_output_dir
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(directory / "render_cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    columns = min(config.demo_columns, len(predictions))
    rows = math.ceil(len(predictions) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, result in zip(axes.flat, predictions):
        if result.ok:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(Path(result.path).expanduser()) as image:
                        axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
                axis.set_title(f"{result.predicted_class} | {result.confidence:.2%}\n{Path(result.path).name}")
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                axis.text(.5, .5, "Prediction saved; preview unavailable", ha="center", va="center", wrap=True)
        else:
            axis.text(.5, .5, f"ERROR\n{Path(result.path).name}\n{result.error}",
                      ha="center", va="center", color="firebrick", wrap=True)
    figure.suptitle(source_description)
    figure.tight_layout()
    destination = directory / "inference_demo.png"
    figure.savefig(destination, dpi=config.plot_dpi)
    plt.close(figure)
    return destination
