"""One-image inference with checkpoint-owned labels and per-file failures."""

from collections.abc import Iterable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Protocol
import warnings

from PIL import Image, ImageOps
import torch

from config import Config
from data.dataset import TransformFactory
from data.transforms import build_transforms
from evaluation.checkpoint import LoadedCheckpoint, load_best_checkpoint
from models.sanity import select_device


LOGGER = logging.getLogger(__name__)


class CheckpointLoader(Protocol):
    """Minimal loading dependency needed by a Predictor."""

    def __call__(self, config: Config) -> LoadedCheckpoint:
        """Restore a selected classifier and its saved input/label contract."""
        ...


def load_prediction_checkpoint(config: Config) -> LoadedCheckpoint:
    """Read class order from the checkpoint instead of assuming current defaults."""
    return load_best_checkpoint(config, match_config_classes=False)


@dataclass(frozen=True)
class ClassProbability:
    """One class's softmax probability, stored as a fraction."""

    class_name: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    """Exactly one success or failure record for an input path."""

    path: str
    predicted_class: str | None = None
    confidence: float | None = None
    top3: tuple[ClassProbability, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this image was successfully classified."""
        return self.error is None


class Predictor:
    """Load the best state once; convert every image to RGB and isolate file errors.

    Confidence is the winning softmax probability, not a calibrated guarantee.
    A checkpoint/configuration loading error fails construction explicitly;
    recoverable errors for individual image paths become Prediction records.
    """

    def __init__(self, config: Config | None = None, *, checkpoint_loader: CheckpointLoader = load_prediction_checkpoint,
                 transform_factory: TransformFactory = build_transforms) -> None:
        requested = config if config is not None else Config()
        loaded = checkpoint_loader(requested)
        self.config = loaded.config
        self.class_names = loaded.config.class_names
        self.provenance = loaded.provenance
        self.device = select_device(self.config.device)
        self.model = loaded.model.to(self.device).eval()
        self.transform = transform_factory("test", self.config)

    def predict(self, path: str | Path) -> Prediction:
        """Return class, confidence and ranked probabilities, or a per-file error."""
        display_path = str(path)
        try:
            source = Path(path).expanduser()
            if not source.exists():
                raise FileNotFoundError(f"Image file does not exist: {source}")
            if not source.is_file():
                raise IsADirectoryError(f"Image path is not a file: {source}")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(source) as image:
                    image.load()
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(tensor)
                if tuple(logits.shape) != (1, len(self.class_names)) or not torch.isfinite(logits).all():
                    raise ValueError("Model returned invalid class logits")
                probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
            # Resolve exact probability ties by saved class order on every device.
            ranking = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
            top = tuple(ClassProbability(self.class_names[i], float(probabilities[i]))
                        for i in ranking[:self.config.prediction_top_k])
            winner = ranking[0]
            return Prediction(display_path, self.class_names[winner], float(probabilities[winner]), top)
        except Exception as exc:
            # Per-image isolation is intentional: decoder/plugin failures must
            # not abort later files. KeyboardInterrupt/SystemExit still propagate.
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Skipping image %s: %s", display_path, error)
            return Prediction(display_path, error=error)

    def predict_many(self, paths: Iterable[str | Path]) -> list[Prediction]:
        """Keep input order and one result per path; one bad file never aborts the batch."""
        return [self.predict(path) for path in paths]
