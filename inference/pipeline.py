"""Select unseen/demo inputs, run resilient predictions and save reusable outputs."""

from collections.abc import Sequence
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any

from config import Config
from data.audit import file_sha1
from data.manifest import read_manifest, save_json
from evaluation.checkpoint import sha256_file
from inference.predictor import Prediction, Predictor
from inference.reporting import log_predictions, save_prediction_grid


LOGGER = logging.getLogger(__name__)


def _folder_files(folder: Path) -> list[Path]:
    """Include visible files regardless of extension, so non-images receive error records."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {folder}")
    return [path for path in sorted(folder.rglob("*")) if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(folder).parts)]


def select_demo_inputs(config: Config, *, paths: Sequence[Path] | None = None,
                       input_dir: Path | None = None) -> tuple[list[Path], str]:
    """Prefer supplied images/new_images; otherwise select test rows, never train/dev."""
    if paths is not None and input_dir is not None:
        raise ValueError("Provide image paths or an input directory, not both")
    if paths is not None:
        return list(paths), "User-provided images"
    if input_dir is not None:
        return _folder_files(input_dir), "User-provided image directory"
    if config.new_images_dir.is_dir():
        return _folder_files(config.new_images_dir), "Images from new_images"
    records = read_manifest(config.manifest_path)
    candidates = {label: [config.dataset_root / row.path for row in records if row.split == "test" and row.label == label]
                  for label in config.class_names}
    selected = [candidates[label][index] for index in range(config.demo_images_per_class)
                for label in config.class_names if index < len(candidates[label])]
    return selected, "Reused final-test images (not new external validation)"


def run_prediction_demo(config: Config, *, paths: Sequence[Path] | None = None,
                        input_dir: Path | None = None) -> dict[str, Any]:
    """Predict once with the selected best state; never tune using demonstration inputs."""
    predictor = Predictor(config)
    effective = predictor.config
    selected, source = select_demo_inputs(effective, paths=paths, input_dir=input_dir)
    if not selected:
        raise ValueError("No input files found; supply --images or a nonempty --input-dir")
    # Refuse exact copies of training/dev files in the demo, even if renamed.
    records = read_manifest(effective.manifest_path)
    training_hashes = {row.group_id for row in records if row.split in {"train", "dev"}}
    training_paths = {row.path for row in records if row.split in {"train", "dev"}}
    predictions: list[Prediction] = []
    for path in selected:
        try:
            resolved = path.resolve()
            root = effective.dataset_root.resolve()
            original_path = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else None
            excluded = original_path in training_paths or (path.is_file() and file_sha1(path) in training_hashes)
        except OSError:
            excluded = False  # Let Predictor report its own per-file access error.
        if excluded:
            predictions.append(Prediction(str(path), error="Demo excludes exact train/dev image copies"))
        else:
            predictions.append(predictor.predict(path))
    log_predictions(predictions)
    image = save_prediction_grid(predictions, effective, source_description=source)
    checkpoint_path = effective.phase4_output_dir / effective.checkpoint_filename
    if sha256_file(checkpoint_path) != predictor.provenance["sha256"]:
        raise ValueError("Checkpoint changed during prediction")
    report: dict[str, Any] = {
        "source": source, "class_names": list(predictor.class_names), "device": str(predictor.device),
        "checkpoint": predictor.provenance, "checkpoint_unchanged": True,
        "success_count": sum(result.ok for result in predictions),
        "error_count": sum(not result.ok for result in predictions),
        "predictions": [asdict(result) for result in predictions], "visual": str(image),
        "confidence_note": "Softmax confidence is not calibrated; reused test samples are not a new independent evaluation.",
    }
    save_json(report, effective.phase6_output_dir / "predictions.json")
    LOGGER.info("Saved inference demo: %s; successful=%d failed=%d", image, report["success_count"], report["error_count"])
    return report
