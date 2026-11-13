"""Load the loss-selected checkpoint and verify selection before test inference."""

import csv
from dataclasses import dataclass, replace
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import Config
from models.builder import build_model


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Restored model, saved input contract and independent selection evidence."""

    model: nn.Module
    config: Config
    provenance: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Hash checkpoint bytes for immutable-evaluation provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_best_checkpoint(config: Config, *, match_config_classes: bool = True) -> LoadedCheckpoint:
    """Verify the checkpoint matches the earliest minimum dev-loss epoch, then restore it.

    The full stored state replaces random initialization, so no ImageNet download
    is needed during evaluation. Current dataset/output paths remain portable;
    model architecture and deterministic preprocessing come from the checkpoint.
    """
    path = config.phase4_output_dir / config.checkpoint_filename
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "epoch", "val_loss", "class_names", "config"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Checkpoint is missing required model/selection metadata")
    names = tuple(payload["class_names"])
    if names != tuple(payload["config"]["class_names"]) or (match_config_classes and names != config.class_names):
        raise ValueError("Checkpoint class names/order differ from the configured dataset")
    with (config.phase4_output_dir / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if not history or any(not math.isfinite(float(row["val_loss"])) for row in history):
        raise ValueError("A nonempty finite dev-validation history is required")
    best = min(history, key=lambda row: (float(row["val_loss"]), int(row["epoch"])))
    if payload["epoch"] != int(best["epoch"]) or not math.isclose(payload["val_loss"], float(best["val_loss"])):
        raise ValueError("Checkpoint does not match the lowest dev-validation loss in history")
    status = json.loads((config.phase4_output_dir / "training_status.json").read_text())
    if (status.get("status") != "completed" or status.get("test_dataset_constructed") is not False
            or status.get("best_epoch") != payload["epoch"]
            or not math.isclose(status.get("best_val_loss", float("inf")), payload["val_loss"])):
        raise ValueError("Training status does not confirm completed dev-only checkpoint selection")
    saved = payload["config"]
    restored = replace(config, architecture=saved["architecture"], num_classes=saved["num_classes"],
                       class_names=names, image_size=saved["image_size"],
                       normalization_mean=tuple(saved["normalization_mean"]),
                       normalization_std=tuple(saved["normalization_std"]),
                       interpolation=saved["interpolation"], freeze_features=saved["freeze_features"],
                       pretrained_weights=None)
    model = build_model(restored)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    LOGGER.info("Restored complete best-checkpoint state: epoch=%d dev_loss=%.8f", payload["epoch"], payload["val_loss"])
    return LoadedCheckpoint(model, restored, {
        "path": str(path), "sha256": sha256_file(path), "epoch": payload["epoch"],
        "dev_val_loss": payload["val_loss"], "dev_val_accuracy": float(best["val_accuracy"]),
        "selection_criterion": "minimum dev-validation loss; earliest epoch on ties",
        "selection_verified_against_history": True, "test_used_for_checkpoint_selection": False,
        "original_pretrained_weights": saved["pretrained_weights"],
    })
