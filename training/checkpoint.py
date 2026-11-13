"""Checkpoint persistence strategies, independent of the training loop."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import Config


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointInfo:
    """The best saved epoch and its selection criterion."""

    path: Path
    epoch: int
    val_loss: float


class CheckpointStrategy(ABC):
    """Small persistence contract supporting alternative selection strategies."""

    @abstractmethod
    def save_if_best(self, model: nn.Module, *, epoch: int, val_loss: float, config: Config) -> CheckpointInfo | None:
        """Return saved checkpoint details, or None when this epoch is not selected."""
        raise NotImplementedError


def _serializable(value: Any) -> Any:
    """Represent config with safe primitive values for torch.load(weights_only=True)."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    return value


class BestLossCheckpointer(CheckpointStrategy):
    """Save only strict dev-loss improvements; ties retain the earliest best epoch."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.best_val_loss = float("inf")

    def save_if_best(self, model: nn.Module, *, epoch: int, val_loss: float, config: Config) -> CheckpointInfo | None:
        """Atomically replace the checkpoint with a detached CPU state snapshot."""
        if not math.isfinite(val_loss) or epoch < 1:
            raise ValueError("Checkpoint selection requires finite loss and a positive epoch")
        if val_loss >= self.best_val_loss:
            return None
        payload = {
            "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "epoch": epoch, "val_loss": val_loss, "class_names": list(config.class_names),
            "config": _serializable(asdict(config)),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(self.path)
        self.best_val_loss = val_loss
        LOGGER.info("Saved best checkpoint: epoch=%d dev_loss=%.6f path=%s", epoch, val_loss, self.path)
        return CheckpointInfo(self.path, epoch, val_loss)
