"""Seed available random generators without requiring the later training stack."""

import importlib
import logging
import random

import numpy as np


LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (when installed), including CUDA devices.

    Phase 1 needs no torch operations. This optional import lets the data-only
    requirements stay small while honoring the same seed in an ML environment.
    Deterministic training-kernel configuration belongs to the training phase.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        LOGGER.info("Seeded random and NumPy with %d; PyTorch is not installed or needed in Phase 1", seed)
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    LOGGER.info("Seeded random, NumPy and PyTorch with %d", seed)
