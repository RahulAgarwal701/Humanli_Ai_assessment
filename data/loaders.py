"""DataLoader composition with explicit ordering and independent RNG generators."""

from collections.abc import Mapping
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import Config


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy using PyTorch's unique, generator-derived worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(
    datasets: Mapping[str, Dataset[tuple[Tensor, int]]], config: Config
) -> dict[str, DataLoader]:
    """Shuffle training only; retain partial batches and deterministic eval ordering."""
    if set(datasets) not in ({"train", "dev"}, {"train", "dev", "test"}):
        raise ValueError("Expected train/dev datasets with an optional final-test dataset")
    return {
        mode: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=mode == "train",
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config.seed),
        )
        for mode, dataset in datasets.items()
    }
