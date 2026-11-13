"""Data-only tensor diagnostics and a denormalized training augmentation preview."""

from collections.abc import Mapping
from dataclasses import asdict
import importlib.metadata
import logging
import platform
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import Config
from data.dataset import RGBImageDataset
from data.manifest import save_json


LOGGER = logging.getLogger(__name__)


def check_datasets(
    datasets: Mapping[str, RGBImageDataset], loaders: Mapping[str, DataLoader], config: Config
) -> dict[str, Any]:
    """Check each loader, exact eval equality and save only training-image previews."""
    report: dict[str, Any] = {
        "config": asdict(config),
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "Pillow")},
        "class_to_idx": dict(datasets["train"].class_to_idx),
        "splits": {},
    }
    for mode, dataset in datasets.items():
        if dataset.class_to_idx != datasets["train"].class_to_idx:
            raise ValueError("Dataset class-index mappings differ")
        images, labels = next(iter(loaders[mode]))
        if images.shape[1:] != (3, config.image_size, config.image_size) or images.dtype != torch.float32:
            raise ValueError(f"Unexpected tensor shape or dtype in {mode}")
        if not torch.isfinite(images).all():
            raise ValueError(f"Nonfinite tensors in {mode}")
        mean_values = torch.tensor(config.normalization_mean, dtype=torch.float32)
        std_values = torch.tensor(config.normalization_std, dtype=torch.float32)
        lower = (-mean_values / std_values).min().item()
        upper = ((1 - mean_values) / std_values).max().item()
        if images.min().item() < lower or images.max().item() > upper:
            raise ValueError(f"Values outside ImageNet-normalized pixel bounds in {mode}")
        summary = {
            "count": len(dataset), "batch_shape": list(images.shape), "dtype": str(images.dtype),
            "min": images.min().item(), "max": images.max().item(),
            "label_dtype": str(labels.dtype), "sampler": type(loaders[mode].sampler).__name__,
        }
        if mode in {"dev", "test"}:
            first, first_label = dataset[0]
            second, second_label = dataset[0]
            equal = torch.equal(first, second) and first_label == second_label
            summary["repeated_image_exactly_equal"] = equal
            if not equal:
                raise ValueError(f"{mode} preprocessing is not deterministic")
        report["splits"][mode] = summary
        LOGGER.info("%s: count=%d; shape=%s; dtype=%s; range=[%.6f, %.6f]; sampler=%s", mode,
                    len(dataset), tuple(images.shape), images.dtype, summary["min"], summary["max"],
                    summary["sampler"])
        if mode in {"dev", "test"}:
            LOGGER.info("%s: repeated image tensors exactly equal = %s", mode, equal)

    training = datasets["train"]
    previews: list[torch.Tensor] = []
    sources: list[dict[str, str | int]] = []
    for target, label in enumerate(config.class_names):
        index = next(i for i, sample in enumerate(training.samples) if sample.target == target)
        for _ in range(config.grid_samples_per_class):
            image, _ = training[index]
            previews.append(image)
        sources.append({"label": label, "path": str(training.samples[index].path),
                        "augmented_views": config.grid_samples_per_class})
    mean = torch.tensor(config.normalization_mean).view(1, 3, 1, 1)
    std = torch.tensor(config.normalization_std).view(1, 3, 1, 1)
    visible = (torch.stack(previews) * std + mean).clamp(0, 1)
    config.phase2_output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = config.phase2_output_dir / "augmented_training_grid.png"
    save_image(visible, grid_path, nrow=config.grid_samples_per_class)
    report["grid"] = {"path": str(grid_path), "row_order": list(config.class_names), "sources": sources}
    report["passed"] = True
    save_json(report, config.phase2_output_dir / "sanity_report.json")
    LOGGER.info("Saved training augmentation grid: %s", grid_path)
    return report
