"""Model-only parameter and mode checks, with no dataset or optimizer access."""

from dataclasses import asdict
import importlib.metadata
import logging
import platform
from typing import Any

import torch
from torch import nn

from config import Config
from data.manifest import save_json
from models.builder import ModelBuilder


LOGGER = logging.getLogger(__name__)


def select_device(requested: str) -> torch.device:
    """Select a supported GPU when available, otherwise CPU; reject unavailable explicit devices."""
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        return torch.device(next(name for name in ("cuda", "mps", "cpu") if available[name]))
    if requested not in available or not available[requested]:
        raise ValueError(f"Requested device is unavailable: {requested}")
    return torch.device(requested)


def inspect_model(config: Config, builder: ModelBuilder) -> dict[str, Any]:
    """Build through the injected interface, verify parameters/modes and forward a dummy batch.

    Both a successful check and a loading failure are persisted as explicit
    evidence. Architecture-only checks have a different filename and never claim
    that ImageNet weights were loaded.
    """
    config.phase3_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = builder.build()
    except (OSError, RuntimeError, ValueError) as exc:
        save_json({"passed": False, "pretrained_weights_loaded": False, "config": asdict(config),
                   "error_type": type(exc).__name__, "error": str(exc)},
                  config.phase3_output_dir / "model_build_error.json")
        raise
    device = select_device(config.device)
    model = model.to(device)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    LOGGER.info("Parameters: total=%s; trainable=%s; frozen=%s", f"{total:,}", f"{trainable:,}", f"{total-trainable:,}")
    LOGGER.info("Trainable parameter names: %s", names)
    if config.freeze_features and set(names) != {"fc.weight", "fc.bias"}:
        raise ValueError(f"Frozen feature-extractor contract failed: {names}")
    model.train()
    batch_norms = {name: module for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)}
    if config.freeze_features and any(module.training for module in batch_norms.values()):
        raise ValueError("Frozen BatchNorm unexpectedly entered training mode")
    before = {name: value.detach().clone() for name, value in model.named_buffers()}
    shape = (config.model_check_batch_size, len(config.normalization_mean), config.image_size, config.image_size)
    dummy = torch.randn(*shape, device=device)
    with torch.no_grad():
        output = model(dummy)
    expected = (config.model_check_batch_size, config.num_classes)
    if tuple(output.shape) != expected or not torch.isfinite(output).all():
        raise ValueError(f"Unexpected/nonfinite dummy output: {tuple(output.shape)}")
    buffers_unchanged = all(torch.equal(before[name], value) for name, value in model.named_buffers())
    if config.freeze_features and not buffers_unchanged:
        raise ValueError("Frozen running-statistic buffers changed during a train-mode forward pass")
    LOGGER.info("Train-mode BatchNorm: %d layers; eval mode=%s; buffers unchanged=%s",
                len(batch_norms), all(not module.training for module in batch_norms.values()), buffers_unchanged)
    LOGGER.info("Dummy forward on %s: %s -> %s", device, shape, tuple(output.shape))
    pretrained = config.pretrained_weights is not None
    report: dict[str, Any] = {
        "passed": True, "verification_mode": "pretrained" if pretrained else "architecture_only",
        "pretrained_weights_loaded": pretrained, "weights": config.pretrained_weights,
        "config": asdict(config), "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "torchvision")},
        "device": str(device), "total_parameters": total, "trainable_parameters": trainable,
        "class_to_idx": {label: index for index, label in enumerate(config.class_names)},
        "frozen_parameters": total-trainable, "trainable_parameter_names": names,
        "input_shape": list(shape), "output_shape": list(output.shape),
        "batchnorm_layer_count": len(batch_norms),
        "batchnorm_all_eval_after_train": all(not module.training for module in batch_norms.values()),
        "buffers_unchanged_after_train_forward": buffers_unchanged,
        "head": str(model.fc),
    }
    filename = "model_report.json" if pretrained else "architecture_check_report.json"
    save_json(report, config.phase3_output_dir / filename)
    return report
