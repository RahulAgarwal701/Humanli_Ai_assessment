"""Compose Phase 4 using train/dev data only, with explicit run status."""

from dataclasses import asdict
import importlib.metadata
import logging
import os
import platform
from typing import Any

import torch
from torch import nn

from config import Config
from data.dataset import build_training_datasets
from data.loaders import build_dataloaders
from data.manifest import save_json
from models.builder import build_model
from models.sanity import select_device
from reproducibility import seed_everything
from training.checkpoint import BestLossCheckpointer
from training.metrics import MetricHistory
from training.reporting import LoggingEpochReporter, assess_learning_curves, save_curves, save_history
from training.trainer import Trainer


LOGGER = logging.getLogger(__name__)


def run_training(config: Config) -> dict[str, Any]:
    """Run a real pretrained training job; do not substitute random frozen features.

    No test Dataset is constructed, no test path is resolved and no test pixels
    are opened. Only train/dev assignment metadata is used from the manifest.
    """
    config.phase4_output_dir.mkdir(parents=True, exist_ok=True)
    status_path = config.phase4_output_dir / "training_status.json"
    pretrained_loaded = False
    history = MetricHistory()
    save_json({"status": "starting", "config": asdict(config), "best_epoch": None,
               "best_val_loss": None}, status_path)
    try:
        if config.pretrained_weights is None:
            raise ValueError("Phase 4 requires pretrained weights; --weights none is only for Phase 3 architecture checks")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(config.deterministic_training)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = config.deterministic_training
        seed_everything(config.seed)
        device = select_device(config.device)
        # Acquire real ImageNet weights before opening even a training image.
        model = build_model(config).to(device)
        pretrained_loaded = True
        datasets = build_training_datasets(config)
        loaders = build_dataloaders(datasets, config)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.lr,
                                     betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = BestLossCheckpointer(config.phase4_output_dir / config.checkpoint_filename)
        trainer = Trainer(model, optimizer, criterion, loaders, strategy, config, device,
                          history, LoggingEpochReporter())
        LOGGER.info("Starting %d epochs on %s: train=%d dev=%d batch_size=%d Adam lr=%g trainable=%d",
                    config.epochs, device, len(datasets["train"]), len(datasets["dev"]), config.batch_size,
                    config.lr, sum(p.numel() for p in model.parameters() if p.requires_grad))
        best = trainer.fit()
        save_history(history.records, config.phase4_output_dir / "history.csv")
        loss_plot, accuracy_plot = save_curves(history.records, config.phase4_output_dir, dpi=config.plot_dpi)
        assessment = assess_learning_curves(history.records, config)
        report: dict[str, Any] = {
            "status": "completed", "config": asdict(config), "python": platform.python_version(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "Pillow", "matplotlib")},
            "best_epoch": best.epoch, "best_val_loss": best.val_loss, "checkpoint": str(best.path),
            "history": history.as_dicts(), "loss_plot": str(loss_plot), "accuracy_plot": str(accuracy_plot),
            "curve_assessment": assessment, "dataset_counts": {mode: len(ds) for mode, ds in datasets.items()},
            "device": str(device), "pretrained_weights_loaded": True,
            "test_dataset_constructed": False,
        }
        save_json(report, status_path)
        LOGGER.info("Best epoch=%d dev_loss=%.6f", best.epoch, best.val_loss)
        LOGGER.info("Learning curves: %s", assessment)
        return report
    except (OSError, ValueError, RuntimeError) as exc:
        save_json({"status": "failed", "config": asdict(config), "error_type": type(exc).__name__,
                   "error": str(exc), "best_epoch": None, "best_val_loss": None,
                   "epochs_completed": len(history.records), "pretrained_weights_loaded": pretrained_loaded,
                   "test_dataset_constructed": False,
                   "note": "A failed run has no final training result; any existing artifacts may belong to an earlier run."}, status_path)
        raise
