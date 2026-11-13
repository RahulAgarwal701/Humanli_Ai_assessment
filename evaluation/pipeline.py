"""One final-test evaluation of the already selected best checkpoint."""

from dataclasses import asdict
import importlib.metadata
import logging
from typing import Any

from torch.utils.data import DataLoader

from config import Config
from data.dataset import TestFolderDataset
from data.loaders import seed_worker
from data.manifest import read_manifest, save_json
from data.transforms import build_transforms
from evaluation.checkpoint import load_best_checkpoint, sha256_file
from evaluation.evaluator import Evaluator
from evaluation.metrics import AccuracyMetric, ClassificationReportMetric, ConfusionMatrixMetric
from evaluation.reporting import log_results, save_confusion_matrix
from models.sanity import select_device
from reproducibility import seed_everything


LOGGER = logging.getLogger(__name__)


def run_evaluation(config: Config) -> dict[str, Any]:
    """Restore saved preprocessing, evaluate only final-test images, and save all metrics."""
    seed_everything(config.seed)
    loaded = load_best_checkpoint(config)
    effective = loaded.config
    device = select_device(effective.device)
    model = loaded.model.to(device)
    dataset = TestFolderDataset(effective.dataset_root, effective.test_folder, effective.class_names,
                                build_transforms("test", effective), read_manifest(effective.manifest_path))
    loader = DataLoader(dataset, batch_size=effective.batch_size, shuffle=False,
                        num_workers=effective.num_workers, pin_memory=effective.pin_memory,
                        drop_last=False, worker_init_fn=seed_worker)
    evaluator = Evaluator(model, [AccuracyMetric(), ClassificationReportMetric(), ConfusionMatrixMetric()], device)
    result = evaluator.evaluate(loader, effective.class_names)
    if len(result.targets) != len(dataset):
        raise ValueError("Evaluation did not cover every final-test image exactly once")
    if sha256_file(effective.phase4_output_dir / effective.checkpoint_filename) != loaded.provenance["sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    log_results(result, effective.class_names, digits=effective.evaluation_digits)
    figure = save_confusion_matrix(result.metrics["confusion_matrix"], effective.class_names, effective)
    errors = [{"path": sample.path.relative_to(effective.dataset_root.resolve()).as_posix(),
               "true_class": effective.class_names[target], "predicted_class": effective.class_names[predicted]}
              for sample, target, predicted in zip(dataset.samples, result.targets, result.predictions)
              if target != predicted]
    matrix = result.metrics["confusion_matrix"]
    confusions = [{"true_class": name, "predicted_class": predicted_name, "count": matrix[i][j]}
                  for i, name in enumerate(effective.class_names)
                  for j, predicted_name in enumerate(effective.class_names) if i != j and matrix[i][j]]
    gap = (result.metrics["accuracy"] - loaded.provenance["dev_val_accuracy"]) * 100
    report: dict[str, Any] = {
        "status": "completed", "class_names": list(effective.class_names), "sample_count": len(dataset),
        "metrics": result.metrics, "checkpoint": loaded.provenance, "device": str(device),
        "preprocessing": {key: asdict(effective)[key] for key in
                          ("image_size", "normalization_mean", "normalization_std", "interpolation")},
        "confusion_matrix_axes": {"rows": "true class", "columns": "predicted class"},
        "confusion_matrix_image": str(figure), "observed_confusions": confusions,
        "misclassified_examples": errors,
        "dev_test_comparison": {"dev_accuracy": loaded.provenance["dev_val_accuracy"],
                                "test_accuracy": result.metrics["accuracy"], "test_minus_dev_percentage_points": gap},
        "test_set_usage": "Final evaluation only; not used for tuning or checkpoint selection in this pipeline.",
        "checkpoint_unchanged": True,
        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "scikit-learn", "matplotlib")},
    }
    save_json(report, effective.phase5_output_dir / "test_metrics.json")
    LOGGER.info("Test minus dev accuracy: %+.4f percentage points", gap)
    LOGGER.info("Test set was not used for tuning or checkpoint selection; selected checkpoint is unchanged.")
    LOGGER.info("Saved metrics: %s", effective.phase5_output_dir / "test_metrics.json")
    return report
