"""Composition root for separately requested data and model phases."""

from dataclasses import asdict
import importlib.metadata
import logging
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

from config import Config
from cli import parse_config, parse_options
from data.audit import audit_dataset
from data.manifest import save_json, save_manifest
from data.records import ManifestRow
from data.validation import validate_split
from reproducibility import seed_everything
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.split import DatasetSplitter


LOGGER = logging.getLogger(__name__)


def run_phase_one(config: Config, splitter: "DatasetSplitter") -> dict[str, Any]:
    """Run only data preparation; no torchvision imports, model or training work.

    Splitting is a pure function of sorted content records and an explicit seed.
    Available Python, NumPy and torch random generators receive the same seed.
    """
    seed_everything(config.seed)
    audit = audit_dataset(config.dataset_root, config.train_folder, config.test_folder, config.class_names)
    save_json(audit.report, config.output_dir / "audit_report.json")
    LOGGER.info("Audit formats/sizes/modes: %s", audit.report["sources"])
    LOGGER.info("Duplicate groups: %d; extra copies: %d; unreadable/layout errors: %d",
                len(audit.report["duplicate_groups"]), audit.report["extra_duplicate_copies"],
                len(audit.report["errors"]))
    if not audit.report["passed"]:
        raise ValueError(
            "Audit failed; inspect audit_report.json. Fix unreadable files, layout, "
            "conflicting labels or train/test hash overlap explicitly before splitting."
        )
    indices = splitter.split(audit.train, dev_ratio=config.dev_ratio, seed=config.seed)
    validation = validate_split(
        audit.train, audit.test, indices, class_names=config.class_names,
        dev_ratio=config.dev_ratio, class_tolerance=config.class_proportion_tolerance,
        ratio_tolerance=config.split_ratio_tolerance,
    )
    rows = [
        ManifestRow(record.path, record.label, record.group_id, split_name)
        for split_name, selection in (("train", indices.train), ("dev", indices.dev))
        for record in (audit.train[i] for i in selection)
    ]
    rows.extend(ManifestRow(record.path, record.label, record.group_id, "test") for record in audit.test)
    rows.sort(key=lambda row: (row.split, row.path))
    save_manifest(rows, config.output_dir / "split_manifest.csv")
    save_json(validation, config.output_dir / "split_report.json")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "scikit-learn")}
    save_json({"config": asdict(config), "python": sys.version, "platform": platform.platform(),
               "dependencies": versions, "strategy": type(splitter).__name__},
              config.output_dir / "run_metadata.json")
    for name, summary in validation["splits"].items():
        LOGGER.info("%s: n=%d; counts=%s; class proportions=%s", name, summary["count"],
                    summary["class_counts"],
                    {label: f"{value:.4%}" for label, value in summary["class_proportions"].items()})
    LOGGER.info("Hash group overlap: %s", validation["overlap_hash_counts"])
    LOGGER.info("Dev fraction: %.4f%%; complete, exactly-once coverage: %s",
                validation["actual_dev_ratio"] * 100, validation["coverage_exactly_once"])
    LOGGER.info("Saved manifest: %s", config.output_dir / "split_manifest.csv")
    return validation


def run_phase_two(config: Config) -> dict[str, Any]:
    """Build datasets/loaders and inspect tensors; never audit again or train a model."""
    from data.dataset import build_datasets
    from data.loaders import build_dataloaders
    from data.sanity import check_datasets

    seed_everything(config.seed)
    datasets = build_datasets(config)
    loaders = build_dataloaders(datasets, config)
    return check_datasets(datasets, loaders, config)


def run_phase_three(config: Config) -> dict[str, Any]:
    """Inspect model construction only; never load data or perform training."""
    from models.builder import create_model_builder
    from models.sanity import inspect_model

    seed_everything(config.seed)
    return inspect_model(config, create_model_builder(config))


def run_phase_four(config: Config) -> dict[str, Any]:
    """Run training and dev validation only, without touching the final-test source."""
    from training.pipeline import run_training

    return run_training(config)


def run_phase_five(config: Config) -> dict[str, Any]:
    """Evaluate the fixed best checkpoint on final-test images only."""
    from evaluation.pipeline import run_evaluation

    return run_evaluation(config)


def run_phase_six(config: Config, *, paths: Sequence[Path] | None = None,
                  input_dir: Path | None = None) -> dict[str, Any]:
    """Run only resilient image predictions and save the labeled demo."""
    from inference.pipeline import run_prediction_demo

    return run_prediction_demo(config, paths=paths, input_dir=input_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """Run exactly the selected phase; Phase 2 never regenerates Phase 1 splits."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        options = parse_options(argv)
        config = options.config
        if config.phase == 2:
            run_phase_two(config)
        elif config.phase == 3:
            run_phase_three(config)
        elif config.phase == 4:
            run_phase_four(config)
        elif config.phase == 5:
            run_phase_five(config)
        elif config.phase == 6:
            run_phase_six(config, paths=options.images, input_dir=options.input_dir)
        else:
            from data.split import StratifiedGroupSplitter

            run_phase_one(config, StratifiedGroupSplitter())
    except (ValueError, OSError, RuntimeError) as exc:
        LOGGER.error("Phase failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
