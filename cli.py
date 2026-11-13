"""CLI parsing only; model/data/training work is composed by main.py."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import Config


COMMAND_PHASES = {"split": 1, "transforms": 2, "model": 3, "train": 4, "evaluate": 5, "predict": 6}


@dataclass(frozen=True)
class CLIOptions:
    """Central config plus optional prediction inputs."""

    config: Config
    images: tuple[Path, ...] | None = None
    input_dir: Path | None = None


def parse_options(argv: Sequence[str] | None = None) -> CLIOptions:
    """Accept readable stage commands and backward-compatible --phase flags."""
    defaults = Config()
    parser = argparse.ArgumentParser(description="Reproducible AFHQ classification: split, train, evaluate and predict.")
    parser.add_argument("command", nargs="?", choices=tuple(COMMAND_PHASES))
    parser.add_argument("--phase", type=int, choices=tuple(COMMAND_PHASES.values()), default=None)
    parser.add_argument("--data-root", type=Path, default=defaults.dataset_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--dev-ratio", type=float, default=defaults.dev_ratio)
    parser.add_argument("--class-names", nargs=3, default=defaults.class_names)
    parser.add_argument("--manifest", type=Path, default=defaults.manifest_path)
    parser.add_argument("--phase2-output-dir", type=Path, default=defaults.phase2_output_dir)
    parser.add_argument("--phase3-output-dir", type=Path, default=defaults.phase3_output_dir)
    parser.add_argument("--phase4-output-dir", type=Path, default=defaults.phase4_output_dir)
    parser.add_argument("--phase5-output-dir", type=Path, default=defaults.phase5_output_dir)
    parser.add_argument("--phase6-output-dir", type=Path, default=defaults.phase6_output_dir)
    parser.add_argument("--checkpoint", type=Path, help="Best checkpoint; history.csv and training_status.json must be alongside it")
    parser.add_argument("--architecture", default=defaults.architecture)
    parser.add_argument("--weights", default=defaults.pretrained_weights,
                        help="ImageNet weights enum name; 'none' is for architecture checks only")
    parser.add_argument("--torch-hub-dir", type=Path, default=defaults.torch_hub_dir)
    parser.add_argument("--freeze-features", action=argparse.BooleanOptionalAction, default=defaults.freeze_features)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=defaults.device)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--images", type=Path, nargs="+", help="Image paths; individual failures are recorded")
    inputs.add_argument("--input-dir", type=Path, help="Predict files from this directory")
    args = parser.parse_args(argv)
    command_phase = COMMAND_PHASES.get(args.command)
    if command_phase is not None and args.phase is not None and command_phase != args.phase:
        parser.error("Command and --phase select different stages")
    phase = command_phase if command_phase is not None else args.phase if args.phase is not None else defaults.phase
    if (args.images is not None or args.input_dir is not None) and phase != COMMAND_PHASES["predict"]:
        parser.error("--images and --input-dir are available only for predict / --phase 6")
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if checkpoint and phase not in {COMMAND_PHASES["evaluate"], COMMAND_PHASES["predict"]}:
        parser.error("--checkpoint is available for evaluate or predict")
    config = Config(
        dataset_root=args.data_root.expanduser().resolve(), output_dir=args.output_dir.expanduser().resolve(),
        seed=args.seed, dev_ratio=args.dev_ratio, class_names=tuple(args.class_names),
        manifest_path=args.manifest.expanduser().resolve(), phase=phase,
        phase2_output_dir=args.phase2_output_dir.expanduser().resolve(),
        phase3_output_dir=args.phase3_output_dir.expanduser().resolve(),
        phase4_output_dir=checkpoint.parent if checkpoint else args.phase4_output_dir.expanduser().resolve(),
        checkpoint_filename=checkpoint.name if checkpoint else defaults.checkpoint_filename,
        phase5_output_dir=args.phase5_output_dir.expanduser().resolve(),
        phase6_output_dir=args.phase6_output_dir.expanduser().resolve(),
        architecture=args.architecture, pretrained_weights=None if args.weights.lower() == "none" else args.weights,
        freeze_features=args.freeze_features, torch_hub_dir=args.torch_hub_dir.expanduser().resolve(),
        device=args.device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers,
    )
    return CLIOptions(config, tuple(path.expanduser() for path in args.images) if args.images is not None else None,
                      args.input_dir.expanduser() if args.input_dir is not None else None)


def parse_config(argv: Sequence[str] | None = None) -> Config:
    """Retain the original parse_config API for callers requiring just configuration."""
    return parse_options(argv).config
