"""Shared, validated configuration for training, evaluation and inference."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    """Central source of paths, class order, hyperparameters and audit tolerances."""

    dataset_root: Path = PROJECT_ROOT / "afhq"
    output_dir: Path = PROJECT_ROOT / "artifacts" / "phase1"
    train_folder: str = "train"
    test_folder: str = "val"
    seed: int = 42
    dev_ratio: float = 0.10
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 5
    lr: float = 0.001
    class_names: tuple[str, ...] = ("dog", "cat", "wild")
    class_proportion_tolerance: float = 0.02
    split_ratio_tolerance: float = 0.01
    manifest_path: Path = PROJECT_ROOT / "artifacts" / "phase1" / "split_manifest.csv"
    phase2_output_dir: Path = PROJECT_ROOT / "artifacts" / "phase2"
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 10.0
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    interpolation: str = "bilinear"
    rotation_fill: int = 0
    num_workers: int = 0
    pin_memory: bool = False
    grid_samples_per_class: int = 4
    phase: int = 1
    architecture: str = "resnet18"
    num_classes: int = 3
    freeze_features: bool = True
    pretrained_weights: str | None = "IMAGENET1K_V1"
    torch_hub_dir: Path = PROJECT_ROOT / "artifacts" / "torch_hub"
    phase3_output_dir: Path = PROJECT_ROOT / "artifacts" / "phase3"
    model_check_batch_size: int = 4
    device: str = "auto"
    phase4_output_dir: Path = PROJECT_ROOT / "checkpoints"
    checkpoint_filename: str = "best_model.pt"
    deterministic_training: bool = True
    plot_dpi: int = 140
    curve_loss_tolerance: float = 0.01
    underfit_accuracy_threshold: float = 0.60
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    phase5_output_dir: Path = PROJECT_ROOT / "artifacts" / "phase5"
    evaluation_digits: int = 4
    phase6_output_dir: Path = PROJECT_ROOT / "artifacts" / "phase6"
    new_images_dir: Path = PROJECT_ROOT / "new_images"
    prediction_top_k: int = 3
    demo_images_per_class: int = 2
    demo_columns: int = 3

    def __post_init__(self) -> None:
        """Reject invalid configuration before touching the dataset."""
        if not isinstance(self.dataset_root, Path) or not isinstance(self.output_dir, Path):
            raise TypeError("dataset_root and output_dir must be pathlib.Path objects")
        if not 0 < self.dev_ratio < 1:
            raise ValueError("dev_ratio must be between 0 and 1")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32) for NumPy compatibility")
        if min(self.image_size, self.batch_size, self.epochs) < 1 or self.lr <= 0:
            raise ValueError("image_size, batch_size, epochs and lr must be positive")
        if self.num_classes < 1 or len(self.class_names) != self.num_classes or len(set(self.class_names)) != self.num_classes:
            raise ValueError("num_classes must match the number of unique class_names")
        names = (*self.class_names, self.train_folder, self.test_folder)
        if any(not n or n.startswith(".") or Path(n).name != n or "\\" in n for n in names):
            raise ValueError("Class and split names must be simple, non-hidden folder names")
        if self.train_folder == self.test_folder:
            raise ValueError("Training and final-test folders must differ")
        if not 0 <= self.class_proportion_tolerance < 1:
            raise ValueError("class_proportion_tolerance must be in [0, 1)")
        if not 0 <= self.split_ratio_tolerance < 1:
            raise ValueError("split_ratio_tolerance must be in [0, 1)")
        if not isinstance(self.manifest_path, Path) or not isinstance(self.phase2_output_dir, Path):
            raise TypeError("manifest_path and phase2_output_dir must be pathlib.Path objects")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("RGB normalization requires three means and three standard deviations")
        if any(value <= 0 for value in self.normalization_std):
            raise ValueError("Normalization standard deviations must be positive")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if min(self.rotation_degrees, self.jitter_brightness, self.jitter_contrast) < 0:
            raise ValueError("Augmentation magnitudes must be nonnegative")
        if self.interpolation not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("interpolation must be nearest, bilinear or bicubic")
        if not 0 <= self.rotation_fill <= 255:
            raise ValueError("rotation_fill must be in [0, 255]")
        if self.num_workers < 0 or self.grid_samples_per_class < 1:
            raise ValueError("num_workers must be nonnegative and grid_samples_per_class positive")
        if self.phase not in {1, 2, 3, 4, 5, 6}:
            raise ValueError("Only phases 1 through 6 are implemented")
        if not isinstance(self.freeze_features, bool):
            raise TypeError("freeze_features must be a bool")
        if not isinstance(self.torch_hub_dir, Path) or not isinstance(self.phase3_output_dir, Path):
            raise TypeError("torch_hub_dir and phase3_output_dir must be pathlib.Path objects")
        if not self.architecture or self.model_check_batch_size < 1:
            raise ValueError("architecture must be nonempty and model_check_batch_size positive")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")
        if not isinstance(self.phase4_output_dir, Path):
            raise TypeError("phase4_output_dir must be a pathlib.Path")
        if not self.checkpoint_filename or Path(self.checkpoint_filename).name != self.checkpoint_filename:
            raise ValueError("checkpoint_filename must be a simple file name")
        if self.plot_dpi < 1 or not isinstance(self.deterministic_training, bool):
            raise ValueError("plot_dpi must be positive and deterministic_training must be a bool")
        if self.curve_loss_tolerance < 0 or not 0 <= self.underfit_accuracy_threshold <= 1:
            raise ValueError("Curve loss tolerance must be nonnegative and accuracy threshold in [0, 1]")
        if len(self.adam_betas) != 2 or any(not 0 <= beta < 1 for beta in self.adam_betas):
            raise ValueError("Adam requires two beta coefficients in [0, 1)")
        if self.adam_eps <= 0 or self.weight_decay < 0:
            raise ValueError("adam_eps must be positive and weight_decay nonnegative")
        if not isinstance(self.phase5_output_dir, Path) or self.evaluation_digits < 1:
            raise ValueError("phase5_output_dir must be a Path and evaluation_digits positive")
        if not isinstance(self.phase6_output_dir, Path) or not isinstance(self.new_images_dir, Path):
            raise TypeError("phase6_output_dir and new_images_dir must be pathlib.Path objects")
        if min(self.prediction_top_k, self.demo_images_per_class, self.demo_columns) < 1:
            raise ValueError("Prediction rank count and demo dimensions must be positive")

    @property
    def train_dir(self) -> Path:
        """Original training pool, from which train and dev will be selected."""
        return self.dataset_root / self.train_folder

    @property
    def test_dir(self) -> Path:
        """Untouched final-test source, named val in AFHQ."""
        return self.dataset_root / self.test_folder
