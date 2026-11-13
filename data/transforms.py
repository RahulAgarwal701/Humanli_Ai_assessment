"""Extensible factories for PIL-to-normalized-tensor preprocessing."""

from collections.abc import Callable
from typing import Protocol

from PIL import Image
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from config import Config


class ImageTransform(Protocol):
    """Small transform contract shared by all image dataset implementations."""

    def __call__(self, image: Image.Image) -> Tensor:
        """Convert an RGB PIL image into a normalized CHW tensor."""
        ...


TransformBuilder = Callable[[Config], ImageTransform]
_BUILDERS: dict[str, TransformBuilder] = {}


def register_transform(mode: str) -> Callable[[TransformBuilder], TransformBuilder]:
    """Register a new mode without changing the factory or existing builders."""
    def register(builder: TransformBuilder) -> TransformBuilder:
        if not mode or mode in _BUILDERS:
            raise ValueError(f"Empty or already registered transform mode: {mode!r}")
        _BUILDERS[mode] = builder
        return builder
    return register


def _resize(config: Config) -> T.Resize:
    """Fix both spatial dimensions; interpolation is explicit and configurable."""
    interpolation = InterpolationMode(config.interpolation)
    return T.Resize((config.image_size, config.image_size), interpolation=interpolation, antialias=True)


@register_transform("train")
def build_train_transform(config: Config) -> ImageTransform:
    """Augment PIL pixels before conversion and ImageNet normalization."""
    return T.Compose([
        _resize(config),
        T.RandomHorizontalFlip(p=config.horizontal_flip_probability),
        T.RandomRotation(config.rotation_degrees, interpolation=InterpolationMode(config.interpolation),
                         fill=config.rotation_fill),
        T.ColorJitter(brightness=config.jitter_brightness, contrast=config.jitter_contrast),
        T.ToTensor(),
        T.Normalize(mean=config.normalization_mean, std=config.normalization_std),
    ])


@register_transform("test")
@register_transform("dev")
def build_eval_transform(config: Config) -> ImageTransform:
    """Deterministic spatial resize, tensor conversion and normalization only."""
    return T.Compose([
        _resize(config),
        T.ToTensor(),
        T.Normalize(mean=config.normalization_mean, std=config.normalization_std),
    ])


def build_transforms(mode: str, config: Config | None = None) -> ImageTransform:
    """Build a registered mode; unknown modes fail instead of silently augmenting."""
    try:
        builder = _BUILDERS[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown transform mode {mode!r}; available: {sorted(_BUILDERS)}") from exc
    return builder(config if config is not None else Config())
