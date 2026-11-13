"""Interchangeable model builders and an extensible architecture factory."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
import logging

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from config import Config
from models.feature_extractor import FeatureExtractorClassifier


LOGGER = logging.getLogger(__name__)


class ModelBuilder(ABC):
    """Minimal model-construction contract, independent of datasets or training."""

    @abstractmethod
    def build(self) -> FeatureExtractorClassifier:
        """Return the shared features/fc classifier contract; propagate loading errors."""
        raise NotImplementedError


BuilderFactory = Callable[[Config], ModelBuilder]
_BUILDERS: dict[str, BuilderFactory] = {}


def register_model(name: str) -> Callable[[BuilderFactory], BuilderFactory]:
    """Register an architecture without changing the existing factory or builders."""
    def register(factory: BuilderFactory) -> BuilderFactory:
        if not name or name in _BUILDERS:
            raise ValueError(f"Empty or already registered architecture: {name!r}")
        _BUILDERS[name] = factory
        return factory
    return register


@register_model("resnet18")
class ResNet18FeatureExtractorBuilder(ModelBuilder):
    """Load ImageNet ResNet18, replace its head and apply configured freezing."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build(self) -> FeatureExtractorClassifier:
        """Use the explicit weights API; None is an intentional offline test mode.

        Errors never cause an automatic random-initialization fallback. The hub
        cache is project-local so no shared or read-only cache needs modification.
        """
        try:
            weights = (ResNet18_Weights[self.config.pretrained_weights]
                       if self.config.pretrained_weights is not None else None)
        except KeyError as exc:
            raise ValueError(f"Unknown ResNet18 weights: {self.config.pretrained_weights!r}") from exc
        if weights is None:
            LOGGER.warning("Building ResNet18 with weights=None: architecture checks only, not pretrained")
        else:
            LOGGER.info("Loading ResNet18 with weights=%s", weights.name)
        previous_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(str(self.config.torch_hub_dir))
        try:
            source = resnet18(weights=weights)
        finally:
            torch.hub.set_dir(previous_hub_dir)
        # Derive 512 from torchvision's architecture rather than duplicating it.
        source.fc = nn.Linear(source.fc.in_features, self.config.num_classes)
        features = nn.Sequential(OrderedDict(
            (name, module) for name, module in source.named_children() if name != "fc"
        ))
        return FeatureExtractorClassifier(features, source.fc, freeze_features=self.config.freeze_features)


def create_model_builder(config: Config) -> ModelBuilder:
    """Resolve a configured architecture to the small abstract builder interface."""
    try:
        factory = _BUILDERS[config.architecture]
    except KeyError as exc:
        raise ValueError(f"Unknown architecture {config.architecture!r}; available: {sorted(_BUILDERS)}") from exc
    return factory(config)


def build_model(config: Config) -> FeatureExtractorClassifier:
    """Convenience factory for callers that only need the composed model."""
    return create_model_builder(config).build()
