"""A classifier that preserves frozen feature-extractor inference behavior."""

import torch
from torch import Tensor, nn


class FeatureExtractorClassifier(nn.Module):
    """Compose a feature extractor and a trainable head behind the nn.Module API.

    Frozen parameters alone do not freeze BatchNorm running-statistic buffers.
    Keeping the entire frozen feature extractor in eval mode protects those
    pretrained statistics even after a trainer calls model.train(). The parent
    classifier and head still receive the requested training/eval mode.
    """

    def __init__(self, features: nn.Module, fc: nn.Linear, *, freeze_features: bool) -> None:
        super().__init__()
        self.features = features
        self.fc = fc
        self.freeze_features = freeze_features
        self.features.requires_grad_(not freeze_features)
        self.fc.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True) -> "FeatureExtractorClassifier":
        """Honor nn.Module's mode contract while retaining frozen feature state."""
        super().train(mode)
        if self.freeze_features:
            self.features.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Produce class logits; leave softmax to later prediction code."""
        features = torch.flatten(self.features(images), start_dim=1)
        return self.fc(features)
