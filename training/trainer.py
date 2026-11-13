"""Training orchestration with injected models, optimization and persistence."""

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from config import Config
from training.checkpoint import CheckpointInfo, CheckpointStrategy
from training.metrics import ClassificationMeter, ClassificationMetrics, MetricRecorder
from training.reporting import EpochReporter


BatchLoader = Iterable[tuple[Tensor, Tensor]]


class Trainer:
    """Run optimization and dev validation; never construct or accept a test loader.

    Move the injected model onto device before constructing the optimizer. The
    injected criterion must produce a mean batch loss for sample-weighted metrics.
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                 loaders: Mapping[str, BatchLoader], checkpointer: CheckpointStrategy,
                 config: Config, device: torch.device, metrics: MetricRecorder,
                 reporter: EpochReporter) -> None:
        if set(loaders) != {"train", "dev"}:
            raise ValueError("Trainer accepts exactly train and dev loaders; test is forbidden")
        trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        optimized_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        if not trainable_ids or set(optimized_ids) != trainable_ids or len(optimized_ids) != len(trainable_ids):
            raise ValueError("Optimizer must contain each trainable parameter exactly once and no frozen parameters")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loaders = loaders
        self.checkpointer = checkpointer
        self.config = config
        self.device = device
        self.metrics = metrics
        self.reporter = reporter

    def _run_epoch(self, *, training: bool) -> ClassificationMetrics:
        """Execute a full split with gradients enabled only for training."""
        if training:
            self.model.train()
        else:
            self.model.eval()
        meter = ClassificationMeter()
        loader = self.loaders["train" if training else "dev"]
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if not torch.isfinite(loss).item():
                    raise ValueError("Nonfinite loss; training stopped before an invalid update")
                if training:
                    loss.backward()
                    self.optimizer.step()
                meter.update(logits, labels, loss.detach().item())
        return meter.compute()

    def fit(self) -> CheckpointInfo:
        """Run configured epochs and return the checkpoint selected by injected policy."""
        best: CheckpointInfo | None = None
        for epoch in range(1, self.config.epochs + 1):
            train = self._run_epoch(training=True)
            dev = self._run_epoch(training=False)
            record = self.metrics.record(epoch, train, dev)
            self.reporter.report(record)
            saved = self.checkpointer.save_if_best(self.model, epoch=epoch, val_loss=dev.loss, config=self.config)
            if saved is not None:
                best = saved
        if best is None:
            raise ValueError("No checkpoint was selected by the checkpoint strategy")
        return best
