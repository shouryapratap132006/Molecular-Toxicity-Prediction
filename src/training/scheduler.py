"""Learning rate scheduling utilities and step tracking."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from src.training.config import TrainingConfig

logger = logging.getLogger(__name__)


def build_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
) -> Optional[LRScheduler | ReduceLROnPlateau]:
    """Instantiate a learning rate scheduler according to config.

    Args:
        optimizer: PyTorch optimizer.
        config: TrainingConfig instance.

    Returns:
        Learning rate scheduler instance or None.
    """
    scheduler_name = config.scheduler.lower().strip()
    if scheduler_name == "reducelronplateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_lr,
        )
    elif scheduler_name in ("none", "", "null"):
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def step_scheduler(
    scheduler: Optional[LRScheduler | ReduceLROnPlateau],
    metric: float,
    current_epoch: int,
) -> Optional[float]:
    """Step the scheduler and log if learning rate changed.

    Args:
        scheduler: Scheduler instance or None.
        metric: Monitored metric (e.g. validation loss).
        current_epoch: Current epoch number.

    Returns:
        Updated current learning rate.
    """
    if scheduler is None:
        return None

    old_lr = scheduler.optimizer.param_groups[0]["lr"]

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()

    new_lr = scheduler.optimizer.param_groups[0]["lr"]
    if new_lr < old_lr:
        logger.info(
            "Epoch %d: Learning rate reduced from %.2e to %.2e",
            current_epoch,
            old_lr,
            new_lr,
        )
    return new_lr
