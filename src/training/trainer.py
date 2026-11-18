"""Training loop and epoch execution for Phase 2 MolecularGNN."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.training.losses import masked_bce_loss

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    device: torch.device,
    loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    gradient_clip: float = 1.0,
) -> tuple[float, int]:
    """Train the model for a single epoch.

    Args:
        model: MolecularGNN instance.
        dataloader: Iterable yielding graph batch dictionaries.
        optimizer: PyTorch optimizer (e.g. AdamW).
        device: Target compute device.
        loss_fn: Loss function (defaults to masked_bce_loss).
        gradient_clip: Max gradient norm for clipping (0 or negative disables).

    Returns:
        tuple of (epoch_loss, total_valid_labels_processed)
    """
    model.train()
    if loss_fn is None:
        loss_fn = masked_bce_loss

    total_weighted_loss = 0.0
    total_valid_labels = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass through model
        logits = model(batch)

        y = batch["y"]
        mask = batch["mask"]

        # Determine valid labels in this batch
        if isinstance(y, np.ndarray):
            y_t = torch.from_numpy(y).to(device=device)
        else:
            y_t = y.to(device=device)

        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask).to(device=device)
        else:
            mask_t = mask.to(device=device)

        valid_mask = mask_t.bool() & torch.isfinite(y_t)
        batch_valid_count = int(valid_mask.sum().item())

        if batch_valid_count == 0:
            logger.debug("Skipping batch with 0 valid labels.")
            continue

        # Compute masked loss
        loss = loss_fn(logits, y, mask)

        if not torch.isfinite(loss):
            logger.warning("Encountered non-finite loss (%s); skipping backward.", loss.item())
            continue

        loss.backward()

        # Gradient clipping for numerical stability
        if gradient_clip is not None and gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()

        total_weighted_loss += loss.item() * batch_valid_count
        total_valid_labels += batch_valid_count

    epoch_loss = total_weighted_loss / max(total_valid_labels, 1)
    return float(epoch_loss), total_valid_labels
