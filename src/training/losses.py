"""Masked Binary Cross-Entropy Loss for Multi-Task Toxicity Prediction.

Handles unlabelled assay endpoints (NaNs) and explicit boolean masks to ensure
that missing labels never contribute to gradient updates.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_bce_loss(
    logits: torch.Tensor,
    targets: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    pos_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> torch.Tensor:
    """Compute masked binary cross-entropy loss over observed multi-task labels.

    Args:
        logits: Raw unnormalized predictions of shape [batch_size, num_tasks].
        targets: Target labels of shape [batch_size, num_tasks], containing 0.0, 1.0, or NaN.
        mask: Boolean mask of shape [batch_size, num_tasks] indicating observed labels.
        pos_weight: Optional positive class weighting tensor of shape [num_tasks].

    Returns:
        Scalar torch.Tensor loss averaged over all observed labels.
    """
    device = logits.device

    # Ensure targets and mask are tensors on the correct device
    if isinstance(targets, np.ndarray):
        targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
    elif isinstance(targets, torch.Tensor):
        targets_t = targets.to(device=device, dtype=torch.float32)
    else:
        targets_t = torch.tensor(targets, device=device, dtype=torch.float32)

    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask).to(device=device, dtype=torch.bool)
    elif isinstance(mask, torch.Tensor):
        mask_t = mask.to(device=device, dtype=torch.bool)
    else:
        mask_t = torch.tensor(mask, device=device, dtype=torch.bool)

    # Replace NaNs with 0.0 in targets to allow standard BCE computation
    # These entries are strictly zeroed out by valid_mask below
    valid_targets = torch.nan_to_num(targets_t, nan=0.0)

    # Valid mask requires BOTH explicit mask is True and target is finite (non-NaN)
    valid_mask = mask_t & torch.isfinite(targets_t)

    # Handle optional per-endpoint positive class weighting
    pw_t: Optional[torch.Tensor] = None
    if pos_weight is not None:
        if isinstance(pos_weight, np.ndarray):
            pw_t = torch.from_numpy(pos_weight).to(device=device, dtype=torch.float32)
        elif isinstance(pos_weight, torch.Tensor):
            pw_t = pos_weight.to(device=device, dtype=torch.float32)
        else:
            pw_t = torch.tensor(pos_weight, device=device, dtype=torch.float32)

    loss_matrix = F.binary_cross_entropy_with_logits(
        logits,
        valid_targets,
        pos_weight=pw_t,
        reduction="none",
    )

    num_valid = valid_mask.sum()
    if num_valid == 0:
        # Zero valid labels in batch: return a zero loss that preserves the gradient graph
        return (logits * 0.0).sum()

    loss = (loss_matrix * valid_mask.float()).sum() / num_valid
    return loss


class MaskedBCELoss(nn.Module):
    """Module wrapper for masked binary cross-entropy loss with optional pos_weight."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone().detach().float())
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: Union[torch.Tensor, np.ndarray],
        mask: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        return masked_bce_loss(logits, targets, mask, pos_weight=self.pos_weight)
