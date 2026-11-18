"""Unit tests for masked binary cross-entropy loss."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.training.losses import MaskedBCELoss, masked_bce_loss


def test_masked_bce_loss_fully_observed():
    logits = torch.tensor([[1.0, -1.0], [0.5, -0.5]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[True, True], [True, True]])

    loss = masked_bce_loss(logits, targets, mask)
    expected = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.isclose(loss, expected)


def test_masked_bce_loss_nan_ignored():
    logits = torch.tensor([[1.0, -1.0], [0.5, -0.5]], requires_grad=True)
    # Entry [0, 1] is NaN and masked False
    targets = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    mask = torch.tensor([[True, False], [True, True]])

    loss = masked_bce_loss(logits, targets, mask)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Compute expected loss on the 3 observed elements only
    obs_logits = torch.tensor([1.0, 0.5, -0.5])
    obs_targets = torch.tensor([1.0, 0.0, 1.0])
    expected = F.binary_cross_entropy_with_logits(obs_logits, obs_targets)
    assert torch.isclose(loss, expected)


def test_masked_bce_loss_mask_filters_valid_targets():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 1.0]])
    # Even though targets[0, 1] is finite, mask says False
    mask = torch.tensor([[True, False]])

    loss = masked_bce_loss(logits, targets, mask)
    expected = F.binary_cross_entropy_with_logits(torch.tensor([0.0]), torch.tensor([1.0]))
    assert torch.isclose(loss, expected)


def test_masked_bce_loss_zero_valid_labels():
    logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[float("nan"), float("nan")]])
    mask = torch.tensor([[False, False]])

    loss = masked_bce_loss(logits, targets, mask)
    assert not torch.isnan(loss)
    assert loss.item() == 0.0

    # Gradient must still propagate without error
    loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad == 0.0)


def test_masked_bce_gradient_flow():
    logits = torch.tensor([[1.0, 2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])  # second task is masked out

    loss = masked_bce_loss(logits, targets, mask)
    loss.backward()

    assert logits.grad is not None
    # First task has non-zero gradient
    assert logits.grad[0, 0] != 0.0
    # Second task has exactly zero gradient because it was masked
    assert logits.grad[0, 1].item() == 0.0


def test_masked_bce_pos_weight():
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, True]])
    pos_weight = torch.tensor([4.0, 4.0])

    loss_weighted = masked_bce_loss(logits, targets, mask, pos_weight=pos_weight)
    loss_unweighted = masked_bce_loss(logits, targets, mask)

    # With positive targets, pos_weight > 1 increases loss
    assert loss_weighted > loss_unweighted


def test_masked_bce_numpy_inputs():
    logits = torch.tensor([[0.5, -0.5]])
    targets = np.array([[1.0, np.nan]])
    mask = np.array([[True, False]])

    loss = masked_bce_loss(logits, targets, mask)
    assert not torch.isnan(loss)
    assert loss.dim() == 0


def test_masked_bce_module_wrapper():
    loss_fn = MaskedBCELoss(pos_weight=torch.tensor([2.0, 1.0]))
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, True]])

    loss = loss_fn(logits, targets, mask)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
