"""Unit tests for class imbalance weighting and learning rate scheduling."""

import numpy as np
import pytest
import torch
import torch.optim as optim

from src.training.config import TrainingConfig
from src.training.losses import compute_endpoint_pos_weights
from src.training.scheduler import build_scheduler, step_scheduler


def test_compute_endpoint_pos_weights():
    # 4 samples, 2 endpoints
    # Task 0: 3 neg, 1 pos -> ratio = 3.0
    # Task 1: 1 neg, 3 pos -> ratio = 1/3 ~ 0.333
    labels = np.array([
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=np.float32)

    weights = compute_endpoint_pos_weights(labels, num_tasks=2, max_weight=50.0)
    assert weights.shape == (2,)
    assert torch.isclose(weights[0], torch.tensor(3.0))
    assert torch.isclose(weights[1], torch.tensor(1.0 / 3.0), atol=1e-3)


def test_compute_endpoint_pos_weights_with_nans_and_edge_cases():
    # Task 0: 2 neg, 1 pos, 1 NaN -> ratio = 2.0
    # Task 1: 0 neg, 2 pos, 2 NaN -> all pos -> fallback 1.0
    # Task 2: 0 pos, 2 neg, 2 NaN -> all neg -> fallback 1.0
    # Task 3: 4 NaN -> fallback 1.0
    labels = np.array([
        [0.0, 1.0, 0.0, np.nan],
        [0.0, 1.0, 0.0, np.nan],
        [1.0, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float32)

    weights = compute_endpoint_pos_weights(labels, num_tasks=4, max_weight=10.0)
    assert weights.shape == (4,)
    assert torch.isclose(weights[0], torch.tensor(2.0))
    assert torch.isclose(weights[1], torch.tensor(1.0))
    assert torch.isclose(weights[2], torch.tensor(1.0))
    assert torch.isclose(weights[3], torch.tensor(1.0))


def test_compute_endpoint_pos_weights_clamping():
    # 1 pos, 100 neg -> ratio = 100.0, max_weight = 25.0
    labels = np.zeros((101, 1), dtype=np.float32)
    labels[0, 0] = 1.0

    weights = compute_endpoint_pos_weights(labels, num_tasks=1, max_weight=25.0)
    assert weights[0].item() == 25.0


def test_build_and_step_scheduler():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = optim.AdamW([param], lr=1e-2)
    config = TrainingConfig(
        scheduler="ReduceLROnPlateau",
        scheduler_factor=0.5,
        scheduler_patience=2,
        min_lr=1e-5,
    )

    scheduler = build_scheduler(optimizer, config)
    assert scheduler is not None

    # Step with non-improving loss to trigger reduction after patience=2
    step_scheduler(scheduler, metric=0.5, current_epoch=1)
    assert optimizer.param_groups[0]["lr"] == 1e-2

    step_scheduler(scheduler, metric=0.5, current_epoch=2)
    assert optimizer.param_groups[0]["lr"] == 1e-2

    step_scheduler(scheduler, metric=0.5, current_epoch=3)
    assert optimizer.param_groups[0]["lr"] == 1e-2

    # Epoch 4: patience exhausted -> lr reduced by factor 0.5
    step_scheduler(scheduler, metric=0.5, current_epoch=4)
    assert optimizer.param_groups[0]["lr"] == 5e-3
