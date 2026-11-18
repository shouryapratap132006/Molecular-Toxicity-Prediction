"""Unit tests for GNN training loop and single-epoch optimization step."""

import numpy as np
import pytest
import torch
import torch.optim as optim

from src.data.graph_dataset import collate_graphs
from src.models import MolecularGNN
from src.training.trainer import train_one_epoch


@pytest.fixture
def synthetic_graph():
    return {
        "mol_id": "MOL1",
        "x": np.ones((4, 19), dtype=np.float32),
        "edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64),
        "edge_attr": np.ones((4, 18), dtype=np.float32),
        "y": np.array([1.0, 0.0, np.nan, 1.0, 0.0, np.nan, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "mask": np.array([True, True, False, True, True, False, True, True, True, True, True, True], dtype=bool),
    }


def test_train_one_epoch_updates_weights(synthetic_graph):
    batch = collate_graphs([synthetic_graph, synthetic_graph])
    dataloader = [batch]

    model = MolecularGNN(hidden_dim=32, num_layers=2)
    device = torch.device("cpu")
    model.to(device)

    # Record parameter values before step
    initial_param = next(model.parameters()).clone().detach()

    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    loss, valid_labels = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        gradient_clip=1.0,
    )

    assert valid_labels == 20  # 10 valid labels per graph * 2 graphs
    assert isinstance(loss, float)
    assert loss > 0.0

    # Ensure parameter was updated
    updated_param = next(model.parameters())
    assert not torch.allclose(initial_param, updated_param)
    assert model.training


def test_train_one_epoch_gradient_clipping(synthetic_graph):
    batch = collate_graphs([synthetic_graph])
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    device = torch.device("cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    loss, valid_count = train_one_epoch(
        model=model,
        dataloader=[batch],
        optimizer=optimizer,
        device=device,
        gradient_clip=0.5,
    )
    assert loss > 0.0
    assert valid_count == 10


def test_train_one_epoch_all_nan_batch_skipped():
    empty_graph = {
        "mol_id": "MOL_EMPTY",
        "x": np.ones((2, 19), dtype=np.float32),
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "edge_attr": np.ones((2, 18), dtype=np.float32),
        "y": np.full(12, np.nan, dtype=np.float32),
        "mask": np.zeros(12, dtype=bool),
    }
    batch = collate_graphs([empty_graph])
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    device = torch.device("cpu")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    initial_param = next(model.parameters()).clone().detach()

    loss, valid_labels = train_one_epoch(
        model=model,
        dataloader=[batch],
        optimizer=optimizer,
        device=device,
    )

    assert valid_labels == 0
    # Weights should remain unchanged because the batch had zero valid labels
    assert torch.allclose(initial_param, next(model.parameters()))
