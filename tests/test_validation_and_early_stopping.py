"""Unit tests for validation evaluation and early stopping."""

import numpy as np
import pytest
import torch

from src.data.graph_dataset import collate_graphs
from src.models import MolecularGNN
from src.training.early_stopping import EarlyStopping
from src.training.trainer import evaluate


@pytest.fixture
def synthetic_graph():
    return {
        "mol_id": "MOL_VAL",
        "x": np.ones((3, 19), dtype=np.float32),
        "edge_index": np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
        "edge_attr": np.ones((3, 18), dtype=np.float32),
        "y": np.array([1.0, 0.0, np.nan, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "mask": np.array([True, True, False, True, True, True, True, True, True, True, True, True], dtype=bool),
    }


def test_evaluate_leaves_weights_unchanged(synthetic_graph):
    batch = collate_graphs([synthetic_graph, synthetic_graph])
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    device = torch.device("cpu")
    model.to(device)

    initial_weights = [p.clone().detach() for p in model.parameters()]

    val_loss, valid_labels = evaluate(
        model=model,
        dataloader=[batch],
        device=device,
    )

    assert isinstance(val_loss, float)
    assert val_loss > 0.0
    assert valid_labels == 22  # 11 valid labels per graph * 2

    # Weights must be completely identical
    for p_init, p_now in zip(initial_weights, model.parameters()):
        assert torch.all(p_init == p_now)

    # Model is in eval mode
    assert not model.training


def test_early_stopping_improvement_and_trigger():
    es = EarlyStopping(patience=3, min_delta=0.01)

    # Epoch 0: initial best
    assert es.step(0.50, epoch=0) is True
    assert es.best_score == 0.50
    assert es.best_epoch == 0
    assert es.counter == 0
    assert es.should_stop is False

    # Epoch 1: no improvement (needs < 0.49)
    assert es.step(0.495, epoch=1) is False
    assert es.counter == 1
    assert es.should_stop is False

    # Epoch 2: improvement (< 0.49)
    assert es.step(0.48, epoch=2) is True
    assert es.best_score == 0.48
    assert es.best_epoch == 2
    assert es.counter == 0

    # Epoch 3: worse
    assert es.step(0.49, epoch=3) is False
    assert es.counter == 1

    # Epoch 4: worse
    assert es.step(0.51, epoch=4) is False
    assert es.counter == 2

    # Epoch 5: worse -> triggers stop at patience 3
    assert es.step(0.50, epoch=5) is False
    assert es.counter == 3
    assert es.should_stop is True


def test_early_stopping_state_dict():
    es = EarlyStopping(patience=5, min_delta=0.005)
    es.step(0.40, epoch=1)
    es.step(0.45, epoch=2)

    state = es.state_dict()
    assert state["best_score"] == 0.40
    assert state["counter"] == 1

    es_loaded = EarlyStopping(patience=2)
    es_loaded.load_state_dict(state)
    assert es_loaded.best_score == 0.40
    assert es_loaded.counter == 1
    assert es_loaded.patience == 5
