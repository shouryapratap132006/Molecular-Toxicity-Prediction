"""Tests for edge-aware molecular message passing layers."""

import pytest
import torch

from src.models.message_passing import EdgeAwareMessagePassingLayer


def test_message_passing_shapes():
    """Verify input-output shapes through message passing."""
    layer = EdgeAwareMessagePassingLayer(hidden_dim=32, activation="relu", dropout=0.0)
    h = torch.randn(6, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 32)

    out = layer(h, edge_index, edge_attr)
    assert out.shape == (6, 32)


def test_message_passing_empty_edges():
    """Verify message passing handles molecules with zero bonds gracefully."""
    layer = EdgeAwareMessagePassingLayer(hidden_dim=32, activation="relu", dropout=0.0)
    h = torch.randn(4, 32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 32))

    out = layer(h, edge_index, edge_attr)
    assert out.shape == (4, 32)
    assert not torch.isnan(out).any()


def test_message_passing_edge_features_matter():
    """Verify that different edge features produce different outputs."""
    torch.manual_seed(42)
    layer = EdgeAwareMessagePassingLayer(hidden_dim=32, activation="relu", dropout=0.0)
    layer.eval()

    h = torch.randn(3, 32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr1 = torch.zeros(2, 32)
    edge_attr2 = torch.ones(2, 32) * 5.0

    out1 = layer(h, edge_index, edge_attr1)
    out2 = layer(h, edge_index, edge_attr2)

    # Outputs on target nodes (1 and 2) must differ
    assert not torch.allclose(out1[1], out2[1])
    assert not torch.allclose(out1[2], out2[2])


def test_message_passing_backpropagation():
    """Verify gradients propagate through node and edge paths."""
    layer = EdgeAwareMessagePassingLayer(hidden_dim=16, activation="relu", dropout=0.0)
    h = torch.randn(4, 16, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 16, requires_grad=True)

    out = layer(h, edge_index, edge_attr)
    loss = out.sum()
    loss.backward()

    assert h.grad is not None
    assert edge_attr.grad is not None
    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(edge_attr.grad).all()
