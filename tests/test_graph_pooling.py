"""Tests for graph-level readout pooling layers."""

import pytest
import torch

from src.models.graph_pooling import GlobalGraphPooling


def test_pooling_initialization_invalid():
    """Verify invalid pooling strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported pooling"):
        GlobalGraphPooling(method="invalid_pooling")  # type: ignore[arg-type]


def test_pooling_sum_mean_max_numerical():
    """Verify exact mathematical aggregation of sum, mean, and max pooling."""
    # 3 graphs: g0 has 2 nodes, g1 has 3 nodes, g2 has 1 node
    h = torch.tensor([
        [1.0, 2.0],    # g0 node 0
        [3.0, 4.0],    # g0 node 1
        [10.0, 20.0],  # g1 node 0
        [30.0, 40.0],  # g1 node 1
        [50.0, 60.0],  # g1 node 2
        [100.0, 200.0] # g2 node 0
    ])
    batch = torch.tensor([0, 0, 1, 1, 1, 2])

    # Sum
    pool_sum = GlobalGraphPooling("sum")
    out_sum = pool_sum(h, batch)
    assert out_sum.shape == (3, 2)
    assert torch.allclose(out_sum[0], torch.tensor([4.0, 6.0]))
    assert torch.allclose(out_sum[1], torch.tensor([90.0, 120.0]))
    assert torch.allclose(out_sum[2], torch.tensor([100.0, 200.0]))

    # Mean
    pool_mean = GlobalGraphPooling("mean")
    out_mean = pool_mean(h, batch)
    assert out_mean.shape == (3, 2)
    assert torch.allclose(out_mean[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(out_mean[1], torch.tensor([30.0, 40.0]))
    assert torch.allclose(out_mean[2], torch.tensor([100.0, 200.0]))

    # Max
    pool_max = GlobalGraphPooling("max")
    out_max = pool_max(h, batch)
    assert out_max.shape == (3, 2)
    assert torch.allclose(out_max[0], torch.tensor([3.0, 4.0]))
    assert torch.allclose(out_max[1], torch.tensor([50.0, 60.0]))
    assert torch.allclose(out_max[2], torch.tensor([100.0, 200.0]))


def test_pooling_single_graph_default_batch():
    """Verify pooling when batch tensor is None (defaults to B=1)."""
    h = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pool_mean = GlobalGraphPooling("mean")
    out = pool_mean(h, batch=None)
    assert out.shape == (1, 2)
    assert torch.allclose(out[0], torch.tensor([2.0, 3.0]))


def test_pooling_empty_nodes():
    """Verify pooling handles empty node tensors."""
    h = torch.empty((0, 16))
    pool = GlobalGraphPooling("mean")
    out = pool(h, num_graphs=3)
    assert out.shape == (3, 16)
    assert torch.allclose(out, torch.zeros((3, 16)))
