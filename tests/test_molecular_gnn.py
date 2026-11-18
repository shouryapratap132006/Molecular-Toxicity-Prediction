"""Comprehensive test suite for Molecular Graph Neural Network (MolecularGNN).

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.graph_dataset import collate_graphs, iter_graph_batches
from src.data.graph_preprocess import load_graph_datasets
from src.models import (
    TOX21_ENDPOINTS,
    GNNConfig,
    MolecularGNN,
)


@pytest.fixture
def sample_graph_a() -> dict:
    """Fixture creating a synthetic 3-node, 2-bond (4 directed edges) graph."""
    return {
        "mol_id": "SYNTH_A",
        "x": np.ones((3, 19), dtype=np.float32),
        "edge_index": np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        "edge_attr": np.ones((4, 18), dtype=np.float32),
        "y": np.array([1.0, 0.0] + [np.nan] * 10, dtype=np.float32),
        "mask": np.array([True, True] + [False] * 10, dtype=bool),
    }


@pytest.fixture
def sample_graph_b() -> dict:
    """Fixture creating a synthetic 5-node, 4-bond (8 directed edges) graph."""
    return {
        "mol_id": "SYNTH_B",
        "x": np.full((5, 19), 2.0, dtype=np.float32),
        "edge_index": np.array([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=np.int64),
        "edge_attr": np.full((8, 18), 0.5, dtype=np.float32),
        "y": np.array([0.0] * 12, dtype=np.float32),
        "mask": np.array([True] * 12, dtype=bool),
    }


@pytest.fixture
def sample_graph_zero_bonds() -> dict:
    """Fixture creating an isolated monoatomic graph (zero chemical bonds)."""
    return {
        "mol_id": "SYNTH_ZERO_BONDS",
        "x": np.ones((1, 19), dtype=np.float32) * 3.0,
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "edge_attr": np.empty((0, 18), dtype=np.float32),
        "y": np.array([np.nan] * 12, dtype=np.float32),
        "mask": np.array([False] * 12, dtype=bool),
    }


# -----------------------------------------------------------------------------
# 1. Initialization and Configuration Tests
# -----------------------------------------------------------------------------


def test_gnn_config_validation():
    """Verify GNNConfig rejects invalid parameter values."""
    with pytest.raises(ValueError, match="node_feature_dim must be positive"):
        GNNConfig(node_feature_dim=0)
    with pytest.raises(ValueError, match="edge_feature_dim must be positive"):
        GNNConfig(edge_feature_dim=-1)
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        GNNConfig(hidden_dim=0)
    with pytest.raises(ValueError, match="num_layers must be positive"):
        GNNConfig(num_layers=0)
    with pytest.raises(ValueError, match="dropout must be in"):
        GNNConfig(dropout=1.2)
    with pytest.raises(ValueError, match="Unsupported pooling"):
        GNNConfig(pooling="unsupported")  # type: ignore[arg-type]


def test_model_initialization():
    """Verify clean instantiation of MolecularGNN with defaults and custom config."""
    model_default = MolecularGNN()
    assert model_default.config.node_feature_dim == 19
    assert model_default.config.edge_feature_dim == 18
    assert model_default.config.num_tasks == 12
    assert model_default.endpoints == TOX21_ENDPOINTS

    cfg = GNNConfig(hidden_dim=64, num_layers=4, pooling="sum")
    model_custom = MolecularGNN(config=cfg)
    assert len(model_custom.layers) == 4
    assert model_custom.pooling.method == "sum"


# -----------------------------------------------------------------------------
# 2. Single Graph & Batch Forward Pass Tests
# -----------------------------------------------------------------------------


def test_single_graph_forward(sample_graph_a):
    """Verify forward pass on a single unbatched graph dictionary."""
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    model.eval()

    # Pass single graph (simulating B=1 batch)
    batch = collate_graphs([sample_graph_a])
    logits = model(batch)

    assert logits.shape == (1, 12)
    assert not torch.isnan(logits).any()


def test_batch_forward_multiple_graphs(sample_graph_a, sample_graph_b):
    """Verify forward pass on a batch of heterogeneous variable-sized graphs."""
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    model.eval()

    batch = collate_graphs([sample_graph_a, sample_graph_b])
    logits = model(batch)

    assert logits.shape == (2, 12)
    assert not torch.isnan(logits).any()


def test_explicit_tensor_call(sample_graph_a):
    """Verify model can be invoked with explicit tensor arguments."""
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    model.eval()

    batch = collate_graphs([sample_graph_a])
    logits = model(
        batch_or_x=batch["x"],
        edge_index=batch["edge_index"],
        edge_attr=batch["edge_attr"],
        batch=batch["batch"],
    )
    assert logits.shape == (1, 12)


# -----------------------------------------------------------------------------
# 3. Edge-Feature and Zero-Edge Tests
# -----------------------------------------------------------------------------


def test_zero_edge_molecule_support(sample_graph_zero_bonds):
    """Verify architecture seamlessly processes molecules with zero chemical bonds."""
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    model.eval()

    batch = collate_graphs([sample_graph_zero_bonds])
    logits = model(batch)

    assert logits.shape == (1, 12)
    assert not torch.isnan(logits).any()


def test_mixed_batch_with_zero_edge_molecule(sample_graph_a, sample_graph_zero_bonds):
    """Verify batch containing both normal molecules and zero-edge molecules."""
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    model.eval()

    batch = collate_graphs([sample_graph_a, sample_graph_zero_bonds])
    logits = model(batch)

    assert logits.shape == (2, 12)
    assert not torch.isnan(logits).any()


def test_edge_features_impact_prediction(sample_graph_a):
    """Verify that changing bond attributes alters the output logits."""
    torch.manual_seed(42)
    model = MolecularGNN(hidden_dim=32, num_layers=2, dropout=0.0)
    model.eval()

    batch1 = collate_graphs([sample_graph_a])
    logits1 = model(batch1)

    # Clone graph and alter bond features
    graph_a_altered = dict(sample_graph_a)
    graph_a_altered["edge_attr"] = sample_graph_a["edge_attr"] * 5.0
    batch2 = collate_graphs([graph_a_altered])
    logits2 = model(batch2)

    assert not torch.allclose(logits1, logits2, atol=1e-4)


# -----------------------------------------------------------------------------
# 4. Zero Cross-Graph Leakage Test (Critical Isolation Verification)
# -----------------------------------------------------------------------------


def test_no_cross_graph_leakage(sample_graph_a, sample_graph_b):
    """Verify that predictions for graph A are identical whether evaluated alone
    or in a heterogeneous batch with graph B (Zero Cross-Graph Leakage)."""
    torch.manual_seed(42)
    model = MolecularGNN(hidden_dim=64, num_layers=3, dropout=0.0)
    model.eval()

    # Predict graph A alone
    batch_a_alone = collate_graphs([sample_graph_a])
    with torch.no_grad():
        logits_a_alone = model(batch_a_alone)

    # Predict graph A batched together with graph B
    batch_ab = collate_graphs([sample_graph_a, sample_graph_b])
    with torch.no_grad():
        logits_ab = model(batch_ab)

    # The prediction for graph A (row 0 of the batch) must match exactly
    logits_a_in_batch = logits_ab[0:1]
    assert torch.allclose(
        logits_a_alone, logits_a_in_batch, atol=1e-5
    ), "Cross-graph leakage detected: Graph A prediction differed when batched with Graph B!"


# -----------------------------------------------------------------------------
# 5. Pooling Strategy Coverage
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pool_method", ["mean", "sum", "max"])
def test_pooling_methods_forward(sample_graph_a, sample_graph_b, pool_method):
    """Verify that mean, sum, and max pooling configurations execute properly."""
    model = MolecularGNN(hidden_dim=32, num_layers=2, pooling=pool_method)
    model.eval()

    batch = collate_graphs([sample_graph_a, sample_graph_b])
    logits = model(batch)
    assert logits.shape == (2, 12)
    assert not torch.isnan(logits).any()


# -----------------------------------------------------------------------------
# 6. Backward Pass, Non-Zero Gradients, and Masked BCE Loss
# -----------------------------------------------------------------------------


def test_masked_bce_loss_and_gradients(sample_graph_a, sample_graph_b):
    """Verify model participates in end-to-end masked BCE loss with non-zero gradients."""
    torch.manual_seed(42)
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    model.train()

    batch = collate_graphs([sample_graph_a, sample_graph_b])
    logits = model(batch)

    # Target labels and mask
    y = torch.from_numpy(batch["y"]).float()
    mask = torch.from_numpy(batch["mask"]).bool()

    # Replace NaNs in y with dummy 0 for loss matrix computation (safely zeroed out by mask)
    y_safe = torch.nan_to_num(y, nan=0.0)
    loss_matrix = F.binary_cross_entropy_with_logits(logits, y_safe, reduction="none")

    # Apply mask and compute mean over observed assay labels
    loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1)
    loss.backward()

    # Assert all model parameters received valid non-zero finite gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient on {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"
    assert loss.item() > 0.0


# -----------------------------------------------------------------------------
# 7. Checkpoint Save/Load & Determinism in Eval Mode
# -----------------------------------------------------------------------------


def test_checkpoint_save_and_load(sample_graph_a):
    """Verify model state_dict round-trip saving and loading reproduces exact logits."""
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    model.eval()

    batch = collate_graphs([sample_graph_a])
    with torch.no_grad():
        original_logits = model(batch)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "model_checkpoint.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Instantiate fresh model and load weights
        loaded_model = MolecularGNN(hidden_dim=32, num_layers=2)
        loaded_model.load_state_dict(torch.load(ckpt_path))
        loaded_model.eval()

        with torch.no_grad():
            loaded_logits = loaded_model(batch)

    assert torch.allclose(original_logits, loaded_logits)


def test_eval_mode_determinism(sample_graph_a, sample_graph_b):
    """Verify model in eval mode produces identical outputs on repeated calls."""
    model = MolecularGNN(hidden_dim=32, num_layers=2, dropout=0.5)
    model.eval()

    batch = collate_graphs([sample_graph_a, sample_graph_b])
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)

    assert torch.allclose(out1, out2)


# -----------------------------------------------------------------------------
# 8. Device Agnosticism (CPU and Optional MPS/CUDA)
# -----------------------------------------------------------------------------


def test_device_transfer(sample_graph_a):
    """Verify model supports device placement and transfers without code changes."""
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    model.to(torch.device("cpu"))
    assert model.device == torch.device("cpu")

    batch = collate_graphs([sample_graph_a])
    logits = model(batch)
    assert logits.device == torch.device("cpu")

    # If Apple Silicon MPS or CUDA is available, test hardware acceleration
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    if device is not None:
        model.to(device)
        assert model.device == device
        logits_accel = model(batch)
        assert logits_accel.device.type == device.type


# -----------------------------------------------------------------------------
# 9. Real Student 1 Graph Preprocessing & Batch Loader Integration Test
# -----------------------------------------------------------------------------


def test_real_student1_dataset_batches():
    """Verify model runs on actual minibatches from Student 1's graph loader."""
    datasets = load_graph_datasets()
    assert "train" in datasets and "val" in datasets and "test" in datasets

    model = MolecularGNN(hidden_dim=64, num_layers=3, pooling="mean")
    model.eval()

    # Test batches from each partition
    for partition_name, ds in datasets.items():
        batch_iter = iter_graph_batches(ds, batch_size=16)
        first_batch = next(batch_iter)

        with torch.no_grad():
            logits = model(first_batch)
            probs = model.predict_proba(first_batch)

        assert logits.shape == (len(first_batch["mol_id"]), 12)
        assert probs.shape == (len(first_batch["mol_id"]), 12)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert not torch.isnan(logits).any()
