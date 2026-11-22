"""End-to-End Integration Tests for Phase 2 GNN Workflow.

Student 5 Module: Integration, Experiments, Final IEEE Report & Presentation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.graph_dataset import collate_graphs, iter_graph_batches
from src.data.graph_preprocess import load_graph_datasets
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_registry import ExperimentRecord, ExperimentRegistry
from src.experiments.run_experiment import execute_experiment
from src.models import TOX21_ENDPOINTS, MolecularGNN
from src.training.config import TrainingConfig
from src.training.losses import MaskedBCELoss
from src.training.reproducibility import set_seed
from src.training.trainer import evaluate, train_one_epoch


@pytest.fixture
def synthetic_graph_batch() -> dict:
    """Fixture providing a deterministic 2-graph synthetic batch."""
    g1 = {
        "mol_id": "SYNTH_1",
        "x": np.ones((4, 19), dtype=np.float32),
        "edge_index": np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64),
        "edge_attr": np.ones((6, 18), dtype=np.float32),
        "y": np.array([1.0, 0.0] + [np.nan] * 10, dtype=np.float32),
        "mask": np.array([True, True] + [False] * 10, dtype=bool),
    }
    g2 = {
        "mol_id": "SYNTH_2",
        "x": np.full((3, 19), 2.0, dtype=np.float32),
        "edge_index": np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        "edge_attr": np.full((4, 18), 0.5, dtype=np.float32),
        "y": np.array([0.0] * 12, dtype=np.float32),
        "mask": np.array([True] * 12, dtype=bool),
    }
    return collate_graphs([g1, g2])


def test_pipeline_data_to_model_forward(synthetic_graph_batch):
    """Verify Student 1 batch feeds directly into Student 2 MolecularGNN."""
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    model.eval()

    with torch.no_grad():
        logits = model(synthetic_graph_batch)

    assert logits.shape == (2, 12), f"Expected [2, 12], got {logits.shape}"
    assert not torch.isnan(logits).any(), "Found NaNs in GNN forward output"


def test_single_training_and_validation_step(synthetic_graph_batch):
    """Verify Student 3 train_one_epoch and evaluate functions run without error."""
    set_seed(42)
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = MaskedBCELoss()
    device = torch.device("cpu")

    # Mock dataloaders yielding the synthetic batch
    mock_loader = [synthetic_graph_batch]

    train_loss, train_count = train_one_epoch(
        model=model,
        dataloader=mock_loader,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        gradient_clip=1.0,
    )
    assert train_loss > 0.0
    assert train_count > 0
    assert not np.isnan(train_loss)

    val_loss, val_count = evaluate(
        model=model,
        dataloader=mock_loader,
        device=device,
        loss_fn=loss_fn,
    )
    assert val_loss > 0.0
    assert val_count > 0
    assert not np.isnan(val_loss)


def test_experiment_config_roundtrip():
    """Verify ExperimentConfig YAML serialization, deserialization, and conversion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = Path(tmp_dir) / "test_config.yaml"
        cfg = ExperimentConfig(
            experiment_id="test_exp",
            hidden_dim=64,
            num_layers=2,
            pooling="sum",
            tags=["unit_test"],
        )
        cfg.save_yaml(yaml_path)

        loaded_cfg = ExperimentConfig.load_yaml(yaml_path)
        assert loaded_cfg.experiment_id == "test_exp"
        assert loaded_cfg.hidden_dim == 64
        assert loaded_cfg.num_layers == 2
        assert loaded_cfg.pooling == "sum"
        assert loaded_cfg.tags == ["unit_test"]

        # Verify conversion to Student 3 TrainingConfig
        t_cfg = loaded_cfg.to_training_config()
        assert t_cfg.hidden_dim == 64
        assert t_cfg.num_layers == 2
        assert t_cfg.pooling == "sum"


def test_experiment_registry_persistence():
    """Verify ExperimentRegistry registers, persists, and lists experiment runs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_path = Path(tmp_dir) / "registry.json"
        registry = ExperimentRegistry(reg_path)

        rec = ExperimentRecord(
            experiment_id="exp_01",
            status="SUCCESS",
            config={"hidden_dim": 64},
            seed=42,
            git_commit="abcdef",
            started_at_utc="2026-11-22T00:00:00Z",
            completed_at_utc="2026-11-22T00:01:00Z",
            runtime_seconds=60.0,
            device="cpu",
            best_epoch=5,
            best_val_loss=0.25,
        )
        registry.register(rec)

        assert "exp_01" in registry.list_experiments()
        retrieved = registry.get("exp_01")
        assert retrieved is not None
        assert retrieved["best_val_loss"] == 0.25

        # Check summary table
        rows = registry.get_summary_table()
        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "exp_01"


def test_controlled_reproducibility():
    """Verify identical seeds produce identical training loss trajectories."""
    def run_mini_training(seed: int) -> float:
        set_seed(seed)
        model = MolecularGNN(hidden_dim=16, num_layers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = MaskedBCELoss()

        # Deterministic dummy batch
        torch.manual_seed(seed)
        x = torch.randn(6, 19)
        e_idx = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        e_attr = torch.randn(3, 18)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        y = torch.tensor([[1.0] * 12, [0.0] * 12], dtype=torch.float32)
        mask = torch.tensor([[True] * 12, [True] * 12], dtype=torch.bool)

        logits = model(batch_or_x=x, edge_index=e_idx, edge_attr=e_attr, batch=batch)
        loss = loss_fn(logits, y, mask)
        loss.backward()
        opt.step()
        return loss.item()

    loss_run1 = run_mini_training(42)
    loss_run2 = run_mini_training(42)
    assert abs(loss_run1 - loss_run2) < 1e-6, "Seed 42 failed exact reproducibility"


def test_endpoint_consistency():
    """Verify endpoint ordering across Phase 1 and Phase 2 matches canonical standard."""
    from src.data.common import load_config
    cfg = load_config()
    cfg_endpoints = list(cfg["endpoints"])

    assert cfg_endpoints == list(TOX21_ENDPOINTS), "Endpoint ordering diverged from Phase 1 config"
