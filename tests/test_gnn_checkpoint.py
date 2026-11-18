"""Unit tests for GNN checkpoint saving, restoration, and loading."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import torch.optim as optim

from src.data.graph_dataset import collate_graphs
from src.models import MolecularGNN
from src.training.checkpoint import (
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)
from src.training.config import TrainingConfig


@pytest.fixture
def sample_batch():
    graph = {
        "mol_id": "CKPT_MOL",
        "x": np.ones((4, 19), dtype=np.float32),
        "edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64),
        "edge_attr": np.ones((4, 18), dtype=np.float32),
        "y": np.zeros(12, dtype=np.float32),
        "mask": np.ones(12, dtype=bool),
    }
    return collate_graphs([graph])


def test_save_and_load_checkpoint(sample_batch):
    model = MolecularGNN(hidden_dim=32, num_layers=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainingConfig(hidden_dim=32, num_layers=2, seed=99)

    model.eval()
    with torch.no_grad():
        original_logits = model(sample_batch)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_gnn.pt"

        # Save checkpoint
        saved_path = save_checkpoint(
            filepath=ckpt_path,
            model=model,
            epoch=5,
            best_val_loss=0.3456,
            optimizer=optimizer,
            config=config,
        )
        assert saved_path.exists()

        # Load into a fresh model with different initial weights
        fresh_model = MolecularGNN(hidden_dim=32, num_layers=2)
        fresh_optimizer = optim.AdamW(fresh_model.parameters(), lr=1e-3)

        state = load_checkpoint(
            filepath=ckpt_path,
            model=fresh_model,
            optimizer=fresh_optimizer,
            device="cpu",
        )

        assert state["epoch"] == 5
        assert state["best_val_loss"] == 0.3456
        assert state["seed"] == 99
        assert len(state["endpoint_order"]) == 12

        # Verify predictions are identical
        fresh_model.eval()
        with torch.no_grad():
            restored_logits = fresh_model(sample_batch)

        assert torch.allclose(original_logits, restored_logits, atol=1e-6)


def test_load_model_from_checkpoint(sample_batch):
    model = MolecularGNN(hidden_dim=32, num_layers=2, dropout=0.2)
    config = TrainingConfig(hidden_dim=32, num_layers=2, dropout=0.2)

    model.eval()
    with torch.no_grad():
        original_logits = model(sample_batch)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "auto_loaded_gnn.pt"
        save_checkpoint(
            filepath=ckpt_path,
            model=model,
            epoch=12,
            best_val_loss=0.28,
            config=config,
        )

        # Use the Student 4 convenience factory
        loaded_model = load_model_from_checkpoint(ckpt_path, device="cpu")
        assert not loaded_model.training
        assert isinstance(loaded_model, MolecularGNN)

        with torch.no_grad():
            loaded_logits = loaded_model(sample_batch)

        assert torch.allclose(original_logits, loaded_logits, atol=1e-6)


def test_load_checkpoint_file_not_found():
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint("non_existent_checkpoint_file.pt")
