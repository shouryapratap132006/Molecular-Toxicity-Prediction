"""Integration tests for Phase 2 GNN training pipeline.

Verifies end-to-end training execution, checkpoint creation, logging,
curve generation, early stopping, and Student 4 model evaluation contract.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from src.data.graph_dataset import GraphDataset, collate_graphs
from src.training.checkpoint import load_checkpoint, load_model_from_checkpoint
from src.training.config import TrainingConfig
from src.training.train import parse_args, run_training


def _make_dummy_dataset(num_graphs: int = 12) -> GraphDataset:
    """Helper to generate small in-memory synthetic graph dataset."""
    graphs = []
    endpoints = [f"task_{i}" for i in range(12)]
    for i in range(num_graphs):
        num_atoms = 5 + (i % 4)
        x = np.random.randn(num_atoms, 19).astype(np.float32)

        # Ring edges
        src = list(range(num_atoms))
        dst = [(k + 1) % num_atoms for k in src]
        edge_index = np.array([src + dst, dst + src], dtype=np.int64)
        edge_attr = np.random.randn(edge_index.shape[1], 18).astype(np.float32)

        # 12 Tox21 tasks with occasional NaN
        y = np.random.choice([0.0, 1.0, np.nan], size=(12,), p=[0.7, 0.2, 0.1]).astype(np.float32)
        mask = np.isfinite(y).astype(bool)

        graphs.append({
            "mol_id": f"dummy_{i}",
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": y,
            "mask": mask,
            "num_nodes": num_atoms,
            "num_edges": edge_index.shape[1],
        })

    return GraphDataset(graphs, endpoints)


def test_end_to_end_training_synthetic(tmp_path: Path):
    """Verify complete training loop runs and creates all expected artifacts."""
    train_data = _make_dummy_dataset(num_graphs=16)
    val_data = _make_dummy_dataset(num_graphs=8)
    datasets = {"train": train_data, "val": val_data}

    ckpt_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    plot_dir = tmp_path / "plots"

    config = TrainingConfig(
        seed=123,
        batch_size=4,
        epochs=3,
        learning_rate=1e-3,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        early_stopping_patience=5,
        device="cpu",
        checkpoint_dir=str(ckpt_dir),
        log_dir=str(log_dir),
        plot_dir=str(plot_dir),
    )

    summary = run_training(config=config, datasets=datasets)

    # 1. Summary checks
    assert summary["total_epochs_trained"] == 3
    assert summary["best_epoch"] in [1, 2, 3]
    assert np.isfinite(summary["best_val_loss"])
    assert np.isfinite(summary["final_train_loss"])
    assert summary["stopped_early"] is False

    # 2. Checkpoint files exist and loadable
    best_ckpt = Path(summary["checkpoints"]["best"])
    assert best_ckpt.is_file()
    last_ckpt = Path(summary["checkpoints"]["last"])
    assert last_ckpt.is_file()

    ckpt_dict = load_checkpoint(best_ckpt)
    assert "model_state_dict" in ckpt_dict
    assert ckpt_dict["epoch"] == summary["best_epoch"]

    # 3. Log files exist and valid
    csv_file = Path(summary["logs"]["history_csv"])
    assert csv_file.is_file()
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3
        assert reader[0]["epoch"] == "1"

    config_json = Path(summary["logs"]["config_json"])
    assert config_json.is_file()
    with open(config_json, mode="r", encoding="utf-8") as f:
        loaded_cfg = json.load(f)
        assert loaded_cfg["hidden_dim"] == 16

    summary_json = log_dir / "training_summary.json"
    assert summary_json.is_file()

    # 4. Plots generated
    curve_plot = Path(summary["plots"]["curves_png"])
    assert curve_plot.is_file()
    assert curve_plot.stat().st_size > 0


def test_student4_evaluation_contract(tmp_path: Path):
    """Verify Student 4 can load the checkpoint and run inference on test batches."""
    train_data = _make_dummy_dataset(num_graphs=8)
    val_data = _make_dummy_dataset(num_graphs=4)
    datasets = {"train": train_data, "val": val_data}

    config = TrainingConfig(
        batch_size=4,
        epochs=2,
        hidden_dim=32,
        num_layers=2,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        plot_dir=str(tmp_path / "plots"),
    )

    summary = run_training(config=config, datasets=datasets)
    best_ckpt = summary["checkpoints"]["best"]

    # Student 4 contract: load model from checkpoint
    model = load_model_from_checkpoint(best_ckpt, device=torch.device("cpu"))
    model.eval()

    # Simulate evaluation on test batch
    test_batch = collate_graphs([val_data[0], val_data[1], val_data[2]])
    with torch.no_grad():
        logits = model(test_batch)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (3, 12)
    assert torch.all(torch.isfinite(logits)), "Inference output must not contain NaNs or Infs"

    # Probabilities via sigmoid
    probs = torch.sigmoid(logits)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))


def test_early_stopping_triggered(tmp_path: Path):
    """Verify early stopping terminates training when validation loss stops improving."""
    train_data = _make_dummy_dataset(num_graphs=8)
    val_data = _make_dummy_dataset(num_graphs=4)
    datasets = {"train": train_data, "val": val_data}

    # High patience threshold with tiny min_delta and intentionally high patience = 2
    config = TrainingConfig(
        epochs=15,
        early_stopping_patience=2,
        early_stopping_min_delta=10.0,  # Unattainable improvement delta guarantees trigger
        batch_size=4,
        hidden_dim=16,
        num_layers=2,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        plot_dir=str(tmp_path / "plots"),
    )

    summary = run_training(config=config, datasets=datasets)
    assert summary["stopped_early"] is True
    assert summary["total_epochs_trained"] <= 4, "Early stopping should have stopped by epoch 3 or 4"


def test_cli_argument_parsing():
    """Verify command line argument parser correctly constructs arguments."""
    args = parse_args([
        "--epochs", "25",
        "--batch-size", "64",
        "--lr", "0.0005",
        "--hidden-dim", "64",
        "--num-layers", "4",
        "--pooling", "sum",
        "--use-pos-weights",
        "--seed", "99",
    ])
    assert args.epochs == 25
    assert args.batch_size == 64
    assert args.lr == 0.0005
    assert args.hidden_dim == 64
    assert args.num_layers == 4
    assert args.pooling == "sum"
    assert args.use_pos_weights is True
    assert args.seed == 99
