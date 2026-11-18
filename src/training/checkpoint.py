"""Checkpoint serialization, restoration, and model loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.models.molecular_gnn import TOX21_ENDPOINTS, MolecularGNN
from src.training.config import TrainingConfig
from src.training.reproducibility import get_system_info

logger = logging.getLogger(__name__)


def save_checkpoint(
    filepath: Optional[Path | str] = None,
    model: Optional[nn.Module] = None,
    epoch: int = 0,
    best_val_loss: Optional[float] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    config: Optional[TrainingConfig] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
    path: Optional[Path | str] = None,
    val_loss: Optional[float] = None,
) -> Path:
    """Save training state and model weights to a PyTorch checkpoint."""
    actual_path = filepath if filepath is not None else path
    if actual_path is None:
        raise ValueError("Either filepath or path must be provided to save_checkpoint.")
    if model is None:
        raise ValueError("model must be provided to save_checkpoint.")

    actual_val_loss = best_val_loss if best_val_loss is not None else (val_loss if val_loss is not None else float("inf"))

    path_obj = Path(actual_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    model_config_dict: Optional[dict[str, Any]] = None
    if hasattr(model, "config") and model.config is not None:
        if hasattr(model.config, "to_dict"):
            model_config_dict = model.config.to_dict()
        elif hasattr(model.config, "__dict__"):
            model_config_dict = vars(model.config)

    endpoint_order = list(getattr(model, "endpoints", TOX21_ENDPOINTS))

    checkpoint = {
        "epoch": epoch,
        "best_val_loss": float(actual_val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config_dict,
        "training_config": config.to_dict() if config is not None else None,
        "seed": config.seed if config is not None else 42,
        "endpoint_order": endpoint_order,
        "system_info": get_system_info(),
        "extra_metadata": extra_metadata or {},
    }

    torch.save(checkpoint, path_obj)
    logger.info("Saved model checkpoint to %s (epoch %d, val_loss %.5f)", path_obj, epoch, actual_val_loss)
    return path_obj


def load_checkpoint(
    filepath: Path | str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint file and optionally restore model, optimizer, and scheduler states.

    Args:
        filepath: Path to saved checkpoint file.
        model: Optional model to restore weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Target device for loading weights.

    Returns:
        Loaded state dictionary.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    map_loc = torch.device(device) if isinstance(device, str) else device
    try:
        state = torch.load(path, map_location=map_loc, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions that don't accept weights_only
        state = torch.load(path, map_location=map_loc)

    if model is not None:
        model.load_state_dict(state["model_state_dict"])
        model.to(map_loc)
        logger.info("Restored model weights from %s", path)

    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return state


def load_model_from_checkpoint(
    filepath: Path | str,
    device: str | torch.device = "cpu",
) -> MolecularGNN:
    """Instantiate a MolecularGNN model directly from a checkpoint file.

    Student 4 Integration Contract:
    Provides an out-of-the-box evaluation-ready model from the saved checkpoint.

    Args:
        filepath: Path to saved checkpoint.
        device: Device to place the restored model on.

    Returns:
        MolecularGNN model placed in eval() mode.
    """
    map_loc = torch.device(device) if isinstance(device, str) else device
    state = load_checkpoint(filepath, device=map_loc)

    # Reconstruct configuration
    train_cfg_dict = state.get("training_config")
    if train_cfg_dict is not None:
        cfg = TrainingConfig.from_dict(train_cfg_dict)
        model = MolecularGNN(**cfg.to_model_kwargs())
    else:
        # Fallback to model_config dict
        model_cfg_dict = state.get("model_config", {})
        model = MolecularGNN(
            hidden_dim=model_cfg_dict.get("hidden_dim", 128),
            num_layers=model_cfg_dict.get("num_layers", 3),
            pooling=model_cfg_dict.get("pooling", "mean"),
            dropout=model_cfg_dict.get("dropout", 0.1),
        )

    model.load_state_dict(state["model_state_dict"])
    model.to(map_loc)
    model.eval()
    return model
