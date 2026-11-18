"""Centralized training configuration for Phase 2 MolecularGNN."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass
class TrainingConfig:
    """Configuration dataclass for GNN training, optimization, and evaluation.

    Attributes:
        seed: Random seed for reproducibility across random, numpy, and torch.
        batch_size: Number of molecular graphs per training minibatch.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for AdamW optimizer.
        weight_decay: Weight decay (L2 penalty) for optimizer.
        hidden_dim: Hidden dimension for GNN node and edge embeddings.
        num_layers: Number of edge-aware message passing layers.
        dropout: Dropout rate applied in message passing and prediction head.
        early_stopping_patience: Number of epochs without validation loss improvement before stopping.
        early_stopping_min_delta: Minimum validation loss decrease to qualify as improvement.
        scheduler: Learning rate scheduler name ("ReduceLROnPlateau" or "none").
        scheduler_factor: Multiplicative factor for learning rate reduction.
        scheduler_patience: Epochs with no improvement before reducing learning rate.
        min_lr: Minimum allowable learning rate.
        gradient_clip: Maximum gradient norm for gradient clipping.
        use_pos_weights: Whether to use positive class weighting in BCE loss.
        max_pos_weight: Upper bound clamp for positive class weight to prevent instability.
        pooling: Graph-level readout pooling method ("mean", "sum", "max").
        activation: Non-linear activation function ("relu", "gelu", "leaky_relu", "silu").
        norm: Normalization layer ("layer", "batch", "none").
        residual: Whether to use residual skip connections in message passing layers.
        num_tasks: Number of toxicity endpoints (default 12 for Tox21).
        device: Device preference ("auto", "cuda", "mps", "cpu").
        checkpoint_dir: Directory where model checkpoints are saved.
        log_dir: Directory where training history logs are saved.
        plot_dir: Directory where training curves are saved.
        extra_metadata: Optional dictionary of additional experiment metadata.
    """

    seed: int = 42
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    scheduler: str = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    use_pos_weights: bool = False
    max_pos_weight: float = 50.0
    pooling: str = "mean"
    activation: str = "relu"
    norm: str = "layer"
    residual: bool = True
    num_tasks: int = 12
    device: str = "auto"
    checkpoint_dir: str = "results/phase2/checkpoints"
    log_dir: str = "results/phase2/logs"
    plot_dir: str = "results/phase2/plots"
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingConfig:
        """Create configuration instance from dictionary, ignoring unexpected keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_json(self, path: Path | str) -> None:
        """Serialize configuration to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: Path | str) -> TrainingConfig:
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_model_kwargs(self) -> dict[str, Any]:
        """Extract arguments required for initializing MolecularGNN."""
        return {
            "node_feature_dim": 19,
            "edge_feature_dim": 18,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_tasks": self.num_tasks,
            "pooling": self.pooling,
            "dropout": self.dropout,
            "residual": self.residual,
            "norm": self.norm,
            "activation": self.activation,
        }
