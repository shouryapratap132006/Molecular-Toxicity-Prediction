"""Configuration dataclass for Molecular Graph Neural Network (GNN).

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GNNConfig:
    """Hyperparameter configuration for MolecularGNN architecture.

    Attributes:
        node_feature_dim: Dimensionality of input atom feature vectors (default 19).
        edge_feature_dim: Dimensionality of input bond feature vectors (default 18).
        hidden_dim: Hidden dimension for node and edge embeddings across layers.
        num_layers: Number of edge-aware message passing layers.
        num_tasks: Number of toxicity prediction endpoints (12 for Tox21).
        dropout: Dropout probability applied after message passing and in prediction head.
        pooling: Graph-level readout pooling strategy ('mean', 'sum', 'max').
        residual: Whether to use residual skip-connections across message passing layers.
        norm: Normalization type applied to node representations ('layer', 'batch', 'none').
        activation: Non-linear activation function ('relu', 'gelu', 'silu', 'leaky_relu').
    """

    node_feature_dim: int = 19
    edge_feature_dim: int = 18
    hidden_dim: int = 128
    num_layers: int = 3
    num_tasks: int = 12
    dropout: float = 0.1
    pooling: Literal["mean", "sum", "max"] = "mean"
    residual: bool = True
    norm: Literal["layer", "batch", "none"] = "layer"
    activation: Literal["relu", "gelu", "silu", "leaky_relu"] = "relu"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.node_feature_dim <= 0:
            raise ValueError(f"node_feature_dim must be positive, got {self.node_feature_dim}")
        if self.edge_feature_dim <= 0:
            raise ValueError(f"edge_feature_dim must be positive, got {self.edge_feature_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {self.num_tasks}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.pooling not in ("mean", "sum", "max"):
            raise ValueError(f"Unsupported pooling: '{self.pooling}'. Choose 'mean', 'sum', or 'max'.")
        if self.norm not in ("layer", "batch", "none"):
            raise ValueError(f"Unsupported norm: '{self.norm}'. Choose 'layer', 'batch', or 'none'.")
        if self.activation not in ("relu", "gelu", "silu", "leaky_relu"):
            raise ValueError(f"Unsupported activation: '{self.activation}'.")
