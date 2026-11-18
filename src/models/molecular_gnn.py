"""Modular Multi-Task Molecular Graph Neural Network (MolecularGNN).

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.

This module implements the complete end-to-end edge-aware Message Passing
Neural Network (MPNN) architecture for predicting all 12 Tox21 toxicity endpoints.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.encoders import EdgeEncoder, NodeEncoder, get_activation
from src.models.graph_pooling import GlobalGraphPooling
from src.models.message_passing import EdgeAwareMessagePassingLayer
from src.models.model_config import GNNConfig

# Canonical Phase 1 & Phase 2 Tox21 endpoints in authoritative order
TOX21_ENDPOINTS: tuple[str, ...] = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


class MultiTaskPredictionHead(nn.Module):
    """Prediction head mapping pooled graph embeddings to 12 endpoint logits.

    Input shape:  [batch_size, hidden_dim]
    Output shape: [batch_size, num_tasks] (raw logits, no sigmoid)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_tasks: int = 12,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        """Compute task logits for each molecule."""
        if h_graph.dim() != 2 or h_graph.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected pooled graph embeddings of shape [B, {self.hidden_dim}], got {h_graph.shape}"
            )
        return self.head(h_graph)


class MolecularGNN(nn.Module):
    """Edge-Aware Multi-Task Molecular Graph Neural Network.

    Architecture pipeline:
        Raw Graph (Atoms x, Bonds edge_attr, Connectivity edge_index)
            ↓
        [NodeEncoder & EdgeEncoder] -> [N, hidden_dim], [E, hidden_dim]
            ↓
        L x [EdgeAwareMessagePassingLayer] -> [N, hidden_dim]
            ↓
        [GlobalGraphPooling] (mean / sum / max) -> [B, hidden_dim]
            ↓
        [MultiTaskPredictionHead] -> [B, 12] (Raw Logits)
    """

    endpoints: tuple[str, ...] = TOX21_ENDPOINTS

    def __init__(
        self,
        node_feature_dim: int = 19,
        edge_feature_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_tasks: int = 12,
        pooling: str = "mean",
        dropout: float = 0.1,
        residual: bool = True,
        norm: str = "layer",
        activation: str = "relu",
        config: Optional[GNNConfig] = None,
    ) -> None:
        super().__init__()

        # Resolve configuration: either explicit dataclass or kwargs
        if config is not None:
            self.config = config
        else:
            self.config = GNNConfig(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_tasks=num_tasks,
                pooling=pooling,  # type: ignore[arg-type]
                dropout=dropout,
                residual=residual,
                norm=norm,  # type: ignore[arg-type]
                activation=activation,  # type: ignore[arg-type]
            )

        cfg = self.config

        # 1. Feature Encoders
        self.node_encoder = NodeEncoder(
            in_dim=cfg.node_feature_dim,
            hidden_dim=cfg.hidden_dim,
            activation=cfg.activation,
            norm=cfg.norm,
        )
        self.edge_encoder = EdgeEncoder(
            in_dim=cfg.edge_feature_dim,
            hidden_dim=cfg.hidden_dim,
            activation=cfg.activation,
            norm=cfg.norm,
        )

        # 2. Message Passing Stack
        self.layers = nn.ModuleList(
            [
                EdgeAwareMessagePassingLayer(
                    hidden_dim=cfg.hidden_dim,
                    activation=cfg.activation,
                    dropout=cfg.dropout,
                    residual=cfg.residual,
                    norm=cfg.norm,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        # 3. Readout Graph Pooling
        self.pooling = GlobalGraphPooling(method=cfg.pooling)

        # 4. Multi-Task Prediction Head (12 Tox21 logits)
        self.prediction_head = MultiTaskPredictionHead(
            hidden_dim=cfg.hidden_dim,
            num_tasks=cfg.num_tasks,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )

    @property
    def device(self) -> torch.device:
        """Device on which the model parameters reside."""
        return next(self.parameters()).device

    def _prepare_tensor(
        self,
        arr: Union[torch.Tensor, np.ndarray],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert input array or tensor and transfer to model device."""
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr)
        elif isinstance(arr, torch.Tensor):
            t = arr
        else:
            t = torch.tensor(arr)
        return t.to(dtype=dtype, device=self.device)

    def forward(
        self,
        batch_or_x: Union[Mapping[str, Any], torch.Tensor, np.ndarray],
        edge_index: Optional[Union[torch.Tensor, np.ndarray]] = None,
        edge_attr: Optional[Union[torch.Tensor, np.ndarray]] = None,
        batch: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass through the molecular graph neural network.

        Accepts either:
        1. A graph batch dictionary (from Student 1's collate_graphs):
           model(batch_dict)
        2. Explicit tensor arguments:
           model(x, edge_index, edge_attr, batch)

        Returns:
            torch.Tensor of shape [batch_size, 12] containing raw prediction logits.
        """
        # Case 1: Dictionary input (Student 1's batch format)
        if isinstance(batch_or_x, Mapping):
            x_raw = batch_or_x["x"]
            edge_index_raw = batch_or_x["edge_index"]
            edge_attr_raw = batch_or_x["edge_attr"]
            batch_raw = batch_or_x.get("batch")
            if num_graphs is None and "ptr" in batch_or_x:
                num_graphs = len(batch_or_x["ptr"]) - 1
            elif num_graphs is None and "mol_id" in batch_or_x:
                num_graphs = len(batch_or_x["mol_id"])
        else:
            # Case 2: Explicit tensor arguments
            x_raw = batch_or_x
            edge_index_raw = edge_index
            edge_attr_raw = edge_attr
            batch_raw = batch

        if edge_index_raw is None:
            raise ValueError("edge_index must be provided.")
        if edge_attr_raw is None:
            raise ValueError("edge_attr must be provided.")

        # Prepare tensors with correct dtypes and device
        x = self._prepare_tensor(x_raw, torch.float32)
        edge_index_t = self._prepare_tensor(edge_index_raw, torch.int64)
        edge_attr_t = self._prepare_tensor(edge_attr_raw, torch.float32)

        batch_t = (
            self._prepare_tensor(batch_raw, torch.int64)
            if batch_raw is not None
            else None
        )

        # 1. Feature encoding
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr_t)

        # 2. Sequential edge-aware message passing
        for layer in self.layers:
            h = layer(h, edge_index_t, e)

        # 3. Graph-level pooling: [num_nodes, D] -> [batch_size, D]
        h_graph = self.pooling(h, batch=batch_t, num_graphs=num_graphs)

        # 4. Multi-task prediction head: [batch_size, D] -> [batch_size, 12]
        logits = self.prediction_head(h_graph)

        return logits

    def predict_proba(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute sigmoid probabilities for each toxicity endpoint.

        Returns:
            torch.Tensor of shape [batch_size, 12] with values in [0.0, 1.0].
        """
        logits = self.forward(*args, **kwargs)
        return torch.sigmoid(logits)
