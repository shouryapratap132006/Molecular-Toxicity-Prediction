"""Graph-level readout pooling layers for molecular graph representations.

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.

This module implements deterministic, vectorised graph pooling operations
('mean', 'sum', 'max') that aggregate atom-level embeddings into a fixed-size
molecular embedding per molecule without cross-graph information leakage.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


class GlobalGraphPooling(nn.Module):
    """Aggregates atom representations h into graph-level representations.

    Supports:
    - 'mean': Arithmetic mean of atom representations per molecule.
    - 'sum': Sum of atom representations per molecule.
    - 'max': Element-wise maximum of atom representations per molecule.

    Input:
        h: Atom representations of shape [num_nodes, hidden_dim].
        batch: 1D integer tensor of shape [num_nodes] mapping each node to
               its graph index in {0, ..., B-1}.
        num_graphs: Optional total number of graphs B. Inferred if not provided.

    Output:
        Graph embeddings of shape [B, hidden_dim].
    """

    def __init__(self, method: Literal["mean", "sum", "max"] = "mean") -> None:
        super().__init__()
        if method not in ("mean", "sum", "max"):
            raise ValueError(
                f"Unsupported pooling method '{method}'. Choose 'mean', 'sum', or 'max'."
            )
        self.method = method

    def forward(
        self,
        h: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        """Pool node features into graph features.

        Args:
            h: [num_nodes, hidden_dim]
            batch: [num_nodes] with values in 0..B-1. If None, treats all nodes
                   as belonging to a single molecule (B=1).
            num_graphs: Optional total count B. If None, inferred as batch.max() + 1.

        Returns:
            [B, hidden_dim] graph-level representation tensor.
        """
        if h.dim() != 2:
            raise ValueError(f"Expected 2D node tensor [N, D], got {h.shape}")

        num_nodes, hidden_dim = h.shape

        if num_nodes == 0:
            b_dim = num_graphs if num_graphs is not None else 0
            return torch.zeros((b_dim, hidden_dim), dtype=h.dtype, device=h.device)

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=h.device)

        if batch.shape != (num_nodes,):
            raise ValueError(f"batch tensor shape {batch.shape} must match num_nodes {num_nodes}")

        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1

        if num_graphs <= 0:
            raise ValueError(f"num_graphs must be positive, got {num_graphs}")

        if self.method == "sum":
            out = torch.zeros((num_graphs, hidden_dim), dtype=h.dtype, device=h.device)
            out.index_add_(0, batch, h)
            return out

        elif self.method == "mean":
            sum_pooled = torch.zeros((num_graphs, hidden_dim), dtype=h.dtype, device=h.device)
            sum_pooled.index_add_(0, batch, h)
            # Count atoms per graph
            counts = torch.bincount(batch, minlength=num_graphs).to(dtype=h.dtype, device=h.device)
            counts = counts.clamp(min=1.0).unsqueeze(1)
            return sum_pooled / counts

        elif self.method == "max":
            # Initialize with large negative values for max pooling
            out = torch.full(
                (num_graphs, hidden_dim),
                float("-inf"),
                dtype=h.dtype,
                device=h.device,
            )
            # Scatter-reduce max in pure PyTorch
            out.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, hidden_dim), h, reduce="amax")
            # Replace -inf for any empty graph indices with 0
            out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
            return out

        else:
            raise NotImplementedError(f"Pooling method '{self.method}' not implemented.")
