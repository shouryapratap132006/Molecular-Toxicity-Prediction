"""Input feature encoders for molecular graph nodes (atoms) and edges (bonds).

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Resolve activation module from name."""
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Supported: {list(activations.keys())}")
    return activations[name]()


class NodeEncoder(nn.Module):
    """Encodes raw atom features into hidden node representations.

    Input shape:  [num_nodes, node_feature_dim] (default 19)
    Output shape: [num_nodes, hidden_dim]
    """

    def __init__(
        self,
        in_dim: int = 19,
        hidden_dim: int = 128,
        activation: str = "relu",
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.activation = get_activation(activation)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        if norm == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform raw node features into latent space."""
        if x.dim() != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected node features of shape [N, {self.in_dim}], got {x.shape}")
        h = self.linear1(x)
        h = self.activation(h)
        h = self.linear2(h)
        return self.norm(h)


class EdgeEncoder(nn.Module):
    """Encodes raw bond features into hidden edge representations.

    Input shape:  [num_edges, edge_feature_dim] (default 18)
    Output shape: [num_edges, hidden_dim]

    Correctly handles molecules with zero bonds (shape [0, 18] -> [0, hidden_dim]).
    """

    def __init__(
        self,
        in_dim: int = 18,
        hidden_dim: int = 128,
        activation: str = "relu",
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.activation = get_activation(activation)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        if norm == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Transform raw bond features into latent space."""
        if edge_attr.dim() != 2 or edge_attr.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected edge features of shape [E, {self.in_dim}], got {edge_attr.shape}"
            )
        if edge_attr.shape[0] == 0:
            # Handle empty edge tensor (molecules with 0 bonds)
            return torch.empty(
                (0, self.hidden_dim),
                dtype=edge_attr.dtype,
                device=edge_attr.device,
            )
        e = self.linear1(edge_attr)
        e = self.activation(e)
        e = self.linear2(e)
        return self.norm(e)
