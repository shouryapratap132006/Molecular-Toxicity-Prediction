"""Edge-aware molecular message passing layers.

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.

This module implements relational Message Passing Neural Network (MPNN) layers
tailored for molecular graphs, incorporating directional bond features,
node-level updates, residual connections, normalization, and handling of
variable graph sizes and zero-bond edge cases.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import get_activation


class EdgeAwareMessagePassingLayer(nn.Module):
    """Single layer of Edge-Aware Molecular Message Passing.

    Computes directional messages m_{u -> v} for every bond (u -> v):
        m_{u -> v} = MLP_msg([h_u, h_v, e_{u,v}])
    Aggregates incoming messages at each atom v:
        m_v = sum_{u in N(v)} m_{u -> v}
    Updates atom representation:
        h_v^{new} = MLP_update([h_v, m_v])
        h_v^{out} = Norm(h_v + Dropout(h_v^{new}))   (if residual=True)

    Handles:
    - Variable-sized molecular graphs
    - Directional bond representations
    - Graphs with zero edges (e.g., isolated atoms / monoatomic salts)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        residual: bool = True,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual

        # Message MLP: consumes [source_node, target_node, edge_attr]
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node update MLP: consumes [current_node, aggregated_message]
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Normalization layer
        if norm == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the message passing layer.

        Args:
            h: Node representations of shape [num_nodes, hidden_dim].
            edge_index: Graph connectivity of shape [2, num_edges].
                        edge_index[0] is source u, edge_index[1] is target v.
            edge_attr: Edge representations of shape [num_edges, hidden_dim].

        Returns:
            Updated node representations of shape [num_nodes, hidden_dim].
        """
        num_nodes = h.shape[0]
        num_edges = edge_index.shape[1] if edge_index.dim() == 2 else 0

        if num_edges > 0:
            src = edge_index[0]
            dst = edge_index[1]

            # 1. Message construction: [h_src, h_dst, e_{src->dst}]
            h_src = h[src]
            h_dst = h[dst]
            msg_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
            messages = self.msg_mlp(msg_input)

            # 2. Deterministic message aggregation into target nodes
            # Shape: [num_nodes, hidden_dim]
            agg_messages = torch.zeros(
                (num_nodes, self.hidden_dim),
                dtype=h.dtype,
                device=h.device,
            )
            agg_messages.index_add_(0, dst, messages)
        else:
            # Safe fallback for zero-bond graphs: no incoming messages
            agg_messages = torch.zeros(
                (num_nodes, self.hidden_dim),
                dtype=h.dtype,
                device=h.device,
            )

        # 3. Node update
        update_input = torch.cat([h, agg_messages], dim=-1)
        h_update = self.update_mlp(update_input)

        # 4. Residual connection, activation, and normalization
        if self.residual:
            h_out = h + self.dropout(h_update)
        else:
            h_out = self.dropout(h_update)

        return self.norm(h_out)
