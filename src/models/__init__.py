"""Model implementations for Molecular Toxicity Prediction.

Phase 1:
    src/models/ml/ — Classical machine learning baseline classifiers
    (Logistic Regression, Random Forest, XGBoost)

Phase 2:
    Molecular Graph Neural Network (GNN) architectures and layers.
"""

from src.models.encoders import EdgeEncoder, NodeEncoder
from src.models.graph_pooling import GlobalGraphPooling
from src.models.message_passing import EdgeAwareMessagePassingLayer
from src.models.model_config import GNNConfig
from src.models.molecular_gnn import (
    TOX21_ENDPOINTS,
    MolecularGNN,
    MultiTaskPredictionHead,
)

__all__ = [
    "MolecularGNN",
    "GNNConfig",
    "NodeEncoder",
    "EdgeEncoder",
    "EdgeAwareMessagePassingLayer",
    "GlobalGraphPooling",
    "MultiTaskPredictionHead",
    "TOX21_ENDPOINTS",
]
