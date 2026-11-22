"""Experiment orchestration and tracking module for Phase 2 GNN.

Student 5 Module: Integration, Experiments, Final IEEE Report & Presentation.
"""

from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_registry import ExperimentRecord, ExperimentRegistry
from src.experiments.run_experiment import execute_experiment

__all__ = [
    "ExperimentConfig",
    "ExperimentRecord",
    "ExperimentRegistry",
    "execute_experiment",
]
