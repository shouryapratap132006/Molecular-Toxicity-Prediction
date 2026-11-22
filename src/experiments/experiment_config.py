"""Experiment configuration dataclass for Phase 2 GNN orchestration.

Student 5 Module: Integration, Experiments, Final IEEE Report & Presentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional
import yaml

from src.training.config import TrainingConfig


@dataclass
class ExperimentConfig:
    """Central configuration governing a Phase 2 GNN experiment run.

    Encapsulates experiment identification, hyperparameter ablation settings,
    training parameters, evaluation controls, and artifact destinations.
    """

    experiment_id: str = "gnn_baseline"
    description: str = "Default Phase 2 MolecularGNN baseline"
    seed: int = 42
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    pooling: str = "mean"
    activation: str = "relu"
    norm: str = "layer"
    residual: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    scheduler: str = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    use_pos_weights: bool = False
    max_pos_weight: float = 50.0
    device: str = "auto"
    num_tasks: int = 12
    evaluate_on_test: bool = False
    output_dir: str = "results/phase2/experiments"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert experiment configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        """Create an ExperimentConfig instance from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load experiment configuration from a YAML file."""
        target = Path(path)
        with open(target, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"YAML at {path} must parse to a dictionary, got {type(data)}")
        return cls.from_dict(data)

    def save_yaml(self, path: Path | str) -> None:
        """Save experiment configuration to a YAML file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def to_training_config(self) -> TrainingConfig:
        """Convert to Student 3 TrainingConfig instance."""
        exp_dir = Path(self.output_dir) / self.experiment_id
        return TrainingConfig(
            seed=self.seed,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            scheduler=self.scheduler,
            scheduler_factor=self.scheduler_factor,
            scheduler_patience=self.scheduler_patience,
            min_lr=self.min_lr,
            gradient_clip=self.gradient_clip,
            use_pos_weights=self.use_pos_weights,
            max_pos_weight=self.max_pos_weight,
            pooling=self.pooling,
            activation=self.activation,
            norm=self.norm,
            residual=self.residual,
            num_tasks=self.num_tasks,
            device=self.device,
            checkpoint_dir=str(exp_dir / "checkpoints"),
            log_dir=str(exp_dir / "logs"),
            plot_dir=str(exp_dir / "plots"),
            extra_metadata={
                "experiment_id": self.experiment_id,
                "description": self.description,
                "tags": self.tags,
            },
        )
