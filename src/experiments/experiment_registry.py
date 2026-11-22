"""Machine-readable experiment registry for Phase 2 GNN tracking.

Student 5 Module: Integration, Experiments, Final IEEE Report & Presentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional


def get_git_commit_hash() -> str:
    """Retrieve current Git commit SHA-1 hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


@dataclass
class ExperimentRecord:
    """Record representing a single executed experiment."""

    experiment_id: str
    status: str
    config: dict[str, Any]
    seed: int
    git_commit: str
    started_at_utc: str
    completed_at_utc: str
    runtime_seconds: float
    device: str
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    total_epochs: Optional[int] = None
    stopped_early: Optional[bool] = None
    checkpoint_path: Optional[str] = None
    training_log_path: Optional[str] = None
    test_metrics: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    """Central registry tracking all Phase 2 experiment executions."""

    def __init__(self, registry_path: Path | str = "results/phase2/experiments/experiment_registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.records: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load registry from disk if it exists."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = {}
        else:
            self.records = {}

    def save(self) -> None:
        """Persist registry to disk with indentation."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, sort_keys=True)

    def register(self, record: ExperimentRecord) -> None:
        """Add or update an experiment record in the registry."""
        self.records[record.experiment_id] = record.to_dict()
        self.save()

    def get(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """Retrieve an experiment record by ID."""
        return self.records.get(experiment_id)

    def list_experiments(self) -> list[str]:
        """List all registered experiment IDs."""
        return sorted(self.records.keys())

    def get_summary_table(self) -> list[dict[str, Any]]:
        """Generate a concise summary row per experiment."""
        rows = []
        for exp_id, rec in sorted(self.records.items()):
            cfg = rec.get("config", {})
            test_m = rec.get("test_metrics", {}) or {}
            rows.append({
                "experiment_id": exp_id,
                "status": rec.get("status"),
                "seed": rec.get("seed"),
                "hidden_dim": cfg.get("hidden_dim"),
                "num_layers": cfg.get("num_layers"),
                "pooling": cfg.get("pooling"),
                "use_pos_weights": cfg.get("use_pos_weights"),
                "best_epoch": rec.get("best_epoch"),
                "best_val_loss": rec.get("best_val_loss"),
                "test_roc_auc": test_m.get("roc_auc_mean"),
                "test_pr_auc": test_m.get("pr_auc_mean"),
                "runtime_s": round(rec.get("runtime_seconds", 0.0), 2),
            })
        return rows
