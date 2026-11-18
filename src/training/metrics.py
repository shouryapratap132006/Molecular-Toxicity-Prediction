"""Training history logging, summary metrics, and visualization utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Tracks epoch-level metrics and produces CSV logs, summary JSON, and plots."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: Optional[float] = None,
        epoch_time_sec: float = 0.0,
        valid_labels_train: int = 0,
        valid_labels_val: int = 0,
        lr: Optional[float] = None,
        train_labels: Optional[int] = None,
        val_labels: Optional[int] = None,
    ) -> None:
        """Record metrics for an epoch."""
        actual_lr = learning_rate if learning_rate is not None else (lr if lr is not None else 0.0)
        actual_train_cnt = valid_labels_train if valid_labels_train else (train_labels if train_labels else 0)
        actual_val_cnt = valid_labels_val if valid_labels_val else (val_labels if val_labels else 0)

        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "learning_rate": float(actual_lr),
            "epoch_time_sec": float(epoch_time_sec),
            "valid_labels_train": int(actual_train_cnt),
            "valid_labels_val": int(actual_val_cnt),
        }
        self.history.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Return history as a pandas DataFrame."""
        return pd.DataFrame(self.history)

    def to_csv(self, path: Path | str) -> Path:
        """Alias for save_history_csv."""
        return self.save_history_csv(path)

    def save_history_csv(self, path: Path | str) -> Path:
        """Save training history to CSV."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(target, index=False)
        logger.info("Saved training history CSV to %s (%d records)", target, len(df))
        return target

    def save_summary_json(
        self,
        path: Path | str,
        best_epoch: int,
        best_val_loss: float,
        total_training_time_sec: float,
        early_stopped: bool,
        extra_metrics: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Save high-level training summary to JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "total_epochs": len(self.history),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "final_train_loss": float(self.history[-1]["train_loss"]) if self.history else None,
            "final_val_loss": float(self.history[-1]["val_loss"]) if self.history else None,
            "total_training_time_sec": float(total_training_time_sec),
            "early_stopped": bool(early_stopped),
            **(extra_metrics or {}),
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        logger.info("Saved training summary JSON to %s", target)
        return target

    def plot_curves(
        self,
        output_path: Path | str,
        best_epoch: Optional[int] = None,
        title: Optional[str] = None,
    ) -> Path:
        """Alias for plot_training_curves."""
        return self.plot_training_curves(output_path, best_epoch=best_epoch)

    def plot_training_curves(
        self,
        output_path: Path | str,
        best_epoch: Optional[int] = None,
    ) -> Path:
        """Generate publication-ready training and validation loss curves.

        Args:
            output_path: Destination PNG path.
            best_epoch: Optional best epoch index to highlight.

        Returns:
            Path of saved plot.
        """
        if len(self.history) == 0:
            raise ValueError("Cannot plot empty training history")

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        epochs = df["epoch"]

        fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        # 1. Loss curves
        ax_loss.plot(epochs, df["train_loss"], label="Train Loss (Masked BCE)", color="#1f77b4", lw=2)
        ax_loss.plot(epochs, df["val_loss"], label="Val Loss (Masked BCE)", color="#ff7f0e", lw=2)

        if best_epoch is not None and best_epoch in df["epoch"].values:
            best_val = df.loc[df["epoch"] == best_epoch, "val_loss"].values[0]
            ax_loss.scatter(
                [best_epoch],
                [best_val],
                color="red",
                s=100,
                zorder=5,
                label=f"Best Model (Epoch {best_epoch}: {best_val:.4f})",
            )
            ax_loss.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6)

        ax_loss.set_ylabel("Masked BCE Loss", fontsize=11)
        ax_loss.set_title("Phase 2 MolecularGNN Training & Validation Loss", fontsize=12, fontweight="bold")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc="upper right", frameon=True)

        # 2. Learning rate curve
        ax_lr.plot(epochs, df["learning_rate"], label="Learning Rate", color="#2ca02c", lw=1.8)
        ax_lr.set_xlabel("Epoch", fontsize=11)
        ax_lr.set_ylabel("Learning Rate", fontsize=11)
        ax_lr.set_yscale("log")
        ax_lr.grid(True, alpha=0.3, which="both")
        ax_lr.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        plt.savefig(target, dpi=300)
        plt.close(fig)
        logger.info("Saved training curves plot to %s", target)
        return target
