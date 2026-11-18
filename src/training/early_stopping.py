"""Early stopping mechanism based on validation performance."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting when validation loss stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change in monitored metric to qualify as an improvement.
        mode: "min" (e.g. loss) or "max" (e.g. ROC-AUC).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.best_epoch: int = -1
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, current_score: float, epoch: int) -> bool:
        """Update early stopping state with the current epoch's metric score.

        Args:
            current_score: Metric value (e.g. validation loss).
            epoch: Current epoch index.

        Returns:
            is_best: True if current score is a new best, False otherwise.
        """
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            logger.debug("Validation metric improved to %.5f at epoch %d", current_score, epoch)
            return True
        else:
            self.counter += 1
            logger.debug(
                "No improvement for %d/%d epochs. Best: %.5f at epoch %d",
                self.counter,
                self.patience,
                self.best_score,
                self.best_epoch,
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered at epoch %d. Best metric: %.5f at epoch %d",
                    epoch,
                    self.best_score,
                    self.best_epoch,
                )
            return False

    def state_dict(self) -> dict:
        """Return serializable state dictionary."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from dictionary."""
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.mode = state["mode"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]
