"""Unit tests for training metrics logging, summary serialization, and visualization."""

import json
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.training.metrics import TrainingLogger


def test_training_logger_logging():
    logger = TrainingLogger()
    logger.log_epoch(
        epoch=0,
        train_loss=0.45,
        val_loss=0.42,
        learning_rate=1e-3,
        epoch_time_sec=2.5,
        valid_labels_train=1000,
        valid_labels_val=200,
    )
    logger.log_epoch(
        epoch=1,
        train_loss=0.40,
        val_loss=0.38,
        learning_rate=1e-3,
        epoch_time_sec=2.4,
        valid_labels_train=1000,
        valid_labels_val=200,
    )

    df = logger.to_dataframe()
    assert len(df) == 2
    assert list(df["epoch"]) == [0, 1]
    assert df["train_loss"].iloc[0] == 0.45
    assert df["val_loss"].iloc[1] == 0.38


def test_training_logger_save_csv_and_summary():
    logger = TrainingLogger()
    for ep in range(3):
        logger.log_epoch(
            epoch=ep,
            train_loss=0.5 - 0.05 * ep,
            val_loss=0.48 - 0.04 * ep,
            learning_rate=1e-3,
            epoch_time_sec=1.5,
            valid_labels_train=500,
            valid_labels_val=100,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "history.csv"
        json_path = Path(tmp_dir) / "summary.json"

        logger.save_history_csv(csv_path)
        assert csv_path.exists()
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == 3

        logger.save_summary_json(
            path=json_path,
            best_epoch=2,
            best_val_loss=0.40,
            total_training_time_sec=4.5,
            early_stopped=False,
            extra_metrics={"model": "MolecularGNN"},
        )
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_epochs"] == 3
        assert data["best_epoch"] == 2
        assert data["best_val_loss"] == 0.40
        assert data["early_stopped"] is False
        assert data["model"] == "MolecularGNN"


def test_training_logger_plot_curves():
    logger = TrainingLogger()
    for ep in range(5):
        logger.log_epoch(
            epoch=ep,
            train_loss=0.5 - 0.02 * ep,
            val_loss=0.45 - 0.015 * ep,
            learning_rate=1e-3,
            epoch_time_sec=1.0,
            valid_labels_train=200,
            valid_labels_val=50,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_path = Path(tmp_dir) / "curves.png"
        saved = logger.plot_training_curves(plot_path, best_epoch=4)
        assert saved.exists()
        assert saved.stat().st_size > 1000  # Non-trivial image file


def test_training_logger_empty_plot_raises():
    logger = TrainingLogger()
    with pytest.raises(ValueError, match="Cannot plot empty"):
        logger.plot_training_curves("non_existent.png")
