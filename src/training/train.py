"""Complete end-to-end training pipeline for Phase 2 MolecularGNN.

Student 3 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional

import torch

from src.data.graph_dataset import iter_graph_batches
from src.data.graph_preprocess import load_graph_datasets
from src.models import MolecularGNN
from src.training.checkpoint import save_checkpoint
from src.training.config import TrainingConfig
from src.training.early_stopping import EarlyStopping
from src.training.losses import MaskedBCELoss, compute_endpoint_pos_weights
from src.training.metrics import TrainingLogger
from src.training.reproducibility import get_device, get_system_info, set_seed
from src.training.scheduler import build_scheduler, step_scheduler
from src.training.trainer import evaluate, train_one_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_training(
    config: Optional[TrainingConfig] = None,
    datasets: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Execute complete MolecularGNN training workflow.

    Args:
        config: TrainingConfig instance (uses defaults if None).
        datasets: Dict of {"train": ..., "val": ..., "test": ...} datasets
                  (loaded via load_graph_datasets() if None).

    Returns:
        Dictionary summarizing training results, metrics, and artifact paths.
    """
    if config is None:
        config = TrainingConfig()

    logger.info("Initializing Phase 2 MolecularGNN training pipeline...")
    logger.info("Configuration: %s", config.to_dict())

    # 1. Ensure deterministic reproducibility
    set_seed(config.seed)

    # 2. Resolve hardware device
    device = get_device(config.device)
    logger.info("Selected compute device: %s", device)

    # 3. Load graph datasets
    if datasets is None:
        logger.info("Loading preprocessed graph datasets from cache...")
        datasets = load_graph_datasets()

    train_data = datasets["train"]
    val_data = datasets["val"]
    logger.info(
        "Datasets loaded: %d train graphs, %d val graphs",
        len(train_data),
        len(val_data),
    )

    # 4. Optional positive class weights for class imbalance
    pos_weight = None
    if config.use_pos_weights:
        pos_weight = compute_endpoint_pos_weights(
            train_data,
            max_weight=config.max_pos_weight,
        )
        logger.info("Class imbalance pos_weights computed: %s", pos_weight.tolist())

    loss_fn = MaskedBCELoss(pos_weight=pos_weight)

    # 5. Initialize MolecularGNN
    model_kwargs = config.to_model_kwargs()
    logger.info("Instantiating MolecularGNN with kwargs: %s", model_kwargs)
    model = MolecularGNN(**model_kwargs).to(device)

    # 6. Initialize Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)

    # 7. Early Stopping and Training Tracker
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )
    tracker = TrainingLogger()

    # Create destination directories
    checkpoint_dir = Path(config.checkpoint_dir)
    log_dir = Path(config.log_dir)
    plot_dir = Path(config.plot_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoint_dir / "best_gnn.pt"
    last_checkpoint_path = checkpoint_dir / "last_gnn.pt"

    logger.info(
        "Beginning training for up to %d epochs (patience=%d)...",
        config.epochs,
        config.early_stopping_patience,
    )

    last_epoch_run = 0
    final_train_loss = float("inf")
    final_val_loss = float("inf")

    # 8. Training Loop
    for epoch in range(1, config.epochs + 1):
        last_epoch_run = epoch
        t0 = time.time()

        # Optimize one epoch on train set
        train_loader = iter_graph_batches(train_data, batch_size=config.batch_size)
        train_loss, train_labels = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            gradient_clip=config.gradient_clip,
        )
        final_train_loss = train_loss

        # Evaluate on validation set (NEVER test set)
        val_loader = iter_graph_batches(val_data, batch_size=config.batch_size)
        val_loss, val_labels = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
        )
        final_val_loss = val_loss
        epoch_time = time.time() - t0

        # Step LR scheduler
        current_lr = step_scheduler(scheduler, val_loss, epoch)
        if current_lr is None:
            current_lr = optimizer.param_groups[0]["lr"]

        # Record metrics in logger
        tracker.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=current_lr,
            epoch_time_sec=epoch_time,
            valid_labels_train=train_labels,
            valid_labels_val=val_labels,
        )

        logger.info(
            "Epoch [%3d/%3d]  Train Loss: %.4f  Val Loss: %.4f  LR: %.2e",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            current_lr,
        )

        # Check for validation improvement
        is_best = early_stopping.step(val_loss, epoch)
        if is_best:
            logger.info(
                "★ Validation loss improved to %.4f (epoch %d). Saving best checkpoint to %s",
                val_loss,
                epoch,
                best_checkpoint_path,
            )
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                epoch=epoch,
                val_loss=val_loss,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                extra_metadata={
                    "train_loss": train_loss,
                    "best_val_loss": early_stopping.best_loss,
                    "early_stopping_counter": early_stopping.counter,
                },
            )

        if early_stopping.should_stop:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d without improvement).",
                epoch,
                config.early_stopping_patience,
            )
            break

    # Save last checkpoint
    save_checkpoint(
        path=last_checkpoint_path,
        model=model,
        epoch=last_epoch_run,
        val_loss=final_val_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    # 9. Save Artifacts and Summary Logs
    csv_path = log_dir / "training_history.csv"
    tracker.to_csv(csv_path)
    logger.info("Saved epoch history to %s", csv_path)

    config_json_path = log_dir / "experiment_config.json"
    config.save_json(config_json_path)
    logger.info("Saved configuration to %s", config_json_path)

    plot_path = plot_dir / "training_curves.png"
    tracker.plot_curves(
        plot_path,
        title="Phase 2 MolecularGNN Training & Validation Dynamics",
    )
    logger.info("Generated training curves plot at %s", plot_path)

    summary: dict[str, Any] = {
        "best_epoch": early_stopping.best_epoch,
        "best_val_loss": early_stopping.best_loss,
        "total_epochs_trained": last_epoch_run,
        "stopped_early": early_stopping.should_stop,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "device": str(device),
        "checkpoints": {
            "best": str(best_checkpoint_path),
            "last": str(last_checkpoint_path),
        },
        "logs": {
            "history_csv": str(csv_path),
            "config_json": str(config_json_path),
        },
        "plots": {
            "curves_png": str(plot_path),
        },
        "system_info": get_system_info(),
    }

    summary_json_path = log_dir / "training_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info("Saved training summary to %s", summary_json_path)

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("Best Epoch: %d | Best Val Loss: %.4f", early_stopping.best_epoch, early_stopping.best_loss)
    logger.info("Best Model Checkpoint: %s", best_checkpoint_path)
    logger.info("=" * 60)

    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for GNN training."""
    parser = argparse.ArgumentParser(
        description="Train Phase 2 MolecularGNN on Tox21 Graph Dataset.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay L2 penalty (default: 1e-4)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden embedding dimension (default: 128)")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of message passing layers (default: 3)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--scheduler-patience", type=int, default=4, help="Scheduler patience (default: 4)")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="Scheduler reduction factor (default: 0.5)")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping max norm (default: 1.0)")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "sum", "max"], help="Readout pooling")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--use-pos-weights", action="store_true", help="Enable endpoint positive class weighting")
    parser.add_argument("--checkpoint-dir", type=str, default="results/phase2/checkpoints")
    parser.add_argument("--log-dir", type=str, default="results/phase2/logs")
    parser.add_argument("--plot-dir", type=str, default="results/phase2/plots")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for MolecularGNN training."""
    args = parse_args(argv)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        early_stopping_patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        gradient_clip=args.gradient_clip,
        pooling=args.pooling,
        device=args.device,
        seed=args.seed,
        use_pos_weights=args.use_pos_weights,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
    )

    run_training(config=config)


if __name__ == "__main__":
    main()
