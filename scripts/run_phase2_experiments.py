"""Orchestration script to execute the required Phase 2 GNN experiment matrix.

Student 5 Module: Integration, Experiments, Final IEEE Report & Presentation.

Executes:
- Experiment A: GNN Baseline (hidden_dim=128, layers=3, seed=42)
- Experiment B: Hidden Dimension Ablation (hidden_dim=64 vs 128)
- Experiment C: Message Passing Depth Ablation (layers=2 vs 3)
- Experiment D: Readout Pooling & Loss Weighting Ablation (sum pooling, pos_weight)
- Experiment E: Reproducibility Verification (seed=42 vs seed=43)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set Matplotlib and OpenMP configs
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.graph_preprocess import load_graph_datasets
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_registry import ExperimentRegistry
from src.experiments.run_experiment import execute_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("run_phase2_experiments")


def run_experiment_matrix(
    configs_dir: Path | str = "configs/phase2",
    registry_path: Path | str = "results/phase2/experiments/experiment_registry.json",
    summary_dir: Path | str = "results/phase2/summaries",
    max_epochs: int = 25,
) -> None:
    """Execute all configured Phase 2 experiments and compile synthesis summaries."""
    configs_dir = Path(configs_dir)
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    registry = ExperimentRegistry(registry_path)

    # Required experiment suite order
    experiment_files = [
        "gnn_baseline.yaml",
        "gnn_hidden64.yaml",
        "gnn_depth2.yaml",
        "gnn_sum_pooling.yaml",
        "gnn_pos_weighted.yaml",
        "gnn_seed43.yaml",
    ]

    logger.info("Loading canonical Tox21 graph datasets once into memory...")
    datasets = load_graph_datasets()

    results = []

    for fname in experiment_files:
        cfg_path = configs_dir / fname
        if not cfg_path.exists():
            logger.warning("Config file %s does not exist, skipping.", cfg_path)
            continue

        exp_cfg = ExperimentConfig.load_yaml(cfg_path)
        # Cap epochs for controlled ablation benchmarking
        exp_cfg.epochs = min(exp_cfg.epochs, max_epochs)
        exp_cfg.early_stopping_patience = 8

        logger.info(">>> Launching Experiment: %s (%s)", exp_cfg.experiment_id, exp_cfg.description)
        summary = execute_experiment(exp_cfg, datasets=datasets, registry=registry)
        results.append(summary)

    # 1. Compile comparison table
    summary_rows = registry.get_summary_table()
    df_summary = pd.DataFrame(summary_rows)
    csv_summary_path = summary_dir / "phase2_ablation_summary.csv"
    df_summary.to_csv(csv_summary_path, index=False)
    logger.info("Saved ablation summary table to %s", csv_summary_path)

    # 2. Generate publication-ready ablation comparison bar charts
    plot_ablation_figures(df_summary, summary_dir)

    # 3. Export JSON summary
    with open(summary_dir / "phase2_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info("Phase 2 Experiment Matrix Execution Complete!")


def plot_ablation_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate publication-ready ablation figures for IEEE report and presentation."""
    if df.empty or "best_val_loss" not in df.columns:
        return

    # Figure 1: Best Validation Loss across Ablation Experiments
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

    exp_ids = df["experiment_id"].tolist()
    val_losses = df["best_val_loss"].tolist()

    bars = ax.bar(range(len(exp_ids)), val_losses, color=colors[:len(exp_ids)], width=0.55, edgecolor="black")
    ax.set_xticks(range(len(exp_ids)))
    ax.set_xticklabels(exp_ids, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylabel("Best Validation Masked BCE Loss", fontsize=12, fontweight="bold")
    ax.set_title("Phase 2 GNN Architecture & Optimization Ablations (Validation Loss)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot_path = output_dir / "ablation_validation_loss_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info("Saved ablation plot to %s", plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="Max epochs per ablation run")
    args = parser.parse_args()
    run_experiment_matrix(max_epochs=args.epochs)
