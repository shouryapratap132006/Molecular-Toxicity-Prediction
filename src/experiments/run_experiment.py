"""Centralized experiment runner and orchestrator for Phase 2 GNN experiments.

Student 5 Module: Integration, Experiments, Final IEEE Report & Presentation.

This module coordinates:
1. Configuration loading and validation.
2. Canonical graph dataset loading (Student 1).
3. Deterministic training execution (Student 3).
4. Experiment registry and artifact tracking.
5. Optional quarantined test evaluation invocation (Student 4).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional

from src.data.common import utc_now
from src.data.graph_preprocess import load_graph_datasets
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_registry import ExperimentRecord, ExperimentRegistry, get_git_commit_hash
from src.training.reproducibility import set_seed
from src.training.train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("src.experiments.run_experiment")


def execute_experiment(
    config: ExperimentConfig,
    datasets: Optional[Mapping[str, Any]] = None,
    registry: Optional[ExperimentRegistry] = None,
) -> dict[str, Any]:
    """Execute a single configured Phase 2 experiment end-to-end.

    Args:
        config: ExperimentConfig instance.
        datasets: Preloaded graph datasets (or None to load canonical).
        registry: ExperimentRegistry instance (or None to use default).

    Returns:
        Dictionary containing experiment execution summary and paths.
    """
    if registry is None:
        registry = ExperimentRegistry()

    exp_id = config.experiment_id
    logger.info("============================================================")
    logger.info("Executing Phase 2 Experiment: %s", exp_id)
    logger.info("Description: %s", config.description)
    logger.info("Parameters: hidden_dim=%d, layers=%d, pool=%s, seed=%d",
                config.hidden_dim, config.num_layers, config.pooling, config.seed)
    logger.info("============================================================")

    start_time = time.time()
    started_at = utc_now()
    git_hash = get_git_commit_hash()

    # 1. Deterministic seed setup
    set_seed(config.seed)

    # 2. Canonical graph dataset loading
    if datasets is None:
        logger.info("Loading canonical Tox21 graph datasets (Student 1)...")
        datasets = load_graph_datasets()

    # 3. Training config conversion
    training_cfg = config.to_training_config()

    try:
        # 4. Invoke Student 3 training API
        train_res = run_training(config=training_cfg, datasets=datasets)

        runtime = time.time() - start_time
        completed_at = utc_now()

        # 5. Optional test evaluation (Student 4 interface)
        test_metrics = None
        if config.evaluate_on_test:
            best_ckpt = train_res["checkpoints"]["best"]
            logger.info("Running quarantined test evaluation on checkpoint: %s", best_ckpt)
            from src.evaluation.phase2_test_evaluation import EvaluationConfig, run_evaluation_pipeline

            eval_cfg = EvaluationConfig(
                checkpoint_path=best_ckpt,
                expected_sha256=None,  # Dynamic checkpoint evaluation
                output_dir=str(Path(config.output_dir) / exp_id / "test_evaluation"),
                figures_dir=str(Path(config.output_dir) / exp_id / "figures"),
                device=config.device,
            )
            eval_res = run_evaluation_pipeline(eval_cfg)
            agg = eval_res["aggregate_metrics"]
            test_metrics = {
                "roc_auc_mean": agg["macro_roc_auc"]["mean"],
                "roc_auc_sem": agg["macro_roc_auc"]["sem"],
                "pr_auc_mean": agg["macro_pr_auc"]["mean"],
                "pr_auc_sem": agg["macro_pr_auc"]["sem"],
            }
            logger.info("Test Evaluation: ROC-AUC=%.4f +/- %.4f, PR-AUC=%.4f +/- %.4f",
                        test_metrics["roc_auc_mean"], test_metrics["roc_auc_sem"],
                        test_metrics["pr_auc_mean"], test_metrics["pr_auc_sem"])

        # 6. Create experiment record
        record = ExperimentRecord(
            experiment_id=exp_id,
            status="SUCCESS",
            config=config.to_dict(),
            seed=config.seed,
            git_commit=git_hash,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            runtime_seconds=runtime,
            device=train_res.get("device", "unknown"),
            best_epoch=train_res.get("best_epoch"),
            best_val_loss=train_res.get("best_val_loss"),
            total_epochs=train_res.get("total_epochs_trained"),
            stopped_early=train_res.get("stopped_early"),
            checkpoint_path=train_res["checkpoints"].get("best"),
            training_log_path=str(train_res.get("logs", {}).get("history_csv")),
            test_metrics=test_metrics,
        )
        registry.register(record)

        summary = {
            "experiment_id": exp_id,
            "status": "SUCCESS",
            "runtime_seconds": runtime,
            "best_epoch": train_res.get("best_epoch"),
            "best_val_loss": train_res.get("best_val_loss"),
            "checkpoint": train_res["checkpoints"].get("best"),
            "test_metrics": test_metrics,
        }

        # Save individual experiment summary
        summary_path = Path(config.output_dir) / exp_id / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    except Exception as exc:
        runtime = time.time() - start_time
        completed_at = utc_now()
        logger.error("Experiment %s failed with exception: %s", exp_id, exc, exc_info=True)

        record = ExperimentRecord(
            experiment_id=exp_id,
            status="FAILED",
            config=config.to_dict(),
            seed=config.seed,
            git_commit=git_hash,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            runtime_seconds=runtime,
            device=config.device,
            error_message=str(exc),
        )
        registry.register(record)
        raise exc


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for executing experiments."""
    parser = argparse.ArgumentParser(description="Phase 2 GNN Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML or JSON config")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model on test partition after training")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (cpu, cuda, mps, auto)")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        parser.exit(1, f"Config file not found: {cfg_path}\n")

    if cfg_path.suffix in (".yaml", ".yml"):
        exp_cfg = ExperimentConfig.load_yaml(cfg_path)
    else:
        with open(cfg_path, "r", encoding="utf-8") as f:
            exp_cfg = ExperimentConfig.from_dict(json.load(f))

    if args.evaluate:
        exp_cfg.evaluate_on_test = True
    if args.device:
        exp_cfg.device = args.device

    summary = execute_experiment(exp_cfg)
    print("\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETED: {summary['experiment_id']}")
    print(f"Status: {summary['status']} | Runtime: {summary['runtime_seconds']:.2f}s")
    print(f"Best Epoch: {summary['best_epoch']} | Best Val Loss: {summary['best_val_loss']:.4f}")
    if summary["test_metrics"]:
        print(f"Test ROC-AUC: {summary['test_metrics']['roc_auc_mean']:.4f} +/- {summary['test_metrics']['roc_auc_sem']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
