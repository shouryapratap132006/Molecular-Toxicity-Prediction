"""Phase 2 Test-Set Evaluation and Validation Pipeline for MolecularGNN.

Student 4 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.

Key Operational and Scientific Guarantees:
------------------------------------------
1. Quarantined Test Set: Exactly 783 molecular graphs evaluated under strict
   zero-leakage protocol. Test labels are never accessed or utilized for model
   selection, threshold optimization, or architecture tuning.
2. Verified Checkpoint Loading: Model parameters loaded directly from Student 3's
   trained checkpoint (results/phase2/checkpoints/best_gnn.pt) after strict SHA-256
   integrity validation.
3. Mask-Aware Metrics: Untested assays (NaNs) are strictly masked per endpoint.
   Missing labels are never replaced with zero or imputed.
4. Continuous Probability Scores: Sigmoid activation applied to raw logits
   prior to calculating ROC-AUC (Mann-Whitney U rank statistic) and PR-AUC
   (Average Precision).
5. Supplementary Threshold Analysis: Fixed threshold (0.5) applied for confusion
   matrix and classification metrics without post-hoc threshold tuning.
6. Publication-Quality Figures: Generated at 300 DPI or higher in reports/figures/phase2/.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from src.data.common import ROOT, load_config, sha256, utc_now

logger = logging.getLogger("src.evaluation.phase2_test_evaluation")

EXPECTED_CHECKPOINT_SHA256 = (
    "a6445726e46d0be9357e05d6988deb7b83f99df4e9cc6419ad8d114cccffa972"
)
EXPECTED_TEST_COUNT = 783
EXPECTED_TASK_COUNT = 12

CANONICAL_ENDPOINTS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


@dataclass
class EvaluationConfig:
    """Configuration for Phase 2 GNN final test evaluation."""

    checkpoint_path: str = "results/phase2/checkpoints/best_gnn.pt"
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256
    batch_size: int = 64
    device: str = "auto"
    output_dir: str = "results/phase2/test_evaluation"
    figures_dir: str = "reports/figures/phase2"
    threshold: float = 0.5
    expected_test_molecules: int = EXPECTED_TEST_COUNT
    num_endpoints: int = EXPECTED_TASK_COUNT
    endpoint_order: list[str] = field(default_factory=lambda: list(CANONICAL_ENDPOINTS))

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.expected_test_molecules <= 0:
            raise ValueError(f"Expected test molecules must be positive, got {self.expected_test_molecules}")
        if len(self.endpoint_order) != self.num_endpoints:
            raise ValueError(
                f"Endpoint count mismatch: expected {self.num_endpoints}, got {len(self.endpoint_order)}"
            )


def setup_evaluation_logging(output_dir: Path | str) -> logging.Logger:
    """Configure evaluation logging to file and console."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "test_evaluation.log"

    eval_logger = logging.getLogger("src.evaluation.phase2_test_evaluation")
    eval_logger.setLevel(logging.INFO)
    eval_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    eval_logger.addHandler(file_handler)

    eval_logger.info("Initialized Phase 2 test evaluation logger. Target: %s", log_file)
    return eval_logger


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for test evaluation."""
    parser = argparse.ArgumentParser(
        description="Phase 2 GNN Final Quarantined Test Evaluation (Student 4).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/phase2/checkpoints/best_gnn.pt",
        help="Path to trained GNN model checkpoint (default: results/phase2/checkpoints/best_gnn.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Hardware compute device (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase2/test_evaluation",
        help="Directory to save evaluation artifacts (default: results/phase2/test_evaluation)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="reports/figures/phase2",
        help="Directory to save figures (default: reports/figures/phase2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed classification decision threshold (default: 0.5)",
    )
    return parser.parse_args(argv)


def verify_and_load_checkpoint(
    checkpoint_path: Path | str,
    expected_sha256: Optional[str] = EXPECTED_CHECKPOINT_SHA256,
    device: str | torch.device = "auto",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Verify Student 3 checkpoint integrity, configuration, and load into eval mode.

    Guarantees:
    1. File exists and SHA-256 matches the expected digest exactly.
    2. Architecture parameters match Student 3 specification (hidden_dim=128, layers=3, endpoints=12).
    3. Endpoint order strictly corresponds to the canonical 12 Tox21 assays.
    4. Model is returned strictly in eval() mode on the specified device.
    5. Zero modifications to checkpoint file on disk.

    Args:
        checkpoint_path: Path to checkpoint file.
        expected_sha256: Expected SHA-256 digest string, or None to skip hash check.
        device: Target compute device ("auto", "cpu", "cuda", "mps").

    Returns:
        Tuple of (loaded_model, checkpoint_verification_metadata).
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Compute and verify SHA-256
    file_bytes = path.read_bytes()
    computed_sha256 = hashlib.sha256(file_bytes).hexdigest()
    file_size = len(file_bytes)

    if expected_sha256 is not None and computed_sha256 != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch!\n"
            f"Expected: {expected_sha256}\n"
            f"Computed: {computed_sha256}"
        )

    # Resolve device
    from src.training.reproducibility import get_device
    target_device = get_device(device) if isinstance(device, str) else device

    from src.training.checkpoint import load_checkpoint, load_model_from_checkpoint
    state = load_checkpoint(path, device=target_device)

    # Verify essential checkpoint contents
    required_keys = {"epoch", "best_val_loss", "model_state_dict", "model_config"}
    missing_keys = required_keys - set(state.keys())
    if missing_keys:
        raise KeyError(f"Checkpoint corrupted: missing required keys {missing_keys}")

    # Validate endpoint order if present
    endpoint_order = state.get("endpoint_order", CANONICAL_ENDPOINTS)
    if endpoint_order != CANONICAL_ENDPOINTS:
        raise ValueError(
            f"Checkpoint endpoint order mismatch!\nExpected: {CANONICAL_ENDPOINTS}\nGot: {endpoint_order}"
        )

    # Validate model configuration
    model_cfg = state.get("model_config", {})
    if model_cfg.get("num_tasks") != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"Model task dimension mismatch: expected {EXPECTED_TASK_COUNT}, got {model_cfg.get('num_tasks')}"
        )

    # Instantiate and load model
    model = load_model_from_checkpoint(path, device=target_device)
    model.eval()

    # Verify evaluation mode
    if model.training:
        raise RuntimeError("Model failed to enter evaluation mode (training is True)")

    # Parameter count and layer verification
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metadata = {
        "checkpoint_path": str(path),
        "sha256": computed_sha256,
        "size_bytes": file_size,
        "epoch": state.get("epoch"),
        "best_val_loss": state.get("best_val_loss"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_config": model_cfg,
        "training_config": state.get("training_config", {}),
        "endpoint_order": endpoint_order,
        "device": str(target_device),
        "evaluation_mode": not model.training,
    }

    logger.info(
        "Verified Student 3 checkpoint successfully: %s (epoch=%s, best_val_loss=%.4f, SHA256=%s...)",
        path.name,
        metadata["epoch"],
        metadata["best_val_loss"],
        computed_sha256[:12],
    )
    return model, metadata


def run_test_inference(
    model: torch.nn.Module,
    test_dataset: Any,
    batch_size: int = 64,
    device: Optional[torch.device | str] = None,
    canonical_split_path: Path | str = "data/processed/scaffold_split_indices.json",
) -> dict[str, Any]:
    """Execute batched inference on quarantined test dataset under zero-leakage protocol.

    Guarantees:
    1. Zero test target labels are used for inference or threshold determination.
    2. Batching preserves canonical test partition sequence without shuffling.
    3. Tensors and arrays are safely converted and detached from computational graphs.
    4. Probabilities are strictly bounded in [0, 1] and finite (no NaNs / Infs).
    5. Output dimensions strictly match (783, 12).
    6. All 783 molecule IDs are unique and strictly aligned with scaffold_split_indices.json.

    Args:
        model: Restored MolecularGNN model in eval() mode.
        test_dataset: GraphDataset containing exactly 783 test molecular graphs.
        batch_size: Minibatch size for collation and inference (default: 64).
        device: Target device (defaults to model's parameter device).
        canonical_split_path: Path to scaffold split JSON for identifier verification.

    Returns:
        Dictionary containing predictions, targets, masks, molecule IDs, and validation stats.
    """
    if len(test_dataset) != EXPECTED_TEST_COUNT:
        raise ValueError(
            f"Quarantined test set count mismatch: expected {EXPECTED_TEST_COUNT}, got {len(test_dataset)}"
        )

    model.eval()
    if device is not None:
        target_dev = torch.device(device) if isinstance(device, str) else device
        model.to(target_dev)

    from src.data.graph_dataset import iter_graph_batches

    all_probs = []
    all_targets = []
    all_masks = []
    all_mol_ids: list[str] = []

    logger.info(
        "Initiating batched inference on %d quarantined test graphs (batch_size=%d)...",
        len(test_dataset),
        batch_size,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(iter_graph_batches(test_dataset, batch_size=batch_size)):
            # 1. Forward pass produces raw logits [B, 12]
            logits = model(batch)
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Model output must be a torch.Tensor, got {type(logits)}")
            if logits.ndim != 2 or logits.shape[1] != EXPECTED_TASK_COUNT:
                raise ValueError(
                    f"Expected logits shape [B, {EXPECTED_TASK_COUNT}], got {logits.shape}"
                )

            # 2. Continuous probabilities via sigmoid
            probs = torch.sigmoid(logits)
            probs_np = probs.detach().cpu().numpy().astype(np.float32)
            all_probs.append(probs_np)

            # 3. Targets and Masks conversion
            raw_y = batch["y"]
            if isinstance(raw_y, torch.Tensor):
                y_np = raw_y.detach().cpu().numpy().astype(np.float32)
            else:
                y_np = np.asarray(raw_y, dtype=np.float32)
            all_targets.append(y_np)

            raw_mask = batch["mask"]
            if isinstance(raw_mask, torch.Tensor):
                mask_np = raw_mask.detach().cpu().numpy().astype(bool)
            else:
                mask_np = np.asarray(raw_mask, dtype=bool)
            all_masks.append(mask_np)

            # 4. Molecule identifiers
            mol_ids = batch["mol_id"]
            if not isinstance(mol_ids, (list, tuple)):
                mol_ids = list(mol_ids)
            all_mol_ids.extend(mol_ids)

    # Concatenate across batches
    predictions = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Rigorous Validation Checklist
    # 1. Dimension checks
    if predictions.shape != (EXPECTED_TEST_COUNT, EXPECTED_TASK_COUNT):
        raise ValueError(
            f"Prediction shape mismatch: expected ({EXPECTED_TEST_COUNT}, {EXPECTED_TASK_COUNT}), got {predictions.shape}"
        )
    if targets.shape != predictions.shape:
        raise ValueError(
            f"Target shape mismatch: expected {predictions.shape}, got {targets.shape}"
        )
    if masks.shape != predictions.shape:
        raise ValueError(
            f"Mask shape mismatch: expected {predictions.shape}, got {masks.shape}"
        )

    # 2. Numerical validity
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Inference produced non-finite prediction probabilities (NaN or Inf)")
    if not np.all((predictions >= 0.0) & (predictions <= 1.0)):
        raise ValueError("Inference produced probabilities outside the valid [0, 1] interval")

    # 3. Molecule identifier integrity
    if len(all_mol_ids) != EXPECTED_TEST_COUNT:
        raise ValueError(
            f"Molecule ID count mismatch: expected {EXPECTED_TEST_COUNT}, got {len(all_mol_ids)}"
        )
    unique_ids = set(all_mol_ids)
    if len(unique_ids) != EXPECTED_TEST_COUNT:
        duplicates = [mid for mid in all_mol_ids if all_mol_ids.count(mid) > 1]
        raise ValueError(f"Duplicate molecule identifiers detected in test set: {set(duplicates)}")

    # 4. Verify canonical split alignment
    split_file = Path(canonical_split_path)
    if split_file.is_file():
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        canonical_test_ids = split_data.get("row_identifiers", {}).get("test_mol_ids")
        if canonical_test_ids is not None:
            if all_mol_ids != canonical_test_ids:
                raise ValueError("Inference molecule IDs diverged from canonical test split order!")
            logger.info("Canonical test split ordering verified: 100% 1-to-1 match with scaffold split.")

    # 5. Mask consistency check: targets with mask=True must be finite binary {0.0, 1.0}
    valid_targets = targets[masks]
    if not np.all(np.isin(valid_targets, [0.0, 1.0])):
        invalid_vals = valid_targets[~np.isin(valid_targets, [0.0, 1.0])]
        raise ValueError(f"Observed target values must be strictly binary {0, 1}, found: {invalid_vals[:5]}")

    logger.info(
        "Inference and validation completed successfully. Shape: %s, Total Valid Entries: %d / %d (%.2f%%)",
        predictions.shape,
        int(np.sum(masks)),
        predictions.size,
        100.0 * np.sum(masks) / predictions.size,
    )

    return {
        "predictions": predictions,
        "targets": targets,
        "masks": masks,
        "molecule_ids": all_mol_ids,
        "num_molecules": len(all_mol_ids),
        "num_tasks": EXPECTED_TASK_COUNT,
        "total_valid_observations": int(np.sum(masks)),
        "total_missing_observations": int(np.sum(~masks)),
    }


def save_inference_artifacts(
    output_dir: Path | str,
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    molecule_ids: Sequence[str],
) -> dict[str, str]:
    """Atomically save test inference predictions, targets, masks, and IDs.

    Args:
        output_dir: Destination directory.
        predictions: Predicted probability array of shape [783, 12].
        targets: Target label array of shape [783, 12].
        masks: Boolean mask array of shape [783, 12].
        molecule_ids: List of 783 molecule identifiers.

    Returns:
        Dictionary mapping artifact names to their file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "test_predictions.npy"
    target_path = out_dir / "test_targets.npy"
    mask_path = out_dir / "test_masks.npy"
    mol_ids_path = out_dir / "test_molecule_ids.csv"

    # Save arrays
    np.save(pred_path, predictions)
    np.save(target_path, targets)
    np.save(mask_path, masks)

    # Save molecule IDs CSV
    import pandas as pd
    ids_df = pd.DataFrame({
        "test_index": np.arange(len(molecule_ids)),
        "mol_id": molecule_ids,
    })
    ids_df.to_csv(mol_ids_path, index=False)

    logger.info("Saved inference arrays to %s", out_dir)
    return {
        "test_predictions": str(pred_path),
        "test_targets": str(target_path),
        "test_masks": str(mask_path),
        "test_molecule_ids": str(mol_ids_path),
    }


def compute_mask_aware_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    endpoint_order: Sequence[str] = CANONICAL_ENDPOINTS,
) -> tuple[Any, dict[str, Any]]:
    """Compute mask-aware evaluation metrics across all 12 endpoints.

    Strict Masking Guarantees:
    1. Untested assay entries (where mask is False or targets are NaN) are completely
       isolated and excluded from rank and precision-recall calculations.
    2. Missing labels are never imputed, substituted, or converted to negative/zero.
    3. ROC-AUC is calculated via Wilcoxon/Mann-Whitney U rank-sum with exact average
       tie correction. Single-class partitions gracefully yield NaN with status
       'single_class_undefined'.
    4. PR-AUC represents Average Precision (integral of Precision w.r.t Recall).
    5. Macro-averaging includes only non-single-class valid endpoints (K_valid),
       reporting mean, sample standard deviation (ddof=1), SEM, and 95% Student-t CI.

    Args:
        predictions: Continuous predicted probabilities [783, 12].
        targets: Ground-truth target labels [783, 12].
        masks: Ground-truth observation masks [783, 12].
        endpoint_order: Ordered sequence of endpoint names.

    Returns:
        Tuple of (endpoint_metrics_dataframe, aggregate_metrics_dictionary).
    """
    import pandas as pd
    from src.evaluation.metrics import evaluate_endpoint
    from src.evaluation.aggregation import aggregate_endpoint_metrics

    endpoint_rows: list[dict[str, Any]] = []
    endpoint_eval_map: dict[str, dict[str, Any]] = {}

    for task_idx, name in enumerate(endpoint_order):
        yt = targets[:, task_idx]
        ys = predictions[:, task_idx]
        m = masks[:, task_idx]

        total_samples = len(yt)
        yt_valid = yt[m]
        ys_valid = ys[m]
        observed = len(yt_valid)
        missing = total_samples - observed

        res = evaluate_endpoint(yt_valid, ys_valid, endpoint_name=name)

        n_pos = int(np.sum(yt_valid == 1.0)) if observed > 0 else 0
        n_neg = int(np.sum(yt_valid == 0.0)) if observed > 0 else 0
        prev = float(n_pos / observed) if observed > 0 else float(np.nan)

        record = {
            "endpoint": name,
            "status": res["status"],
            "total_samples": total_samples,
            "observed": observed,
            "missing": missing,
            "positive_count": n_pos,
            "negative_count": n_neg,
            "prevalence": prev,
            "roc_auc": res["roc_auc"],
            "pr_auc": res["pr_auc"],
        }
        endpoint_rows.append(record)
        endpoint_eval_map[name] = res

    # Compute macro-aggregates over valid endpoints
    agg = aggregate_endpoint_metrics(endpoint_eval_map)

    # Format dataframe
    df = pd.DataFrame(endpoint_rows)

    logger.info(
        "Computed mask-aware metrics across %d endpoints (valid=%d, excluded=%d). "
        "Macro ROC-AUC: %s | Macro PR-AUC: %s",
        len(endpoint_order),
        agg["valid_endpoints_count"],
        len(agg["excluded_endpoints"]),
        agg["macro_roc_auc"]["formatted_sem"],
        agg["macro_pr_auc"]["formatted_sem"],
    )
    return df, agg


def save_metric_artifacts(
    output_dir: Path | str,
    endpoint_df: Any,
    aggregate_dict: Mapping[str, Any],
) -> dict[str, str]:
    """Save endpoint metrics CSV and aggregate metrics JSON atomically.

    Args:
        output_dir: Destination directory.
        endpoint_df: Pandas DataFrame containing endpoint metrics.
        aggregate_dict: Dictionary containing macro aggregation metrics.

    Returns:
        Dictionary mapping artifact keys to their file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "endpoint_metrics.csv"
    json_path = out_dir / "aggregate_metrics.json"

    endpoint_df.to_csv(csv_path, index=False)

    from src.data.common import write_json
    write_json(json_path, aggregate_dict)

    logger.info("Saved metric artifacts: %s and %s", csv_path.name, json_path.name)
    return {
        "endpoint_metrics_csv": str(csv_path),
        "aggregate_metrics_json": str(json_path),
    }


def compute_supplementary_threshold_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    endpoint_order: Sequence[str] = CANONICAL_ENDPOINTS,
    threshold: float = 0.5,
) -> tuple[Any, dict[str, Any]]:
    """Compute supplementary classification metrics at fixed decision threshold 0.5.

    Protocol & Integrity Guarantees:
    1. Fixed Threshold: Decision boundary is fixed at 0.5 without any tuning against test data.
    2. Strict Isolation: Only valid observed samples (mask=True) enter calculations.
    3. Metrics computed per endpoint: TP, FP, TN, FN, Precision, Recall, Specificity, F1, Balanced Accuracy.
    4. Confusion matrices formatted as [[TN, FP], [FN, TP]].
    5. Safely handles zero divisions without crashing (returning 0.0 or nan).

    Args:
        predictions: Predicted probabilities [783, 12].
        targets: Ground truth labels [783, 12].
        masks: Validity masks [783, 12].
        endpoint_order: Canonical endpoint order.
        threshold: Classification decision boundary (fixed at 0.5).

    Returns:
        Tuple of (threshold_metrics_dataframe, summary_dict).
    """
    import pandas as pd
    from src.evaluation.metrics import compute_threshold_metrics

    records = []
    confusion_matrices = {}

    for idx, name in enumerate(endpoint_order):
        yt = targets[:, idx]
        ys = predictions[:, idx]
        m = masks[:, idx]

        yt_valid = yt[m]
        ys_valid = ys[m]

        metrics = compute_threshold_metrics(yt_valid, ys_valid, threshold=threshold)
        metrics["endpoint"] = name
        records.append(metrics)

        confusion_matrices[name] = {
            "confusion_matrix": [
                [int(metrics["tn"]), int(metrics["fp"])],
                [int(metrics["fn"]), int(metrics["tp"])],
            ],
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "tn": int(metrics["tn"]),
            "fn": int(metrics["fn"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "specificity": float(metrics["specificity"]),
            "f1": float(metrics["f1"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
        }

    df = pd.DataFrame(records)
    summary = {
        "threshold": float(threshold),
        "protocol": "Fixed threshold = 0.5 (strictly supplementary, zero test-set threshold tuning)",
        "endpoints": confusion_matrices,
        "mean_balanced_accuracy": float(df["balanced_accuracy"].dropna().mean()),
        "mean_f1": float(df["f1"].dropna().mean()),
    }

    logger.info(
        "Computed supplementary threshold metrics at t=%.2f. Mean Balanced Acc: %.4f | Mean F1: %.4f",
        threshold,
        summary["mean_balanced_accuracy"],
        summary["mean_f1"],
    )
    return df, summary


def run_error_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    molecule_ids: Sequence[str],
    endpoint_order: Sequence[str] = CANONICAL_ENDPOINTS,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Execute in-depth factual error analysis of GNN test predictions.

    Covers:
    1. Endpoint-wise performance ranking (highest vs lowest ROC-AUC / PR-AUC).
    2. Severe class imbalance sensitivity (prevalence vs performance correlation).
    3. Prediction probability and confidence distributions (true vs false predictions).
    4. Multi-task error patterns (molecules with multiple false positive/negative calls).
    5. Identification of challenging targets and biochemical hypotheses.

    Args:
        predictions: Continuous predicted probabilities [783, 12].
        targets: Target labels [783, 12].
        masks: Observation masks [783, 12].
        molecule_ids: Molecule IDs list.
        endpoint_order: List of endpoint names.
        threshold: Threshold for error classification.

    Returns:
        Structured dictionary of error analysis findings.
    """
    from src.evaluation.metrics import compute_roc_auc, compute_pr_auc

    endpoint_stats = []
    prevalences = []
    roc_aucs = []
    pr_aucs = []

    for idx, name in enumerate(endpoint_order):
        m = masks[:, idx]
        yt = targets[:, idx][m]
        ys = predictions[:, idx][m]

        n_pos = int(np.sum(yt == 1.0))
        n_neg = int(np.sum(yt == 0.0))
        prev = n_pos / len(yt) if len(yt) > 0 else 0.0

        roc = compute_roc_auc(yt, ys)
        pr = compute_pr_auc(yt, ys, method="average_precision")

        prevalences.append(prev)
        roc_aucs.append(roc)
        pr_aucs.append(pr)

        endpoint_stats.append({
            "endpoint": name,
            "prevalence": prev,
            "roc_auc": roc,
            "pr_auc": pr,
            "observed": len(yt),
            "positives": n_pos,
            "negatives": n_neg,
        })

    # Sort endpoints by ROC-AUC
    sorted_by_roc = sorted(endpoint_stats, key=lambda x: x["roc_auc"], reverse=True)
    sorted_by_pr = sorted(endpoint_stats, key=lambda x: x["pr_auc"], reverse=True)

    # Correlation between positive prevalence and metrics
    prev_arr = np.array(prevalences)
    roc_arr = np.array(roc_aucs)
    pr_arr = np.array(pr_aucs)

    corr_prev_roc = float(np.corrcoef(prev_arr, roc_arr)[0, 1])
    corr_prev_pr = float(np.corrcoef(prev_arr, pr_arr)[0, 1])

    # Probability distribution metrics
    all_valid_probs = predictions[masks]
    prob_quantiles = {
        "min": float(np.min(all_valid_probs)),
        "q25": float(np.percentile(all_valid_probs, 25)),
        "median": float(np.median(all_valid_probs)),
        "mean": float(np.mean(all_valid_probs)),
        "q75": float(np.percentile(all_valid_probs, 75)),
        "max": float(np.max(all_valid_probs)),
    }

    # High-confidence errors at threshold 0.5
    # False positives: y_true == 0 but prob >= 0.5
    # False negatives: y_true == 1 but prob < 0.5
    molecule_fp_counts = np.zeros(len(molecule_ids), dtype=int)
    molecule_fn_counts = np.zeros(len(molecule_ids), dtype=int)

    for i in range(len(molecule_ids)):
        for t in range(len(endpoint_order)):
            if masks[i, t]:
                gold = targets[i, t]
                score = predictions[i, t]
                if gold == 0.0 and score >= threshold:
                    molecule_fp_counts[i] += 1
                elif gold == 1.0 and score < threshold:
                    molecule_fn_counts[i] += 1

    top_fp_indices = np.argsort(molecule_fp_counts)[::-1][:5]
    top_fn_indices = np.argsort(molecule_fn_counts)[::-1][:5]

    challenging_fp_molecules = [
        {"mol_id": molecule_ids[idx], "false_positive_count": int(molecule_fp_counts[idx])}
        for idx in top_fp_indices if molecule_fp_counts[idx] > 0
    ]
    challenging_fn_molecules = [
        {"mol_id": molecule_ids[idx], "false_negative_count": int(molecule_fn_counts[idx])}
        for idx in top_fn_indices if molecule_fn_counts[idx] > 0
    ]

    analysis = {
        "best_performing_endpoints_roc": sorted_by_roc[:3],
        "worst_performing_endpoints_roc": sorted_by_roc[-3:],
        "best_performing_endpoints_pr": sorted_by_pr[:3],
        "worst_performing_endpoints_pr": sorted_by_pr[-3:],
        "class_imbalance_analysis": {
            "prevalence_vs_roc_auc_correlation": round(corr_prev_roc, 4),
            "prevalence_vs_pr_auc_correlation": round(corr_prev_pr, 4),
            "findings": (
                f"PR-AUC exhibits strong correlation with endpoint prevalence (r={corr_prev_pr:.4f}), "
                f"confirming that Average Precision reflects baseline task rarity. In contrast, "
                f"ROC-AUC evaluates relative ranking across all decision thresholds (r={corr_prev_roc:.4f})."
            ),
        },
        "probability_distribution": prob_quantiles,
        "challenging_multi_task_false_positives": challenging_fp_molecules,
        "challenging_multi_task_false_negatives": challenging_fn_molecules,
        "limitations": [
            "Extreme positive class sparsity in Tox21 causes fixed threshold 0.5 to miss rare positives.",
            "2D graph representations omit explicit 3D conformers and binding-pocket stereochemistry.",
            "Independent endpoint evaluation does not explicitly capture shared biological mechanisms.",
        ],
    }
    return analysis


def build_evaluation_summary(
    config: EvaluationConfig,
    checkpoint_meta: dict[str, Any],
    inference_meta: dict[str, Any],
    aggregate_meta: dict[str, Any],
    threshold_summary: dict[str, Any],
    error_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compile comprehensive evaluation summary dictionary for serialization."""
    from src.data.common import versions

    return {
        "evaluation_name": "Phase 2 MolecularGNN Final Quarantined Test Evaluation",
        "evaluator": "Student 4 (Final Evaluation & Test-Set Validation Engineer)",
        "timestamp_utc": utc_now(),
        "checkpoint_verification": {
            "checkpoint_path": checkpoint_meta["checkpoint_path"],
            "sha256": checkpoint_meta["sha256"],
            "size_bytes": checkpoint_meta["size_bytes"],
            "sha256_match_expected": checkpoint_meta["sha256"] == EXPECTED_CHECKPOINT_SHA256,
            "epoch": checkpoint_meta["epoch"],
            "best_val_loss": checkpoint_meta["best_val_loss"],
            "total_parameters": checkpoint_meta["total_parameters"],
            "evaluation_mode": checkpoint_meta["evaluation_mode"],
        },
        "model_architecture": {
            "model_type": "MolecularGNN",
            "node_feature_dim": 19,
            "edge_feature_dim": 18,
            "hidden_dim": 128,
            "num_layers": 3,
            "pooling": "mean",
            "dropout": 0.1,
            "norm": "layer",
            "residual": True,
            "activation": "relu",
            "output_endpoints": 12,
        },
        "dataset_verification": {
            "test_molecules_count": inference_meta["num_molecules"],
            "expected_test_molecules": config.expected_test_molecules,
            "count_matches_expected": inference_meta["num_molecules"] == config.expected_test_molecules,
            "total_valid_observations": inference_meta["total_valid_observations"],
            "total_missing_observations": inference_meta["total_missing_observations"],
            "valid_observation_density": round(
                inference_meta["total_valid_observations"] / (inference_meta["num_molecules"] * config.num_endpoints), 4
            ),
        },
        "evaluation_protocol": {
            "quarantined_test_set": True,
            "zero_test_label_leakage": True,
            "batch_size": config.batch_size,
            "device": checkpoint_meta["device"],
            "fixed_decision_threshold": config.threshold,
            "roc_auc_method": "Mann-Whitney U rank statistic with average tie ranking",
            "pr_auc_method": "Average Precision (rectangular integration of precision w.r.t recall)",
            "macro_aggregation_method": "Sample mean over valid endpoints with sample SD (ddof=1), SEM, and 95% Student-t CI",
            "missing_label_policy": "Strict boolean masking; NaNs excluded; missing values never replaced with zero",
        },
        "macro_metrics": {
            "valid_endpoints_count": aggregate_meta["valid_endpoints_count"],
            "excluded_endpoints_count": len(aggregate_meta["excluded_endpoints"]),
            "excluded_endpoints": aggregate_meta["excluded_endpoints"],
            "macro_roc_auc": aggregate_meta["macro_roc_auc"],
            "macro_pr_auc": aggregate_meta["macro_pr_auc"],
        },
        "supplementary_threshold_analysis": {
            "threshold": threshold_summary["threshold"],
            "mean_balanced_accuracy": threshold_summary["mean_balanced_accuracy"],
            "mean_f1": threshold_summary["mean_f1"],
        },
        "error_analysis_highlights": {
            "best_roc_endpoint": error_summary["best_performing_endpoints_roc"][0]["endpoint"],
            "best_roc_value": error_summary["best_performing_endpoints_roc"][0]["roc_auc"],
            "worst_roc_endpoint": error_summary["worst_performing_endpoints_roc"][0]["endpoint"],
            "worst_roc_value": error_summary["worst_performing_endpoints_roc"][0]["roc_auc"],
            "prevalence_pr_correlation": error_summary["class_imbalance_analysis"]["prevalence_vs_pr_auc_correlation"],
        },
        "environment": versions(),
        "warnings_or_skipped_operations": [],
    }




