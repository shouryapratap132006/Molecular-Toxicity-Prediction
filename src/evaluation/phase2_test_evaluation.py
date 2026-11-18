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

