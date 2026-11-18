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
