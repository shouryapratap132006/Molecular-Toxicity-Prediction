"""Comprehensive test suite for Phase 2 GNN Final Quarantined Test Evaluation.

Student 4 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.

Covers:
1. Exact quarantined test set size (783 molecules).
2. Prediction, target, and mask dimensions (783, 12).
3. Probability range [0, 1] and finiteness (zero NaNs/Infs).
4. Strict missing-label isolation and boolean masking.
5. Molecule identifier uniqueness, ordering, and duplicate detection.
6. Edge case robustness (zero valid observations, single-class partitions).
7. Artifact integrity (reloadability of npy, csv, json, log).
8. Reproducibility under identical checkpoint and data (numerical tolerance <= 1e-5).
9. Checkpoint SHA-256 and configuration invariants.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.common import ROOT, read_json
from src.data.graph_dataset import collate_graphs, iter_graph_batches
from src.data.graph_preprocess import DEFAULT_SPLIT, load_graph_datasets
from src.evaluation.metrics import compute_pr_auc, compute_roc_auc, compute_threshold_metrics, evaluate_endpoint
from src.evaluation.phase2_test_evaluation import (
    CANONICAL_ENDPOINTS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TASK_COUNT,
    EXPECTED_TEST_COUNT,
    EvaluationConfig,
    compute_mask_aware_metrics,
    compute_supplementary_threshold_analysis,
    run_test_inference,
    verify_and_load_checkpoint,
)


def test_quarantined_test_set_size_and_disjointness():
    """Verify test partition contains exactly 783 molecular graphs with 0 leakage."""
    datasets = load_graph_datasets()
    assert "test" in datasets
    test_data = datasets["test"]
    assert len(test_data) == EXPECTED_TEST_COUNT, (
        f"Test set size must be exactly {EXPECTED_TEST_COUNT}, got {len(test_data)}"
    )

    split = read_json(DEFAULT_SPLIT)
    test_indices = split["indices"]["test"]
    train_indices = split["indices"]["train"]
    val_indices = split["indices"]["val"]

    assert len(test_indices) == EXPECTED_TEST_COUNT
    assert set(test_indices).isdisjoint(set(train_indices))
    assert set(test_indices).isdisjoint(set(val_indices))

    # Verify molecule IDs match 1-to-1 in canonical order
    test_mol_ids = [test_data[i]["mol_id"] for i in range(len(test_data))]
    expected_ids = split["row_identifiers"]["test_mol_ids"]
    assert test_mol_ids == expected_ids


def test_checkpoint_integrity_and_sha256():
    """Verify Student 3 best GNN checkpoint SHA-256, configuration, and evaluation mode."""
    ckpt_path = Path("results/phase2/checkpoints/best_gnn.pt")
    if not ckpt_path.is_file():
        pytest.skip("Checkpoint best_gnn.pt not found on disk")

    model, meta = verify_and_load_checkpoint(ckpt_path, expected_sha256=EXPECTED_CHECKPOINT_SHA256)
    assert meta["sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert not model.training, "Loaded model must be in eval() mode"

    # Architecture configuration checks
    cfg = meta["model_config"]
    assert cfg["hidden_dim"] == 128
    assert cfg["num_layers"] == 3
    assert cfg["num_tasks"] == EXPECTED_TASK_COUNT
    assert cfg["pooling"] == "mean"
    assert meta["endpoint_order"] == CANONICAL_ENDPOINTS


def test_prediction_target_mask_dimensions_and_finiteness():
    """Verify saved arrays strictly match (783, 12) with finite probabilities in [0, 1]."""
    out_dir = Path("results/phase2/test_evaluation")
    pred_file = out_dir / "test_predictions.npy"
    target_file = out_dir / "test_targets.npy"
    mask_file = out_dir / "test_masks.npy"

    if not (pred_file.is_file() and target_file.is_file() and mask_file.is_file()):
        pytest.skip("Inference artifacts not yet generated")

    preds = np.load(pred_file)
    targets = np.load(target_file)
    masks = np.load(mask_file)

    # Shape checks
    assert preds.shape == (EXPECTED_TEST_COUNT, EXPECTED_TASK_COUNT)
    assert targets.shape == (EXPECTED_TEST_COUNT, EXPECTED_TASK_COUNT)
    assert masks.shape == (EXPECTED_TEST_COUNT, EXPECTED_TASK_COUNT)

    # Numerical validity
    assert np.all(np.isfinite(preds)), "All predictions must be finite"
    assert np.all((preds >= 0.0) & (preds <= 1.0)), "Predictions must be in [0, 1]"
    assert preds.dtype == np.float32

    # Target validity on observed subset
    observed_targets = targets[masks]
    assert np.all(np.isin(observed_targets, [0.0, 1.0])), "Observed targets must be strictly binary 0 or 1"

    # Missing label masking invariant: wherever mask is False, value must not be assumed 0
    assert np.sum(masks) < masks.size, "Dataset must contain untested missing labels"


def test_molecule_id_uniqueness_and_canonical_order():
    """Verify 783 test molecule identifiers are strictly unique and match split order."""
    csv_path = Path("results/phase2/test_evaluation/test_molecule_ids.csv")
    if not csv_path.is_file():
        pytest.skip("test_molecule_ids.csv not found")

    df = pd.read_csv(csv_path)
    assert len(df) == EXPECTED_TEST_COUNT
    assert "mol_id" in df.columns

    mol_ids = df["mol_id"].tolist()
    assert len(set(mol_ids)) == EXPECTED_TEST_COUNT, "Test molecule IDs must have zero duplicates"

    split = read_json(DEFAULT_SPLIT)
    assert mol_ids == split["row_identifiers"]["test_mol_ids"]


def test_mask_aware_metric_calculation_isolation():
    """Verify that unobserved labels (NaNs/mask=False) have zero mathematical impact on metrics."""
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.choice([0.0, 1.0], size=n, p=[0.8, 0.2])
    y_score = rng.uniform(0.0, 1.0, size=n)

    # Base metric with 100% observed
    base_roc = compute_roc_auc(y_true, y_score)
    base_pr = compute_pr_auc(y_true, y_score)

    # Append 50 missing values (NaNs)
    y_true_with_nans = np.concatenate([y_true, [np.nan] * 50])
    y_score_with_nans = np.concatenate([y_score, rng.uniform(0.0, 1.0, size=50)])

    nan_roc = compute_roc_auc(y_true_with_nans, y_score_with_nans)
    nan_pr = compute_pr_auc(y_true_with_nans, y_score_with_nans)

    # Invariance check: NaNs must produce exact same result
    assert np.isclose(base_roc, nan_roc, atol=1e-12)
    assert np.isclose(base_pr, nan_pr, atol=1e-12)


def test_single_class_and_zero_observed_edge_cases():
    """Verify single-class and unobserved slices return NaN without raising exceptions."""
    # All negative (no positive class)
    y_all_neg = np.zeros(50)
    scores = np.linspace(0.1, 0.9, 50)
    res_neg = evaluate_endpoint(y_all_neg, scores, endpoint_name="all_neg")
    assert np.isnan(res_neg["roc_auc"])
    assert np.isnan(res_neg["pr_auc"])
    assert res_neg["status"] == "single_class_undefined"

    # All positive (no negative class)
    y_all_pos = np.ones(50)
    res_pos = evaluate_endpoint(y_all_pos, scores, endpoint_name="all_pos")
    assert np.isnan(res_pos["roc_auc"])
    assert np.isnan(res_pos["pr_auc"])
    assert res_pos["status"] == "single_class_undefined"

    # All missing (NaNs)
    y_all_nan = np.full(50, np.nan)
    res_nan = evaluate_endpoint(y_all_nan, scores, endpoint_name="all_nan")
    assert np.isnan(res_nan["roc_auc"])
    assert np.isnan(res_nan["pr_auc"])
    assert res_nan["status"] == "no_observed_samples"


def test_saved_artifacts_integrity_and_reloading():
    """Verify CSV, JSON, and NPY evaluation artifacts are parseable and mathematically consistent."""
    out_dir = Path("results/phase2/test_evaluation")
    summary_path = out_dir / "evaluation_summary.json"
    csv_path = out_dir / "endpoint_metrics.csv"
    agg_path = out_dir / "aggregate_metrics.json"

    if not (summary_path.is_file() and csv_path.is_file() and agg_path.is_file()):
        pytest.skip("Evaluation artifacts not present")

    # JSON loading
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["dataset_verification"]["test_molecules_count"] == EXPECTED_TEST_COUNT
    assert summary["macro_metrics"]["valid_endpoints_count"] == EXPECTED_TASK_COUNT

    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)
    assert agg["valid_endpoints_count"] == EXPECTED_TASK_COUNT
    assert "macro_roc_auc" in agg
    assert "macro_pr_auc" in agg

    # CSV loading
    df = pd.read_csv(csv_path)
    assert len(df) == EXPECTED_TASK_COUNT
    assert df["endpoint"].tolist() == CANONICAL_ENDPOINTS
    assert (df["status"] == "valid").all()
    assert (df["observed"] + df["missing"] == EXPECTED_TEST_COUNT).all()

    # Recalculated macro mean matches aggregate JSON
    mean_roc_csv = float(df["roc_auc"].mean())
    mean_pr_csv = float(df["pr_auc"].mean())
    assert np.isclose(mean_roc_csv, agg["macro_roc_auc"]["mean"], atol=1e-5)
    assert np.isclose(mean_pr_csv, agg["macro_pr_auc"]["mean"], atol=1e-5)


def test_inference_repeatability_numerical_tolerance():
    """Verify repeated inference on identical data produces identical predictions within 1e-5."""
    ckpt_path = Path("results/phase2/checkpoints/best_gnn.pt")
    if not ckpt_path.is_file():
        pytest.skip("Checkpoint best_gnn.pt not found on disk")

    model, _ = verify_and_load_checkpoint(ckpt_path)
    datasets = load_graph_datasets()
    test_batch = next(iter_graph_batches(datasets["test"], batch_size=32))

    with torch.no_grad():
        logits_1 = model(test_batch)
        probs_1 = torch.sigmoid(logits_1).cpu().numpy()

        logits_2 = model(test_batch)
        probs_2 = torch.sigmoid(logits_2).cpu().numpy()

    np.testing.assert_allclose(probs_1, probs_2, rtol=1e-5, atol=1e-6)


def test_all_seven_figures_exist_and_non_empty():
    """Verify all seven required publication figures exist and are valid PNG images."""
    fig_dir = Path("reports/figures/phase2")
    required_figures = [
        "endpoint_roc_auc.png",
        "endpoint_pr_auc.png",
        "endpoint_roc_curves.png",
        "endpoint_precision_recall_curves.png",
        "prediction_probability_distributions.png",
        "endpoint_confusion_matrices.png",
        "endpoint_valid_label_distribution.png",
    ]

    for fig_name in required_figures:
        p = fig_dir / fig_name
        assert p.is_file(), f"Missing required figure: {fig_name}"
        assert p.stat().st_size > 10000, f"Figure file suspiciously small: {p.stat().st_size} bytes"

