# Phase 2 Final Graph Neural Network Evaluation & Test-Set Validation Report

**Student Role:** Student 4 — Final GNN Evaluation & Test-Set Validation Engineer  
**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Independent Test-Set Evaluation of Student 3's Best-Trained `MolecularGNN` on the Quarantined Test Partition ($N=783$)  
**Status:** **PASSED — Full Independent Validation Complete & Verified**  
**Evaluation Date:** 2026-11-19  

---

## 1. Executive Summary & Student 4 Responsibility

As **Student 4**, my core responsibility in Phase 2 is to conduct an independent, mathematically defensible, and leak-free final evaluation of the best-trained Graph Neural Network (GNN) model produced by Student 3, evaluated strictly against the quarantined test dataset containing exactly **783 molecular graphs**.

### 1.1 Strict Protection Rules & Quarantined Protocol
1. **Zero Data Leakage:** The test dataset was quarantined throughout model architecture exploration, hyperparameter tuning, and training. Test labels were never accessed during training, early stopping, or threshold tuning.
2. **Model Invariance:** Student 3's best checkpoint (`results/phase2/checkpoints/best_gnn.pt`) was evaluated in inference mode (`eval()`) with no retraining, fine-tuning, or parameter modification.
3. **Mask Preservation:** Tox21 assay missingness (untested compounds) was preserved via boolean masks. Missing labels were never imputed or replaced with zero.
4. **Deterministic Reproducibility:** Repeated inference using the verified checkpoint and canonical data yields identical prediction arrays within numerical tolerance ($\text{max } |\Delta| < 10^{-6}$).

### 1.2 Summary of Headline Findings
- **Total Test Molecular Graphs Evaluated:** Exactly **783** ($100.0\%$ accounted for, 0 duplicates, 0 missing).
- **Total Endpoints Evaluated:** Exactly **12** canonical Tox21 endpoints.
- **Valid Assay Observations:** $7,175$ valid labels out of $9,396$ total potential entries ($76.36\%$ observed assay density).
- **Macro-Averaged Test ROC-AUC:** **$0.7395 \pm 0.0182$ (SEM)** across all $K=12$ valid endpoints ($\text{SD} = 0.0629$, $95\%\text{ CI } [0.6995, 0.7795]$).
- **Macro-Averaged Test PR-AUC (Average Precision):** **$0.3118 \pm 0.0331$ (SEM)** across all $K=12$ valid endpoints ($\text{SD} = 0.1147$, $95\%\text{ CI } [0.2389, 0.3847]$).
- **Supplementary Fixed-Threshold (0.5) Analysis:** Mean Balanced Accuracy = $0.5599$, Mean $F_1$ Score = $0.1777$ (severely impacted by extreme positive class sparsity, highlighting the necessity of threshold-free rank metrics).

---

## 2. Dataset Partitioning & Split Integrity Verification

The evaluation operates on the canonical Bemis–Murcko scaffold split established in Phase 1 and preserved across Phase 2:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Cleaned Chemical Universe (N=7,823)                   │
│         data/processed/tox21_cleaned.csv (SHA-256: 2a30e42b...)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            Canonical Bemis-Murcko Scaffold Partitioning                │
│             data/processed/scaffold_split_indices.json                 │
├───────────────────────┬────────────────────────┬───────────────────────┤
│    Train Partition    │  Validation Partition  │    Test Partition     │
│  6,258 graphs (80.0%) │   782 graphs (10.0%)   │   783 graphs (10.0%)  │
│  Used by Student 3    │  Used for Val Loss &   │  QUARANTINED FOR      │
│  for optimization     │  Early Stopping        │  STUDENT 4 EVALUATION │
└───────────────────────┴────────────────────────┴───────────────────────┘
```

### 2.1 Split Partition Accounting
- **Train Split:** 6,258 molecular graphs ($79.995\%$).
- **Validation Split:** 782 molecular graphs ($9.996\%$).
- **Test Split (Quarantined):** Exactly **783 molecular graphs** ($10.009\%$).
- **Scaffold Isolation:** Re-verified zero non-empty scaffold overlap between partitions ($\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Row Alignment:** The 783 test graph identifiers in `data/processed/phase2/graphs.npz` strictly match `split["row_identifiers"]["test_mol_ids"]` in 1-to-1 order with zero transposition or duplicate indices.

---

## 3. Trained Model Checkpoint Verification

Before running inference, Student 3's best checkpoint was independently validated for cryptographic and architectural integrity:

### 3.1 Checkpoint Metadata
| Checkpoint Property | Value | Verification Status |
| :--- | :--- | :---: |
| **Artifact Path** | `results/phase2/checkpoints/best_gnn.pt` | Verified on disk |
| **File Size** | 5,126,861 bytes (~4.89 MB) | Exact match |
| **Cryptographic Hash (SHA-256)** | `a6445726e46d0be9357e05d6988deb7b83f99df4e9cc6419ad8d114cccffa972` | Verified match |
| **Optimal Checkpoint Epoch** | Epoch 20 (Early stopping patience 10) | Verified |
| **Best Validation Loss (Masked BCE)** | $0.2708488698684029$ | Verified |
| **Training Random Seed** | 42 | Verified |
| **Optimizer & Scheduler** | AdamW ($\text{lr}=10^{-3}$, $\text{wd}=10^{-4}$), ReduceLROnPlateau | Verified |

### 3.2 Model Architecture Breakdown
The loaded model is `MolecularGNN` (`src/models/molecular_gnn.py`):
- **Node Input Dimension ($d_v$):** 19 atom features (element one-hot, formal charge, hybridization, aromaticity, hydrogen count).
- **Edge Input Dimension ($d_e$):** 18 bond features (bond type one-hot, conjugation, ring membership, stereochemistry).
- **Hidden Embedding Dimension ($d_h$):** 128.
- **Message-Passing Layers:** 3 $\times$ `EdgeAwareMessagePassingLayer` with residual skip connections, Layer Normalization, and ReLU activations.
- **Global Readout Pooling:** Graph-level `mean` pooling across constituent atom representations $\to \mathbb{R}^{B \times 128}$.
- **Multi-Task Prediction Head:** 2-layer MLP (`Linear(128, 128) -> LayerNorm -> ReLU -> Dropout(0.1) -> Linear(128, 12)`).
- **Total Model Parameters:** 402,956 (100% trainable).
- **Inference Mode:** Verified `model.training is False` with device placement on CPU / CUDA.

---

## 4. Quarantined Test-Set Evaluation Methodology

The evaluation pipeline (`src/evaluation/phase2_test_evaluation.py`) executes under a strict multi-stage verification contract:

1. **Deterministic Batched Inference:** Minibatches of size 64 are collated using `iter_graph_batches` from `src.data.graph_dataset`.
2. **Sigmoid Probability Calibration:** Raw logits $\mathbf{Z} \in \mathbb{R}^{B \times 12}$ are converted to continuous predicted probabilities via $\hat{p}_{b, t} = \sigma(z_{b, t}) \in [0, 1]$.
3. **Array Validation:**
   - Prediction array shape: exactly $(783, 12)$.
   - Target array shape: exactly $(783, 12)$.
   - Validity mask array shape: exactly $(783, 12)$.
   - All probabilities verified finite ($0.0 \le \hat{p} \le 1.0$, zero NaNs, zero Infs).
4. **Mask-Aware Metric Computation:**
   - For each endpoint $t \in \{1, \dots, 12\}$, only observations where $\text{mask}_{b, t} = \text{True}$ are evaluated.
   - Missing targets (NaNs) are strictly filtered out without distortion.
   - **ROC-AUC:** Mann-Whitney U rank statistic with average tie correction.
   - **PR-AUC:** Average Precision (integral of Precision w.r.t Recall).

---

## 5. Comprehensive Test-Set Evaluation Results

### 5.1 Endpoint-Wise Performance & Accounting Table
Across all 12 endpoints, zero single-class slices or unobserved assays were encountered ($K_{\text{valid}} = 12$):

| Toxicity Endpoint | Status | Total Graphs | Observed Valid | Missing (NaN) | Positives ($N_+$) | Negatives ($N_-$) | Positive Prevalence | Test ROC-AUC | Test PR-AUC (AP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NR-AR** | `valid` | 783 | 722 | 61 | 25 | 697 | 3.46% | **0.7230** | 0.3221 |
| **NR-AR-LBD** | `valid` | 783 | 631 | 152 | 14 | 617 | 2.22% | **0.7680** | 0.3206 |
| **NR-AhR** | `valid` | 783 | 632 | 151 | 92 | 540 | 14.56% | **0.8337** | 0.4823 |
| **NR-Aromatase** | `valid` | 783 | 517 | 266 | 41 | 476 | 7.93% | **0.6295** | 0.1726 |
| **NR-ER** | `valid` | 783 | 566 | 217 | 69 | 497 | 12.19% | **0.6918** | 0.3167 |
| **NR-ER-LBD** | `valid` | 783 | 663 | 120 | 27 | 636 | 4.07% | **0.8186** | 0.3528 |
| **NR-PPAR-gamma** | `valid` | 783 | 579 | 204 | 28 | 551 | 4.84% | **0.6949** | 0.1024 |
| **SR-ARE** | `valid` | 783 | 472 | 311 | 99 | 373 | 20.97% | **0.6610** | 0.3551 |
| **SR-ATAD5** | `valid` | 783 | 663 | 120 | 25 | 638 | 3.77% | **0.7754** | 0.2547 |
| **SR-HSE** | `valid` | 783 | 561 | 222 | 44 | 517 | 7.84% | **0.7301** | 0.2602 |
| **SR-MMP** | `valid` | 783 | 527 | 256 | 114 | 413 | 21.63% | **0.7965** | 0.5169 |
| **SR-p53** | `valid` | 783 | 642 | 141 | 62 | 580 | 9.66% | **0.7515** | 0.2851 |
| **Total / Overall** | — | **783** | **7,175** | **2,221** | **640** | **6,535** | **8.92%** | — | — |

### 5.2 Macro-Averaged Evaluation & Uncertainty Quantities
Macro-averaging was conducted strictly across the $K=12$ valid endpoints according to project reporting standards:

$$\text{Macro Mean} = \frac{1}{K} \sum_{k=1}^K M_k, \quad \text{SD} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (M_k - \text{Mean})^2}, \quad \text{SEM} = \frac{\text{SD}}{\sqrt{K}}$$

- **Macro Test ROC-AUC:**
  - $\text{Mean} \pm \text{SEM}$: **$0.7395 \pm 0.0182$ (SEM, K=12)**
  - $\text{Mean} \pm \text{SD}$: $0.7395 \pm 0.0629$ (SD, K=12)
  - $95\%$ Student-$t$ Confidence Interval: **$[0.6995, 0.7795]$** ($t_{\text{crit}} = 2.201, df=11$)
- **Macro Test PR-AUC (Average Precision):**
  - $\text{Mean} \pm \text{SEM}$: **$0.3118 \pm 0.0331$ (SEM, K=12)**
  - $\text{Mean} \pm \text{SD}$: $0.3118 \pm 0.1147$ (SD, K=12)
  - $95\%$ Student-$t$ Confidence Interval: **$[0.2389, 0.3847]$** ($t_{\text{crit}} = 2.201, df=11$)

---

## 6. Supplementary Fixed-Threshold Analysis (Threshold = 0.5)

To provide an empirical perspective on binary decision boundaries without introducing test-label leakage, predictions were evaluated at a **fixed threshold of 0.5**:

$$\hat{y}_{b, t} = \mathbb{I}(\hat{p}_{b, t} \ge 0.5)$$

| Endpoint | True Pos (TP) | False Pos (FP) | True Neg (TN) | False Neg (FN) | Precision | Recall (Sens.) | Specificity | $F_1$ Score | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NR-AR** | 8 | 6 | 691 | 17 | 0.5714 | 0.3200 | 0.9914 | 0.4103 | 0.6557 |
| **NR-AR-LBD** | 5 | 7 | 610 | 9 | 0.4167 | 0.3571 | 0.9887 | 0.3846 | 0.6729 |
| **NR-AhR** | 16 | 12 | 528 | 76 | 0.5714 | 0.1739 | 0.9778 | 0.2667 | 0.5758 |
| **NR-Aromatase** | 0 | 0 | 476 | 41 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 |
| **NR-ER** | 10 | 9 | 488 | 59 | 0.5263 | 0.1449 | 0.9819 | 0.2273 | 0.5634 |
| **NR-ER-LBD** | 7 | 2 | 634 | 20 | 0.7778 | 0.2593 | 0.9969 | 0.3889 | 0.6281 |
| **NR-PPAR-gamma** | 0 | 0 | 551 | 28 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 |
| **SR-ARE** | 6 | 6 | 367 | 93 | 0.5000 | 0.0606 | 0.9839 | 0.1081 | 0.5223 |
| **SR-ATAD5** | 0 | 0 | 638 | 25 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 |
| **SR-HSE** | 0 | 0 | 517 | 44 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 |
| **SR-MMP** | 27 | 15 | 398 | 87 | 0.6429 | 0.2368 | 0.9637 | 0.3462 | 0.6003 |
| **SR-p53** | 0 | 0 | 580 | 62 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 |
| **Mean / Aggregate** | — | — | — | — | **0.3339** | **0.1294** | **0.9904** | **0.1777** | **0.5599** |

### 6.1 Critical Interpretation of Thresholded Metrics
1. **Severe Imbalance Penalty:** For assays with low positive prevalence ($< 5\%$, such as `NR-Aromatase`, `NR-PPAR-gamma`, `SR-ATAD5`, `SR-HSE`, and `SR-p53`), the model predicts continuous probabilities below $0.5$ for virtually all molecules. This results in $100\%$ specificity but zero true positives at $t=0.5$.
2. **High Specificity vs Low Sensitivity:** Mean specificity is $99.04\%$, while mean sensitivity is only $12.94\%$. When the model does predict $\hat{p} \ge 0.5$ (e.g. `NR-ER-LBD` precision $77.78\%$, `SR-MMP` precision $64.29\%$), confidence is high.
3. **Threshold-Free Justification:** This empirical finding confirms that thresholded metrics are uninformative for Tox21 under uncalibrated default thresholds ($0.5$), establishing ROC-AUC and PR-AUC as the only scientifically valid benchmark metrics.

---

## 7. In-Depth Error Analysis & Scientific Discussion

### 7.1 Best-Performing Endpoints
1. **`NR-AhR` (Aryl Hydrocarbon Receptor, ROC-AUC = $0.8337$, PR-AUC = $0.4823$):**
   - *Biochemical Rationale:* AhR binds rigid, planar polycyclic aromatic hydrocarbons (PAHs) and halogenated dioxins. The edge-aware MPNN effectively captures conjugated aromatic ring topologies through its aromaticity atom flag and conjugated bond feature.
2. **`NR-ER-LBD` (Estrogen Receptor Ligand-Binding Domain, ROC-AUC = $0.8186$, PR-AUC = $0.3528$):**
   - *Biochemical Rationale:* Steroidal pharmacophores contain a characteristic cyclopentanoperhydrophenanthrene 4-ring skeleton with specific phenolic hydrogen-bond donors, easily resolved by message-passing layers.
3. **`SR-MMP` (Mitochondrial Membrane Potential Disruption, ROC-AUC = $0.7965$, PR-AUC = $0.5169$):**
   - *Biochemical Rationale:* Mitochondrial uncouplers and ionophores frequently possess lipophilic cationic or ionizable acidic substructures that alter membrane potential, resulting in the highest PR-AUC ($0.5169$) across all endpoints.

### 7.2 Challenging & Underperforming Endpoints
1. **`NR-Aromatase` (Cytochrome P450 CYP19A1, ROC-AUC = $0.6295$, PR-AUC = $0.1726$):**
   - *Failure Mechanism:* CYP19A1 inhibition involves complex catalytic heme coordination ($Fe^{2+}$-porphyrin complexation) and steroidal substrate competitive antagonism. Standard 2D molecular graphs without 3D geometric orientation struggle to resolve non-planar coordinate covalent bonding.
2. **`SR-ARE` (Antioxidant Response Element, ROC-AUC = $0.6610$, PR-AUC = $0.3551$):**
   - *Failure Mechanism:* Nrf2-ARE is an indirect cellular stress pathway triggered by diverse electrophilic mechanisms (Michael acceptors, redox cycling quinones, alkylating agents). The absence of a single unified structural scaffold causes broad dispersion in latent graph space.
3. **`NR-PPAR-gamma` (PR-AUC = $0.1024$, ROC-AUC = $0.6949$):**
   - *Failure Mechanism:* Despite a fair ROC-AUC, PR-AUC is low due to extreme positive sparsity (only 28 positive samples in test set, 4.84% prevalence).

### 7.3 Prevalence vs Performance Sensitivity
Statistical correlation across the 12 endpoints demonstrates the mathematical divergence between ROC-AUC and PR-AUC:
- **Prevalence vs ROC-AUC ($r = +0.0898$):** Near-zero correlation. Confirms that ROC-AUC evaluates relative ranking across all possible operating thresholds and is mathematically invariant to positive class prevalence.
- **Prevalence vs PR-AUC ($r = +0.5056$):** Strong positive correlation. Because the random baseline for PR-AUC equals positive prevalence ($P / (P+N)$), endpoints with higher prevalence naturally attain higher Average Precision scores.

---

## 8. Benchmark Comparison: Phase 1 Baselines vs Phase 2 GNN

Under identical Bemis–Murcko scaffold split indices ($N_{\text{test}} = 783$), we compare Student 3's `MolecularGNN` against the Phase 1 classical machine learning models:

| Model Architecture | Input Representation | Macro Test ROC-AUC | Uncertainty ($\text{SEM}$) | $95\%$ Confidence Interval |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | Morgan Fingerprints (2,048) | 0.6983 | $\pm 0.0210$ | $[0.6521, 0.7445]$ |
| **Logistic Regression** | Physicochemical Descriptors (9) | 0.6412 | $\pm 0.0234$ | $[0.5897, 0.6927]$ |
| **Logistic Regression** | Combined (Morgan + Descriptors) | 0.7065 | $\pm 0.0205$ | $[0.6614, 0.7516]$ |
| **Random Forest** | Morgan Fingerprints (2,048) | 0.7412 | $\pm 0.0179$ | $[0.7018, 0.7806]$ |
| **Random Forest** | Physicochemical Descriptors (9) | 0.6834 | $\pm 0.0215$ | $[0.6361, 0.7307]$ |
| **Random Forest (Champion)** | Combined (Morgan + Descriptors) | **0.7488** | $\pm 0.0181$ | $[0.7090, 0.7886]$ |
| **XGBoost** | Combined (Morgan + Descriptors) | 0.7345 | $\pm 0.0192$ | $[0.6922, 0.7768]$ |
| **MolecularGNN (Student 3)** | **Molecular Graph ($V, E$)** | **0.7395** | $\pm 0.0182$ | $[0.6995, 0.7795]$ |

### 8.1 Architectural & Comparative Insights
1. **Competitive Parity:** `MolecularGNN` achieves **$0.7395 \pm 0.0182$**, performing within $0.0093$ of the Champion Random Forest Combined model ($0.7488 \pm 0.0181$). The confidence intervals overlap extensively, confirming that the GNN provides competitive discriminative ranking.
2. **Feature Learning vs Fixed Bit Hashing:** While Random Forest relies on fixed bit hashing collisions (ECFP4), `MolecularGNN` learns continuous atom and bond representations directly from molecular topology without feature engineering.
3. **Multi-Task Synergies:** The shared message-passing encoder enables parameter sharing across all 12 tasks, excelling on highly correlated nuclear receptor targets (e.g. `NR-AhR` 0.8337, `NR-ER-LBD` 0.8186).

---

## 9. Publication-Quality Figure Inventory

All seven figures were generated programmatically and saved to `reports/figures/phase2/` at $\ge 300\text{ DPI}$:

| Figure File | Dimensions / DPI | Description & Visual Contents |
| :--- | :---: | :--- |
| **`endpoint_roc_auc.png`** | 3000 $\times$ 1500 px (300 DPI) | Endpoint-wise ROC-AUC bar chart with horizontal macro-mean line ($0.7395$) and random guess baseline ($0.50$). |
| **`endpoint_pr_auc.png`** | 3000 $\times$ 1500 px (300 DPI) | Endpoint-wise PR-AUC (Average Precision) bar chart contrasted against diamond scatter markers of positive prevalence. |
| **`endpoint_roc_curves.png`** | 4200 $\times$ 3000 px (300 DPI) | 3 $\times$ 4 grid of Receiver Operating Characteristic curves (FPR vs TPR) with shaded area and individual endpoint AUCs. |
| **`endpoint_precision_recall_curves.png`** | 4200 $\times$ 3000 px (300 DPI) | 3 $\times$ 4 grid of Precision-Recall curves with horizontal dashed lines indicating baseline positive class prevalence. |
| **`prediction_probability_distributions.png`** | 4200 $\times$ 3000 px (300 DPI) | 3 $\times$ 4 grid comparing continuous predicted probability densities for ground-truth Inactive ($y=0$) vs Active ($y=1$) molecules. |
| **`endpoint_confusion_matrices.png`** | 4200 $\times$ 3000 px (300 DPI) | 3 $\times$ 4 grid of annotated $2 \times 2$ confusion matrices at fixed threshold 0.5 displaying TP, FP, TN, FN counts and $F_1$ scores. |
| **`endpoint_valid_label_distribution.png`** | 3300 $\times$ 1800 px (300 DPI) | Stacked horizontal bar chart showing observed active, observed inactive, and missing assay counts out of 783. |

---

## 10. Generated Artifact Inventory

All evaluation outputs were serialized atomically to `results/phase2/test_evaluation/`:

| Artifact Path | Format | Size | Description |
| :--- | :---: | :---: | :--- |
| **`test_predictions.npy`** | NumPy binary | 37 KB | Continuous predicted probabilities of shape `(783, 12)` in float32. |
| **`test_targets.npy`** | NumPy binary | 37 KB | Ground-truth assay labels of shape `(783, 12)` in float32 with NaNs for unobserved entries. |
| **`test_masks.npy`** | NumPy binary | 9.3 KB | Boolean validity masks of shape `(783, 12)` (`True` = observed valid assay). |
| **`test_molecule_ids.csv`** | CSV text | 9.6 KB | Exact 1-to-1 canonical alignment of all 783 test molecule IDs. |
| **`endpoint_metrics.csv`** | CSV text | 1.2 KB | Complete table of endpoint-wise samples, counts, prevalence, ROC-AUC, and PR-AUC. |
| **`aggregate_metrics.json`** | JSON text | 1.9 KB | Serialized macro-average metrics with SD, SEM, and 95% Student-$t$ Confidence Intervals. |
| **`evaluation_summary.json`** | JSON text | 3.3 KB | Complete audit summary including checkpoint verification, architecture, protocol, and environment. |
| **`test_evaluation.log`** | Text log | 3.0 KB | Detailed timestamped execution log of the full test evaluation pipeline. |

---

## 11. Automated Test Suite Verification

A dedicated automated test suite was created in `tests/test_phase2_test_evaluation.py` to ensure regression testing and long-term maintainability:

| Test Name | Focus Area | Status |
| :--- | :--- | :---: |
| `test_quarantined_test_set_size_and_disjointness` | Exact 783 count, zero scaffold overlap, split index disjointness | **PASS** |
| `test_checkpoint_integrity_and_sha256` | Checkpoint SHA-256 validation, config integrity, eval mode | **PASS** |
| `test_prediction_target_mask_dimensions_and_finiteness` | Array shapes `(783, 12)`, probability range $[0, 1]$, finite values | **PASS** |
| `test_molecule_id_uniqueness_and_canonical_order` | 783 unique IDs, zero duplicates, 1-to-1 scaffold split match | **PASS** |
| `test_mask_aware_metric_calculation_isolation` | Invariance to missing labels, zero leakage from NaNs | **PASS** |
| `test_single_class_and_zero_observed_edge_cases` | Robust handling of single-class slices without division by zero | **PASS** |
| `test_saved_artifacts_integrity_and_reloading` | Full roundtrip reloadability and mathematical consistency of CSV/JSON/NPY | **PASS** |
| `test_inference_repeatability_numerical_tolerance` | Repeated inference on identical data produces identical scores ($\le 10^{-6}$) | **PASS** |
| `test_all_seven_figures_exist_and_non_empty` | Verification of all 7 figures in `reports/figures/phase2/` ($> 100\text{ KB}$) | **PASS** |

Running the full repository test suite demonstrates zero regressions across all components:
```bash
pytest -q
# ======================== 211 passed, 1 skipped in 13.11s ========================
```

---

## 12. Reproducibility Guide

To reproduce the complete Phase 2 evaluation pipeline from first principles:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Build or verify graph cache against canonical scaffold split
python3 -m src.data.graph_preprocess

# 3. Execute Phase 2 final quarantined test evaluation
python3 -m src.evaluation.phase2_test_evaluation

# 4. Run automated test validation
pytest tests/test_phase2_test_evaluation.py -v
```

---

## 13. Progressive Git Commit Log

In accordance with Student 4 contribution guidelines, exactly eight meaningful, progressive commits were executed:

| Commit # | Commit Hash | Conventional Commit Message | Focus Area |
| :---: | :---: | :--- | :--- |
| **1** | `b9686b5` | `feat(evaluation): add phase 2 evaluation pipeline structure` | Pipeline architecture, config dataclass, logging, CLI options. |
| **2** | `a593d0a` | `feat(evaluation): integrate student 3 checkpoint loading` | Checkpoint SHA-256 verification, config check, eval mode setup. |
| **3** | `95e7399` | `feat(evaluation): implement quarantined test inference` | Batched inference (batch size 64), sigmoid, ID tracking, array saving. |
| **4** | `2cb56e3` | `feat(metrics): add mask-aware toxicity evaluation metrics` | Mask-aware ROC-AUC, Average Precision, macro SEM/CI aggregation. |
| **5** | `6110524` | `feat(analysis): add endpoint-wise error analysis` | Supplementary threshold 0.5 metrics, confusion matrices, error analysis. |
| **6** | `ef13076` | `feat(visualization): add phase 2 evaluation figures` | Generation of all 7 publication-quality figures at 300 DPI. |
| **7** | `e23e5f9` | `test(evaluation): add reproducibility and artifact validation tests` | 9 automated tests verifying dimensions, metrics, and reproducibility. |
| **8** | *(Current)* | `docs(evaluation): finalize phase 2 test evaluation report` | Final comprehensive evaluation report and benchmark reconciliation. |

---

## 14. Limitations and Future Directions

1. **Extreme Class Sparsity:** Endpoints with positive prevalence $< 5\%$ suffer severe false-negative rates under uncalibrated thresholding. Future work should investigate threshold calibration on validation curves or cost-sensitive focal loss message passing.
2. **2D Graph Representations:** Current atom/bond graphs do not encode 3D conformers, dihedral angles, or spatial pharmacophore coordinates, which are critical for binding pockets like CYP19A1 (`NR-Aromatase`).
3. **Multi-Task Transfer Learning:** Incorporating explicit task-relation graphs or cross-attention message passing could enhance prediction sharing between correlated receptor assays (e.g., AhR, ER, and AR).

---

## 15. Final Declaration

I confirm that this report reflects the actual, measured, and uncorrupted outputs of Student 3's best-trained `MolecularGNN` on the quarantined Tox21 test set ($N=783$). No metrics were fabricated, no test labels were leaked, and the checkpoint SHA-256 remained intact throughout all operations.
