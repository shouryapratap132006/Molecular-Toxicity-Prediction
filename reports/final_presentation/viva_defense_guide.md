# B.Tech Project Viva Defense & Faculty Examination Guide

**Project Title:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Department:** Computer Science & Artificial Intelligence, Newton School of Technology  
**Team Members:**  
- **Student 1 (Data & Preprocessing Owner):** Shourya Pratap — RDKit sanitization, Bemis–Murcko scaffold partitioning, atom/bond featurization, graph collation.  
- **Student 2 (Architecture Owner):** Bulbul Agarwalla — MolecularGNN design, edge-conditioned message passing, residual layer normalization, graph readout pooling.  
- **Student 3 (Training Pipeline Owner):** Amogha Raj Sindur — Masked BCE multi-task loss, AdamW optimization, learning rate schedulers, early stopping, checkpointing.  
- **Student 4 (Evaluation & Validation Owner):** Kanishka Dubey — Quarantined test evaluation ($N=783$), mask-aware ROC-AUC and PR-AUC, calibration, error diagnostics.  
- **Student 5 (Integration, Experiments & Documentation Owner):** Aditya Yadav — End-to-end integration, centralized experiment runner, systematic ablation matrix, cross-phase benchmarking, IEEE report, and presentation deck.  

---

## 1. Quick Project Cheat Sheet (Numbers You Must Know by Heart)

| Dimension | Exact Value |
| :--- | :--- |
| **Cleaned Molecules ($N$)** | **7,823** (from 7,831 raw records; 8 rejected by RDKit sanitization) |
| **Canonical Scaffold Splits** | **Train:** 6,258 (80%) \| **Val:** 782 (10%) \| **Test:** 783 (10%) |
| **Assay Endpoints** | Exactly **12** (7 Nuclear Receptors + 5 Stress Response pathways) |
| **Observed Assay Density** | **76.36%** ($7,175$ valid labels out of $9,396$ potential test entries; 23.64% missing) |
| **Positive Class Prevalence** | Ranging from **1.5%** (NR-AR-LBD) to **15.2%** (SR-MMP) |
| **Atom / Node Features** | **19 dimensions** (atomic identity, degree, charge, aromaticity, hydrogens, hybridization) |
| **Bond / Edge Features** | **18 dimensions** (bond order, conjugation, ring membership, stereochemistry) |
| **Phase 1 Champion** | **Random Forest (Combined FP + Descriptors):** ROC-AUC **0.7488 ± 0.0185**, PR-AUC **0.3891 ± 0.0442** |
| **Phase 2 Champion** | **MolecularGNN (Edge-MPNN):** ROC-AUC **0.7395 ± 0.0182**, PR-AUC **0.3118 ± 0.0331** |
| **Statistical Comparison** | **Statistical Parity:** Overlapping 95% CIs ($[0.6995, 0.7795]$ vs $[0.7125, 0.7851]$); $\Delta \text{ROC-AUC} = 0.0093 < \text{SEM}$ |

---

## 2. Top 10 Viva Questions & Model Answers

### Q1: Why did your GNN not dramatically outperform Random Forest? Is your GNN a failure?
> **Model Answer:**  
> "Absolutely not. In fact, matching Random Forest's performance within statistical parity ($\Delta \text{ROC-AUC} = 0.0093$, with overlapping 95% confidence intervals) under a strict Bemis–Murcko scaffold split is a major scientific achievement.  
> First, Random Forest had access to 2,057 heavily engineered dimensions—2,048 circular fingerprint bits (ECFP4) and 9 curated RDKit physicochemical descriptors. In contrast, our `MolecularGNN` received zero handcrafted molecular properties; it learned representations directly from raw atom and bond features via iterative message passing.  
> Second, our finding is completely consistent with modern cheminformatics literature, such as Jiang et al. (2021) in *Nature Machine Intelligence*, which showed that across 11 molecular benchmarks, classical tree ensembles frequently match or outperform graph neural networks on small-to-medium tabular datasets ($N < 10,000$). Deep graph architectures shine during transfer learning or pre-training on millions of molecules, but on 6,258 training graphs, reaching parity with an engineered ensemble validates our graph representation learning pipeline."

---

### Q2: What is Bemis–Murcko scaffold splitting, and why did you not use a simple random train/test split?
> **Model Answer:**  
> "A random split in molecular datasets causes severe 'scaffold leakage' or optimistic evaluation bias. In chemistry, molecules cluster into congeneric series sharing identical central core structures (scaffolds). If we split randomly, structural analogues of test molecules end up in the training set. The model would merely memorize scaffold-activity correlations rather than learning true molecular toxicity mechanisms.  
> Bemis–Murcko scaffold splitting isolates the core ring systems and linkers from side chains, clustering compounds by their frameworks. We then sort scaffold clusters and allocate entire scaffolds strictly into either train, validation, or test partitions ($S_{\text{train}} \cap S_{\text{test}} = \emptyset$). This simulates real-world lead discovery: predicting toxicity on completely novel chemical series that medicinal chemists have never synthesized before."

---

### Q3: Why is 23.6% of the Tox21 dataset missing, and how did you handle these missing labels? Did you impute them?
> **Model Answer:**  
> "The Tox21 benchmark is a compilation of robotic high-throughput screening bioassays. Not every compound was tested in every assay due to compound volatility, autofluorescence, limited compound volume, or assay incompatibility.  
> We strictly rejected zero-imputation or mean-imputation. Imputing unobserved assays as 'inactive' (0) introduces massive false-negative contamination into the training loss. Instead, we implemented **mask-aware multi-task training**:  
> Each sample is paired with a binary observation mask $m \in \{0, 1\}^{12}$. In the loss function:  
> $$\mathcal{L} = \frac{\sum_{k=1}^{12} m_k \cdot \text{BCE}(y_k, \sigma(\hat{z}_k))}{\sum_{k=1}^{12} m_k}$$  
> Only observed endpoints contribute gradients. Similarly, during Student 4's test evaluation, metrics are computed only over valid observations ($7,175$ labels across 783 test molecules)."

---

### Q4: Why did you report ROC-AUC and PR-AUC instead of standard Classification Accuracy or F1-Score?
> **Model Answer:**  
> "Accuracy is mathematically meaningless under extreme class imbalance. In Tox21, positive toxicity prevalence is as low as 1.5% (e.g., NR-AR-LBD). A trivial dummy classifier that always predicts 'non-toxic' achieves 98.5% accuracy while completely failing to detect any toxic compound!  
> F1-score requires fixing an arbitrary decision threshold (e.g., $t = 0.5$). However, because positive prevalence is low, continuous predicted probabilities are naturally calibrated to lower ranges, causing an arbitrary 0.5 threshold to collapse recall.  
> ROC-AUC evaluates the ranking capability: the probability that a randomly chosen toxic molecule receives a higher predicted risk score than a non-toxic molecule, independent of threshold.  
> PR-AUC (Average Precision) specifically evaluates precision across recall levels on the rare positive class, penalizing false positives far more severely than ROC-AUC."

---

### Q5: How does your Message Passing Neural Network actually compute molecular representations? Walk us through the math.
> **Model Answer:**  
> "Our `MolecularGNN` operates in four distinct stages:  
> 1. **Initial Embedding:** Raw node features $\mathbf{x}_i \in \mathbb{R}^{19}$ and edge features $\mathbf{e}_{ij} \in \mathbb{R}^{18}$ are projected into a $d=128$ hidden space: $\mathbf{h}_i^{(0)} = \text{ReLU}(\mathbf{W}_v \mathbf{x}_i + \mathbf{b}_v)$ and $\mathbf{h}_{ij}^e = \text{ReLU}(\mathbf{W}_e \mathbf{e}_{ij} + \mathbf{b}_e)$.  
> 2. **Edge-Aware Message Construction:** For layer $l \in \{1, 2, 3\}$, atom $j$ sends a message to atom $i$ conditioned on the bond between them: $\mathbf{m}_{ij}^{(l)} = \text{MLP}([\mathbf{h}_j^{(l-1)} \,\|\, \mathbf{h}_{ij}^e])$.  
> 3. **Aggregation & Residual Update:** Neighbor messages are aggregated via summation $\mathbf{\bar{m}}_i^{(l)} = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(l)}$ and used to update node states with a skip-connection and LayerNorm:  
>    $$\mathbf{h}_i^{(l)} = \text{LayerNorm}(\mathbf{h}_i^{(l-1)} + \text{Dropout}(\text{ReLU}(\mathbf{W}_u [\mathbf{h}_i^{(l-1)} \,\|\, \mathbf{\bar{m}}_i^{(l)}])))$$  
> 4. **Graph Readout Pooling:** Size-invariant mean pooling averages node vectors across the molecule: $\mathbf{h}_G = \frac{1}{|V|} \sum_{i \in V} \mathbf{h}_i^{(3)}$.  
> 5. **Multi-Task Head:** A linear projection generates 12 independent logits: $\hat{\mathbf{z}} = \mathbf{W}_{\text{out}} \mathbf{h}_G + \mathbf{b}_{\text{out}}$."

---

### Q6: In your Phase 2 ablations, why was Mean Pooling superior to Sum Pooling ($0.2771$ vs $0.3047$)?
> **Model Answer:**  
> "Molecules vary greatly in size. In our dataset, molecules range from tiny fragments with 2–4 heavy atoms up to large macrocycles with 50+ atoms.  
> Sum pooling scales linearly with the number of atoms: $|V| \cdot \bar{h}$. As a result, large non-toxic molecules produce vector norms an order of magnitude larger than small toxic molecules, introducing an irrelevant structural size bias directly into the classification layer.  
> Mean pooling divides by $|V|$, normalizing for molecular size and isolating chemical feature density. This stabilization directly produced lower validation loss ($0.2771$ vs $0.3047$)."

---

### Q7: Why did you choose 3 message passing layers instead of 5 or 10?
> **Model Answer:**  
> "In molecular graphs, $L$ layers correspond to an $L$-hop receptive field. In organic molecules, the typical topological diameter of functional groups and aromatic ring systems is 3 to 5 bonds.  
> With $L=3$, an atom can gather context from adjacent rings, heteroatoms, and functional pharmacophores.  
> Increasing depth beyond 3 or 4 layers in GNNs frequently causes **over-smoothing**, where node representations converge toward identical average vectors due to repeated Laplacian-style averaging across connected components. Furthermore, with only 6,258 training molecules, deeper networks dramatically increase parameter count and risk overfitting."

---

### Q8: What did positive class weighting (`pos_weight`) do, and why did you not use it for your baseline test evaluation?
> **Model Answer:**  
> "In `gnn_pos_weighted`, we scaled positive errors by the inverse prevalence ratio: $w_k = \frac{N_{\text{neg}}}{N_{\text{pos}}}$, where $w_k$ ranged from 5.6 to 39.4.  
> While positive weighting forces the optimizer to prioritize sensitivity (recall) at a default threshold of $0.5$, it artificially skews the calibration of raw predicted probabilities toward 1.0.  
> For threshold-free rank metrics like ROC-AUC and PR-AUC, unweighted cross-entropy provides smoother, better-calibrated probability distributions. Hence, the standard unweighted loss is the theoretically sound baseline for benchmark evaluations."

---

### Q9: How did you ensure zero data leakage between Phase 1 and Phase 2?
> **Model Answer:**  
> "Zero data leakage was enforced through five strict engineering measures:  
> 1. **Immutable Partition Indices:** Scaffold split assignments were frozen in `data/processed/scaffold_split_indices.json` in Phase 1 and imported read-only into Phase 2.  
> 2. **Feature Scaler Fitting:** All continuous descriptor normalization (StandardScaler) was fitted strictly on training rows ($N=6,258$).  
> 3. **Quarantined Test Split:** The 783 test graphs were never touched during model design, training, loss calculation, or early stopping.  
> 4. **Model Invariance:** Evaluation used `model.eval()` on frozen checkpoint weights verified with SHA-256 checksums.  
> 5. **Independent Evaluator:** Student 4 ran evaluation as an isolated pipeline that consumed checkpoints and outputted immutable metric artifacts."

---

### Q10: What are the main limitations of your research, and what would you do next?
> **Model Answer:**  
> "Our study has three key limitations:  
> 1. **2D Topology vs. 3D Conformations:** Our GNN operates strictly on 2D chemical connectivity. True biological receptor-ligand binding depends on 3D spatial conformations, stereoelectronic surface charges, and dihedral angles. Incorporating 3D equivariant GNNs (such as EGNN or SchNet) would be the logical next step.  
> 2. **Benchmark Scope:** Our findings are grounded on the Tox21 dataset. Evaluating generalization to other benchmarks (e.g., ClinTox, SIDER, ToxCast) would determine if parity with Random Forest holds universally.  
> 3. **Pre-training:** Our model was trained from scratch on 6,258 molecules. Pre-training on 2 million molecules from ChEMBL or ZINC using self-supervised masked atom prediction before fine-tuning would likely propel the GNN beyond classical tree models."

---

## 3. Student-by-Student Viva Defense Script

### If Asked: "Who did what on the team?"
- **Shourya Pratap (Student 1):**  
  *"I owned the data engineering and preprocessing pipeline. I curated the raw Tox21 dataset using RDKit, stripping salts, normalizing stereochemistry, and generating the canonical Bemis–Murcko scaffold split. In Phase 2, I featurized each molecule into a graph with 19 node and 18 edge features, and implemented PyTorch collation."*
- **Bulbul Agarwalla (Student 2):**  
  *"I designed and implemented the `MolecularGNN` architecture in PyTorch. I built the node and edge embedding layers, the edge-conditioned message passing layers with residual connections and layer normalization, the size-invariant global mean pooling, and the 12-task prediction head."*
- **Amogha Raj Sindur (Student 3):**  
  *"I developed the multi-task training and optimization pipeline. I implemented the masked BCEWithLogitsLoss to handle missing assay data without imputation, integrated the Adam optimizer and ReduceLROnPlateau scheduler, and built the early stopping, checkpointing, and reproducibility tracking systems."*
- **Kanishka Dubey (Student 4):**  
  *"I owned independent test evaluation and validation. I evaluated the best checkpoint on the 783 quarantined test graphs, computed mask-aware ROC-AUC and PR-AUC with standard errors and 95% confidence intervals, performed threshold and error diagnostics, and generated publication figures."*
- **Aditya Yadav (Student 5):**  
  *"I was responsible for integration, experiment orchestration, and final documentation. I built the central YAML experiment runner, executed the systematic ablation matrix across 6 controlled settings, audited cross-phase performance parity, authored the final IEEE research manuscript, and prepared the final viva defense slide deck."*
