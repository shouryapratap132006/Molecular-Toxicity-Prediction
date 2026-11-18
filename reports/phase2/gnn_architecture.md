# Phase 2: Molecular Graph Neural Network (GNN) Architecture

**Author:** Student 2 (Bulbul Agarwalla)  
**Role:** Molecular Graph Neural Network Architecture Owner  
**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Phase 2 — Relational Message Passing Neural Network (MPNN) Architecture  
**Status:** Implemented, Tested, and Validated against Student 1 Graph Pipeline  

---

## 1. Executive Summary & Architecture Flow

This report defines the Phase 2 molecular Graph Neural Network (GNN) architecture designed to consume preprocessed molecular graphs from Student 1 and predict all 12 Tox21 toxicity endpoints. 

Unlike Phase 1 classical baselines that operate on fixed, lossy chemical fingerprints (Morgan/ECFP4) and handcrafted physicochemical descriptors, the Phase 2 GNN operates directly on molecular graph topologies, learning continuous representations through edge-aware message passing.

```
                  ┌────────────────────────────────────────────────────────┐
                  │               Input SMILES String                      │
                  └──────────────────────────┬─────────────────────────────┘
                                             │ (Student 1 RDKit Preprocessing)
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Molecular Graph: Atoms (19-dim), Bonds (18-dim)       │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Atom & Bond Encoders (NodeEncoder & EdgeEncoder)       │
                  │   x: [N, 19] -> [N, D]  |  edge_attr: [E, 18] -> [E, D]│
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ L x Edge-Aware Message Passing Layers                  │
                  │   - Directional bond-conditioned messages              │
                  │   - Deterministic index_add aggregation                │
                  │   - Node update + Residual + LayerNorm + Dropout       │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Graph-Level Readout Pooling (mean / sum / max)         │
                  │   h: [N_total, D] -> h_graph: [B, D]                   │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Multi-Task Prediction Head                             │
                  │   h_graph: [B, D] -> Linear -> ReLU -> Linear -> [B,12]│
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ 12 Raw Toxicity Logits (BCEWithLogitsLoss Ready)       │
                  └────────────────────────────────────────────────────────┘
```

---

## 2. Input Feature Dimensions & Representation

The architecture strictly adheres to Student 1's version 1 feature schema:

| Feature Tensor | Conceptual Shape | Description | Source Details |
| :--- | :--- | :--- | :--- |
| **`x`** | `[num_nodes, 19]` | Atom-level features (atomic number, degree, formal charge, aromaticity, attached H, in-ring, one-hot hybridization, one-hot chirality). | `src/data/graph_features.py:ATOM_FEATURES` |
| **`edge_attr`** | `[num_edges, 18]` | Bond-level features (conjugated, in-ring, 7 bond-type flags, 9 stereochemistry flags). | `src/data/graph_features.py:BOND_FEATURES` |
| **`edge_index`** | `[2, num_edges]` | Directed edge index pairs $(u \to v)$ in `int64`. Each chemical bond contributes two directed edges. | `src/data/graph_dataset.py` |
| **`batch`** | `[num_nodes]` | 1D index mapping each node to its graph in $\{0, \dots, B-1\}$. | `collate_graphs` |
| **`y`** | `[B, 12]` | Target ground-truth toxicity indicators. Missing assays are `NaN`. | `normalize_labels` |
| **`mask`** | `[B, 12]` | Boolean indicators ($1 = \text{observed}, 0 = \text{missing}$). | `normalize_labels` |

---

## 3. Detailed Component Formulation

### 3.1 Input Feature Encoders (`src/models/encoders.py`)

Raw atom and bond features are projected from their discrete/heterogeneous input spaces into a unified latent space of dimension $D$ (`hidden_dim`):

$$\mathbf{h}_v^{(0)} = \text{LayerNorm}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{x}_v + \mathbf{b}_1) + \mathbf{b}_2) \in \mathbb{R}^D$$
$$\mathbf{e}_{uv} = \text{LayerNorm}(\mathbf{W}_{e2} \cdot \text{ReLU}(\mathbf{W}_{e1} \mathbf{edge\_attr}_{uv} + \mathbf{b}_{e1}) + \mathbf{b}_{e2}) \in \mathbb{R}^D$$

- **Zero-Bond Handling:** If a molecule is monoatomic (e.g., an isolated ion with 0 bonds), `edge_attr` has shape `[0, 18]`. The `EdgeEncoder` safely returns an empty tensor `[0, D]` without breaking tensor shapes or gradients.

### 3.2 Relational Message Passing Layer (`src/models/message_passing.py`)

Each layer $l \in \{1, \dots, L\}$ performs edge-conditioned message passing:

1. **Message Generation:**  
   For every directed edge $u \to v$ connecting source atom $u$ to target atom $v$ across bond $e_{uv}$:
   $$\mathbf{m}_{u \to v}^{(l)} = \text{MLP}_{\text{msg}}\left( \left[ \mathbf{h}_u^{(l-1)} \,\|\, \mathbf{h}_v^{(l-1)} \,\|\, \mathbf{e}_{uv} \right] \right) \in \mathbb{R}^D$$
   where $[\cdot \,\|\, \cdot]$ denotes concatenation of the $3 \times D$ dimensional input.

2. **Message Aggregation:**  
   Incoming messages are aggregated at each target atom $v$ via deterministic summation:
   $$\mathbf{m}_v^{(l)} = \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{u \to v}^{(l)}$$
   Implemented via PyTorch's native `index_add_` over target indices `edge_index[1]`. If an atom has degree 0 (no incoming edges), $\mathbf{m}_v^{(l)} = \mathbf{0}$.

3. **Node State Update:**  
   The atom combines its previous hidden state with aggregated messages:
   $$\mathbf{h}_v^{\text{update}} = \text{MLP}_{\text{update}}\left( \left[ \mathbf{h}_v^{(l-1)} \,\|\, \mathbf{m}_v^{(l)} \right] \right) \in \mathbb{R}^D$$

4. **Residual Connection & Layer Normalization:**  
   To prevent vanishing gradients and over-smoothing across deep layers:
   $$\mathbf{h}_v^{(l)} = \text{LayerNorm}\left( \mathbf{h}_v^{(l-1)} + \text{Dropout}(\mathbf{h}_v^{\text{update}}) \right)$$

### 3.3 Graph-Level Readout Pooling (`src/models/graph_pooling.py`)

To obtain a single molecular embedding vector $\mathbf{h}_G \in \mathbb{R}^D$ from the variable number of constituent atoms $N_G$:

- **Mean Pooling (Default):**
  $$\mathbf{h}_G = \frac{1}{|V_G|} \sum_{v \in V_G} \mathbf{h}_v^{(L)}$$
- **Sum Pooling:**
  $$\mathbf{h}_G = \sum_{v \in V_G} \mathbf{h}_v^{(L)}$$
- **Max Pooling:**
  $$\mathbf{h}_G = \max_{v \in V_G} \mathbf{h}_v^{(L)}$$

**Leakage Guarantee:** Pooling is indexed strictly by the `batch` tensor. Automated unit test `test_no_cross_graph_leakage` mathematically verifies that evaluating graph $A$ alone produces logits identical to within $10^{-5}$ tolerance compared to evaluating graph $A$ inside a heterogeneous batch with graph $B$.

### 3.4 Multi-Task Prediction Head (`src/models/molecular_gnn.py`)

The pooled molecular embedding $\mathbf{h}_G$ is passed through a 2-layer prediction MLP:

$$\mathbf{z}_G = \mathbf{W}_{\text{out2}} \cdot \text{ReLU}\left( \mathbf{W}_{\text{out1}} \mathbf{h}_G + \mathbf{b}_{\text{out1}} \right) + \mathbf{b}_{\text{out2}} \in \mathbb{R}^{12}$$

**Key Properties:**
- **Raw Logits:** The head outputs raw unconstrained logits $\mathbf{z}_G \in \mathbb{R}^{12}$, NOT sigmoid probabilities. This is strictly required for numerical stability with `torch.nn.functional.binary_cross_entropy_with_logits`.
- **Inference Probabilities:** For evaluation or deployment, `model.predict_proba(batch)` returns $\sigma(\mathbf{z}_G) \in [0.0, 1.0]^{12}$.
- **Canonical Endpoint Order:** The 12 outputs correspond strictly to the Phase 1 canonical Tox21 ordering:
  1. `NR-AR`
  2. `NR-AR-LBD`
  3. `NR-AhR`
  4. `NR-Aromatase`
  5. `NR-ER`
  6. `NR-ER-LBD`
  7. `NR-PPAR-gamma`
  8. `SR-ARE`
  9. `SR-ATAD5`
  10. `SR-HSE`
  11. `SR-MMP`
  12. `SR-p53`

---

## 4. Tensor Shapes at Each Stage

For a minibatch of $B$ molecules containing a total of $N = \sum_{i=1}^B N_i$ atoms and $E = \sum_{i=1}^B E_i$ directed bonds:

| Stage | Input Tensor(s) | Output Tensor | Shape |
| :--- | :--- | :--- | :--- |
| **Input** | `x`, `edge_index`, `edge_attr`, `batch` | — | `[N, 19]`, `[2, E]`, `[E, 18]`, `[N]` |
| **Node Encoding** | `x` | `h_0` | `[N, hidden_dim]` |
| **Edge Encoding** | `edge_attr` | `e` | `[E, hidden_dim]` |
| **Message Passing (L layers)**| `h_{l-1}`, `edge_index`, `e` | `h_l` | `[N, hidden_dim]` |
| **Graph Pooling** | `h_L`, `batch` | `h_graph` | `[B, hidden_dim]` |
| **Prediction Head** | `h_graph` | `logits` | `[B, 12]` |

---

## 5. Model Configuration & Hyperparameter Defaults

All parameters are encapsulated in `GNNConfig` (`src/models/model_config.py`):

```python
from src.models import GNNConfig, MolecularGNN

config = GNNConfig(
    node_feature_dim=19,      # Raw atom features from Student 1
    edge_feature_dim=18,      # Raw bond features from Student 1
    hidden_dim=128,           # Latent dimension across all layers
    num_layers=3,             # Depth of message passing
    num_tasks=12,             # Number of Tox21 assays
    dropout=0.1,              # Regularization dropout probability
    pooling="mean",           # Readout aggregation: 'mean', 'sum', 'max'
    residual=True,            # Enable skip-connections
    norm="layer",             # Normalization: 'layer', 'batch', 'none'
    activation="relu",        # Activation: 'relu', 'gelu', 'silu', 'leaky_relu'
)

model = MolecularGNN(config=config)
```

---

## 6. How Student 3 (Training Owner) Should Use This Model

Student 3 can instantiate and train this model directly using standard PyTorch idioms.

### 6.1 Instantiation & Minibatch Forward Pass

```python
import torch
import torch.nn.functional as F
from src.data.graph_dataset import iter_graph_batches
from src.data.graph_preprocess import load_graph_datasets
from src.models import MolecularGNN

# 1. Load preprocessed graph partitions (Student 1)
datasets = load_graph_datasets()
train_dataset = datasets["train"]

# 2. Instantiate model and place on target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MolecularGNN(hidden_dim=128, num_layers=3, dropout=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 3. Minibatch Training Step with Masked Multi-Task Loss
model.train()
for batch in iter_graph_batches(train_dataset, batch_size=32):
    optimizer.zero_grad()
    
    # Forward pass: model accepts Student 1 batch dictionary directly
    logits = model(batch)  # shape: [B, 12]
    
    # Prepare targets and observation masks
    y = torch.from_numpy(batch["y"]).to(device)       # [B, 12]
    mask = torch.from_numpy(batch["mask"]).to(device) # [B, 12]
    
    # Replace NaNs in y with dummy 0.0 (safely eliminated by mask)
    y_safe = torch.nan_to_num(y, nan=0.0)
    
    # Element-wise BCE with logits
    loss_matrix = F.binary_cross_entropy_with_logits(logits, y_safe, reduction="none")
    
    # Optional: class imbalance pos_weight per endpoint
    # loss_matrix = loss_matrix * pos_weight.unsqueeze(0)
    
    # Compute masked average over valid observed assay labels
    loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1.0)
    
    loss.backward()
    optimizer.step()
```

### 6.2 Checkpointing

`MolecularGNN` uses standard `torch.nn.Module` state dicts:

```python
# Save checkpoint
torch.save(model.state_dict(), "results/models/gnn_checkpoint.pt")

# Load checkpoint
model.load_state_dict(torch.load("results/models/gnn_checkpoint.pt", map_location=device))
model.eval()
```

---

## 7. Verification & Automated Test Results

The architecture is covered by 25 automated unit and integration tests across three test modules:

- `tests/test_graph_pooling.py` (4 tests)
- `tests/test_message_passing.py` (4 tests)
- `tests/test_molecular_gnn.py` (17 tests)

### Test Execution Command
```bash
.venv/bin/pytest -v tests/test_graph_pooling.py tests/test_message_passing.py tests/test_molecular_gnn.py
```

### Summary of Passing Verifications
1. **Model Initialization:** Validates parameter bounds, rejects negative/zero dimensions or invalid pooling keys.
2. **Single Graph & Batched Inferences:** Verifies forward pass on single molecules ($B=1$) and multi-molecule disjoint union minibatches.
3. **Exact Output Shape:** Validates `[B, 12]` tensor output for all configurations.
4. **Variable Graph Sizes:** Verifies heterogeneous batches with varying atom counts and edge counts.
5. **Zero-Bond Edge Cases:** Confirms isolated monoatomic graphs execute cleanly without NaN or runtime exceptions.
6. **Edge Feature Utilization:** Validates that modifying bond attributes alters node representations and final logits.
7. **Zero Cross-Graph Leakage:** Evaluates molecule $A$ alone and batched with molecule $B$; outputs for molecule $A$ match within $10^{-5}$ tolerance.
8. **Masked BCE Loss Backpropagation:** Verifies finite non-zero gradients on all weights without NaN propagation.
9. **Checkpoint Round-Trip:** Verifies exact numerical reproduction after `state_dict` save/load.
10. **Device Agnosticism:** Successfully executes on CPU and detects Apple Silicon MPS / CUDA when available.
11. **Real Student 1 Batch Execution:** Successfully executes forward and backward passes using live batches generated by `iter_graph_batches` on `data/processed/phase2/graphs.npz`.

**Overall Repository Test Suite:** **167 tests passing** (142 existing tests + 25 Phase 2 GNN tests).

---

## 8. Limitations & Design Decisions

1. **No External Heavy Dependencies:** The architecture is implemented in pure PyTorch (using native `index_add_` and `scatter_reduce_`), avoiding fragile compilation dependencies on PyTorch Geometric C++ extensions while preserving full MPNN relational expressiveness.
2. **2D Graph Topology:** The architecture operates on molecular 2D topological bonds and stereo attributes provided by Student 1. It does not incorporate 3D Euclidean conformer coordinates, maintaining strict parity with Phase 1 input information.
3. **Comparative Hypothesis:** In accordance with research integrity guidelines, no claims are made regarding whether this GNN outperforms the Phase 1 Random Forest baseline ($0.7488$ ROC-AUC) until Student 3 trains the model and Student 4 conducts the full statistical evaluation.
