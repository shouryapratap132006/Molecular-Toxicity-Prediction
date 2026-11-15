# Phase 2 molecular graph preprocessing

A molecular graph represents each RDKit atom as a node and each chemical bond as a pair of directed edges. Nodes carry element and local-environment information. Edges tell a GNN which atoms can exchange messages and describe the bond between them. Each graph retains its Phase 1 molecule ID and the assay labels for that source record.

The pipeline parses the cleaned CSV's canonical isomeric SMILES with the Phase 1 RDKit sanitizer. It keeps RDKit's atom order and connected components, including salts. It adds no explicit hydrogens and retains distinct source records that share the same structure. A structure that fails parsing or contains no atoms produces an error entry with its ID, cleaned CSV index, original source row, SMILES and reason.

## Feature schema, version 1

The feature names and order are defined by `ATOM_FEATURES` and `BOND_FEATURES` in `src/data/graph_features.py`. Each flag is 0 or 1. Every categorical block has exactly one active feature. `OTHER` covers unlisted RDKit categories.

| Array | Width | Ordered columns |
| --- | ---: | --- |
| `x` | 19 | Atomic number, explicit neighbor count (degree), signed formal charge, aromatic flag, attached hydrogen count (including hydrogen neighbors), ring flag, 9 hybridization indicators, 4 chirality indicators |
| `edge_attr` | 18 | Conjugation flag, ring flag, 7 bond-type indicators, 9 bond-stereo indicators |
| `y` | 12 | Phase 1 toxicity endpoints in the order below |
| `mask` | 12 | Availability of the corresponding endpoints |

The hybridization block is ordered `UNSPECIFIED, S, SP, SP2, SP3, SP2D, SP3D, SP3D2, OTHER`. The chirality block is `CHI_UNSPECIFIED, CHI_TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CCW, OTHER`. These tetrahedral tags are relative to RDKit's atom order.

Bond types are ordered `SINGLE, DOUBLE, TRIPLE, AROMATIC, DATIVE, ZERO, OTHER`. Stereo indicators are ordered `STEREONONE, STEREOANY, STEREOZ, STEREOE, STEREOCIS, STEREOTRANS, STEREOATROPCW, STEREOATROPCCW, OTHER`. Stereo describes the chemical bond, so reverse edges share the same attributes. A dative bond carries its type in both directions.

## Graph and batch interface

| Key | Dtype | Individual graph | Batch of `B` graphs |
| --- | --- | --- | --- |
| `x` | NumPy `float32` | `[N, 19]` | `[sum(N), 19]` |
| `edge_index` | NumPy `int64` | `[2, E]` with source and destination rows | `[2, sum(E)]` with node offsets applied |
| `edge_attr` | NumPy `float32` | `[E, 18]` | `[sum(E), 18]` |
| `y` | NumPy `float32` | `[12]` | `[B, 12]` |
| `mask` | NumPy `bool` | `[12]` | `[B, 12]` |
| `mol_id` | Python string | Original Phase 1 ID | Ordered list of `B` IDs |
| `batch` | NumPy `int64` | — | `[sum(N)]` graph index for each node |
| `ptr` | NumPy `int64` | — | `[B+1]` cumulative node offsets, starting at zero |

`N` counts atoms and `E` counts directed edge entries. Each bond contributes adjacent forward and reverse entries with matching feature rows. Bondless graphs have edge arrays shaped `[2, 0]` and `[0, 18]`. Batching uses a disjoint union. For graph `i`, its nodes occupy `ptr[i]:ptr[i+1]`, and both endpoints of every edge fall within that slice.

`dataset.endpoints` is a tuple copied from `configs/data_config.yaml` in Phase 1 order.

```text
NR-AR  NR-AR-LBD  NR-AhR  NR-Aromatase  NR-ER  NR-ER-LBD  NR-PPAR-gamma
SR-ARE  SR-ATAD5  SR-HSE  SR-MMP  SR-p53
```

Observed labels are exactly 0 or 1 with a true mask. Missing labels are NaN with a false mask. Select observed elements before evaluating a loss. Multiplying a loss computed against NaN targets by a mask still propagates NaNs. When a batch contains zero observed labels, skip its supervised loss.

## Run preprocessing and load a partition

Put the original Phase 1 `tox21_cleaned.csv` and `scaffold_split_indices.json` under `data/processed/`. The split JSON's source checksum must match the exact CSV. It must also cover every cleaned index once, and its per-partition IDs and source rows must match the CSV. The pipeline preserves the partition and ordering in that artifact. Every structure error is reported separately, and its source partition stays intact.

```bash
python -m pip install -r requirements.txt
python -m src.data.graph_preprocess
python -m pytest -q tests/test_graph_builder.py tests/test_graph_dataset.py tests/test_graph_preprocess.py
```

Use `--csv` and `--split` to point to the original inputs in another directory. The cache defaults to `data/processed/phase2/graphs.npz`, and the validation, statistics and processing errors JSON files default to `results/phase2/`. Use `--cache` and `--reports-dir` to select other output paths.

```python
from src.data.graph_dataset import iter_graph_batches
from src.data.graph_preprocess import load_graph_datasets

datasets = load_graph_datasets()
train = datasets["train"]
test = datasets["test"]
first = test[0]
for batch in iter_graph_batches(train, batch_size=64):
    x = batch["x"]
    edge_index = batch["edge_index"]
    y = batch["y"]
    observed = batch["mask"]
```

For Students 2 and 3, pass `collate_graphs` as `collate_fn` to a PyTorch `DataLoader` if using its samplers. Convert the resulting arrays with `torch.from_numpy`, keeping `edge_index`, `batch` and `ptr` as integer tensors and `mask` as boolean. Keep `mol_id` as Python strings for evaluation. Use the ordinary PyTorch loader with this custom collator. The arrays already contain the node offsets and graph membership that a GNN batch needs. Torch and PyG are optional consumer-side dependencies, and this pipeline adds no package pins.

A matching cache can be reused. Loading disables pickle and checks source hashes, feature names, preprocessing version, RDKit version, graph IDs, offsets, features, label values and masks. If the inputs or schema change, choose a new cache path. The current loader materializes packed arrays in memory. Stereochemistry follows RDKit's SMILES representation, and the schema includes no 3D coordinates or conformers. The ordered split artifact remains the authoritative source of partition membership.
