"""Convert Phase 1 canonical SMILES and ordered labels into NumPy graph records."""

from functools import lru_cache

import numpy as np

from src.data.graph_features import ATOM_FEATURES, BOND_FEATURES, atom_features, bond_features
from src.data.inspect_dataset import parse_structure


class StructureError(ValueError):
    """A source molecule could not be parsed into a nonempty graph."""


@lru_cache(maxsize=256)
def _encode(smiles):
    molecule, _, reason = parse_structure(smiles)
    if reason:
        raise StructureError(reason)
    x = np.asarray([atom_features(atom) for atom in molecule.GetAtoms()], dtype=np.float32)
    edges, attributes = [], []
    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        edges.extend(((start, end), (end, start)))
        attributes.extend((features, features))
    edge_index = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T.copy()
    edge_attr = np.asarray(attributes, dtype=np.float32).reshape(-1, len(BOND_FEATURES))
    return x, edge_index, edge_attr


def build_graph(smiles: str, mol_id: str, labels) -> dict:
    """Use twelve source-ordered values, with NaN in unobserved positions."""
    if not isinstance(mol_id, str) or not mol_id.strip():
        raise ValueError("A nonempty molecule ID is required")
    if not isinstance(smiles, str):
        raise StructureError("missing_smiles" if smiles is None else "non_string_smiles")
    y = np.asarray(labels, dtype=np.float64)
    mask = ~np.isnan(y)
    if y.shape != (12,) or not np.isin(y[mask], [0, 1]).all():
        raise ValueError(f"{mol_id}: expected twelve binary or NaN labels")
    x, edge_index, edge_attr = _encode(smiles)
    return {
        "mol_id": mol_id, "x": x.copy(), "edge_index": edge_index.copy(),
        "edge_attr": edge_attr.copy(), "y": y.astype(np.float32), "mask": mask,
    }


def validate_graph(graph: dict) -> None:
    """Check graph shape, index bounds, paired edges, and the source-label convention."""
    x, edges, attrs, y, mask = (graph[key] for key in ("x", "edge_index", "edge_attr", "y", "mask"))
    mol_id = graph["mol_id"]
    if not isinstance(mol_id, str) or not mol_id.strip():
        raise ValueError("Invalid molecule identifier")
    if x.ndim != 2 or not len(x) or x.shape[1] != len(ATOM_FEATURES) or x.dtype != np.float32:
        raise ValueError(f"{mol_id}: invalid node feature shape or dtype")
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != np.int64:
        raise ValueError(f"{mol_id}: invalid edge index shape or dtype")
    count = edges.shape[1]
    if attrs.shape != (count, len(BOND_FEATURES)) or attrs.dtype != np.float32:
        raise ValueError(f"{mol_id}: invalid edge feature shape or dtype")
    if not np.isfinite(x).all() or not np.isfinite(attrs).all():
        raise ValueError(f"{mol_id}: nonfinite features")
    for block in (x[:, 3:6:2], x[:, 6:], attrs):
        if not np.isin(block, [0, 1]).all():
            raise ValueError(f"{mol_id}: nonbinary feature indicators")
    for group in (x[:, 6:15], x[:, 15:], attrs[:, 2:9], attrs[:, 9:]):
        if not (group.sum(axis=1) == 1).all():
            raise ValueError(f"{mol_id}: invalid one-hot features")
    if (edges < 0).any() or (edges >= len(x)).any() or count % 2:
        raise ValueError(f"{mol_id}: invalid edge indices")
    if not np.array_equal(edges[:, ::2], edges[::-1, 1::2]) or not np.array_equal(attrs[::2], attrs[1::2]):
        raise ValueError(f"{mol_id}: reverse edges or attributes are misaligned")
    bonds = np.sort(edges[:, ::2], axis=0).T
    if (bonds[:, 0] == bonds[:, 1]).any() or len(np.unique(bonds, axis=0)) != len(bonds):
        raise ValueError(f"{mol_id}: self-loop or repeated bond")
    if y.shape != (12,) or y.dtype != np.float32 or mask.shape != (12,) or mask.dtype != np.bool_:
        raise ValueError(f"{mol_id}: invalid label or mask shape or dtype")
    if not np.array_equal(mask, ~np.isnan(y)) or not np.isin(y[mask], [0, 1]).all():
        raise ValueError(f"{mol_id}: invalid labels or mask")
