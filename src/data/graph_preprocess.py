"""Build or validate a Tox21 graph cache against the existing Phase 1 scaffold split."""

from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from rdkit import rdBase

from src.data.common import ROOT, atomic_write, load_config, read_json, sha256, write_json
from src.data.graph_builder import StructureError, build_graph, validate_graph
from src.data.graph_dataset import GraphDataset, collate_graphs
from src.data.graph_features import ATOM_FEATURES, BOND_FEATURES
from src.data.inspect_dataset import normalize_labels, read_dataset

VERSION = 1
PARTITIONS = ("train", "val", "test")
DEFAULT_SPLIT = ROOT / "data/processed/scaffold_split_indices.json"
DEFAULT_CACHE = ROOT / "data/processed/phase2/graphs.npz"
DEFAULT_REPORTS = ROOT / "results/phase2"


def _source(csv_path, split_path, endpoints):
    signature = {
        "version": VERSION, "rdkit": rdBase.rdkitVersion, "csv_sha256": sha256(csv_path),
        "split_sha256": sha256(split_path), "endpoints": list(endpoints),
        "atom_features": list(ATOM_FEATURES), "bond_features": list(BOND_FEATURES),
    }
    frame = read_dataset(csv_path)
    split = read_json(split_path)
    if sha256(csv_path) != signature["csv_sha256"] or sha256(split_path) != signature["split_sha256"]:
        raise ValueError("Phase 1 inputs changed during loading")
    metadata = split["dataset_metadata"]
    if (split["status"] != "success" or metadata["sha256"] != signature["csv_sha256"]
            or metadata["total_records"] != len(frame) or metadata["smiles_column"] != "canonical_smiles"
            or split["split_configuration"]["method"] != "bemis_murcko_scaffold"):
        raise ValueError("Phase 1 scaffold split does not match the cleaned CSV")
    if not frame.columns.is_unique:
        raise ValueError("Cleaned CSV contains duplicate columns")
    ids = frame["mol_id"].tolist()
    if frame.empty or any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("Cleaned CSV requires a nonempty molecule ID for every row")
    if len(set(ids)) != len(ids):
        raise ValueError("Cleaned CSV contains duplicate molecule IDs")
    source_rows = frame["source_row"].astype(int).tolist()
    if min(source_rows) < 1 or len(set(source_rows)) != len(source_rows):
        raise ValueError("Cleaned CSV source rows must be positive and unique")
    indices = split["indices"]
    if set(indices) != set(PARTITIONS):
        raise ValueError("Phase 1 split must contain train, val and test")
    positions = [i for name in PARTITIONS for i in indices[name]]
    if any(type(i) is not int for i in positions) or sorted(positions) != list(range(len(frame))):
        raise ValueError("Split indices must cover each cleaned CSV row exactly once")
    for name, rows in indices.items():
        identifiers = split["row_identifiers"]
        if (identifiers[f"{name}_mol_ids"] != [ids[i] for i in rows]
                or identifiers[f"{name}_source_rows"] != [source_rows[i] for i in rows]
                or split["partition_counts"][name] != len(rows)):
            raise ValueError(f"Phase 1 {name} identifiers or count do not match the cleaned CSV")
    labels, errors = normalize_labels(frame[list(endpoints)], list(endpoints))
    if errors:
        raise ValueError(f"Malformed source endpoint value: {errors[0]}")
    y = labels.to_numpy(dtype=np.float32, na_value=np.nan)
    smiles = [value if isinstance(value, str) else None for value in frame["canonical_smiles"]]
    return signature, ids, source_rows, smiles, y, indices


def _build(ids, source_rows, smiles, y):
    groups = defaultdict(list)
    for i, value in enumerate(smiles):
        groups[value].append(i)
    records, errors = {}, []
    for value, rows in groups.items():
        try:
            for i in rows:
                graph = build_graph(value, ids[i], y[i])
                validate_graph(graph)
                records[i] = graph
        except StructureError as error:
            errors.extend({"index": i, "mol_id": ids[i], "source_row": source_rows[i],
                           "smiles": value, "reason": str(error)} for i in rows)
    return [records[i] for i in sorted(records)], sorted(errors, key=lambda error: error["index"])


def _save(cache_path, signature, graphs, errors):
    if not graphs:
        raise ValueError("Cannot serialize a dataset with zero valid molecular graphs")
    packed = collate_graphs(graphs)
    edge_ptr = np.insert(np.cumsum([g["edge_attr"].shape[0] for g in graphs], dtype=np.int64), 0, 0)
    packed.pop("batch")
    payload = BytesIO()
    np.savez_compressed(payload, **packed, edge_ptr=edge_ptr,
                        signature=json.dumps(signature), errors=json.dumps(errors))
    atomic_write(cache_path, payload.getvalue())


def _load(cache_path, signature, ids, source_rows, smiles, y):
    with np.load(cache_path, allow_pickle=False) as archive:
        if json.loads(archive["signature"].item()) != signature:
            raise ValueError("Graph cache differs from Phase 1 inputs or feature schema")
        errors = json.loads(archive["errors"].item())
        omitted = set()
        for item in errors:
            i = item["index"]
            if (type(i) is not int or not 0 <= i < len(ids) or i in omitted
                    or (item["mol_id"], item["source_row"], item["smiles"]) != (ids[i], source_rows[i], smiles[i])
                    or not isinstance(item["reason"], str) or not item["reason"]):
                raise ValueError("Graph cache contains an invalid structure error record")
            omitted.add(i)
        rows = [i for i in range(len(ids)) if i not in omitted]
        cache_ids = archive["mol_id"]
        if not rows or cache_ids.tolist() != [ids[i] for i in rows]:
            raise ValueError("Graph cache IDs do not match Phase 1 records and exclusions")
        x, edges, attrs, cached_y, mask, ptr, edge_ptr = (
            archive[key] for key in ("x", "edge_index", "edge_attr", "y", "mask", "ptr", "edge_ptr")
        )
    if (edges.shape != (2, len(attrs)) or edges.dtype != np.int64
            or (edges < 0).any() or (edges >= len(x)).any()
            or cached_y.shape != (len(rows), 12)
            or mask.shape != cached_y.shape or not np.array_equal(cached_y, y[rows], equal_nan=True)):
        raise ValueError("Graph cache arrays differ from source labels or expected shapes")
    for offsets, length, minimum in ((ptr, len(x), 1), (edge_ptr, len(attrs), 0)):
        if (offsets.shape != (len(rows) + 1,) or offsets.dtype != np.int64
                or offsets[0] != 0 or offsets[-1] != length
                or (offsets < 0).any() or (offsets > length).any() or (np.diff(offsets) < minimum).any()):
            raise ValueError("Graph cache contains invalid node or edge offsets")

    graphs = {}
    for j, i in enumerate(rows):
        a, b = ptr[j:j + 2]
        c, d = edge_ptr[j:j + 2]
        graph = {"mol_id": ids[i], "x": x[a:b], "edge_index": edges[:, c:d] - a,
                 "edge_attr": attrs[c:d], "y": cached_y[j], "mask": mask[j]}
        validate_graph(graph)
        graphs[i] = graph
    return graphs, errors


def _datasets(graphs, indices, endpoints):
    return {name: GraphDataset((graphs[i] for i in indices[name] if i in graphs), endpoints)
            for name in PARTITIONS}


def load_graph_datasets(cache_path=DEFAULT_CACHE, *, csv_path=None, split_path=DEFAULT_SPLIT):
    """Load train, val and test GraphDatasets after checking the cache against Phase 1."""
    config = load_config()
    signature, ids, source_rows, smiles, y, indices = _source(
        csv_path or config["processed_path"], split_path, config["endpoints"]
    )
    graphs, _ = _load(cache_path, signature, ids, source_rows, smiles, y)
    return _datasets(graphs, indices, signature["endpoints"])


def preprocess_graphs(*, csv_path, split_path, cache_path, reports_dir):
    """Prepare graphs, reuse a matching cache, and write quality and exclusion reports."""
    signature, ids, source_rows, smiles, y, indices = _source(
        csv_path, split_path, load_config()["endpoints"]
    )
    if not cache_path.exists():
        records, errors = _build(ids, source_rows, smiles, y)
        if not records:
            write_json(reports_dir / "graph_processing_errors.json", errors)
            raise ValueError("No valid molecular graphs. See graph_processing_errors.json")
        _save(cache_path, signature, records, errors)
    graphs, errors = _load(cache_path, signature, ids, source_rows, smiles, y)
    datasets = _datasets(graphs, indices, signature["endpoints"])
    partitions = {}
    for name in PARTITIONS:
        rows = indices[name]
        expected_ids = [ids[i] for i in rows if i in graphs]
        if [g["mol_id"] for g in datasets[name]] != expected_ids:
            raise ValueError(f"Graph IDs diverged from Phase 1 {name}")
        partitions[name] = {"source": len(rows), "graphs": len(expected_ids), "excluded": len(rows) - len(expected_ids)}
    empty = sum(item["reason"] in {"empty_smiles", "missing_smiles", "zero_atom_structure"} for item in errors)
    report = {
        "status": "passed_with_structure_errors" if errors else "passed",
        "signature": signature, "source_records": len(ids), "valid_graphs": len(graphs),
        "invalid_structures": len(errors) - empty, "empty_or_missing_structures": empty,
        "duplicate_graph_ids": 0, "unaccounted_rows": len(ids) - len(graphs) - len(errors),
        "partitions": partitions, "split_ids_and_order_match": True, "split_disjoint_and_complete": True,
        "serialized_graphs_valid": True, "labels_and_masks_match_source": True,
    }
    nodes = [len(graph["x"]) for graph in graphs.values()]
    edges = [graph["edge_attr"].shape[0] for graph in graphs.values()]
    observed = np.stack([graph["mask"] for graph in graphs.values()]).sum(axis=0)
    stats = {
        "node_feature_dim": len(ATOM_FEATURES), "edge_feature_dim": len(BOND_FEATURES),
        "nodes": {"total": sum(nodes), "min": min(nodes), "max": max(nodes)},
        "directed_edges": {"total": sum(edges), "min": min(edges), "max": max(edges)},
        "labels": {name: {"observed": int(count), "missing": len(graphs) - int(count)}
                   for name, count in zip(signature["endpoints"], observed)},
    }
    write_json(reports_dir / "graph_statistics.json", stats)
    write_json(reports_dir / "graph_processing_errors.json", errors)
    write_json(reports_dir / "graph_validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(load_config()["processed_path"]))
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    try:
        report = preprocess_graphs(csv_path=args.csv, split_path=args.split,
                                   cache_path=args.cache, reports_dir=args.reports_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(f"{report['status']}: {report['valid_graphs']}/{report['source_records']} molecular graphs")


if __name__ == "__main__":
    main()
