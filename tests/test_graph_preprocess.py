import numpy as np
import pandas as pd
import pytest

from src.data.common import read_json, sha256, write_json
from src.data.graph_builder import build_graph
from src.data.graph_preprocess import load_graph_datasets, main, preprocess_graphs


@pytest.fixture
def artifacts(tmp_path, config):
    csv = tmp_path / "cleaned.csv"
    split = tmp_path / "split.json"
    indices = {"train": [2, 0], "val": [3], "test": [5, 1, 4]}
    smiles = ["CCO", "C(", "c1ccccc1", "C1CC1", None, "[Na+]"]
    endpoints = config["endpoints"]
    labels = [1, np.nan, 0, 1, 0, 1, 0, np.nan, np.nan, 1, 0, 0]
    rows = [{"mol_id": f"mol-{i}", "source_row": 10 + i, "canonical_smiles": value,
             **dict(zip(endpoints, labels))} for i, value in enumerate(smiles)]
    pd.DataFrame(rows, columns=[*reversed(endpoints), "mol_id", "source_row", "canonical_smiles"]).to_csv(
        csv, index=False
    )
    write_json(split, {
        "status": "success",
        "dataset_metadata": {"sha256": sha256(csv), "total_records": len(rows), "smiles_column": "canonical_smiles"},
        "split_configuration": {"method": "bemis_murcko_scaffold"},
        "indices": indices, "partition_counts": {name: len(pos) for name, pos in indices.items()},
        "row_identifiers": {
            f"{name}_{field}": [rows[i][key] for i in pos]
            for name, pos in indices.items() for field, key in (("mol_ids", "mol_id"), ("source_rows", "source_row"))
        },
    })
    return {"csv_path": csv, "split_path": split, "cache_path": tmp_path / "graphs.npz",
            "reports_dir": tmp_path / "reports"}, labels


def test_round_trip_original_ids_order_and_masks(artifacts, config):
    args, labels = artifacts
    hashes = [sha256(args[key]) for key in ("csv_path", "split_path")]
    report = preprocess_graphs(**args)
    assert report["valid_graphs"] == 4
    assert report["invalid_structures"] == report["empty_or_missing_structures"] == 1
    assert report["unaccounted_rows"] == 0
    assert report["partitions"]["test"] == {"source": 3, "graphs": 1, "excluded": 2}
    errors = read_json(args["reports_dir"] / "graph_processing_errors.json")
    assert [e["mol_id"] for e in errors] == ["mol-1", "mol-4"]
    assert errors[0]["reason"].startswith("unparsable_smiles")
    assert errors[1]["reason"] == "missing_smiles"
    assert read_json(args["reports_dir"] / "graph_validation_report.json") == report
    stats = read_json(args["reports_dir"] / "graph_statistics.json")
    assert stats["labels"]["NR-AR-LBD"] == {"observed": 0, "missing": 4}
    datasets = load_graph_datasets(**{key: value for key, value in args.items() if key != "reports_dir"})
    assert [g["mol_id"] for g in datasets["train"]] == ["mol-2", "mol-0"]
    assert [g["mol_id"] for g in datasets["val"]] == ["mol-3"]
    assert [g["mol_id"] for g in datasets["test"]] == ["mol-5"]
    assert datasets["val"].endpoints == tuple(config["endpoints"])
    loaded = datasets["val"][0]
    expected = build_graph("C1CC1", "mol-3", labels)
    for key in ("x", "edge_index", "edge_attr", "y", "mask"):
        np.testing.assert_array_equal(loaded[key], expected[key])
    before = sha256(args["cache_path"])
    assert preprocess_graphs(**args) == report
    assert sha256(args["cache_path"]) == before
    assert [sha256(args[key]) for key in ("csv_path", "split_path")] == hashes


@pytest.mark.parametrize("change, message", [("overlap", "exactly once"), ("id", "identifiers"), ("hash", "does not match")])
def test_rejects_wrong_split(artifacts, change, message):
    args, _ = artifacts
    split = read_json(args["split_path"])
    if change == "overlap":
        split["indices"]["test"][0] = 0
    elif change == "id":
        split["row_identifiers"]["train_mol_ids"][0] = "wrong"
    else:
        split["dataset_metadata"]["sha256"] = "wrong"
    write_json(args["split_path"], split)
    with pytest.raises(ValueError, match=message):
        preprocess_graphs(**args)


@pytest.mark.parametrize("mutation, message", [
    ("split", "schema"), ("labels", "labels"), ("offset", "offsets"), ("cross_edge", "edge indices"),
])
def test_rejects_stale_or_corrupted_cache(artifacts, mutation, message):
    args, _ = artifacts
    preprocess_graphs(**args)
    if mutation == "split":
        split = read_json(args["split_path"])
        split["generated_at_utc"] = "changed"
        write_json(args["split_path"], split)
    else:
        with np.load(args["cache_path"], allow_pickle=False) as archive:
            arrays = dict(archive)
        if mutation == "labels":
            arrays["y"][0, 0] = 0
        elif mutation == "offset":
            arrays["ptr"][1] = -1
        else:
            arrays["edge_index"][0, 0] = arrays["ptr"][1]
        np.savez_compressed(args["cache_path"], **arrays)
    with pytest.raises(ValueError, match=message):
        preprocess_graphs(**args)


def test_command_writes_all_reports(artifacts, monkeypatch, capsys):
    args, _ = artifacts
    monkeypatch.setattr("sys.argv", [
        "graph_preprocess", "--csv", str(args["csv_path"]), "--split", str(args["split_path"]),
        "--cache", str(args["cache_path"]), "--reports-dir", str(args["reports_dir"]),
    ])
    main()
    assert "4/6 molecular graphs" in capsys.readouterr().out
    assert {path.name for path in args["reports_dir"].glob("*.json")} == {
        "graph_processing_errors.json", "graph_statistics.json", "graph_validation_report.json",
    }
