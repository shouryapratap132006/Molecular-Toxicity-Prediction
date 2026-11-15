import numpy as np
import pytest

from src.data.graph_builder import build_graph
from src.data.graph_dataset import GraphDataset, collate_graphs, iter_graph_batches


def test_batch_offsets_and_supervision(config):
    graphs = [
        build_graph("[Na+]", "ion", [np.nan] * 12),
        build_graph("CO", "alcohol", [0] * 12),
        build_graph("C#N", "nitrile", [1] * 12),
    ]
    dataset = GraphDataset(graphs, config["endpoints"])
    batch = next(iter_graph_batches(dataset, 3))
    assert dataset[1]["mol_id"] == "alcohol"
    assert dataset.endpoints == tuple(config["endpoints"])
    assert batch["mol_id"] == ["ion", "alcohol", "nitrile"]
    assert batch["x"].shape == (5, 19)
    assert batch["edge_attr"].shape == (4, 18)
    assert batch["y"].shape == batch["mask"].shape == (3, 12)
    np.testing.assert_array_equal(batch["ptr"], [0, 1, 3, 5])
    np.testing.assert_array_equal(batch["batch"], [0, 1, 1, 2, 2])
    np.testing.assert_array_equal(batch["edge_index"], [[1, 2, 3, 4], [2, 1, 4, 3]])
    src, tgt = batch["edge_index"]
    np.testing.assert_array_equal(batch["batch"][src], batch["batch"][tgt])
    np.testing.assert_array_equal(batch["edge_attr"], np.concatenate([graph["edge_attr"] for graph in graphs]))
    np.testing.assert_array_equal(batch["mask"].sum(axis=1), [0, 12, 12])
    assert np.isnan(batch["y"][0]).all()
    assert [batch["mol_id"] for batch in iter_graph_batches(dataset, 2)] == [["ion", "alcohol"], ["nitrile"]]


def test_empty_and_bondless_batches():
    graph = build_graph("[Na+]", "ion", [0] * 12)
    assert collate_graphs([graph, graph])["edge_index"].shape == (2, 0)
    with pytest.raises(ValueError, match="empty"):
        collate_graphs([])
    with pytest.raises(ValueError, match="positive"):
        list(iter_graph_batches(GraphDataset([], []), 0))


def test_numpy_backed_batches():
    graph = build_graph("CC", "sample", [0] * 12)
    batches = list(iter_graph_batches(np.array([graph] * 3, dtype=object), 2))
    assert [len(batch["mol_id"]) for batch in batches] == [2, 1]
