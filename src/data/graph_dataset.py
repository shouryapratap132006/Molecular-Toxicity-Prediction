"""Indexable molecular graphs and disjoint NumPy minibatches."""

import numpy as np


class GraphDataset:
    """A partition in the original Phase 1 index order."""

    def __init__(self, records, endpoints):
        self._records = tuple(records)
        self.endpoints = tuple(endpoints)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        return self._records[index]


def collate_graphs(graphs) -> dict:
    """Concatenate graphs and offset edges into the batch's node index space."""
    if len(graphs) == 0:
        raise ValueError("Cannot collate an empty batch")
    counts = [len(graph["x"]) for graph in graphs]
    ptr = np.insert(np.cumsum(counts, dtype=np.int64), 0, 0)
    return {
        "mol_id": [graph["mol_id"] for graph in graphs],
        "x": np.concatenate([graph["x"] for graph in graphs]),
        "edge_index": np.concatenate(
            [graph["edge_index"] + offset for graph, offset in zip(graphs, ptr)], axis=1
        ),
        "edge_attr": np.concatenate([graph["edge_attr"] for graph in graphs]),
        "y": np.stack([graph["y"] for graph in graphs]),
        "mask": np.stack([graph["mask"] for graph in graphs]),
        "batch": np.repeat(np.arange(len(graphs), dtype=np.int64), counts),
        "ptr": ptr,
    }


def iter_graph_batches(dataset, batch_size: int):
    """Yield batches in dataset order, including a final smaller batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(dataset), batch_size):
        yield collate_graphs(dataset[start:start + batch_size])
