import numpy as np
import pytest

from src.data.graph_builder import StructureError, build_graph, validate_graph
from src.data.graph_features import ATOM_FEATURES, BOND_FEATURES, one_hot


@pytest.mark.parametrize("smiles, reason", [(None, "missing"), (" ", "empty"), ("C(", "unparsable")])
def test_bad_smiles(smiles, reason):
    with pytest.raises(StructureError, match=reason):
        build_graph(smiles, "example", [0] * 12)


def test_single_atom_and_missing_labels():
    graph = build_graph("[Na+]", "original-id", [1, np.nan, 0] * 4)
    validate_graph(graph)
    assert graph["mol_id"] == "original-id"
    assert graph["x"].shape == (1, 19)
    assert graph["x"][0, :3].tolist() == [11, 0, 1]
    assert graph["edge_index"].shape == (2, 0)
    assert graph["edge_attr"].shape == (0, 18)
    np.testing.assert_array_equal(graph["mask"], [True, False, True] * 4)
    np.testing.assert_array_equal(graph["y"], [1, np.nan, 0] * 4)


def test_aromatic_ring_and_bond_types():
    graph = build_graph("c1ccccc1.CCO.C=C.C#N", "types", [0] * 12)
    validate_graph(graph)
    x, attrs, edges = graph["x"], graph["edge_attr"], graph["edge_index"]
    assert set(x[:, 0]) == {6, 7, 8}
    assert x[:, ATOM_FEATURES.index("aromatic")].sum() == 6
    assert x[:, ATOM_FEATURES.index("in_ring")].sum() == 6
    assert attrs[:, BOND_FEATURES.index("in_ring")].sum() == 12
    for name, count in {"AROMATIC": 12, "SINGLE": 4, "DOUBLE": 2, "TRIPLE": 2}.items():
        assert attrs[:, BOND_FEATURES.index(f"type:{name}")].sum() == count
    np.testing.assert_array_equal(edges[:, ::2], edges[::-1, 1::2])
    edges[0, 0] = len(x)
    with pytest.raises(ValueError, match="edge indices"):
        validate_graph(graph)


def test_stereochemistry_and_fallback():
    graph = build_graph("F/C=C/F.N[C@@H](C)O", "stereo", [1] * 12)
    assert graph["edge_attr"][:, BOND_FEATURES.index("stereo:STEREOE")].sum() == 2
    chiral_columns = [i for i, name in enumerate(ATOM_FEATURES) if name.startswith("chirality:CHI_TETRAHEDRAL")]
    assert graph["x"][:, chiral_columns].sum() == 1
    assert one_hot("future_category", ("known",)) == [False, True]


def test_repeated_smiles_does_not_share_mutable_graph_arrays():
    first = build_graph("CCO", "first", [0] * 12)
    first["x"][0, 0] = -1
    second = build_graph("CCO", "second", [1] * 12)
    assert second["x"][0, 0] == 6
    assert second["mol_id"] == "second"
    assert second["y"].sum() == 12


def test_invalid_labels_are_rejected_before_rounding():
    with pytest.raises(ValueError, match="binary"):
        build_graph("CC", "invalid labels", [1.00000001] * 12)


def test_serialized_graph_cannot_contain_self_loops():
    graph = build_graph("CC", "self-loop", [0] * 12)
    graph["edge_index"][:] = 0
    with pytest.raises(ValueError, match="self-loop"):
        validate_graph(graph)
