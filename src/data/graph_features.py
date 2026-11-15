"""Ordered atom and bond feature definitions for molecular graphs."""

HYBRIDIZATION = ("UNSPECIFIED", "S", "SP", "SP2", "SP3", "SP2D", "SP3D", "SP3D2")
CHIRALITY = ("CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW")
BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE", "ZERO")
BOND_STEREO = (
    "STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS",
    "STEREOATROPCW", "STEREOATROPCCW",
)

ATOM_FEATURES = (
    "atomic_number", "degree", "formal_charge", "aromatic", "attached_hydrogens", "in_ring",
    *(f"hybridization:{value}" for value in (*HYBRIDIZATION, "OTHER")),
    *(f"chirality:{value}" for value in (*CHIRALITY, "OTHER")),
)
BOND_FEATURES = (
    "conjugated", "in_ring",
    *(f"type:{value}" for value in (*BOND_TYPES, "OTHER")),
    *(f"stereo:{value}" for value in (*BOND_STEREO, "OTHER")),
)


def one_hot(value, categories):
    name = str(value)
    return [name == category for category in categories] + [name not in categories]


def atom_features(atom):
    return [
        atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
        atom.GetIsAromatic(), atom.GetTotalNumHs(includeNeighbors=True), atom.IsInRing(),
        *one_hot(atom.GetHybridization(), HYBRIDIZATION),
        *one_hot(atom.GetChiralTag(), CHIRALITY),
    ]


def bond_features(bond):
    return [
        bond.GetIsConjugated(), bond.IsInRing(),
        *one_hot(bond.GetBondType(), BOND_TYPES),
        *one_hot(bond.GetStereo(), BOND_STEREO),
    ]
