#!/usr/bin/env python3
"""CLI runner script for training Phase 2 MolecularGNN on Tox21 dataset.

Student 3 Module: Molecular Toxicity Prediction via Chemical Fingerprints
and Molecular Graph Networks.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import main

if __name__ == "__main__":
    main()
