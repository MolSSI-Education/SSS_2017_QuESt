"""
This file tests the Molecule in the quest.
"""

import quest
import pytest
import psi4
import numpy as np


def test_molecule():
    molecule = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

    basis = "sto-3g"

    testmol = quest.molecule.Molecule()

    testmol.set_geometry("molecule")

    testmol.set_basis("sto-3g")

    assert testmol.nel == 10

