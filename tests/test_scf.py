"""
This file tests the SCF module
"""

import quest
import pytest
import psi4
import numpy as np

def test_scf():

    geom = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """
    )

    bas = 'sto-3g'

    molecule = quest.RHF(geom, bas)

    pass
