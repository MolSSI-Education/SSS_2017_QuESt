"""
This file tests the Molecule in the quest.
"""

import quest
import pytest
import psi4
import numpy as np
# import Molecule

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

def test_mp2():
    pass