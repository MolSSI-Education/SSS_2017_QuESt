"""
This file tests the MP2 in the quest.
"""

import quest
import pytest
import psi4
import numpy as np
from quest import mp2

psi4.set_output_file("output.dat", True)
geometry = ["""
       O
       H 1 1.1
       H 1 1.1 2 104
       symmetry c1
       """,
       """
           O
       H 1 1.3
       H 1 1.3 2 104
       symmetry c1
       """]

# geometry = ["""
#         O
#         H 1 1.1
#         H 1 1.1 2 104
#         symmetry c1
#         """]
basis_set = ["sto-3g"]
#basis2 = "aug-cc-pvdz"

@pytest.mark.parametrize("geometry", geometry)
@pytest.mark.parametrize("basis_set", basis_set)
def test_mp2(geometry, basis_set):
    mol = psi4.geometry(geometry)
    mp2_e = mp2.mp2(mol, basis_set)
    psi4.energy('MP2')
    result = psi4.compare_values(psi4.core.get_variable('MP2 TOTAL ENERGY'), mp2_e, 6, 'MP2 Energy')
    assert result == True




#    @pytest.mark.parametrize("geometry,basis, expected",[(geometry1, basis1, True),
                                                     #    (geometry2, basis1, True),
                                                         #(geometry1, basis2, True)])
   # pytest.mark.parametrize("geometry,basis, expected",[(geometry[i], basis[i], True)]

