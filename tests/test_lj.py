import pytest
import quest
import psi4
import numpy as np

def test_lj_fit():
    A = 148.0
    B = 1.0
    distances = np.arange(2.0, 10.0, 0.25)
    energies = A/(distances ** 12.0) - B/(distances ** 6.0)
    A, B = quest.lj.fit_lj(distances, energies)
    assert np.isclose(A, 148.0) and np.isclose(B, -1.0)

def test_get_lj_params():
    mol_psi = "He"
    test = quest.molecule.Molecule(mol_psi, "aug-cc-pvdz")
    s, a, b, e, d = quest.lj.build_lj_params(test, True)
    print(s)
    assert abs(s - 2.8) <= 1
