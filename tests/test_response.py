"""
This file tests the response module
"""

import quest
import pytest
import numpy as np

def test_response():


    mol_str = quest.mollib["h2o"]
    basis = 'cc-pVDZ'
 
    molecule = quest.Molecule(mol_str, basis)
    wfn = quest.Wavefunction(molecule, {})

    # Compute RHF
    scf_energy = quest.scf_module.compute_rhf(wfn, df=False, diis=False)

    polarizability = quest.response(wfn)

    test_polarizability = np.array([[2.97055439, 0.000, 0.000], [0.000, 10.5001391, 0.000], [0.000, 0.000, 6.64751258]])

    assert np.allclose(test_polarizability, polarizability)

    pass
