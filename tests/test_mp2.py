"""
This file tests MP2
"""

import quest
import pytest
import psi4
import numpy as np


def test_mp2():
    mol_str = quest.mollib["h2o"]
    basis = "sto-3g"

    rhf_options = \
    {
        'e_conv': 1.e-8,
        'd_conv': 1.e-8,
        'diis': True,
        'max_diis': 7,
        'max_iter': 100,
    }

    molecule = quest.Molecule(mol_str, basis)
    wfn = quest.Wavefunction(molecule, rhf_options)

    scf_energy = quest.scf_module.compute_rhf(wfn)
    mp2_energy = quest.mp2.mp2(wfn)

    psi4.set_options({"scf_type": "pk", "mp2_type":"conv"})
    ref_energy = psi4.energy("MP2" + "/" + basis, molecule=molecule.mol)

    assert np.allclose(mp2_energy, ref_energy)
