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

    molecule = quest.Molecule(mol_str, basis)
    wfn = quest.Wavefunction(molecule, {})

    scf_energy = quest.scf_module.compute_rhf(wfn, df=False, diis=False)
    mp2_energy = quest.mp2(wfn)

    psi4.set_options({"scf_type": "pk"})
    ref_energy = psi4.energy("MP2" + "/" + basis, molecule=molecule.mol())

    assert np.allclose(mp2_energy, ref_energy)
