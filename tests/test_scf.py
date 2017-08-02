"""
This file tests the SCF module
"""

import quest
import pytest
import numpy as np
import psi4

def test_scf():

    mol_str = quest.mollib["h2o"]
    basis = 'sto-3g'

    molecule = quest.Molecule(mol_str, basis)
    wfn = quest.Wavefunction(molecule, {})

    # Compute RHF
    scf_energy = quest.scf_module.compute_rhf(wfn, df=False, diis=False)

    psi4.set_options({"scf_type": "pk"})
    ref_energy = psi4.energy("SCF" + "/" + basis, molecule=molecule.mol)

    assert np.allclose(ref_energy, scf_energy)

    pass

def test_scf_driver():

    mol_str = quest.mollib["h2o"]
    basis = 'sto-3g'
    scf_energy, scf_wfn = quest.driver.compute_rhf(mol_str, basis)

    psi4.set_options({"scf_type": "pk"})
    ref_energy = psi4.energy("SCF" + "/" + basis, molecule=scf_wfn.mol.mol)

    assert np.allclose(ref_energy, scf_energy)