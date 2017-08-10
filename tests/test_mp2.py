"""
This file tests MP2
"""

import quest
import pytest
import psi4
import numpy as np


@pytest.mark.parametrize("mol_str", ["h2o", "co2"])
def test_mp2(mol_str):
    mol_str = quest.mollib[mol_str]
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
    mp2_energy = quest.mp2_module.mp2(wfn)

    psi4.set_options({"scf_type": "pk", "mp2_type": "conv"})
    ref_energy = psi4.energy("MP2" + "/" + basis, molecule=molecule.mol)

    assert np.allclose(mp2_energy, ref_energy)


@pytest.mark.parametrize("mol_str", ["h2o", "co"])
def test_df_mp2(mol_str):
    geometry = psi4.geometry(quest.mollib[mol_str])
    basis = "STO-3G"
    rhf_options = \
    {
        'e_conv': 1.e-8,
        'd_conv': 1.e-8,
        'diis': True,
        'max_diis': 7,
        'max_iter': 100,
    }
    mol = quest.Molecule(geometry, basis)
    wafu = quest.Wavefunction(mol, rhf_options)
    scf_energy = quest.scf_module.compute_rhf(wafu)
    mp2_energy = quest.mp2_module.df_mp2(wafu)

    psi4.set_options({"scf_type": "df"})
    psi4_mp2_energy = psi4.energy('mp2/' + basis, molecule=geometry)
