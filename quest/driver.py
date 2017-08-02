"""
Drives modules
"""

from . import mp2
from . import scf_module
from . import molecule
from . import wavefunction
from .response import response


def compute_scf_response(wavefunction):
    return response(wavefunction)


def compute_mp2(wavefunction):
    return mp2.mp2(wavefunction)


def compute_rhf(mol, basis_name="aug-cc-pvdz", numpy_memory=1.e9, maxiter=12,
                E_conv=1.e-6, D_conv=1.e-4, diis=True, max_diis=7):
    mol = molecule.Molecule(mol, basis_name)
    rhf_options = \
    {
        'e_conv': E_conv,
        'd_conv': D_conv,
        'diis': diis,
        'max_diis': max_diis,
        'max_iter': maxiter,
    }

    wfn = wavefunction.Wavefunction(mol, rhf_options)

    # Compute RHF
    scf_energy = scf_module.compute_rhf(wfn)
    return (scf_energy, wfn)


def compute_mc(sigma, epsilon, coord_file_path, params):
    pass
