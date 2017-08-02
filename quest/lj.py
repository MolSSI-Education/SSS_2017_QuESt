import numpy as np
import psi4
from . import driver
from . import molecule

"""
This module takes a quest Molecule object and returns 
the Lennard-Jones potential parameters (sigma, A, B) 
"""
 
def build_lj_params(mol, returnAll=False):
    """
    Builds the Lennard-Jones coefficients/parameters/potential.

    Parameters
    ----------
    atom : Molecule
           Input molecule (should be one atom)
    returnAll : boolean
                Whether to return coefficients and energies/distances in addition to sigma
                False by default

    Returns
    -------
    sigma, A, B, energies, distances

    Examples
    --------
    s = build_lj_params(molecule)
    s, A, B, e, d = build_lj_params(molecule, ReturnAll=True)
    """
    # set up distances for PES
    start = 2.0
    stop = 10.0
    step = 0.75
    distances = np.arange(start, stop, step)
    energies = np.zeros(distances.size)
    # set up geometry stuff -- make new geom object to avoid overwriting the old one
    atom_str = mol.mol.create_psi4_string_from_molecule().splitlines()[2]
    #atom_str = mol.create_psi4_string_from_molecule().splitlines()[2]
    new_mol = molecule.Molecule(mol=mol.mol, bas=mol.bas_name)
    geom_string = '{:s}\n{:s} 1   {:2.5f}'
    # do MP2 on each distance in the array
    for i, distance in enumerate(distances):
        # construct molecule w/ correct distance
        mol_geom = geom_string.format(atom_str, atom_str, distance)
        #new_mol.set_geometry(mol_geom)
        #new_mol.set_basis(mol.bas_name)
        # call MP2 on molecule and get energy
        energy, wfn = driver.compute_rhf(mol_geom, mol.bas_name)
        # add MP2 energy to energies list
        energies[i] = energy
    # doing the fit
    A,B = fit_lj(distances, energies)
    # calculate sigma
    sigma = np.power(A/(-B), 1.0/6.0)
    # returning the correct values
    if(returnAll):
        return sigma, A, B, energies, distances
    else:
        return sigma


def fit_lj(distances, energies):
    """
    Takes a list of distances and corresponding energies 
    perform a list squares fitting and 
    returns coefficients A for the r^12 term and B for the r^6 term

    Parameters
    ----------
    distances : list of doubles
    energies : list of doubles

    Returns
    -------
    A, B

    Examples
    --------
    A,B = fit_lj(array_1, array_2)
    """

    powers = [-12.0, -6.0]
    r_power = np.power(np.array(distances).reshape(-1,1), powers) 
    A,B = np.linalg.lstsq(r_power, energies)[0]
    return A,B

