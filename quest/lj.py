import numpy as np
import psi4
from . import driver
from . import molecule

"""
This module takes a quest Molecule object and returns 
the Lennard-Jones potential parameters (sigma, A, B) 
"""
 
def build_lj_params(mol, returnAll=False, method='MP2', start=2.0, stop=7.0, step=0.5):
    """
    Builds the Lennard-Jones coefficients/parameters/potential.

    Parameters
    ----------
    atom : Molecule
           Input molecule (should be one atom)
    returnAll : boolean
                Whether to return coefficients and energies/distances in addition to sigma (default is False)
    method : string
             Which method to use for energy calculations (currently supports MP2 and SCF, default is MP2)
    start : double
            The intermolecular distance at which to begin calculating energies (default is 2.0)
    stop : double
           The intermolecular distance at which to finish calculating energies (default is 7.0)
    step : double
           The step size between intermolecular distances (default is 0.5) 

    Returns
    -------
    sigma, A, B, energies, distances

    Examples
    --------
    s = build_lj_params(molecule)
    s, A, B, e, d = build_lj_params(molecule, ReturnAll=True)
    """
    # set up storage for distance and energy values
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
        # perform initial SCF calculation
        energy, wfn = driver.compute_rhf(mol_geom, mol.bas_name)
        # if SCF, add energy to list
        if(method.upper() == 'SCF'):
            energies[i] = energy
        # if MP2, calculate MP2 energy and add to list
        elif(method.upper() == 'MP2'):
            energy_mp2 = driver.compute_mp2(wfn)
            energies[i] = energy_mp2
        # only SCF/MP2 are currently supported
        else:
            raise ValueError('Method name for Lennard-Jones calculation not recognized!');
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
    perform a least squares fitting and 
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

